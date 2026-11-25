"""Thread TTL: config resolution and expiry sweep.

Threads opt into a TTL via a ``thread_ttl`` row (server default on creation or
per-thread override). A background sweep, running whether or not the server
sets defaults, claims expired rows with ``FOR UPDATE SKIP LOCKED`` and applies
the row's strategy:

- ``delete``  — remove checkpoints then the thread row (cascades runs/crons).
- ``keep_latest`` — prune checkpoint history, keep the latest state, re-arm.

The sweep never cancels runs: threads with pending/running runs are skipped
and picked up on a later tick once their runs settle.
"""

from collections.abc import Collection, Sequence
from datetime import UTC, datetime, timedelta

import structlog
from psycopg import Error as PsycopgError
from sqlalchemy import ColumnElement, Select, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.config import ThreadTTLConfig, load_config
from zorvex_api.core.database import db_manager
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import ThreadTTL as ThreadTTLORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES, PruneStrategy
from zorvex_api.observability.metrics import THREAD_TTL_SWEPT
from zorvex_api.services.periodic import PeriodicTask

logger = structlog.getLogger(__name__)

# Rows claimed per transaction. Bounds lock-hold time and the window a user
# DELETE on a claimed thread blocks on the thread_ttl cascade.
_CLAIM_BATCH = 100


def _sweep_config() -> ThreadTTLConfig:
    """How the sweep runs: the server's ``checkpointer.ttl``, else the defaults."""
    return get_thread_ttl_config() or ThreadTTLConfig()


def get_thread_ttl_config() -> ThreadTTLConfig | None:
    """zorvex.json ``checkpointer.ttl``; None leaves a thread created without its own ttl to live on."""
    return load_config().checkpointer.ttl


async def _prune_checkpoint_history(thread_id: str) -> bool:
    """Keep only the latest checkpoint of each namespace; DeltaChannel threads stay whole."""
    try:
        await db_manager.get_checkpointer().aprune([thread_id], strategy="keep_latest")
    except NotImplementedError:
        # A DeltaChannel value is rebuilt from ancestor writes; pruning them empties it.
        logger.warning("keep_latest skipped: thread uses a DeltaChannel", thread_id=thread_id)
        return False
    return True


async def _apply_strategy(
    session: AsyncSession, *, thread_id: str, strategy: str, ttl_minutes: float, now: datetime
) -> str:
    """Apply one expired row's strategy inside the claim transaction; return the outcome label."""
    if strategy == "keep_latest":
        await _prune_checkpoint_history(thread_id)
        # Re-arm: keep_latest is periodic compaction, not a one-shot.
        await session.execute(
            update(ThreadTTLORM)
            .where(ThreadTTLORM.thread_id == thread_id)
            .values(expires_at=now + timedelta(minutes=ttl_minutes))
        )
        return "pruned"

    # Checkpoints first — ordering rationale from the delete_thread route: a
    # failure here leaves the thread row intact and retryable, never orphans.
    await db_manager.get_checkpointer().adelete_thread(thread_id)
    await session.execute(delete(ThreadORM).where(ThreadORM.thread_id == thread_id))
    return "deleted"


def _expired_claim_stmt(
    *,
    now: datetime,
    limit: int,
    exclude_ids: Collection[str] = (),
) -> Select[str, PruneStrategy, float]:
    """Claim query for expired thread_ttl rows, locking thread_ttl AND thread.

    ``skip_locked`` partitions work across instances. Threads with active runs are skipped, not cancelled.
    Locking the thread row too closes the check-then-act race with run
    creation: a run INSERT takes FOR KEY SHARE on the thread (FK), so it waits
    for the claim transaction instead of slipping in after the active-run
    check and being cascade-deleted mid-flight.
    """
    active_runs_exist = (
        select(RunORM.run_id)
        .where(
            RunORM.thread_id == ThreadTTLORM.thread_id,
            RunORM.status.in_(ACTIVE_RUN_STATUSES),
        )
        .exists()
    )
    stmt = (
        select(ThreadTTLORM.thread_id, ThreadTTLORM.strategy, ThreadTTLORM.ttl_minutes)
        .join(ThreadORM, ThreadORM.thread_id == ThreadTTLORM.thread_id)
        .where(
            ThreadTTLORM.expires_at <= now,
            ~active_runs_exist,
        )
    )
    if exclude_ids:
        # Rows that already failed this pass sit first in expiry order; without
        # the exclusion they'd be re-claimed every batch and starve later rows.
        stmt = stmt.where(ThreadTTLORM.thread_id.not_in(exclude_ids))
    return (
        stmt.order_by(ThreadTTLORM.expires_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True, of=(ThreadTTLORM, ThreadORM))
    )


async def _process_expired_batch(
    session: AsyncSession, stmt: Select[str, PruneStrategy, float], now: datetime
) -> tuple[int, int, int, list[str]]:
    """Claim and process one batch in a single transaction.

    Returns (claimed, deleted, pruned, failed_ids). The thread-row DELETE must
    run in the claim transaction: it cascades into the locked thread_ttl row,
    and a separate session would wait forever on a lock this transaction holds.
    """
    rows = (await session.execute(stmt)).all()
    if not rows:
        return 0, 0, 0, []

    deleted = 0
    pruned = 0
    failed_ids: list[str] = []
    for thread_id, strategy, ttl_minutes in rows:
        try:
            outcome = await _apply_strategy(
                session, thread_id=thread_id, strategy=strategy, ttl_minutes=ttl_minutes, now=now
            )
        except (PsycopgError, OSError):
            # Checkpointer-side failure on the other pool: this transaction is
            # untouched — skip the item, retry it on a later claim.
            THREAD_TTL_SWEPT.labels(outcome="error").inc()
            logger.exception("Thread TTL item failed", thread_id=thread_id, strategy=strategy)
            failed_ids.append(thread_id)
            continue
        if outcome == "deleted":
            deleted += 1
        else:
            pruned += 1
        THREAD_TTL_SWEPT.labels(outcome=outcome).inc()

    await session.commit()
    return len(rows), deleted, pruned, failed_ids


async def prune_threads_by_id(
    session: AsyncSession,
    *,
    thread_ids: Sequence[str],
    strategy: PruneStrategy,
    user_id: str,
    auth_filter: ColumnElement[bool],
) -> int:
    """Apply a prune strategy to the caller's named threads now; return how many were pruned.

    Ids the caller cannot see, and threads with active runs, are skipped, as in the TTL sweep.
    """
    active_runs_exist = (
        select(RunORM.run_id)
        .where(RunORM.thread_id == ThreadORM.thread_id, RunORM.status.in_(ACTIVE_RUN_STATUSES))
        .exists()
    )
    stmt = select(ThreadORM.thread_id).where(
        ThreadORM.thread_id.in_(thread_ids), ThreadORM.user_id == user_id, ~active_runs_exist, auth_filter
    )
    # SKIP LOCKED: a thread mid run-creation holds a key-share lock; waiting on it would
    # recheck active runs against a stale snapshot and cascade-delete the new run.
    owned = list((await session.scalars(stmt.with_for_update(skip_locked=True, of=ThreadORM))).all())

    if strategy == "delete":
        await db_manager.get_checkpointer().aprune(owned, strategy="delete")
        await session.execute(delete(ThreadORM).where(ThreadORM.thread_id.in_(owned)))
        await session.commit()
        return len(owned)

    pruned = 0
    for thread_id in owned:
        pruned += await _prune_checkpoint_history(thread_id)
    await session.commit()
    return pruned


class ThreadTTLSweeper(PeriodicTask):
    """Deletes or compacts expired threads."""

    name = "thread TTL sweeper"

    def interval(self) -> float:
        return _sweep_config().sweep_interval_minutes * 60

    async def tick(self) -> None:
        """Sweep up to sweep_limit expired threads in sub-batched transactions."""
        config = _sweep_config()
        maker = get_session_maker()
        claimed_total = 0
        deleted_total = 0
        pruned_total = 0
        failed: set[str] = set()
        while claimed_total < config.sweep_limit:
            now = datetime.now(UTC)
            limit = min(_CLAIM_BATCH, config.sweep_limit - claimed_total)
            stmt = _expired_claim_stmt(now=now, limit=limit, exclude_ids=failed)
            async with maker() as session:
                claimed, deleted, pruned, failed_ids = await _process_expired_batch(session, stmt, now)
            if claimed == 0:
                break
            claimed_total += claimed
            deleted_total += deleted
            pruned_total += pruned
            failed.update(failed_ids)

        if claimed_total:
            logger.info(
                "Thread TTL sweep completed",
                claimed=claimed_total,
                deleted=deleted_total,
                pruned=pruned_total,
            )


thread_ttl_sweeper = ThreadTTLSweeper()
