"""Guarded run and thread status transitions, shared by the API and the executors."""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from sqlalchemy import ColumnElement, CursorResult, exists, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES, RunStatus, ThreadStatus
from zorvex_api.models.run_job import RunJob
from zorvex_api.services.webhook_delivery import RunOutcome, report_run

logger = structlog.getLogger(__name__)


async def start_run(run_id: str, *, user_id: str) -> bool:
    """Move an active run to running without reviving a terminal run."""
    maker = get_session_maker()
    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.user_id == user_id,
                RunORM.status.in_(ACTIVE_RUN_STATUSES),
            )
            .values(status="running", updated_at=datetime.now(UTC))
            .returning(RunORM.run_id)
        )
        started = result.scalar_one_or_none() is not None
        if started:
            await session.commit()
        else:
            await session.rollback()
        return started


async def settle_threads(
    session: AsyncSession,
    thread_ids: Collection[str],
    status: ThreadStatus,
    *,
    user_id: str,
) -> None:
    """A run on each thread ended: its state changed, and its status becomes *status* unless another run is active.

    Does not commit so callers can keep the run and thread transitions in
    one transaction.
    """
    if not thread_ids:
        return
    now = datetime.now(UTC)
    await session.execute(
        update(ThreadORM)
        .where(ThreadORM.thread_id.in_(thread_ids), ThreadORM.user_id == user_id)
        .values(state_updated_at=now)
    )

    active_run_exists = exists(
        select(RunORM.run_id)
        .where(
            RunORM.thread_id == ThreadORM.thread_id,
            RunORM.user_id == user_id,
            RunORM.status.in_(ACTIVE_RUN_STATUSES),
        )
        .correlate(ThreadORM)
    )
    await session.execute(
        update(ThreadORM)
        .where(
            ThreadORM.thread_id.in_(thread_ids),
            ThreadORM.user_id == user_id,
            ~active_run_exists,
        )
        .values(status=status, updated_at=datetime.now(UTC))
    )


def unowned(now: datetime) -> ColumnElement[bool]:
    """A run no live worker holds: never claimed, or its lease run out before *now*."""
    return or_(RunORM.claimed_by.is_(None), RunORM.lease_expires_at < now)


async def interrupt_unowned_run(
    session: AsyncSession,
    run_id: str,
    thread_id: str,
    *,
    user_id: str,
    webhook_job: RunJob | None,
) -> bool:
    """Interrupt an active run only when it has no live database owner, and report it.

    The ownership predicate is checked in the UPDATE so a worker that renews
    or claims the run concurrently cannot be overwritten by the API process.
    If a live worker merely missed its lease, the caller asks it to stop through
    the broker, and guarded finalization rejects any late worker write.
    """
    now = datetime.now(UTC)
    result = cast(
        CursorResult,
        await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user_id,
                RunORM.status.in_(ACTIVE_RUN_STATUSES),
                unowned(now),
            )
            .values(
                status="interrupted",
                claimed_by=None,
                lease_expires_at=None,
                updated_at=now,
            )
            .returning(RunORM.run_id)
        ),
    )
    if result.scalar_one_or_none() is None:
        return False

    await settle_threads(session, [thread_id], "idle", user_id=user_id)
    await session.commit()
    report_run(webhook_job, RunOutcome(status="interrupted"), run_id=run_id)
    logger.info("Interrupted unowned run", run_id=run_id, thread_id=thread_id)
    return True


@dataclass(frozen=True)
class Ending:
    """How a run ended, and where it left its thread.

    *error* is the run's ``error`` event (``streaming_service.error_event``), kept
    for a wait or join that comes after the run's events have expired; *state*,
    the thread's ``values`` and ``interrupts`` where the run stopped, goes on the
    thread's row (None while unknown). A thread with another active run stays
    busy whatever *thread_status* says.
    """

    status: RunStatus
    thread_status: ThreadStatus
    error: dict[str, str] | None = None
    state: dict[str, Any] | None = None


async def finalize_run(
    run_id: str, thread_id: str, ending: Ending, *, user_id: str, webhook_job: RunJob | None
) -> bool:
    """Conditionally update run and thread status in one transaction, then report the run.

    Returns false when another actor has already made the run terminal. This
    prevents an expired worker from overwriting a reconciled cancellation, and
    leaves the run's webhook to that actor, so every run is reported once.
    """
    maker = get_session_maker()
    status = ending.status
    run_values: dict[str, Any] = {"status": status, "updated_at": datetime.now(UTC)}
    if ending.error is not None:
        run_values["error"] = ending.error

    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.user_id == user_id,
                RunORM.status.in_(ACTIVE_RUN_STATUSES),
            )
            .values(**run_values)
            .returning(RunORM.run_id)
        )
        if result.scalar_one_or_none() is None:
            await session.rollback()
            logger.info("Skipped finalizing terminal run", run_id=run_id, status=status)
            return False
        if ending.state is not None:
            await session.execute(update(ThreadORM).where(ThreadORM.thread_id == thread_id).values(**ending.state))
        await settle_threads(session, [thread_id], ending.thread_status, user_id=user_id)
        await session.commit()
        report_run(webhook_job, RunOutcome(status=status, error=ending.error), run_id=run_id)

    logger.info("Finalized run", run_id=run_id, status=status, thread_status=ending.thread_status)
    return True
