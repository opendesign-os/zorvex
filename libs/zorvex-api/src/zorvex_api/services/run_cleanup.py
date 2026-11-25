"""Stopping runs, and deleting threads together with the runs on them."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

import structlog
from psycopg import Error as PsycopgError
from redis import RedisError
from sqlalchemy import case, delete, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.active_runs import active_runs, executing_runs
from zorvex_api.core.database import db_manager
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models import Run
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES, RunCancellationAction
from zorvex_api.services import streaming_service
from zorvex_api.services.base_broker import cancelled_error
from zorvex_api.services.executor import executor
from zorvex_api.services.run_cancel import rollback_run
from zorvex_api.services.run_status import Ending, finalize_run, interrupt_unowned_run, unowned
from zorvex_api.services.webhook_delivery import job_for_run
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

# Strong refs so fire-and-forget deletions survive GC until done.
_background_tasks: set[asyncio.Task[None]] = set()

# Transient infra failures a background deletion logs instead of raising.
_CLEANUP_ERRORS: tuple[type[BaseException], ...] = (RedisError, SQLAlchemyError, OSError, PsycopgError)


# Why a run whose client left is cancelled, as the Agent Server words it.
_DISCONNECT_REASON = "client disconnected from the run stream (on_disconnect=cancel)"


async def request_run_interruption(
    session: AsyncSession,
    run_orm: RunORM,
    *,
    action: RunCancellationAction,
    reason: str,
) -> None:
    """Cancel an active run, as the Agent Server cancels: interrupt it, or roll it back.

    The cancel is recorded on the run first, where its worker finds it wherever
    it runs and whenever it gets there: a rollback outranks an interrupt, and
    the first reason given stands unless a rollback replaces it. A run this
    process executes, or one a live worker owns, is asked to stop, and its
    executor ends it: interrupted, or deleted with the checkpoints it wrote.
    One no executor owns (pending, or its worker gone) is ended here, and the
    next run on its thread may start.
    """
    if run_orm.status in TERMINAL_RUN_STATUSES:
        return
    recorded = await _record_cancel(session, run_orm.run_id, action, reason)
    if recorded is None:
        return  # it ended meanwhile
    kept_action, kept_reason = recorded
    error = cancelled_error(kept_action, kept_reason)
    if run_orm.run_id in executing_runs:
        await streaming_service.cancel_run(run_orm.run_id, end_error=error)
        return

    if kept_action == "rollback":
        ended = await rollback_run(
            run_orm.run_id,
            run_orm.thread_id,
            unowned(datetime.now(UTC)),
            user_id=run_orm.user_id,
            reason=kept_reason,
            webhook_job=job_for_run(run_orm),
        )
    else:
        ended = await interrupt_unowned_run(
            session,
            run_orm.run_id,
            run_orm.thread_id,
            user_id=run_orm.user_id,
            webhook_job=job_for_run(run_orm),
        )

    if ended:
        # A delayed worker can outlive an expired lease, and a local run may still wait
        # its turn: ask either to stop. The streams close here, as no executor owns the run.
        await streaming_service.cancel_run(run_orm.run_id, end_error=None)
        try:
            await streaming_service.end_run(run_orm.run_id, "interrupted", error)
        except (RedisError, OSError):
            # The database has already committed. A broker outage must not turn
            # the successful cancel into an API error.
            logger.exception("Failed to signal reconciled run cancel", run_id=run_orm.run_id)
        await executor.wake(run_orm.thread_id)
        return

    await session.refresh(run_orm)
    if run_orm.status in TERMINAL_RUN_STATUSES:
        return

    await streaming_service.cancel_run(run_orm.run_id, end_error=error)


async def cancel_on_disconnect(run_id: str) -> None:
    """Interrupt the run its client left, as ``on_disconnect="cancel"`` asks; a missing run is left alone.

    Best effort: an infrastructure failure is logged, as a stream's teardown or a wait's
    end must not fail on it, and the run's worker or the lease reaper settles the run.
    """
    try:
        async with get_session_maker()() as session:
            run_orm = await session.get(RunORM, run_id)
            if run_orm is not None:
                await request_run_interruption(session, run_orm, action="interrupt", reason=_DISCONNECT_REASON)
    except _CLEANUP_ERRORS:
        logger.exception("Failed to cancel run on client disconnect", run_id=run_id)


async def _record_cancel(
    session: AsyncSession, run_id: str, action: RunCancellationAction, reason: str | None
) -> tuple[str | None, str | None] | None:
    """Record the cancel on the active run and commit; returns the cancel it now carries, or None once it ended."""
    # A rollback replaces the reason; an interrupt keeps the one given first.
    kept_reason = (
        reason if action == "rollback" else case((RunORM.cancel_action.is_(None), reason), else_=RunORM.cancel_reason)
    )
    result = await session.execute(
        update(RunORM)
        .where(RunORM.run_id == run_id, RunORM.status.in_(ACTIVE_RUN_STATUSES))
        .values(
            cancel_action=case((RunORM.cancel_action == "rollback", "rollback"), else_=action),
            cancel_reason=kept_reason,
        )
        .returning(RunORM.cancel_action, RunORM.cancel_reason)
        .execution_options(synchronize_session=False)
    )
    row = result.one_or_none()
    await session.commit()
    if row is None:
        return None
    return row[0], row[1]


async def stop_run_for_delete(run: RunORM) -> None:
    """Finalize an active run as interrupted, then stop it, so its row can be deleted.

    Finalizing first still reports the run: a worker in another process would
    find the row gone and could neither finalize nor report it.
    """
    await finalize_run(
        run.run_id, run.thread_id, Ending("interrupted", "idle"), user_id=run.user_id, webhook_job=job_for_run(run)
    )
    await streaming_service.cancel_run(run.run_id, end_error=cancelled_error())
    task = active_runs.get(run.run_id)
    if task is not None and not task.done():
        task.cancel()
        await asyncio.wait({task})


async def delete_threads(session: AsyncSession, thread_ids: Sequence[str], *, user_id: str) -> int:
    """Delete the user's threads among *thread_ids*, stopping their active runs first; returns how many.

    Checkpoints go before the rows: when deleting them fails the threads
    survive and the delete can be retried, where the reverse order would
    orphan LangGraph checkpoint rows. Does not commit.
    """
    owned = list(
        (
            await session.scalars(
                select(ThreadORM.thread_id).where(ThreadORM.thread_id.in_(thread_ids), ThreadORM.user_id == user_id)
            )
        ).all()
    )
    if not owned:
        return 0

    active = await session.scalars(
        select(RunORM).where(RunORM.thread_id.in_(owned), RunORM.status.in_(ACTIVE_RUN_STATUSES))
    )
    for run in active.all():
        await stop_run_for_delete(run)

    await db_manager.get_checkpointer().aprune(owned, strategy="delete")
    await session.execute(delete(ThreadORM).where(ThreadORM.thread_id.in_(owned)))
    return len(owned)


async def delete_thread_by_id(thread_id: str, user_id: str) -> None:
    """Delete one thread in a session of its own, logging infrastructure failures instead of raising."""
    try:
        async with get_session_maker()() as session:
            await delete_threads(session, [thread_id], user_id=user_id)
            await session.commit()
    except _CLEANUP_ERRORS:
        logger.exception("Failed to delete thread", thread_id=thread_id)


async def _delete_thread_after_run(run: Run, user_id: str) -> None:
    try:
        await executor.wait_for_completion(run.run_id, timeout=settings.worker.BG_JOB_TIMEOUT_SECS)
    except TimeoutError:
        # The run outlived the job timeout, so its executor gives up on it too.
        pass
    except _CLEANUP_ERRORS:
        logger.exception("Error waiting for run before deleting its thread", run_id=run.run_id)
    await delete_thread_by_id(run.thread_id, user_id)


def delete_thread_after_run(run: Run, user_id: str) -> None:
    """Delete the run's thread, which *user_id* owns, in the background once the run has finished."""
    task = asyncio.create_task(_delete_thread_after_run(run, user_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
