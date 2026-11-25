"""Stopping runs: cancelling one, wherever it executes, as the Agent Server cancels."""

import structlog
from redis import RedisError
from sqlalchemy import case, update
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.active_runs import executing_runs
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES, RunCancellationAction
from zorvex_api.models.run_job import RunJob
from zorvex_api.services import streaming_service
from zorvex_api.services.base_broker import cancelled_error
from zorvex_api.services.executor import executor
from zorvex_api.services.run_cancel import rollback_run
from zorvex_api.services.run_status import interrupt_unowned_run, unowned
from zorvex_api.services.thread_deletion import CLEANUP_ERRORS, delete_temporary_thread
from zorvex_api.services.webhook_delivery import job_for_run

logger = structlog.getLogger(__name__)

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
            unowned(),
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
        await delete_temporary_thread(RunJob.from_run_orm(run_orm))
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
    except CLEANUP_ERRORS:
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
