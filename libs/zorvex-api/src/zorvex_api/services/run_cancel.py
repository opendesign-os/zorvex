"""How a cancelled run ends, as on the Agent Server: interrupted, or rolled back.

A cancel is recorded on the run (``cancel_action``, ``cancel_reason``) before
its executor is asked to stop (services/run_cleanup.py), so the executor, in
whichever process it runs, ends the run the way it was asked to. A rollback
deletes the run with the checkpoints it wrote, leaving the thread with the
state the runs before it made; the thread is then busy while another run is
active on it, else idle, as on the Agent Server.
"""

import structlog
from sqlalchemy import ColumnElement, delete, select

from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES
from zorvex_api.models.run_job import RunJob
from zorvex_api.services import checkpoints
from zorvex_api.services.base_broker import cancelled_error
from zorvex_api.services.run_status import Ending, finalize_run, settle_threads
from zorvex_api.services.webhook_delivery import RunOutcome, report_run

logger = structlog.getLogger(__name__)


async def cancel_of(run_id: str) -> tuple[str | None, str | None]:
    """The cancel recorded on the run: its action and reason; ``(None, None)`` for none, or no run."""
    async with get_session_maker()() as session:
        row = (
            await session.execute(select(RunORM.cancel_action, RunORM.cancel_reason).where(RunORM.run_id == run_id))
        ).one_or_none()
    return (row[0], row[1]) if row is not None else (None, None)


async def end_cancelled_run(
    run_id: str, thread_id: str, ending: Ending, *, user_id: str, webhook_job: RunJob | None
) -> tuple[bool, dict[str, str]]:
    """End a run its executor stopped on a cancel, as the cancel asked; returns whether it ended it, and the error.

    An interrupted run ends as *ending* has it; a rolled-back one is gone. False
    means another actor ended the run first. The error is what the run's stream
    ends with.
    """
    action, reason = await cancel_of(run_id)
    error = cancelled_error(action, reason)
    if action == "rollback":
        ended = await rollback_run(run_id, thread_id, user_id=user_id, reason=reason, webhook_job=webhook_job)
    else:
        ended = await finalize_run(run_id, thread_id, ending, user_id=user_id, webhook_job=webhook_job)
    return ended, error


async def rollback_run(
    run_id: str,
    thread_id: str,
    *where: ColumnElement[bool],
    user_id: str,
    reason: str | None,
    webhook_job: RunJob | None,
) -> bool:
    """Delete an active run and the checkpoints it wrote, settle its thread, and report the run.

    Returns False when the run is no longer active (one that finished first
    keeps its checkpoints) or fails a *where* condition, such as
    ``run_status.unowned``. The run's row stays locked while its checkpoints
    go, so nobody else ends it meanwhile.
    """
    conditions = [RunORM.run_id == run_id, RunORM.user_id == user_id, RunORM.status.in_(ACTIVE_RUN_STATUSES), *where]
    async with get_session_maker()() as session:
        if await session.scalar(select(RunORM.run_id).where(*conditions).with_for_update()) is None:
            await session.rollback()
            return False
        await checkpoints.delete_run(thread_id, run_id)
        await session.execute(delete(RunORM).where(RunORM.run_id == run_id))
        await settle_threads(session, [thread_id], "idle", user_id=user_id)
        await session.commit()
    report_run(webhook_job, RunOutcome(status="rollback", error=cancelled_error("rollback", reason)), run_id=run_id)
    logger.info("Rolled back run", run_id=run_id, thread_id=thread_id, reason=reason)
    return True
