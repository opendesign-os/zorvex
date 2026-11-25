"""Deleting threads, together with the runs on them and their checkpoints.

A stateless run's thread is temporary: whoever ends the run deletes it, as the Agent Server's worker
does, so the thread goes even when the instance that created the run is gone by then.
"""

import asyncio
from collections.abc import Sequence

import structlog
from psycopg import Error as PsycopgError
from redis import RedisError
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.active_runs import active_runs
from zorvex_api.core.database import db_manager
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES
from zorvex_api.models.run_job import RunJob
from zorvex_api.services import streaming_service
from zorvex_api.services.base_broker import cancelled_error
from zorvex_api.services.run_status import Ending, finalize_run
from zorvex_api.services.webhook_delivery import job_for_run

logger = structlog.getLogger(__name__)

# Transient infra failures a background deletion logs instead of raising.
CLEANUP_ERRORS: tuple[type[BaseException], ...] = (RedisError, SQLAlchemyError, OSError, PsycopgError)


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
    except CLEANUP_ERRORS:
        logger.exception("Failed to delete thread", thread_id=thread_id)


async def delete_temporary_thread(job: RunJob) -> None:
    """Delete the thread of *job*, a run that has ended, when the run was a stateless one's."""
    if job.kwargs.temporary:
        await delete_thread_by_id(job.identity.thread_id, job.user.identity)
