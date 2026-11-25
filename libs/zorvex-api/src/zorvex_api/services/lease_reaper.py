"""Background task that recovers runs with expired worker leases.

Periodically scans the runs table for rows where
``status='running' AND lease_expires_at < now()``. It atomically either
returns them to ``pending`` or marks their retry budget exhausted, then
re-enqueues only retryable run IDs.
"""

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta

import structlog
from redis import RedisError
from sqlalchemy import select, update

from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.run_job import RunJob
from zorvex_api.observability.metrics import REAPER_RECOVERED_RUNS
from zorvex_api.services import job_queue
from zorvex_api.services.broker import broker_manager
from zorvex_api.services.executor import executor
from zorvex_api.services.run_queue import runnable
from zorvex_api.services.run_status import settle_threads
from zorvex_api.services.streaming_service import streaming_service
from zorvex_api.services.webhook_delivery import RunOutcome, job_for_webhook, report_run
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

_MAX_RETRIES_ERROR = {"error": "Error", "message": "Max retries exceeded after repeated worker failures"}


class LeaseReaper:
    """Recovers runs whose worker leases have expired."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Lease reaper started",
            interval_seconds=settings.worker.REAPER_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Lease reaper stopped")

    async def _loop(self) -> None:
        interval = settings.worker.REAPER_INTERVAL_SECONDS
        while self._running:
            await asyncio.sleep(interval)
            try:
                await self._reap()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in lease reaper")

    async def _reap(self) -> None:
        """Find crashed workers and stuck pending runs, recover them."""
        crashed, stuck_pending = await self._find_recoverable()

        if not crashed and not stuck_pending:
            return

        if crashed:
            logger.warning("Reaping crashed worker runs", count=len(crashed), run_ids=crashed)
            retryable, exhausted, freed_threads = await self._recover_crashed_runs(crashed)
            if exhausted:
                REAPER_RECOVERED_RUNS.labels(outcome="crashed_exhausted").inc(len(exhausted))
                for run_id in exhausted:
                    await _end_failed_run(run_id)
                for thread_id in freed_threads:
                    await executor.wake(thread_id)
            if retryable:
                pushed = await self._reenqueue(retryable)
                REAPER_RECOVERED_RUNS.labels(outcome="crashed_retried").inc(len(pushed))

        lost: list[str] = []
        if stuck_pending:
            lost = await self._missing_from_queue(stuck_pending)
            if lost:
                logger.warning("Re-enqueueing stuck pending runs", count=len(lost), run_ids=lost)
                pushed = await self._reenqueue(lost)
                REAPER_RECOVERED_RUNS.labels(outcome="stuck_pending").inc(len(pushed))

        logger.info(
            "Lease recovery complete",
            crashed_recovered=len(crashed),
            stuck_reenqueued=len(lost),
        )

    @staticmethod
    async def _find_recoverable() -> tuple[list[str], list[str]]:
        """Find two categories: crashed workers (expired lease) and stuck pending runs.

        Returns (crashed_run_ids, stuck_pending_run_ids) separately so retry
        budget is only charged to crashed runs, not stuck pending ones.
        """
        now = datetime.now(UTC)
        maker = get_session_maker()
        async with maker() as session:
            crashed_result = await session.execute(
                select(RunORM.run_id).where(
                    RunORM.status == "running",
                    RunORM.lease_expires_at.isnot(None),
                    RunORM.lease_expires_at < now,
                )
            )
            crashed = [row[0] for row in crashed_result.fetchall()]

            # Only runs whose turn has come: one waiting behind another on its thread is not stuck.
            stuck_result = await session.execute(
                select(RunORM.run_id).where(
                    RunORM.claimed_by.is_(None),
                    runnable(now),
                    RunORM.created_at < now - timedelta(seconds=settings.worker.STUCK_PENDING_THRESHOLD_SECONDS),
                )
            )
            stuck_pending = [row[0] for row in stuck_result.fetchall()]

        return crashed, stuck_pending

    @staticmethod
    async def _recover_crashed_runs(run_ids: list[str]) -> tuple[list[str], list[str], set[str]]:
        """Atomically classify expired runs and apply their next state.

        Returns the runs to retry, the runs failed for good, and the threads those leave free.
        """
        now = datetime.now(UTC)
        max_retries = settings.worker.BG_JOB_MAX_RETRIES
        retryable: list[str] = []
        exhausted: list[str] = []
        exhausted_threads_by_user: dict[str, set[str]] = {}
        failed_jobs: list[RunJob] = []

        maker = get_session_maker()
        async with maker() as session:
            locked_result = await session.execute(
                select(
                    RunORM.run_id,
                    RunORM.thread_id,
                    RunORM.user_id,
                    RunORM.execution_params,
                    RunORM.assistant_id,
                    RunORM.created_at,
                    RunORM.retries,
                )
                .where(
                    RunORM.run_id.in_(run_ids),
                    RunORM.status == "running",
                    RunORM.lease_expires_at < now,
                )
                .with_for_update(skip_locked=True)
            )

            for row in locked_result.fetchall():
                run_id, thread_id, user_id, params, assistant_id, created_at, retries = row
                retry_count = retries + 1

                values: dict[str, object] = {
                    "retries": retry_count,
                    "claimed_by": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
                is_exhausted = retry_count > max_retries
                if is_exhausted:
                    values.update(
                        status="error",
                        error=_MAX_RETRIES_ERROR,
                    )
                else:
                    values["status"] = "pending"

                update_result = await session.execute(
                    update(RunORM)
                    .where(
                        RunORM.run_id == run_id,
                        RunORM.user_id == user_id,
                        RunORM.status == "running",
                        RunORM.lease_expires_at < now,
                    )
                    .values(**values)
                    .returning(RunORM.run_id)
                )
                if update_result.scalar_one_or_none() is None:
                    continue

                if is_exhausted:
                    exhausted.append(run_id)
                    exhausted_threads_by_user.setdefault(user_id, set()).add(thread_id)
                    failed_job = job_for_webhook(
                        params, run_id=run_id, thread_id=thread_id, assistant_id=assistant_id, created_at=created_at
                    )
                    if failed_job is not None:
                        failed_jobs.append(failed_job)
                    logger.error(
                        "Run exceeded max retries, marking as permanently failed",
                        run_id=run_id,
                        retries=retry_count,
                        max_retries=max_retries,
                    )
                else:
                    retryable.append(run_id)
                    logger.info(
                        "Incrementing retry count",
                        run_id=run_id,
                        retry_count=retry_count,
                        max_retries=max_retries,
                    )

            for user_id, thread_ids in exhausted_threads_by_user.items():
                await settle_threads(
                    session,
                    thread_ids,
                    "error",
                    user_id=user_id,
                )
            await session.commit()
            # This commit made the exhausted runs terminal, so their reports come from here.
            for job in failed_jobs:
                outcome = RunOutcome(status="error", error=_MAX_RETRIES_ERROR)
                report_run(job, outcome, run_id=job.identity.run_id)

        freed = {thread_id for thread_ids in exhausted_threads_by_user.values() for thread_id in thread_ids}
        return retryable, exhausted, freed

    @staticmethod
    async def _missing_from_queue(run_ids: list[str]) -> list[str]:
        """Pending runs gone from the job queue; a queued one is only waiting for a worker."""
        try:
            queued = await job_queue.queued(run_ids)
        except RedisError:
            # Nothing can be pushed either; workers fall back to the Postgres poll.
            logger.warning("Redis unavailable; cannot check stuck pending runs against the queue", run_ids=run_ids)
            return []
        return [run_id for run_id in run_ids if run_id not in queued]

    @staticmethod
    async def _reenqueue(run_ids: list[str]) -> list[str]:
        """Queue the runs again, each at its priority and due time. Returns the IDs queued."""
        async with get_session_maker()() as session:
            rows = (
                await session.execute(
                    select(RunORM.run_id, RunORM.priority, RunORM.created_at).where(RunORM.run_id.in_(run_ids))
                )
            ).all()
        entries = {row.run_id: job_queue.score(row.priority, row.created_at) for row in rows}
        try:
            await job_queue.push(entries)
        except RedisError:
            logger.warning(
                "Redis unavailable during re-enqueue; workers will pick up via Postgres poll", run_ids=run_ids
            )
            return []
        logger.info("Re-enqueued recovered runs", run_ids=list(entries))
        return list(entries)


async def _end_failed_run(run_id: str) -> None:
    """End the stream of a run the reaper failed, and tell its waiters; best effort, as they also watch its status."""
    try:
        await streaming_service.signal_run_error(run_id, _MAX_RETRIES_ERROR)
        await broker_manager.publish_done(run_id)
    except RedisError:
        logger.warning("Could not end the stream of a failed run", run_id=run_id)


lease_reaper = LeaseReaper()
