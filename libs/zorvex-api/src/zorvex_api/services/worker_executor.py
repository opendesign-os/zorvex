"""Redis-backed executor with concurrent async execution and lease-based crash recovery.

Redis mode (REDIS_URI set). Each worker loop dequeues
run_ids from the Redis job queue by priority and spawns up to N_JOBS_PER_WORKER
concurrent asyncio tasks. Each task acquires a lease, executes the
graph with periodic heartbeats, and releases the lease on completion.
If a worker crashes, the lease expires and a background reaper
re-enqueues the run.
"""

import asyncio
import contextlib
import contextvars
import os
import socket
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import structlog
from pydantic import ValidationError
from redis import RedisError
from sqlalchemy import select, update

from zorvex_api.core.active_runs import active_runs, explicit_run_cancellations
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.core.redis_manager import redis_manager
from zorvex_api.models.enums import TERMINAL_RUN_STATUSES
from zorvex_api.models.run_job import RunJob
from zorvex_api.services import job_queue
from zorvex_api.services.base_executor import BaseExecutor
from zorvex_api.services.broker import broker_manager
from zorvex_api.services.redis_broker import RedisBrokerManager
from zorvex_api.services.run_cancel import end_cancelled_run
from zorvex_api.services.run_executor import (
    _TIMEOUT_ERROR,
    _lease_loss_cancellations,
    _shutdown_cancellations,
    _timeout_cancellations,
    execute_run,
)
from zorvex_api.services.run_queue import lock_line, next_on, runnable
from zorvex_api.services.run_status import Ending, finalize_run
from zorvex_api.services.webhook_delivery import job_for_run
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

# How often a wait re-checks a run that may have ended without a done message.
_DONE_RECHECK_SECONDS = 5.0


def _is_valid_run_id(value: str) -> bool:
    """Whether a queue entry is a run id (a UUID), not garbage pushed onto the queue."""
    try:
        UUID(value)
    except ValueError:
        return False
    return True


async def _cleanup_cancelled_execution(run_id: str, execution_task: asyncio.Task[None]) -> None:
    """Stop execution and end an explicitly cancelled run its execution did not get to end."""
    if not execution_task.done():
        execution_task.cancel()
    await asyncio.gather(execution_task, return_exceptions=True)

    if run_id not in explicit_run_cancellations:
        return

    try:
        identity = await _get_run_identity(run_id)
        if identity is None:
            return

        thread_id, user_id = identity
        await end_cancelled_run(
            run_id,
            thread_id,
            Ending("interrupted", "idle"),
            user_id=user_id,
            webhook_job=await _load_webhook_job(run_id),
        )
        await _wake(thread_id)
    except Exception:
        logger.exception("Failed to persist explicit run cancellation", run_id=run_id)


async def _await_cancellation_cleanup(cleanup_task: asyncio.Task[None]) -> None:
    """Let cleanup finish even if the owning task is cancelled repeatedly."""
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            if cleanup_task.cancelled():
                raise
    await cleanup_task


class WorkerExecutor(BaseExecutor):
    """Dispatches runs via the Redis job queue; workers take its head, by priority, within a semaphore."""

    def __init__(self) -> None:
        self._worker_tasks: list[asyncio.Task[None]] = []
        # Task -> run_id, mapped at creation: a task cancelled before it ever
        # runs has no active_runs entry, yet its run still needs the drain requeue.
        self._job_tasks: dict[asyncio.Task[None], str] = {}
        self._running = False
        self._instance_id = f"{socket.gethostname()}-{os.getpid()}"

    # ------------------------------------------------------------------
    # Submit (API side)
    # ------------------------------------------------------------------

    async def submit(self, job: RunJob) -> None:
        run_id = job.identity.run_id
        entry = {run_id: job_queue.score(job.priority, job.identity.created_at)}
        delay = (job.identity.created_at - datetime.now(UTC)).total_seconds()
        if delay > 0:
            # Queued once due. Should this process stop first, the lease reaper finds the run pending.
            task = asyncio.create_task(_enqueue_when_due(entry, delay))
            _delayed_enqueues.add(task)
            task.add_done_callback(_delayed_enqueues.discard)
            return
        await job_queue.push(entry)
        logger.info("Enqueued run_id to job queue", run_id=run_id, priority=job.priority)

    async def wake(self, thread_id: str) -> None:
        await _wake(thread_id)

    # ------------------------------------------------------------------
    # Wait for completion (API side)
    # ------------------------------------------------------------------

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        """Wait until the run is done, woken by its done message.

        The done key and the run's status are checked first, for a run that
        finished before the wait began, and again every few seconds, for one
        that ended without a message (the lease reaper failing it, a crash).
        """
        manager = cast(RedisBrokerManager, broker_manager)
        done = manager.watch_done(run_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while not await _is_done(manager, run_id):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(done.wait(), timeout=min(_DONE_RECHECK_SECONDS, remaining))
                if done.is_set():
                    return
        finally:
            manager.unwatch_done(run_id, done)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        count = settings.worker.WORKER_COUNT
        if count == 0:
            logger.warning(
                "WORKER_COUNT=0: no workers on this instance, runs will queue until another instance picks them up"
            )
        for idx in range(count):
            name = f"{self._instance_id}-worker-{idx}"
            task = asyncio.create_task(self._worker_loop(name))
            self._worker_tasks.append(task)

        max_concurrent = count * settings.worker.N_JOBS_PER_WORKER
        logger.info(
            "Worker executor started",
            worker_count=count,
            jobs_per_worker=settings.worker.N_JOBS_PER_WORKER,
            max_concurrent=max_concurrent,
            instance=self._instance_id,
        )

    async def stop(self) -> None:
        self._running = False
        drain_timeout = settings.worker.WORKER_DRAIN_TIMEOUT

        # Wait for in-flight job tasks to finish
        drained: list[str] = []
        if self._job_tasks:
            logger.info("Draining in-flight jobs", count=len(self._job_tasks))
            _, pending = await asyncio.wait(set(self._job_tasks), timeout=drain_timeout)
            if pending:
                # Requeue drain survivors instead of finalizing work that a
                # plain crash would recover.
                drained = [
                    run_id
                    for task, run_id in self._job_tasks.items()
                    if task in pending and run_id not in explicit_run_cancellations
                ]
                _shutdown_cancellations.update(drained)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        # Cancel worker loops before the requeue push: a loop still blocked on
        # the queue would steal the handed-off jobs back onto this dying instance.
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        if drained:
            await _requeue_drained_runs(drained)

        self._worker_tasks.clear()
        self._job_tasks.clear()
        logger.info("Worker executor stopped", instance=self._instance_id)

    # ------------------------------------------------------------------
    # Worker loop (dequeue + spawn concurrent tasks)
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_name: str) -> None:
        """Dequeue run_ids and spawn concurrent execution tasks.

        Each worker loop manages a semaphore that limits concurrent runs
        to N_JOBS_PER_WORKER. When all slots are busy, the loop blocks
        on semaphore.acquire until a slot frees up.
        """
        semaphore = asyncio.Semaphore(settings.worker.N_JOBS_PER_WORKER)
        logger.info(
            "Worker started",
            worker=worker_name,
            max_concurrent=settings.worker.N_JOBS_PER_WORKER,
        )

        while self._running:
            try:
                await semaphore.acquire()

                if not self._running:
                    semaphore.release()
                    break

                head = await self._dequeue()
                if head is None:
                    semaphore.release()
                    continue
                run_id, score = head

                if not _is_valid_run_id(run_id):
                    logger.warning("Invalid run_id dequeued, discarding", value=run_id[:64])
                    semaphore.release()
                    continue

                if not self._running:
                    # Dequeued while shutdown was already underway: hand it back
                    # rather than start untracked work on a terminating instance.
                    await _push_back({run_id: score})
                    semaphore.release()
                    break

                task = asyncio.create_task(self._execute_and_release(run_id, worker_name, semaphore))
                self._job_tasks[task] = run_id
                task.add_done_callback(lambda t: self._job_tasks.pop(t, None))

            except asyncio.CancelledError:
                break
            except Exception:
                semaphore.release()
                logger.exception("Unexpected error in worker loop", worker=worker_name)
                await asyncio.sleep(1.0)

        logger.info("Worker stopped", worker=worker_name)

    async def _execute_and_release(
        self,
        run_id: str,
        worker_name: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a run with lease + timeout, then release the semaphore slot."""
        # Register in active_runs so cancel-on-disconnect and explicit
        # cancel can find and cancel this specific job task.
        current_task = asyncio.current_task()
        if current_task is not None:
            active_runs[run_id] = current_task
        execution_task = asyncio.create_task(self._execute_with_lease(run_id, worker_name))
        try:
            done, _ = await asyncio.wait(
                {execution_task},
                timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
            )
            if execution_task in done:
                await execution_task
            else:
                logger.error(
                    "Job exceeded timeout, killing",
                    worker=worker_name,
                    run_id=run_id,
                    timeout_secs=settings.worker.BG_JOB_TIMEOUT_SECS,
                )
                _timeout_cancellations.add(run_id)
                execution_task.cancel()
                await asyncio.gather(execution_task, return_exceptions=True)

                identity = await _get_run_identity(run_id)
                if identity is not None:
                    thread_id, user_id = identity
                    error = {"error": "TimeoutError", "message": _TIMEOUT_ERROR}
                    await finalize_run(
                        run_id,
                        thread_id,
                        Ending("timeout", "error", error),
                        user_id=user_id,
                        webhook_job=await _load_webhook_job(run_id),
                    )
                await _release_lease(run_id, worker_name)
                if identity is not None:
                    await self.wake(identity[0])
        except asyncio.CancelledError:
            logger.info("Job task cancelled", worker=worker_name, run_id=run_id)
            cleanup_task = asyncio.create_task(_cleanup_cancelled_execution(run_id, execution_task))
            await _await_cancellation_cleanup(cleanup_task)
            raise
        except Exception:
            logger.exception("Unexpected error in job execution", run_id=run_id)
        finally:
            _timeout_cancellations.discard(run_id)
            _shutdown_cancellations.discard(run_id)
            explicit_run_cancellations.discard(run_id)
            active_runs.pop(run_id, None)
            semaphore.release()

    # ------------------------------------------------------------------
    # Job execution (lease + heartbeat)
    # ------------------------------------------------------------------

    async def _dequeue(self) -> tuple[str, float] | None:
        """The queue's head and its score. Any Redis error, timeout included, falls back to Postgres:
        the block may have popped a run whose reply never arrived."""
        try:
            return await job_queue.pop(settings.worker.QUEUE_WAIT_SECONDS)
        except RedisError as exc:
            logger.warning("Redis queue pop failed, falling back to Postgres poll", error=str(exc))
            await asyncio.sleep(settings.worker.POSTGRES_POLL_INTERVAL_SECONDS)
            return await self._poll_postgres()

    async def _execute_with_lease(self, run_id: str, worker_name: str) -> None:
        """Acquire lease, load job from DB, execute with heartbeat."""
        lease_acquired_at = datetime.now(UTC)
        job = await _acquire_and_load(run_id, worker_name)
        if job is None:
            logger.debug("Lease not acquired or job missing, skipping", run_id=run_id, worker=worker_name)
            return

        logger.info(
            "Worker picked up run",
            worker=worker_name,
            run_id=run_id,
            graph_id=job.identity.graph_id,
        )
        # Wrap execute_run in a task so the heartbeat can cancel it on
        # lease loss, preventing double execution by a second worker.
        job_task = asyncio.create_task(execute_run(job))
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(run_id, worker_name, job_task=job_task),
            context=contextvars.copy_context(),
        )

        try:
            await job_task
        except asyncio.CancelledError:
            logger.info("Worker job cancelled", worker=worker_name, run_id=run_id)
        except Exception:
            logger.exception("Worker job failed", worker=worker_name, run_id=run_id)
        finally:
            # Cancel both child tasks — job_task may still be running if
            # this coroutine was cancelled by wait_for timeout (CancelledError
            # is delivered to `await job_task`, but the Task itself is not
            # cancelled automatically).
            if not job_task.done():
                job_task.cancel()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(job_task, heartbeat_task, return_exceptions=True)
            await _release_lease(run_id, worker_name)
            await self.wake(job.identity.thread_id)

            elapsed = (datetime.now(UTC) - lease_acquired_at).total_seconds()
            logger.info(
                "Worker finished run",
                worker=worker_name,
                run_id=run_id,
                execution_seconds=round(elapsed, 2),
            )

    @staticmethod
    async def _poll_postgres() -> tuple[str, float] | None:
        """Pick the run the queue would, from Postgres: of those that may start now, by priority, then the longest waiting."""
        maker = get_session_maker()
        async with maker() as session:
            row = (
                await session.execute(
                    select(RunORM.run_id, RunORM.priority, RunORM.created_at)
                    .where(RunORM.claimed_by.is_(None), runnable(datetime.now(UTC)))
                    .order_by(RunORM.priority, RunORM.created_at)
                    .limit(1)
                )
            ).first()
        return None if row is None else (row.run_id, job_queue.score(row.priority, row.created_at))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _load_webhook_job(run_id: str) -> RunJob | None:
    """The run's job when it has a webhook, read before the wrapper finalizes it.

    Once the run is terminal, stateless cleanup may delete its row, so a later read could find nothing.
    """
    maker = get_session_maker()
    async with maker() as session:
        run = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
    return job_for_run(run) if run is not None else None


async def _get_run_identity(run_id: str) -> tuple[str, str] | None:
    """Look up the thread and tenant identity for a run."""
    maker = get_session_maker()
    async with maker() as session:
        result = await session.execute(select(RunORM.thread_id, RunORM.user_id).where(RunORM.run_id == run_id))
        row = result.one_or_none()
        if row is None:
            return None
        return row.thread_id, row.user_id


# ------------------------------------------------------------------
# Lease operations (module-level for reuse by LeaseReaper)
# ------------------------------------------------------------------


async def _acquire_and_load(run_id: str, worker_name: str) -> RunJob | None:
    """Acquire lease and load job in a single DB session.

    The lease UPDATE returns the claimed row, so the job is read in the same
    statement. A row whose execution_params cannot be rebuilt into a job is
    marked errored.
    """
    now = datetime.now(UTC)
    lease_until = now + timedelta(seconds=settings.worker.LEASE_DURATION_SECONDS)
    maker = get_session_maker()
    async with maker() as session:
        # With its thread locked, the check that no other run on it runs holds until this claim commits.
        await lock_line(session, run_id)
        run_orm = await session.scalar(
            update(RunORM)
            .where(RunORM.run_id == run_id, RunORM.claimed_by.is_(None), runnable(now))
            .values(claimed_by=worker_name, lease_expires_at=lease_until, status="running")
            .returning(RunORM)
        )
        if run_orm is None:
            # Claimed by another worker, no longer pending, deleted, or not its turn yet: a run
            # waiting in line is queued again when the run ahead of it ends (wake), or when due.
            await session.rollback()
            return None
        await session.commit()

        try:
            job = RunJob.from_run_orm(run_orm)
        except (KeyError, ValidationError):
            logger.warning("Run's execution_params do not describe a job, failing it", run_id=run_id)
            await session.execute(
                update(RunORM)
                .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
                .values(
                    claimed_by=None,
                    lease_expires_at=None,
                    status="error",
                    error={"error": "Error", "message": "Run's execution_params do not describe a job"},
                )
            )
            await session.commit()
            return None
        return job


async def _push_back(entries: Mapping[str, float]) -> None:
    """Best-effort queueing of pending runs at their scores: one dequeued but not started, or one whose turn has come."""
    try:
        await job_queue.push(entries)
    except RedisError:
        # Rows are still pending/unclaimed; the stuck-pending reaper recovers them.
        logger.warning("Could not queue pending runs", run_ids=list(entries))


# Strong refs to the waits of runs submitted with after_seconds, until they are queued.
_delayed_enqueues: set[asyncio.Task[None]] = set()


async def _enqueue_when_due(entries: Mapping[str, float], delay: float) -> None:
    await asyncio.sleep(delay)
    await _push_back(entries)


async def _wake(thread_id: str) -> None:
    """Queue the run next in line on *thread_id* if it may start now; best effort, as the reaper also finds it."""
    async with get_session_maker()() as session:
        row = await next_on(session, thread_id, datetime.now(UTC))
    if row is not None:
        await _push_back({row.run_id: job_queue.score(row.priority, row.created_at)})


async def _requeue_drained_runs(run_ids: list[str]) -> None:
    """Hand runs cancelled at the drain deadline back to the queue.

    Rows still 'running' were mid-execution with finalize skipped; rows still
    'pending' were dequeued but never claimed. Both must reach another instance.
    """
    maker = get_session_maker()
    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(RunORM.run_id.in_(run_ids), RunORM.status.in_(["running", "pending"]))
            .values(status="pending", claimed_by=None, lease_expires_at=None)
            .returning(RunORM.run_id, RunORM.priority, RunORM.created_at)
        )
        reset = {row.run_id: job_queue.score(row.priority, row.created_at) for row in result.fetchall()}
        await session.commit()
    reset_ids = list(reset)

    if not reset_ids:
        return

    logger.info("Requeueing drained runs for another instance", count=len(reset_ids), run_ids=reset_ids)
    try:
        await job_queue.push(reset)
    except RedisError:
        # Rows are already pending + unclaimed: the stuck-pending reaper or the
        # Postgres poll fallback on a surviving instance picks them up.
        logger.warning("Redis unavailable during drain requeue, reaper will recover", run_ids=reset_ids)


async def _release_lease(run_id: str, worker_name: str) -> None:
    """Clear lease fields after job completion, only if this worker still owns the lease."""
    maker = get_session_maker()
    async with maker() as session:
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
            .values(claimed_by=None, lease_expires_at=None)
        )
        await session.commit()


async def _heartbeat_loop(
    run_id: str,
    worker_name: str,
    *,
    job_task: asyncio.Task[None] | None = None,
) -> None:
    """Extend lease periodically while the job is running.

    If the lease is lost (another worker claimed the run), cancels
    ``job_task`` to prevent double execution.
    """
    interval = settings.worker.heartbeat_interval_seconds
    duration = settings.worker.LEASE_DURATION_SECONDS
    maker = get_session_maker()

    while True:
        await asyncio.sleep(interval)
        try:
            new_expiry = datetime.now(UTC) + timedelta(seconds=duration)
            async with maker() as session:
                renewed = await session.scalar(
                    update(RunORM)
                    .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
                    .values(lease_expires_at=new_expiry)
                    .returning(RunORM.run_id)
                )
                await session.commit()
            if renewed is None:
                logger.warning(
                    "Lease lost, cancelling job to prevent double execution",
                    run_id=run_id,
                    worker=worker_name,
                )
                if job_task is not None and not job_task.done():
                    _lease_loss_cancellations.add(run_id)
                    job_task.cancel()
                return
            logger.debug("Lease extended", run_id=run_id, worker=worker_name)
        except Exception:
            logger.warning("Heartbeat lease extension failed", run_id=run_id, worker=worker_name)


async def _is_done(manager: RedisBrokerManager, run_id: str) -> bool:
    """Whether the run left its done key, or its row says it has ended."""
    try:
        if await redis_manager.get_client().exists(manager.done_key(run_id)):
            return True
    except RedisError:
        pass
    return await _is_run_terminal(run_id)


async def _is_run_terminal(run_id: str) -> bool:
    """Check if a run has reached a terminal state in the DB."""
    maker = get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
        if run_orm is None:
            return True
        return run_orm.status in TERMINAL_RUN_STATUSES
