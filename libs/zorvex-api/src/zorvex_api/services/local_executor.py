"""In-process executor using asyncio tasks.

Used without Redis (REDIS_URI unset). Runs execute as background coroutines
in the same event loop as the API server. As on the Agent Server, a thread's
runs take turns: each waits until it is due (``created_at``) and then for the
runs ahead of it on its thread.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from zorvex_api.core.active_runs import active_runs
from zorvex_api.models.run_job import RunJob
from zorvex_api.services.base_executor import BaseExecutor

logger = structlog.getLogger(__name__)


@dataclass
class _Line:
    """The runs of one thread waiting or running here; the lock serves them in turn."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    runs: int = 0


class LocalExecutor(BaseExecutor):
    """Runs graphs as local asyncio tasks (single-instance dev mode)."""

    def __init__(self) -> None:
        self._lines: dict[str, _Line] = {}

    async def submit(self, job: RunJob) -> None:
        task = asyncio.create_task(self._run_in_line(job))
        active_runs[job.identity.run_id] = task
        logger.info("Submitted run to local executor", run_id=job.identity.run_id, task_id=id(task))

    async def _run_in_line(self, job: RunJob) -> None:
        # Deferred import: run_executor imports services that reference
        # the executor singleton, creating a circular chain at module level.
        from zorvex_api.services.run_executor import execute_run

        run_id, thread_id = job.identity.run_id, job.identity.thread_id
        try:
            delay = (job.identity.created_at - datetime.now(UTC)).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
            line = self._lines.setdefault(thread_id, _Line())
            line.runs += 1
            try:
                async with line.lock:
                    await execute_run(job)
            finally:
                line.runs -= 1
                if not line.runs:
                    self._lines.pop(thread_id, None)
        finally:
            # A run cancelled while it waited never reached execute_run, which pops it otherwise.
            if active_runs.get(run_id) is asyncio.current_task():
                active_runs.pop(run_id, None)

    async def wake(self, thread_id: str) -> None:
        """Nothing to hand over: a run waiting here takes its turn as the lock of its line frees."""

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        """Raises TimeoutError past *timeout*, so callers can tell a slow run from a finished one."""
        task = active_runs.get(run_id)
        if task is None:
            return
        # Shielded: the run outlives a caller that stops waiting. Its own failure
        # is recorded by execute_run, so only cancellation reaches here.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def start(self) -> None:
        logger.info("Local executor started (in-process asyncio tasks)")

    async def stop(self) -> None:
        tasks_to_cancel = [task for task in active_runs.values() if not task.done()]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            logger.info("Draining cancelled tasks", count=len(tasks_to_cancel))
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        logger.info("Local executor stopped")
