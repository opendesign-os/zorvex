"""Work the server repeats in the background while it runs: the cron scheduler, the thread TTL sweep, the lease reaper."""

import asyncio
import contextlib

import structlog

logger = structlog.getLogger(__name__)


class PeriodicTask:
    """Calls ``tick`` every ``interval()`` seconds, from ``start()`` until ``stop()``.

    The first tick runs at once, so what fell due while the server was down is
    handled without waiting. A tick that returns True is followed by the next
    at once; one that fails is logged and waited out like any other, so an
    outage cannot spin the loop.
    """

    name = "periodic task"

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None

    def interval(self) -> float:
        """Seconds between ticks."""
        raise NotImplementedError

    async def tick(self) -> bool | None:
        """One pass; True to start the next one at once."""
        raise NotImplementedError

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name=self.name)
        logger.info("Started", task=self.name, interval_seconds=self.interval())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        logger.info("Stopped", task=self.name)

    async def _loop(self) -> None:
        while True:
            again = False
            try:
                again = bool(await self.tick())
            except Exception:
                logger.exception("Tick failed", task=self.name)
            if not again:
                await asyncio.sleep(self.interval())
