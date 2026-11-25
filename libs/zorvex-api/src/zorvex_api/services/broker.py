"""The in-memory broker, for one process, and the process's broker manager: this one, or Redis's with REDIS_URI."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from zorvex_api.core.active_runs import active_runs
from zorvex_api.services.base_broker import (
    END_EVENT_GRACE_SECONDS,
    REPLAY_RETENTION_SECONDS,
    BaseBrokerManager,
    BaseRunBroker,
    event_id,
    is_end,
)
from zorvex_api.services.periodic import PeriodicTask
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

# Sweep often enough that an expired broker is freed close to its deadline,
# the short one of a run without stream_resumable included.
_SWEEP_INTERVAL_SECONDS = 10


class RunBroker(BaseRunBroker):
    """One run's events in memory, with a queue per follower.

    Each ``aiter()`` gets its own queue fed by ``put``, so several followers of
    one run (the v2 SDK's event stream and its lifecycle watcher, say) each get
    every event, as on Redis pub/sub.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.finished = asyncio.Event()
        self._events: list[tuple[str, Any]] = []
        self._followers: set[asyncio.Queue[tuple[str, Any]]] = set()
        self._finished_at: float | None = None
        # Seconds the events stay after the run finished (BrokerManager.retain).
        self.retention: float = REPLAY_RETENTION_SECONDS

    async def put(self, event_id: str, payload: Any) -> None:
        if self.finished.is_set():
            logger.warning("Dropped an event put after the run's end", run_id=self.run_id, event_id=event_id)
            return
        self._events.append((event_id, payload))
        for queue in list(self._followers):
            queue.put_nowait((event_id, payload))
        if is_end(payload):
            self.mark_finished()

    async def aiter(self) -> AsyncGenerator[tuple[str, Any]]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        # Following first, then the stored events: one put in between is in both, never in neither.
        self._followers.add(queue)
        try:
            for stored in list(self._events):
                yield stored
                if is_end(stored[1]):
                    return
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    if self.finished.is_set() and queue.empty():
                        return
                    continue
                yield item
                if is_end(item[1]):
                    return
        finally:
            self._followers.discard(queue)

    async def stored(self) -> list[tuple[str, Any]]:
        return list(self._events)

    def mark_finished(self) -> None:
        if not self.finished.is_set():
            self._finished_at = asyncio.get_running_loop().time()
            self.finished.set()

    def is_finished(self) -> bool:
        return self.finished.is_set()

    def expired(self) -> bool:
        """Whether the run finished longer ago than its events are kept, and nobody follows it."""
        if self._finished_at is None or any(not queue.empty() for queue in self._followers):
            return False
        return asyncio.get_running_loop().time() - self._finished_at > self.retention


class BrokerManager(BaseBrokerManager):
    """The runs' brokers in this process, swept once their events are no longer kept."""

    def __init__(self) -> None:
        self._brokers: dict[str, RunBroker] = {}
        self._sequences: dict[str, int] = {}
        self._sweeper = _Sweeper(self)

    def get_or_create_broker(self, run_id: str) -> RunBroker:
        if run_id not in self._brokers:
            self._brokers[run_id] = RunBroker(run_id)
        return self._brokers[run_id]

    def cleanup_broker(self, run_id: str) -> None:
        # The events stay until the sweep: followers still read them.
        if broker := self._brokers.get(run_id):
            broker.mark_finished()

    def sweep(self) -> None:
        """Drop the brokers whose events are no longer kept."""
        for run_id in [run_id for run_id, broker in self._brokers.items() if broker.expired()]:
            del self._brokers[run_id]
            self._sequences.pop(run_id, None)
            logger.debug("Dropped an expired broker", run_id=run_id)

    async def start(self) -> None:
        await self._sweeper.start()

    async def stop(self) -> None:
        await self._sweeper.stop()

    async def request_cancel(self, run_id: str, *, end_error: dict[str, str] | None) -> None:
        task = active_runs.get(run_id)
        if task is None or task.done():
            return
        logger.info("Cancelling the run's task", run_id=run_id)
        task.cancel()
        if end_error is not None:
            await self.end_stream(run_id, "interrupted", end_error)

    async def publish_done(self, run_id: str) -> None:
        """Nothing to tell: in-process waiters watch the run's task."""

    async def events(self, run_id: str) -> list[tuple[str, Any]]:
        broker = self._brokers.get(run_id)
        return await broker.stored() if broker is not None else []

    async def retain(self, run_id: str, seconds: float) -> None:
        if broker := self._brokers.get(run_id):
            broker.retention = seconds

    async def allocate_event_id(self, run_id: str) -> str:
        sequence = self._sequences.get(run_id, 0) + 1
        self._sequences[run_id] = sequence
        return event_id(run_id, sequence)


class _Sweeper(PeriodicTask):
    name = "broker sweeper"

    def __init__(self, manager: BrokerManager) -> None:
        super().__init__()
        self._manager = manager

    def interval(self) -> float:
        return _SWEEP_INTERVAL_SECONDS

    async def tick(self) -> None:
        self._manager.sweep()


def _create_broker_manager() -> BaseBrokerManager:
    if settings.redis.enabled:
        # Imported only with Redis on.
        from zorvex_api.services.redis_broker import RedisBrokerManager

        logger.info("Using Redis broker for SSE streaming")
        return RedisBrokerManager()
    logger.info("Using in-memory broker for SSE streaming")
    return BrokerManager()


broker_manager: BaseBrokerManager = _create_broker_manager()


async def live_events(run_id: str, *, ended_at: datetime | None = None) -> AsyncGenerator[tuple[str, Any]]:
    """The run's events, stored then live, up to its end.

    For a run already recorded as ended (*ended_at*) the end is only waited for
    until ``END_EVENT_GRACE_SECONDS`` past that time: one whose end does not
    come, because its events are no longer kept, stops there instead of waiting
    for ever. No broker is registered for a run past that deadline.
    """
    deadline = None if ended_at is None else ended_at + timedelta(seconds=END_EVENT_GRACE_SECONDS)
    if deadline is not None and deadline <= datetime.now(UTC):
        return
    events = broker_manager.get_or_create_broker(run_id).aiter()
    try:
        while True:
            if deadline is None:
                item = await anext(events, None)
            else:
                remaining = (deadline - datetime.now(UTC)).total_seconds()
                if remaining <= 0:
                    return
                try:
                    # The timeout covers this wait alone, never the caller's work between items.
                    item = await asyncio.wait_for(anext(events, None), remaining)
                except TimeoutError:
                    return
            if item is None:
                return
            yield item
    finally:
        await events.aclose()
