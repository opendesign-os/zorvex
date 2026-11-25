"""Event broker for managing run-specific event queues.

The broker handles both live event broadcast (via queue) and replay storage
(via replay buffer) for SSE reconnection support.
"""

import asyncio
import contextlib
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
)
from zorvex_api.settings import settings
from zorvex_api.utils import generate_event_id

logger = structlog.getLogger(__name__)

# Sweep often enough that an expired broker is freed close to its deadline,
# the short one of a run without stream_resumable included.
_SWEEP_INTERVAL_SECONDS = 10


class RunBroker(BaseRunBroker):
    """In-memory broker with per-subscriber fan-out + replay buffer.

    Each ``aiter()`` caller gets its own queue fed by ``put`` — so multiple
    concurrent SSE consumers on one run (e.g. the v2 SDK's main event stream and
    its separate lifecycle-watcher) each receive every event, matching the Redis
    pub/sub backend. The replay buffer keeps resumable events for reconnect.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.finished = asyncio.Event()
        self._replay_buffer: list[tuple[str, Any]] = []
        self._subscribers: set[asyncio.Queue[tuple[str, Any]]] = set()
        self._finished_at: float | None = None
        # Seconds the events stay after the run finished (BrokerManager.retain).
        self.retention: float = REPLAY_RETENTION_SECONDS

    async def put(self, event_id: str, payload: Any) -> None:
        if self.finished.is_set():
            logger.warning(f"Attempted to put event {event_id} into finished broker for run {self.run_id}")
            return

        self._replay_buffer.append((event_id, payload))

        for queue in list(self._subscribers):
            queue.put_nowait((event_id, payload))

        # Check if this is an end event
        if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
            self.mark_finished()

    async def aiter(self) -> AsyncGenerator[tuple[str, Any]]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        # Register, then replay the buffer into this iterator: any event put
        # between a caller's separate replay() and this registration is still in
        # the buffer, so nothing is dropped. Callers dedup the overlap by event_id.
        self._subscribers.add(queue)
        backlog = list(self._replay_buffer)
        try:
            for event_id, payload in backlog:
                yield event_id, payload
                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    return
            while True:
                try:
                    event_id, payload = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    if self.finished.is_set() and queue.empty():
                        break
                    continue
                yield event_id, payload
                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    break
        finally:
            self._subscribers.discard(queue)

    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        if not self._replay_buffer:
            return []

        if last_event_id is None:
            return list(self._replay_buffer)

        # Find the index after last_event_id
        for i, (eid, _) in enumerate(self._replay_buffer):
            if eid == last_event_id:
                return list(self._replay_buffer[i + 1 :])

        # last_event_id not found — return all
        return list(self._replay_buffer)

    def mark_finished(self) -> None:
        if self.finished.is_set():
            return
        self._finished_at = asyncio.get_running_loop().time()
        self.finished.set()
        logger.debug(f"Broker for run {self.run_id} marked as finished")

    def is_finished(self) -> bool:
        return self.finished.is_set()

    def is_empty(self) -> bool:
        return all(queue.empty() for queue in self._subscribers)

    def get_finished_age(self) -> float | None:
        """Seconds since the run finished, or None while it is still running."""
        if self._finished_at is None:
            return None
        return asyncio.get_running_loop().time() - self._finished_at


class BrokerManager(BaseBrokerManager):
    """Manages multiple RunBroker instances with periodic cleanup."""

    def __init__(self) -> None:
        self._brokers: dict[str, RunBroker] = {}
        self._event_counters: dict[str, int] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    def get_or_create_broker(self, run_id: str) -> RunBroker:
        if run_id not in self._brokers:
            self._brokers[run_id] = RunBroker(run_id)
            logger.debug(f"Created new broker for run {run_id}")
        return self._brokers[run_id]

    def get_broker(self, run_id: str) -> RunBroker | None:
        return self._brokers.get(run_id)

    def cleanup_broker(self, run_id: str) -> None:
        if run_id in self._brokers:
            self._brokers[run_id].mark_finished()
            logger.debug(f"Marked broker for run {run_id} for cleanup")

    def remove_broker(self, run_id: str) -> None:
        if run_id in self._brokers:
            self._brokers[run_id].mark_finished()
            del self._brokers[run_id]
            self._event_counters.pop(run_id, None)
            logger.debug(f"Removed broker for run {run_id}")

    async def start(self) -> None:
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_brokers())

    async def stop(self) -> None:
        """Stop the periodic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def request_cancel(
        self,
        run_id: str,
        *,
        emit_end_event: bool = True,
        error: dict[str, str] | None = None,
    ) -> None:
        """Cancel a run locally by cancelling its asyncio task."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return

        logger.info(f"Cancelling run {run_id} (local task)")
        task.cancel()
        if emit_end_event:
            await self.end_cancelled(run_id, error)

    async def publish_done(self, run_id: str) -> None:
        """Nothing to tell: in-process waiters watch the run's task."""

    async def events(self, run_id: str) -> list[tuple[str, Any]]:
        broker = self._brokers.get(run_id)
        return await broker.replay(None) if broker is not None else []

    async def retain(self, run_id: str, seconds: float) -> None:
        if broker := self._brokers.get(run_id):
            broker.retention = seconds

    async def allocate_event_id(self, run_id: str) -> str:
        """Allocate the next event ID using an in-memory counter."""
        seq = self._event_counters.get(run_id, 0) + 1
        self._event_counters[run_id] = seq
        return generate_event_id(run_id, seq)

    def sweep_expired_brokers(self) -> list[str]:
        """Drop finished brokers past their replay window. Returns the removed run ids."""
        expired = [
            run_id
            for run_id, broker in self._brokers.items()
            if broker.is_empty()
            and (finished_age := broker.get_finished_age()) is not None
            and finished_age > broker.retention
        ]
        for run_id in expired:
            self.remove_broker(run_id)
            logger.info(f"Cleaned up expired broker for run {run_id}")
        return expired

    async def _cleanup_old_brokers(self) -> None:
        while True:
            try:
                await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)
                self.sweep_expired_brokers()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in broker cleanup task")


def _create_broker_manager() -> BaseBrokerManager:
    """Select broker backend based on settings.

    Returns RedisBrokerManager when REDIS_URI is set,
    otherwise the default in-memory BrokerManager.
    """
    if settings.redis.enabled:
        # Conditional import: redis is only required when the Redis broker is enabled
        from zorvex_api.services.redis_broker import RedisBrokerManager

        logger.info("Using Redis broker for SSE streaming")
        return RedisBrokerManager()
    logger.info("Using in-memory broker for SSE streaming")
    return BrokerManager()


# Global broker manager instance
broker_manager: BaseBrokerManager = _create_broker_manager()


async def live_events(run_id: str, *, ended_at: datetime | None = None) -> AsyncGenerator[tuple[str, Any]]:
    """The run's events as they come, up to its end.

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
