"""Abstract base classes for the broker system"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

# How long a finished run's events stay replayable when it was created with
# stream_resumable, both backends. Reconnects happen in seconds.
REPLAY_RETENTION_SECONDS = 600
# How long a finished run created without stream_resumable keeps its events: for
# those already following it (a wait or join reading its result, a thread stream
# catching up on it), not for a client that comes back later. The Agent Server
# keeps none of them.
UNRESUMABLE_RETENTION_SECONDS = 30
# How soon after a run is recorded as ended its end event is stored. A stream
# that finds no end for a run ended longer ago stops waiting for one: its events
# are no longer kept.
END_EVENT_GRACE_SECONDS = 10


def is_end(payload: Any) -> bool:
    """Whether a broker event is a run's ``end``."""
    return isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end"


def cancelled_error(action: str | None = None, reason: str | None = None) -> dict[str, str]:
    """The error a cancelled run's stream carries before its end, as the Agent Server words it.

    A wait or join on the run answers with it: ``UserRollback`` for a run rolled
    back, else ``UserInterrupt``, its message why the run was cancelled.
    """
    if action == "rollback":
        return {"error": "UserRollback", "message": reason or "User requested rollback of the run"}
    return {"error": "UserInterrupt", "message": reason or "User interrupted the run"}


class BaseRunBroker(ABC):
    """Abstract base class for a run-specific event broker.

    Handles both live event broadcast and replay storage for SSE reconnection.
    """

    @abstractmethod
    async def put(self, event_id: str, payload: Any) -> None:
        """Store an event for replay on reconnect and publish it to live subscribers.

        Args:
            event_id: Unique event identifier (format: {run_id}_event_{seq}).
            payload: The event data (typically a tuple like ("values", {...})).
        """

    @abstractmethod
    def aiter(self) -> AsyncGenerator[tuple[str, Any]]:
        """Async generator yielding (event_id, payload) pairs; a caller that stops early closes it."""
        ...

    @abstractmethod
    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        """Return stored events for replay on reconnect.

        Args:
            last_event_id: If provided, return events after this ID.
                           If None, return all stored events.

        Returns:
            List of (event_id, payload) tuples in order.
        """

    @abstractmethod
    def mark_finished(self) -> None:
        """Mark this broker as finished."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Check if this broker is finished."""


class BaseBrokerManager(ABC):
    """Abstract base class for managing multiple RunBroker instances"""

    @abstractmethod
    def get_or_create_broker(self, run_id: str) -> BaseRunBroker:
        """Get or create a broker for a run"""

    @abstractmethod
    def get_broker(self, run_id: str) -> BaseRunBroker | None:
        """Get an existing broker or None"""

    @abstractmethod
    def cleanup_broker(self, run_id: str) -> None:
        """Clean up a broker for a run"""

    @abstractmethod
    def remove_broker(self, run_id: str) -> None:
        """Remove a broker completely"""

    @abstractmethod
    async def start(self) -> None:
        """Start background tasks (cleanup loop, cancel listener, etc.)"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop background tasks and release resources"""

    @abstractmethod
    async def request_cancel(
        self,
        run_id: str,
        *,
        emit_end_event: bool = True,
        error: dict[str, str] | None = None,
    ) -> None:
        """Request cancellation of a run.

        In single-instance mode, cancels the local asyncio task directly.
        In multi-instance mode, broadcasts via Redis pub/sub so the owning
        instance can act on it. ``emit_end_event`` is disabled when the API
        has already reconciled the database and owns terminal signaling;
        *error* is what the ended stream says (``cancelled_error``).
        """

    @abstractmethod
    async def events(self, run_id: str) -> list[tuple[str, Any]]:
        """The run's stored events while they are kept, without registering a broker for it.

        A finished run's wait or join reads its result from them.
        """

    async def replay(self, run_id: str, last_event_id: str | None) -> list[tuple[str, Any]]:
        """The run's stored events after *last_event_id*, without registering a broker for it.

        All of them when *last_event_id* is None or no longer stored, as a
        broker's own ``replay`` answers.
        """
        events = await self.events(run_id)
        if last_event_id is not None:
            for index, (event_id, _) in enumerate(events):
                if event_id == last_event_id:
                    return events[index + 1 :]
        return events

    @abstractmethod
    async def retain(self, run_id: str, seconds: float) -> None:
        """Keep the run's stored events for *seconds* after its stream ended, and no longer."""

    @abstractmethod
    async def publish_done(self, run_id: str) -> None:
        """Tell the run's waiters in other processes it is done: finalized, its stream ended."""

    async def end_cancelled(self, run_id: str, error: dict[str, str] | None = None) -> None:
        """End the run's stream as a cancelled run's: its error (``cancelled_error``), then its end. Once only."""
        broker = self.get_or_create_broker(run_id)
        if broker.is_finished():
            return
        await broker.put(await self.allocate_event_id(run_id), ("error", error or cancelled_error()))
        await broker.put(await self.allocate_event_id(run_id), ("end", {"status": "interrupted"}))

    @abstractmethod
    async def allocate_event_id(self, run_id: str) -> str:
        """The id of the run's next event: its sequence number, allocated atomically."""
