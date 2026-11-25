"""The broker: each run's events, stored for replay and sent live to whoever follows the run.

Two backends implement it: in memory for one process (broker.py), Redis for
several (redis_broker.py). An event is ``(mode, payload)``, and its id,
``{run_id}_event_{seq}``, numbers the run's events in order. A follower that
reads the stored events and then the live ones drops those it has seen by
their sequence.
"""

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


def event_id(run_id: str, sequence: int) -> str:
    """The id of the run's event number *sequence*."""
    return f"{run_id}_event_{sequence}"


def event_sequence(event_id: str) -> int:
    """The sequence number in an event id; 0 for an id without one."""
    try:
        return int(event_id.rpartition("_event_")[2])
    except ValueError:
        return 0


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
    """One run's events: stored for replay, and sent to the run's live followers."""

    @abstractmethod
    async def put(self, event_id: str, payload: Any) -> None:
        """Store the event and send it to the live followers."""

    @abstractmethod
    def aiter(self) -> AsyncGenerator[tuple[str, Any]]:
        """The stored events, then the live ones, up to the end; a follower drops repeats by sequence."""
        ...

    @abstractmethod
    async def stored(self) -> list[tuple[str, Any]]:
        """The run's stored events, in order."""

    @abstractmethod
    def mark_finished(self) -> None:
        """Take no more events."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Whether the broker takes no more events."""


class BaseBrokerManager(ABC):
    """The brokers of the runs this process streams, and what concerns a run wherever it executes."""

    @abstractmethod
    def get_or_create_broker(self, run_id: str) -> BaseRunBroker:
        """The run's broker in this process."""

    @abstractmethod
    def cleanup_broker(self, run_id: str) -> None:
        """Release the run's broker in this process; its stored events stay as long as they are kept."""

    @abstractmethod
    async def start(self) -> None:
        """Start the background work: sweeping old brokers, or listening for cancels."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the background work."""

    @abstractmethod
    async def request_cancel(self, run_id: str, *, end_error: dict[str, str] | None) -> None:
        """Stop the run's task, wherever it executes; with *end_error* (``cancelled_error``), end its stream too.

        None leaves the stream to the caller, which has already recorded how the run ended.
        """

    @abstractmethod
    async def events(self, run_id: str) -> list[tuple[str, Any]]:
        """The run's stored events while they are kept, without registering a broker for it.

        A finished run's wait or join reads its result from them.
        """

    async def replay(self, run_id: str, last_event_id: str | None) -> list[tuple[str, Any]]:
        """The run's stored events after *last_event_id*, without registering a broker for it.

        All of them when *last_event_id* is None or no longer stored.
        """
        events = await self.events(run_id)
        if last_event_id is not None:
            for index, (stored_id, _) in enumerate(events):
                if stored_id == last_event_id:
                    return events[index + 1 :]
        return events

    @abstractmethod
    async def retain(self, run_id: str, seconds: float) -> None:
        """Keep the run's stored events for *seconds* after its stream ended, and no longer."""

    @abstractmethod
    async def publish_done(self, run_id: str) -> None:
        """Tell the run's waiters in other processes it is done: finalized, its stream ended."""

    @abstractmethod
    async def allocate_event_id(self, run_id: str) -> str:
        """The id of the run's next event: its sequence number, allocated atomically."""

    async def publish(self, run_id: str, event: tuple[str, Any]) -> None:
        """Add the run's next event."""
        await self.get_or_create_broker(run_id).put(await self.allocate_event_id(run_id), event)

    async def end_stream(self, run_id: str, status: str, error: dict[str, str] | None = None) -> None:
        """End the run's stream, once: its *error* first when it has one, then its ``end`` with *status*."""
        broker = self.get_or_create_broker(run_id)
        if broker.is_finished():
            return
        if error is not None:
            await broker.put(await self.allocate_event_id(run_id), ("error", error))
        await broker.put(await self.allocate_event_id(run_id), ("end", {"status": status}))
