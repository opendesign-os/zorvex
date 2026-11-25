"""The Redis broker, for several instances: a run's events in a Redis list, sent live over pub/sub.

The list (its replay buffer) expires on its own a while after the last event,
so no sweep is needed. A run's sequence numbers come from a counter key that
any instance can increment atomically.
"""

import asyncio
import contextlib
import json
import random
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Any

import structlog
from redis import RedisError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from zorvex_api.core.active_runs import active_runs, explicit_run_cancellations
from zorvex_api.core.redis_manager import redis_manager
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.services.base_broker import (
    REPLAY_RETENTION_SECONDS,
    BaseBrokerManager,
    BaseRunBroker,
    event_id,
    event_sequence,
    is_end,
)
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)


# Every write refreshes the TTL, so the buffer expires this long after the last
# event, matching the in-memory post-completion window.
_REPLAY_TTL_SECONDS = REPLAY_RETENTION_SECONDS
# Max events in the replay buffer (prevents unbounded growth)
_REPLAY_MAX_EVENTS = 10_000
# How long a done run's key answers waiters that start after its message.
_DONE_KEY_TTL_SECONDS = 3600

# Reconnect backoff for Redis pub/sub listeners
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 30.0
_BACKOFF_FACTOR = 2.0

# Bounded retry for the write path (put). The read path (aiter) retries
# RedisError indefinitely; the write path must be bounded so a producer can't
# block forever, but it should still retry a transient blip rather than drop the
# event — a dropped event is a permanently lost SSE token.
_PUT_MAX_ATTEMPTS = 3


def _deserialize_payload(raw: Any) -> Any:
    """Convert JSON-deserialized data back to expected Python types.

    Event payloads are tuples like ("values", {...}). JSON has no tuple type,
    so they arrive as lists. We convert top-level lists with a string first
    element back to tuples.
    """
    if isinstance(raw, list) and len(raw) >= 1 and isinstance(raw[0], str):
        return tuple(raw)
    return raw


def _backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(_BACKOFF_BASE * (_BACKOFF_FACTOR**attempt), _BACKOFF_MAX)
    jitter = random.uniform(0, delay * 0.25)  # nosec B311
    return delay + jitter


class RedisRunBroker(BaseRunBroker):
    """One run's events in Redis.

    A put appends the event to the run's list (RPUSH) and publishes it on the
    run's channel; a follower subscribes, reads the list, then takes what the
    channel sends past the last event it read.
    """

    def __init__(self, run_id: str, channel: str, cache_key: str, counter_key: str) -> None:
        self.run_id = run_id
        self._channel = channel
        self._cache_key = cache_key
        self._counter_key = counter_key
        self._finished = False
        # Serializes writes for this run so concurrent put() calls (e.g. a
        # cancel/error signal racing a stream write) keep cache-then-publish
        # ordering — without it, a backoff sleep on one put could let a later
        # event's publish overtake an earlier one.
        self._write_lock = asyncio.Lock()

    async def put(self, event_id: str, payload: Any) -> None:
        """Append an event to the replay buffer and publish it to live subscribers.

        Writes are retried with bounded backoff (see ``_write_with_retry``), so
        delivery is **at-least-once**: in the rare case a write times out *after*
        Redis already applied it, a retry may re-deliver the event. ``event_id``
        is stable (allocated once upstream), so consumers de-duplicate on
        ``Last-Event-ID`` during replay.
        """
        if self._finished:
            logger.warning(f"Attempted to put event {event_id} into finished broker for run {self.run_id}")
            return

        message = json.dumps({"event_id": event_id, "payload": payload}, default=to_jsonable)
        ends = is_end(payload)

        async with self._write_lock:
            # Cache and publish are retried independently so a publish failure
            # never re-runs the cache pipeline (which would double-write the
            # replay buffer and double-increment the sequence counter).
            operation = "cache"
            try:
                await self._write_with_retry(self._cache_event, message)

                operation = "publish"
                await self._write_with_retry(self._publish_event, message)

                if ends:
                    self._finished = True
            except RedisError as e:
                logger.error(
                    f"Redis {operation} write failed for run {self.run_id} after {_PUT_MAX_ATTEMPTS} attempts: {e}"
                )
                # An end that could not be written still ends the broker, so followers stop waiting for it.
                if ends:
                    self._finished = True

    async def _cache_event(self, message: str) -> None:
        """Append the event to the replay buffer; the buffer and the sequence counter live on as long."""
        client = redis_manager.get_client()
        pipe = client.pipeline()
        pipe.rpush(self._cache_key, message)
        pipe.ltrim(self._cache_key, -_REPLAY_MAX_EVENTS, -1)
        pipe.expire(self._cache_key, _REPLAY_TTL_SECONDS)
        pipe.expire(self._counter_key, _REPLAY_TTL_SECONDS)
        await pipe.execute()

    async def _publish_event(self, message: str) -> None:
        """Broadcast the event to live subscribers."""
        client = redis_manager.get_client()
        await client.publish(self._channel, message)

    async def _write_with_retry(self, op: Callable[[str], Awaitable[None]], message: str) -> None:
        """Run a Redis write, retried with bounded exponential backoff on a Redis error.

        A transient blip the client's own retry does not cover (a failover's
        read-only replica, a longer timeout) is retried instead of dropping the
        event; after ``_PUT_MAX_ATTEMPTS`` the error reaches ``put()``, so a
        producer never blocks for good.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_PUT_MAX_ATTEMPTS),
            wait=wait_exponential_jitter(initial=_BACKOFF_BASE, max=_BACKOFF_MAX, exp_base=_BACKOFF_FACTOR),
            retry=retry_if_exception_type(RedisError),
            before_sleep=lambda state: logger.warning(
                "Redis write failed, retrying", run_id=self.run_id, attempt=state.attempt_number
            ),
            reraise=True,
        ):
            with attempt:
                await op(message)

    async def aiter(self) -> AsyncGenerator[tuple[str, Any]]:
        attempt = 0
        while not self._finished:
            try:
                async for event_id, payload in self._subscribe_and_listen():
                    attempt = 0
                    yield event_id, payload
                # Clean exit from _subscribe_and_listen (end event or finished)
                break
            except RedisError as e:
                attempt += 1
                delay = _backoff_delay(attempt)
                logger.warning(
                    f"Redis pub/sub error for run {self.run_id}, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def _subscribe_and_listen(self) -> AsyncIterator[tuple[str, Any]]:
        """The stored events, then the live ones past them, up to the end.

        Subscribed first, then the list read: an event put in between is in
        the list, sent again on the channel, and dropped there by its sequence.
        """
        pubsub = redis_manager.get_client().pubsub()
        await pubsub.subscribe(self._channel)
        last = 0
        try:
            for stored_id, payload in await self.stored():
                last = event_sequence(stored_id)
                yield stored_id, payload
                if is_end(payload):
                    self._finished = True
                    return
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                if message is None:
                    if self._finished:
                        return
                    continue
                if message["type"] != "message":
                    continue
                data = json.loads(message["data"])
                live_id: str = data["event_id"]
                # Stored already, or a put's retried publish.
                if event_sequence(live_id) <= last:
                    continue
                last = event_sequence(live_id)
                payload = _deserialize_payload(data["payload"])
                yield live_id, payload
                if is_end(payload):
                    self._finished = True
                    return
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.aclose()

    async def stored(self) -> list[tuple[str, Any]]:
        try:
            client = redis_manager.get_client()
            raw_messages = await client.lrange(self._cache_key, 0, _REPLAY_MAX_EVENTS - 1)  # type: ignore[invalid-await]
        except RedisError as e:
            logger.error(f"Redis replay failed for run {self.run_id}: {e}")
            return []
        events: list[tuple[str, Any]] = []
        for raw in raw_messages:
            data = json.loads(raw)
            # A put retried after Redis had applied it stores its event twice, side by side
            # (the run's write lock keeps them adjacent): one of them goes.
            if events and events[-1][0] == data["event_id"]:
                continue
            events.append((data["event_id"], _deserialize_payload(data["payload"])))
        return events

    def mark_finished(self) -> None:
        self._finished = True
        logger.debug(f"Redis broker for run {self.run_id} marked as finished")

    def is_finished(self) -> bool:
        return self._finished


class RedisBrokerManager(BaseBrokerManager):
    """Manages RedisRunBroker instances with cross-instance cancel support.

    The local _brokers dict is a cache for status checks. Redis is the source
    of truth for event data. Cancel commands are broadcast via a dedicated
    pub/sub channel so any instance can cancel a run.
    """

    def __init__(self) -> None:
        self._brokers: dict[str, RedisRunBroker] = {}
        self._channel_prefix = settings.redis.channel_prefix
        self._cache_prefix = f"{self._channel_prefix}cache:"
        self._counter_prefix = f"{self._channel_prefix}counter:"
        self._cancel_channel = f"{self._channel_prefix}cancel"
        self._done_channel = f"{self._channel_prefix}done"
        # Waiters in this process, by run: set when the run's done message arrives.
        self._done_watchers: dict[str, set[asyncio.Event]] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._running: bool = False

    def _make_broker(self, run_id: str) -> RedisRunBroker:
        """Create a RedisRunBroker for a run_id."""
        channel = f"{self._channel_prefix}{run_id}"
        cache_key = f"{self._cache_prefix}{run_id}"
        counter_key = f"{self._counter_prefix}{run_id}"
        return RedisRunBroker(run_id, channel, cache_key, counter_key)

    def get_or_create_broker(self, run_id: str) -> RedisRunBroker:
        if run_id not in self._brokers:
            self._brokers[run_id] = self._make_broker(run_id)
            logger.debug(f"Created Redis broker for run {run_id}")
        return self._brokers[run_id]

    async def events(self, run_id: str) -> list[tuple[str, Any]]:
        # Another instance may have run it: the events are in Redis either way.
        return await (self._brokers.get(run_id) or self._make_broker(run_id)).stored()

    async def retain(self, run_id: str, seconds: float) -> None:
        # Called once the run has ended, after its end event set both TTLs to the full window.
        try:
            client = redis_manager.get_client()
            pipe = client.pipeline()
            pipe.expire(f"{self._cache_prefix}{run_id}", int(seconds))
            pipe.expire(f"{self._counter_prefix}{run_id}", int(seconds))
            await pipe.execute()
        except RedisError as e:
            logger.warning(f"Could not shorten the replay window of run {run_id}: {e}")

    def cleanup_broker(self, run_id: str) -> None:
        broker = self._brokers.pop(run_id, None)
        if broker:
            broker.mark_finished()
            logger.debug(f"Cleaned up Redis broker for run {run_id}")

    async def start(self) -> None:
        """Start the cancel command listener."""
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_for_cancel_commands())
        logger.info("Redis broker manager started", cancel_channel=self._cancel_channel)

    async def stop(self) -> None:
        """Stop the cancel command listener."""
        self._running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None
        logger.debug("Redis broker manager stopped")

    async def request_cancel(self, run_id: str, *, end_error: dict[str, str] | None) -> None:
        """Broadcast the cancel, so the instance executing the run stops it."""
        message = json.dumps({"run_id": run_id, "end_error": end_error})
        try:
            client = redis_manager.get_client()
            await client.publish(self._cancel_channel, message)
            logger.info(f"Published cancel command for run {run_id}")
        except RedisError as e:
            logger.error(f"Failed to publish cancel for run {run_id}: {e}")
            # Fall back to local execution - if the task is on this instance,
            # we can still cancel it even if Redis publish fails.
            await self._execute_cancel(run_id, end_error)

    def done_key(self, run_id: str) -> str:
        """The key a done run leaves, for the waiters that start after its message."""
        return f"{self._channel_prefix}done:{run_id}"

    async def publish_done(self, run_id: str) -> None:
        client = redis_manager.get_client()
        await client.set(self.done_key(run_id), "1", ex=_DONE_KEY_TTL_SECONDS)
        await client.publish(self._done_channel, run_id)

    def watch_done(self, run_id: str) -> asyncio.Event:
        """An event set when the run's done message arrives; release it with :meth:`unwatch_done`."""
        event = asyncio.Event()
        self._done_watchers.setdefault(run_id, set()).add(event)
        return event

    def unwatch_done(self, run_id: str, event: asyncio.Event) -> None:
        watchers = self._done_watchers.get(run_id)
        if watchers is not None:
            watchers.discard(event)
            if not watchers:
                del self._done_watchers[run_id]

    async def allocate_event_id(self, run_id: str) -> str:
        """The run's next event id, its sequence from an INCR any instance can make. Raises RedisError."""
        counter_key = f"{self._counter_prefix}{run_id}"
        client = redis_manager.get_client()
        sequence = await client.incr(counter_key)  # ty: ignore[invalid-await]  # redis.asyncio stubs
        await client.expire(counter_key, _REPLAY_TTL_SECONDS)
        return event_id(run_id, int(sequence))

    async def _listen_for_cancel_commands(self) -> None:
        """Background task: subscribe to the cancel and done channels, reconnecting with backoff."""
        attempt = 0
        while self._running:
            try:
                await self._subscribe_and_handle_cancels()
                attempt = 0
            except asyncio.CancelledError:
                break
            except RedisError as e:
                attempt += 1
                delay = _backoff_delay(attempt)
                logger.error(
                    f"Cancel listener disconnected, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def _subscribe_and_handle_cancels(self) -> None:
        """Take cancel commands and done messages until stopped."""
        client = redis_manager.get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(self._cancel_channel, self._done_channel)

        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.5,
                )
                if message is None:
                    continue
                if message["type"] != "message":
                    continue
                if message["channel"] == self._done_channel:
                    for event in self._done_watchers.pop(message["data"], ()):
                        event.set()
                    continue

                try:
                    data = json.loads(message["data"])
                    end_error = data["end_error"]
                    if end_error is not None and not isinstance(end_error, dict):
                        raise ValueError("end_error must be an object")
                    await self._execute_cancel(data["run_id"], end_error)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid cancel message: {e}")
        finally:
            await pubsub.unsubscribe(self._cancel_channel, self._done_channel)
            await pubsub.aclose()

    async def _execute_cancel(self, run_id: str, end_error: dict[str, str] | None) -> None:
        """Stop the run's task when this instance executes it, and with *end_error* end its stream."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return
        logger.info("Cancelling the run's task", run_id=run_id)
        explicit_run_cancellations.add(run_id)
        task.cancel()
        # The broker stays: execute_run releases it, and its finished state keeps the stream from ending twice.
        if end_error is not None:
            await self.end_stream(run_id, "interrupted", end_error)
