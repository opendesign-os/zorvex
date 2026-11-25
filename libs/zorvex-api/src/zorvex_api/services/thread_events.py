"""Thread events that belong to no run, live to whoever follows the thread: its state updates.

In process without Redis; over Redis pub/sub with it, so a stream served by
any instance sees them. Live only: what is published while a follower's
subscription is down does not reach it.
"""

import asyncio
import contextlib
import json
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from redis import RedisError

from zorvex_api.core.redis_manager import redis_manager
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

Event = tuple[str, Any]

# Backoff between subscribing again after a lost subscription: doubling from 1s, up to 30s.
_RESUBSCRIBE_MAX_SECONDS = 30.0

_followers: dict[str, set[asyncio.Queue[Event]]] = defaultdict(set)


def _channel(thread_id: str) -> str:
    return f"{settings.redis.REDIS_KEY_PREFIX}thread:{thread_id}"


async def publish(thread_id: str, event: str, data: Any) -> None:
    """Send *event* to the thread's followers."""
    payload = to_jsonable(data)
    if settings.redis.enabled:
        await redis_manager.get_client().publish(_channel(thread_id), json.dumps([event, payload]))
        return
    for queue in list(_followers.get(thread_id, ())):
        queue.put_nowait((event, payload))


@contextlib.asynccontextmanager
async def follow(thread_id: str) -> AsyncGenerator[asyncio.Queue[Event]]:
    """A queue of the thread's events from now on, while the context lasts."""
    queue: asyncio.Queue[Event] = asyncio.Queue()
    if not settings.redis.enabled:
        _followers[thread_id].add(queue)
        try:
            yield queue
        finally:
            _followers[thread_id].discard(queue)
            if not _followers[thread_id]:
                _followers.pop(thread_id, None)
        return
    channel = _channel(thread_id)
    pubsub = redis_manager.get_client().pubsub()
    await pubsub.subscribe(channel)

    async def pump() -> None:
        # redis-py reconnects and subscribes again within its own retries; past them, a
        # new subscription takes over, so the follower keeps getting what comes after.
        nonlocal pubsub
        failures = 0
        while True:
            try:
                if not pubsub.subscribed:
                    await pubsub.subscribe(channel)
                async for message in pubsub.listen():
                    failures = 0
                    if message.get("type") == "message":
                        data = message["data"]
                        event, payload = json.loads(data.decode() if isinstance(data, bytes) else data)
                        queue.put_nowait((event, payload))
                return  # listen() ends only once unsubscribed
            except RedisError as exc:
                failures += 1
                logger.warning("Thread event subscription lost; subscribing again", thread_id=thread_id, error=str(exc))
                await asyncio.sleep(min(2.0 ** (failures - 1), _RESUBSCRIBE_MAX_SECONDS))
                with contextlib.suppress(RedisError):
                    await pubsub.aclose()
                pubsub = redis_manager.get_client().pubsub()

    task = asyncio.create_task(pump())
    try:
        yield queue
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with contextlib.suppress(RedisError):
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
