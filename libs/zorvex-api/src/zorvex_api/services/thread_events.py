"""Thread events that belong to no run, live to whoever follows the thread: its state updates.

In process without Redis; over Redis pub/sub with it, so a stream served by
any instance sees them.
"""

import asyncio
import contextlib
import json
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

from zorvex_api.core.redis_manager import redis_manager
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.settings import settings

Event = tuple[str, Any]

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
    pubsub = redis_manager.get_client().pubsub()
    await pubsub.subscribe(_channel(thread_id))

    async def pump() -> None:
        async for message in pubsub.listen():
            if message.get("type") == "message":
                data = message["data"]
                event, payload = json.loads(data.decode() if isinstance(data, bytes) else data)
                queue.put_nowait((event, payload))

    task = asyncio.create_task(pump())
    try:
        yield queue
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await pubsub.unsubscribe(_channel(thread_id))
        await pubsub.aclose()
