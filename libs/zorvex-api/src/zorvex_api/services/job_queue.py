"""The worker job queue: a Redis sorted set of run ids, lowest score first, which workers take with BZPOPMIN.

A run's score puts it in its priority's band, -5 first and 5 last, and orders the band by when its runs fall
due: runs start by priority, then in turn. Queueing a run already queued leaves one entry.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime

from zorvex_api.core.redis_manager import redis_manager
from zorvex_api.settings import settings

# Wider than any epoch milliseconds, while every band's scores stay exact in a double.
_BAND = 10**13


def score(priority: int, due: datetime) -> float:
    """Where a run of *priority* that falls due at *due* stands in the queue."""
    return (priority + 5) * _BAND + due.timestamp() * 1000


async def push(entries: Mapping[str, float]) -> None:
    """Queue runs at their scores. Raises RedisError."""
    if entries:
        await redis_manager.get_client().zadd(settings.redis.queue_key, dict(entries))


async def pop(timeout: float) -> tuple[str, float] | None:
    """The queue's head and its score, waiting up to *timeout* seconds; None when the wait ends empty.

    Raises RedisError, a timed-out read included.
    """
    result = await redis_manager.get_blocking_client().bzpopmin(settings.redis.queue_key, timeout=timeout)
    return None if result is None else (str(result[1]), float(result[2]))


async def queued(run_ids: Sequence[str]) -> set[str]:
    """The runs among *run_ids* the queue holds. Raises RedisError."""
    if not run_ids:
        return set()
    # One round trip, in a command every Redis has (ZMSCORE needs 6.2).
    pipe = redis_manager.get_client().pipeline(transaction=False)
    for run_id in run_ids:
        pipe.zscore(settings.redis.queue_key, run_id)
    scores = await pipe.execute()
    return {run_id for run_id, score in zip(run_ids, scores, strict=True) if score is not None}
