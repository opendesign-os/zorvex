"""Redis connection manager for the event broker."""

from urllib.parse import urlparse

import redis.asyncio as aioredis
import structlog
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from zorvex_api.settings import settings

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_CONNECTIONS = 250
# PING pooled connections idle longer than this (seconds) before reuse, so
# server-side idle disconnects don't surface as ConnectionError.
_DEFAULT_HEALTH_CHECK_INTERVAL = 30


class RedisManager:
    """Manages Redis connection pool lifecycle.

    Follows the same pattern as DatabaseManager: a global singleton
    initialized during app lifespan and closed on shutdown.
    """

    def __init__(self) -> None:
        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None
        self._blocking_pool: aioredis.ConnectionPool | None = None
        self._blocking_client: aioredis.Redis | None = None

    async def initialize(self) -> None:
        """Create connection pool and verify connectivity."""
        if self._client is not None:
            return
        url = settings.redis.REDIS_URI
        if url is None:
            raise RuntimeError("REDIS_URI is not set")

        # Directly constructed pools default to zero retries; bounded retries survive
        # idle disconnects/failovers. Lease acquisition deduplicates RPUSH replays.
        # max_connections and health_check_interval are defaults that REDIS_URI
        # query parameters of the same name override.
        self._pool = aioredis.ConnectionPool.from_url(
            url,
            max_connections=_DEFAULT_MAX_CONNECTIONS,
            decode_responses=True,
            health_check_interval=_DEFAULT_HEALTH_CHECK_INTERVAL,
            retry=Retry(
                ExponentialBackoff(cap=1.0, base=0.05),
                settings.redis.REDIS_RETRY_ATTEMPTS,
            ),
            retry_on_error=[RedisConnectionError, RedisTimeoutError],
        )
        self._client = aioredis.Redis(connection_pool=self._pool)

        # A blocking pop needs a read timeout longer than its block. No retry: a reissued pop would hide
        # a reply lost with the old connection, so errors surface and _dequeue polls Postgres.
        self._blocking_pool = aioredis.ConnectionPool.from_url(
            url,
            decode_responses=True,
            health_check_interval=_DEFAULT_HEALTH_CHECK_INTERVAL,
        )
        # Set after from_url, whose URL query parameters would override them: one connection
        # per worker loop plus spare, and a read timeout that outlasts the block.
        self._blocking_pool.max_connections = settings.worker.WORKER_COUNT + 2
        self._blocking_pool.connection_kwargs.update(
            socket_timeout=settings.worker.blocking_socket_timeout_seconds,
            retry=None,
            retry_on_timeout=False,
            retry_on_error=[],
        )
        self._blocking_client = aioredis.Redis(connection_pool=self._blocking_pool)

        await self._client.ping()  # type: ignore[invalid-await]  # redis.asyncio stubs
        # Log only host info, not full URL which may contain credentials
        parsed = urlparse(url)
        logger.info("Redis broker initialized", host=parsed.hostname, port=parsed.port)

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._blocking_client:
            await self._blocking_client.aclose()
            self._blocking_client = None
        if self._blocking_pool:
            await self._blocking_pool.disconnect()
            self._blocking_pool = None
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis broker connections closed")

    def get_client(self) -> aioredis.Redis:
        """Return the shared async Redis client."""
        if self._client is None:
            raise RuntimeError("Redis not initialized. Set REDIS_URI and ensure Redis is running.")
        return self._client

    def get_blocking_client(self) -> aioredis.Redis:
        """Return the client for blocking commands, whose read timeout outlasts the block."""
        if self._blocking_client is None:
            raise RuntimeError("Redis not initialized. Set REDIS_URI and ensure Redis is running.")
        return self._blocking_client


# Global Redis manager instance
redis_manager = RedisManager()
