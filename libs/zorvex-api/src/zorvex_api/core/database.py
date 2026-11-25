"""The database: a SQLAlchemy engine for Zorvex's tables, and the pool LangGraph's checkpointer and store share.

Both connect with psycopg, so libpq parses DATABASE_URI for each of them: multiple
hosts, sslmode, sslrootcert, target_session_attrs and the PG* environment variables
work as in libpq. Prepared statements are off, so PgBouncer in transaction mode works.
"""

from typing import cast

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import TTLConfig
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from zorvex_api.config import load_config
from zorvex_api.settings import settings

# The pool LangGraph's checkpointer and store share: dict rows, as LangGraph reads them.
Pool = AsyncConnectionPool[AsyncConnection[DictRow]]

logger = structlog.get_logger(__name__)

# Items without a ttl never expire; those with one are hidden once expired and swept every 5 minutes.
_STORE_TTL_DEFAULTS: TTLConfig = {"refresh_on_read": True, "omit_expired": True, "sweep_interval_minutes": 5}


async def _connect() -> AsyncConnection:
    return await AsyncConnection.connect(settings.db.DATABASE_URI, prepare_threshold=None)


class DatabaseManager:
    """Manages database connections and LangGraph persistence components"""

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None

        # Shared pool for LangGraph components (Checkpointer + Store)
        self.lg_pool: Pool | None = None
        self._checkpointer: AsyncPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None

    async def initialize(self) -> None:
        """Initialize database connections and LangGraph components"""
        # Idempotency check: if already initialized, do nothing
        if self.engine:
            return

        try:
            await self._initialize_components()
        except BaseException:
            # BaseException so a cancelled startup also releases its pools before propagating.
            await self._release_after_failed_init()
            raise

    async def _release_after_failed_init(self) -> None:
        """Close whatever a failed initialize() opened, without masking the original error."""
        try:
            await self.close()
        except Exception:
            logger.exception("Cleanup after failed database initialization also failed")

    async def _initialize_components(self) -> None:
        self.engine = create_async_engine(
            "postgresql+psycopg://",
            async_creator=_connect,
            pool_size=settings.pool.SQLALCHEMY_POOL_SIZE,
            max_overflow=settings.pool.SQLALCHEMY_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=settings.db.DB_ECHO_LOG,
        )

        lg_max = settings.pool.LANGGRAPH_POSTGRES_POOL_MAX_SIZE
        lg_kwargs = {"autocommit": True, "prepare_threshold": None, "row_factory": dict_row}

        # Opened explicitly: a pool opened in its constructor warns.
        pool = Pool(
            conninfo=settings.db.DATABASE_URI,
            min_size=settings.pool.LANGGRAPH_CHECKPOINTER_POSTGRES_POOL_MIN_SIZE,
            max_size=lg_max,
            open=False,
            kwargs=lg_kwargs,
            connection_class=AsyncConnection[DictRow],
            check=Pool.check_connection,
        )
        await pool.open()
        self.lg_pool = pool

        # The checkpointer and store share the pool instead of opening their own.
        logger.info(f"Initializing LangGraph components with shared pool (max {lg_max} conns)...")

        self._checkpointer = AsyncPostgresSaver(conn=pool, serde=load_config().checkpointer.serde.serializer())
        await self._checkpointer.setup()  # Ensure tables exist

        store_config = load_config().store
        index_config = (
            cast(PostgresIndexConfig, store_config.index.model_dump(exclude_none=True)) if store_config.index else None
        )
        # Without a ttl config LangGraph neither hides nor deletes expired items, so the ttl
        # a put sets would never apply: TTL support is always on, tunable via store.ttl.
        ttl_config: TTLConfig = {**_STORE_TTL_DEFAULTS, **(store_config.ttl or {})}

        self._store = AsyncPostgresStore(conn=pool, index=index_config, ttl=ttl_config)
        await self._store.setup()  # Ensure tables exist
        await self._store.start_ttl_sweeper()

        if index_config:
            embed_model = index_config.get("embed", "unknown")
            logger.info(f"Semantic store enabled with embeddings: {embed_model}")

        logger.info("✅ Database and LangGraph components initialized")

    async def close(self) -> None:
        """Close database connections"""
        engine, pool, store = self.engine, self.lg_pool, self._store
        # Cleared up front so a failed close cannot leave initialize() short-circuiting on a stale engine.
        self.engine = None
        self.lg_pool = None
        self._checkpointer = None
        self._store = None

        try:
            if store:
                await store.stop_ttl_sweeper(timeout=5)
            if engine:
                await engine.dispose()
        finally:
            if pool:
                await pool.close()

        logger.info("✅ Database connections closed")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """Return the live AsyncPostgresSaver instance."""
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._checkpointer

    def get_store(self) -> AsyncPostgresStore:
        """Return the live AsyncPostgresStore instance."""
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    def get_pool(self) -> Pool:
        """The pool LangGraph's checkpointer and store share."""
        if self.lg_pool is None:
            raise RuntimeError("Database not initialized")
        return self.lg_pool

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine


# Global database manager instance
db_manager = DatabaseManager()
