"""SQLAlchemy models for Zorvex's own tables, and the async session factory.

Alembic migrations own the schema; these models describe it. LangGraph's
checkpoint and store tables are created by the library and are not mapped here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

import structlog
from fastapi import Depends
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from zorvex_api.models.enums import PruneStrategy

_logger = structlog.getLogger(__name__)

# Safety net against pathological/adversarial nesting. Real agent JSON rarely
# exceeds a few dozen levels; Python's default frame limit is ~1000. 200 is
# well above any legitimate payload and well below the interpreter ceiling.
_MAX_STRIP_DEPTH = 200


def _strip_null_bytes(value: Any, _depth: int = 0) -> Any:
    """Recursively strip U+0000 from strings inside JSON-compatible structures.

    Postgres JSONB rejects \\u0000 with UntranslatableCharacterError; agent
    output can contain literal NULL bytes from untrusted input or model
    hallucination. Stripping at the type boundary protects every JSONB column.

    Beyond ``_MAX_STRIP_DEPTH`` the value is returned untouched — a deeper
    payload than that is almost certainly adversarial, and letting Postgres
    reject it surfaces a clearer signal than a RecursionError at bind time.
    """
    if _depth >= _MAX_STRIP_DEPTH:
        _logger.warning("jsonb_strip_depth_exceeded", depth=_depth, type=type(value).__name__)
        return value
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for k, v in value.items():
            stripped_k = _strip_null_bytes(k, _depth + 1)
            if stripped_k in result:
                # Two distinct raw keys collapsed to the same stripped key — the
                # earlier value is being dropped by last-wins. Surface so silent
                # data loss is visible in logs.
                _logger.warning("jsonb_strip_key_collision", stripped_key=stripped_k)
            result[stripped_k] = _strip_null_bytes(v, _depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_strip_null_bytes(v, _depth + 1) for v in value]
    return value


class JsonbSafe(TypeDecorator):
    """JSONB column that strips NULL bytes from string values before write.

    Drop-in replacement for ``JSONB``. Read path is untouched — only
    ``process_bind_param`` runs, so existing rows are unaffected.
    """

    impl = JSONB
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        return _strip_null_bytes(value)


class Base(DeclarativeBase):
    pass


def _jsonb_gin_index(name: str, column: str) -> Index:
    """A GIN index over a JSONB column for containment (``@>``) metadata filters."""
    return Index(name, column, postgresql_using="gin", postgresql_ops={column: "jsonb_path_ops"})


class Assistant(Base):
    __tablename__ = "assistant"

    # gen_random_uuid() is in Postgres 13+ core; no extension needed.
    assistant_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    context: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    metadata_dict: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"), name="metadata")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    __table_args__ = (
        Index("idx_assistant_user", "user_id"),
        _jsonb_gin_index("idx_assistant_metadata_gin", "metadata"),
    )


class AssistantVersion(Base):
    __tablename__ = "assistant_versions"

    assistant_id: Mapped[str] = mapped_column(
        Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JsonbSafe, nullable=False, server_default=text("'{}'::jsonb"))
    context: Mapped[dict] = mapped_column(JsonbSafe, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    metadata_dict: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"), name="metadata")
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)


class Thread(Base):
    __tablename__ = "thread"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, server_default=text("'idle'"))
    metadata_dict: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"), name="metadata")
    # The config the thread's graph is built with: its last run's assistant config and
    # run config, without the run's own configurable (services/run_preparation.py).
    config: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    # The state its last run or state update left: what thread reads show and value searches match.
    values: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    interrupts: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    # When the thread's state last changed: a run ending on it, a state update.
    state_updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    # The priority settings the priority callback gave the thread at its first run (services/priority.py).
    run_priority: Mapped[int | None] = mapped_column(Integer)
    fairness_group: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("idx_thread_user", "user_id"),
        _jsonb_gin_index("idx_thread_metadata_gin", "metadata"),
    )


class Run(Base):
    __tablename__ = "run"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    thread_id: Mapped[str] = mapped_column(Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), nullable=False)
    assistant_id: Mapped[str] = mapped_column(Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    # The error event a failed, timed-out or abandoned run ended with ({"error", "message"}):
    # what a wait or join answers once the run's events have expired.
    error: Mapped[dict | None] = mapped_column(JsonbSafe)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    # Its thread's priority, -5 first to 5 last, while a priority callback is set; else 0 (services/job_queue.py).
    priority: Mapped[int] = mapped_column(Integer, server_default=text("0"))

    # The run as requested: the serialized RunJob (input, config, context, stream and
    # interrupt options, webhook, metadata). The only copy; workers rebuild the job from it.
    execution_params: Mapped[dict] = mapped_column(JsonbSafe, nullable=False)

    # Lease-based crash recovery: tracks which worker owns a run and
    # when the lease expires. A background reaper re-enqueues runs
    # whose leases have expired (worker crashed).
    claimed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    # Times the lease reaper put the run back after its worker stopped renewing the lease.
    retries: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    # A cancel, recorded where the run's worker finds it: "interrupt", or "rollback" to
    # delete the run and its checkpoints; and why the run was cancelled.
    cancel_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_run_thread_id", "thread_id"),
        Index("idx_run_user", "user_id"),
        Index("idx_run_assistant_id", "assistant_id"),
        Index("idx_run_created_at", "created_at"),
        # Also serves status-only filters through its leading column.
        Index("idx_run_lease_reaper", "status", "lease_expires_at"),
    )


class Cron(Base):
    __tablename__ = "cron"

    cron_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    assistant_id: Mapped[str] = mapped_column(
        Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), nullable=True
    )
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    schedule: Mapped[str] = mapped_column(Text, nullable=False)
    # IANA zone the schedule is evaluated in; NULL means UTC.
    timezone: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The run each firing creates (input, config, context, stream and durability options).
    payload: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    metadata_dict: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"), name="metadata")
    on_run_completed: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, server_default=text("true"), nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    next_run_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    claimed_until: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    __table_args__ = (
        Index("idx_cron_user", "user_id"),
        Index("idx_cron_assistant_id", "assistant_id"),
        Index("idx_cron_thread_id", "thread_id"),
        Index("idx_cron_next_run", "enabled", "next_run_date"),
    )


class ThreadTTL(Base):
    __tablename__ = "thread_ttl"

    thread_id: Mapped[str] = mapped_column(Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), primary_key=True)
    strategy: Mapped[PruneStrategy] = mapped_column(Text, nullable=False, server_default=text("'delete'"))
    ttl_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    __table_args__ = (Index("idx_thread_ttl_expires_at", "expires_at"),)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Return a cached async_sessionmaker bound to db_manager.engine."""
    global async_session_maker
    if async_session_maker is None:
        from zorvex_api.core.database import db_manager

        engine = db_manager.get_engine()
        async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
    return async_session_maker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an AsyncSession."""
    maker = get_session_maker()
    async with maker() as session:
        yield session


# The request's session. It closes once the endpoint has returned, before a
# streamed body goes out, so a stream or a long wait holds no pool connection.
SessionDep = Annotated[AsyncSession, Depends(get_session, scope="function")]
