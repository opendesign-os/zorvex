"""Alembic environment for Zorvex's own tables; LangGraph creates and migrates its checkpoint and store tables."""

import threading
from logging.config import fileConfig

import psycopg
from alembic import context
from sqlalchemy import create_engine, pool

from zorvex_api.core.orm import Base
from zorvex_api.settings import settings

config = context.config

# Only from the alembic command line: run in a thread by the server (asyncio.to_thread),
# fileConfig deadlocks with the application's logging (sqlalchemy/alembic#1483).
if config.config_file_name is not None and threading.current_thread() is threading.main_thread():
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _include_name(name: str | None, type_: str, _parent_names: object) -> bool:
    """Autogenerate compares Zorvex's tables only."""
    return type_ != "table" or name in target_metadata.tables


def run_migrations_offline() -> None:
    """Emit the migration SQL instead of running it."""
    context.configure(
        dialect_name="postgresql",
        target_metadata=target_metadata,
        include_name=_include_name,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run the migrations on DATABASE_URI, which libpq parses, as the server's connections do."""
    engine = create_engine(
        "postgresql+psycopg://",
        creator=lambda: psycopg.connect(settings.db.DATABASE_URI, prepare_threshold=None),
        poolclass=pool.NullPool,
    )
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, include_name=_include_name)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
