"""Alembic environment configuration for Zorvex database migrations."""

import asyncio
import threading
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection

from alembic import context
from zorvex_api.core.database import create_db_engine
from zorvex_api.core.orm import Base

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# Only reconfigure logging when running from CLI (main thread).
# When invoked programmatically via asyncio.to_thread(), fileConfig()
# causes a cross-thread deadlock with the application's logging.
# See: https://github.com/sqlalchemy/alembic/discussions/1483
if config.config_file_name is not None and threading.current_thread() is threading.main_thread():
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _include_name(name: str | None, type_: str, _parent_names: object) -> bool:
    """Autogenerate compares Zorvex's tables only; LangGraph creates and migrates its own."""
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


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata, include_name=_include_name)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run the migrations on a connection to settings' database."""
    # The URL never goes through alembic.ini: configparser would reject the %XX
    # escapes of a quoted password.
    connectable = create_db_engine(poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
