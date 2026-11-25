"""Alembic migrations of Zorvex's own tables, from the alembic.ini bundled with the package.

The server applies them as it starts (``run_migrations_async``) unless
RUN_MIGRATIONS_ON_STARTUP is false; ``zorvex db`` runs them out of band.
"""

import asyncio
from pathlib import Path

import psycopg
import structlog
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from zorvex_api.settings import settings

logger = structlog.get_logger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def find_alembic_ini() -> Path:
    """The package's alembic.ini: inside the installed package, or beside ``src/`` in a checkout.

    Never one in the working directory, which may be a host project's own.
    """
    for candidate in (_PACKAGE_DIR / "alembic.ini", _PACKAGE_DIR.parent.parent / "alembic.ini"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("alembic.ini is missing from the zorvex-api installation")


def get_alembic_config() -> Config:
    """Alembic's config, its script location absolute so it works from any working directory."""
    ini_path = find_alembic_ini()
    cfg = Config(str(ini_path))
    script_location = cfg.get_main_option("script_location")
    if script_location and not Path(script_location).is_absolute():
        cfg.set_main_option("script_location", str((ini_path.parent / script_location).resolve()))
    return cfg


def _current_revision() -> str | None:
    """The database's revision, read without alembic's lock; None before the first migration."""
    with psycopg.connect(settings.db.DATABASE_URI) as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT version_num FROM alembic_version LIMIT 1")
        except psycopg.errors.UndefinedTable:
            return None
        row = cur.fetchone()
        return row[0] if row else None


def run_migrations_if_needed() -> None:
    """Upgrade to head unless the database is there already, so replicas starting together do not queue."""
    cfg = get_alembic_config()
    head = ScriptDirectory.from_config(cfg).get_current_head()
    try:
        if _current_revision() == head:
            logger.debug("database already at migration head")
            return
    except psycopg.Error as exc:
        logger.debug("revision check failed; upgrading", error=str(exc))

    logger.info("running database migrations")
    command.upgrade(cfg, "head")
    logger.info("database migrations completed")


async def run_migrations_async() -> None:
    """``run_migrations_if_needed`` off the event loop: alembic is synchronous."""
    await asyncio.to_thread(run_migrations_if_needed)
