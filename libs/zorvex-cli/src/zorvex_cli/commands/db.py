"""``zorvex db``: Alembic migrations of the server's tables, on the DATABASE_URI it would use.

zorvex_api is imported once the project's environment is applied, as its settings read it on import.
"""

from pathlib import Path

import click

from zorvex_cli.project import load_project, project_options


@click.group()
def db() -> None:
    """Database migrations, on the database the server uses.

    The server applies them as it starts unless RUN_MIGRATIONS_ON_STARTUP is false; these
    commands run them out of band, such as once per release.
    """


def _alembic(config_path: Path, env_file: Path | None, command_name: str, *args: str, **kwargs: bool) -> None:
    """Run Alembic's *command_name* on the server's migrations, in the project's environment."""
    load_project(config_path, env_file)
    from alembic import command

    from zorvex_api.core.migrations import get_alembic_config

    try:
        getattr(command, command_name)(get_alembic_config(), *args, **kwargs)
    except Exception as exc:
        raise click.ClickException(f"alembic {command_name} failed: {exc}") from exc


@db.command()
@project_options
def upgrade(config_path: Path, env_file: Path | None) -> None:
    """Apply all pending migrations."""
    _alembic(config_path, env_file, "upgrade", "head")


@db.command(context_settings={"ignore_unknown_options": True})
@project_options
@click.argument("revision", default="-1")
def downgrade(config_path: Path, env_file: Path | None, revision: str) -> None:
    """Downgrade to REVISION: steps back (-1, the default), a revision id, or base.

    Examples:

    \b
        zorvex db downgrade          # Downgrade by one revision
        zorvex db downgrade -2       # Downgrade by two revisions
        zorvex db downgrade base     # Remove every migration
    """
    _alembic(config_path, env_file, "downgrade", revision)


@db.command()
@project_options
def current(config_path: Path, env_file: Path | None) -> None:
    """Show the database's current revision."""
    _alembic(config_path, env_file, "current")


@db.command()
@project_options
@click.option("--verbose", "-v", is_flag=True, help="Show detailed migration information.")
def history(config_path: Path, env_file: Path | None, verbose: bool) -> None:
    """Show the migration history."""
    _alembic(config_path, env_file, "history", verbose=verbose)
