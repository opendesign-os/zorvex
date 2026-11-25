"""The project a command works on: its zorvex.json, read once, and its environment, applied by one rule."""

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console

console = Console()

config_option = click.option(
    "--config",
    "-c",
    "config_path",
    envvar="ZORVEX_CONFIG",
    show_envvar=True,
    default="zorvex.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The project's zorvex.json.",
)
env_file_option = click.option(
    "--env-file",
    "-e",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Environment file to load instead of zorvex.json's env or the .env beside zorvex.json.",
)


def project_options[F: Callable[..., Any]](command: F) -> F:
    """``--config`` and ``--env-file``, the options of a command that runs with the project's environment."""
    return config_option(env_file_option(command))


@dataclass(frozen=True)
class Project:
    config_path: Path
    """zorvex.json, absolute."""
    config: dict[str, Any]
    """zorvex.json, parsed."""
    env_file: Path | None
    """The environment file loaded, if one was."""

    @property
    def root(self) -> Path:
        return self.config_path.parent


def load_project(config_path: Path, env_file: Path | None = None, *, create_env: bool = False) -> Project:
    """Read zorvex.json and apply the project's environment to os.environ.

    The environment is *env_file*, else zorvex.json's ``env`` (a file relative to it, or the
    variables), else the .env beside zorvex.json, which *create_env* (the commands that run the
    project locally) first copies from .env.example when missing. Variables already set keep
    their values. ZORVEX_CONFIG is set to the file, so the server reads the same one.
    """
    path = config_path.resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise click.ClickException(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(config, dict):
        raise click.ClickException(f"{path} does not hold a JSON object.")
    os.environ["ZORVEX_CONFIG"] = str(path)

    env = config.get("env")
    if env_file is None and isinstance(env, dict):
        for name, value in env.items():
            os.environ.setdefault(name, str(value))
        return Project(path, config, None)
    if env_file is None and isinstance(env, str) and env:
        env_file = path.parent / env
    elif env_file is None:
        env_file = path.parent / ".env"
        example = path.parent / ".env.example"
        if create_env and not env_file.exists() and example.is_file():
            shutil.copyfile(example, env_file)
            console.print(f"[cyan]Created[/cyan] {env_file} [dim](from .env.example)[/dim]")
    if not env_file.is_file():
        return Project(path, config, None)
    load_dotenv(env_file)
    return Project(path, config, env_file.resolve())
