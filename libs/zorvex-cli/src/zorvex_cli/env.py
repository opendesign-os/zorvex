"""Environment file loading."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


def load_env_file(env_file: Path | None) -> Path | None:
    """Load a .env file into os.environ, keeping variables that are already set.

    Args:
        env_file: The file to load; None means .env in the current directory.

    Returns:
        The file loaded, or None when it does not exist.
    """
    target = env_file if env_file is not None else Path.cwd() / ".env"
    if not target.is_file():
        return None
    load_dotenv(target)
    return target


def config_env(config_path: Path) -> Path | dict[str, str] | None:
    """zorvex.json's ``env``, as langgraph.json has it: a .env file (relative to the config),
    or the variables themselves."""
    try:
        env = json.loads(config_path.read_text(encoding="utf-8")).get("env")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(env, str) and env:
        return (config_path.parent / env).resolve()
    if isinstance(env, dict):
        return {str(name): str(value) for name, value in env.items()}
    return None


def apply_env(variables: dict[str, str]) -> None:
    """Set *variables* in os.environ, keeping those already set, as a .env file is loaded."""
    for name, value in variables.items():
        os.environ.setdefault(name, value)
