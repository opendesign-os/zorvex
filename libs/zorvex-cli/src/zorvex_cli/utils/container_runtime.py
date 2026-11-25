"""Docker or Podman, whichever runs the project's Compose file."""

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import click
from rich.console import Console

console = Console()

ENV_CONTAINER_RUNTIME = "ZORVEX_CONTAINER_RUNTIME"


class _Runtime(NamedTuple):
    label: str
    install: str
    compose: str
    """What ``<runtime> compose`` needs."""
    start: dict[str, tuple[str, ...]]
    """platform.system() -> the command that starts the runtime there."""


RUNTIMES = {
    "docker": _Runtime(
        label="Docker",
        install="https://docs.docker.com/get-docker/",
        compose="the Docker Compose plugin (https://docs.docker.com/compose/install/)",
        start={
            "Darwin": ("open", "-a", "Docker"),
            "Linux": ("systemctl", "start", "docker"),
            "Windows": ("docker", "desktop", "start"),
        },
    ),
    "podman": _Runtime(
        label="Podman",
        install="https://podman.io/docs/installation",
        compose="docker-compose or podman-compose installed as its provider",
        start={
            "Darwin": ("podman", "machine", "start"),
            "Linux": ("systemctl", "--user", "start", "podman.socket"),
            "Windows": ("podman", "machine", "start"),
        },
    ),
}


def _succeeds(*command: str, timeout: float = 10) -> bool:
    """Whether *command* exits 0 within *timeout* seconds; a missing program or a hang fails."""
    try:
        return subprocess.run(command, capture_output=True, timeout=timeout, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _running(runtime: str) -> bool:
    return _succeeds(runtime, "info")


def _start(runtime: str) -> bool:
    """Start *runtime* with this platform's command, then give it a minute to answer."""
    command = RUNTIMES[runtime].start.get(platform.system())
    if command is None:
        return False
    label = RUNTIMES[runtime].label
    console.print(f"[yellow]{label} is not running; starting it with[/yellow] {' '.join(command)}")
    if not _succeeds(*command, timeout=120):
        return False
    deadline = time.monotonic() + 60
    with console.status(f"Waiting for {label}..."):
        while time.monotonic() < deadline:
            if _running(runtime):
                return True
            time.sleep(2)
    return False


def select_runtime() -> str:
    """The runtime to run Compose with, started when it is not running.

    ZORVEX_CONTAINER_RUNTIME (docker or podman) chooses it. Unset, it is Docker when its daemon
    runs, else Podman when installed, else Docker.

    Raises:
        click.ClickException: When it is not installed, does not start, or has no Compose.
    """
    chosen = os.environ.get(ENV_CONTAINER_RUNTIME, "").strip().lower()
    if chosen and chosen not in RUNTIMES:
        raise click.ClickException(f"{ENV_CONTAINER_RUNTIME} is {chosen!r}; set it to docker or podman.")
    running = not chosen and _running("docker")
    runtime = chosen or ("docker" if running or not shutil.which("podman") else "podman")
    details = RUNTIMES[runtime]
    if shutil.which(runtime) is None:
        if chosen:
            raise click.ClickException(f"{details.label} is not installed: {details.install}")
        raise click.ClickException(
            f"Neither Docker nor Podman is installed. Install Docker ({details.install}) "
            f"or Podman ({RUNTIMES['podman'].install})."
        )
    if not (running or _running(runtime) or _start(runtime)):
        command = details.start.get(platform.system())
        how = f" ({' '.join(command)})" if command else ""
        raise click.ClickException(f"{details.label} is not running. Start it{how}, then run the command again.")
    if not _succeeds(runtime, "compose", "version"):
        raise click.ClickException(f"`{runtime} compose` does not work: it needs {details.compose}.")
    return runtime


def compose_argv(runtime: str, compose_file: Path, *args: str) -> list[str]:
    return [runtime, "compose", "-f", str(compose_file), *args]


def run_compose(runtime: str, compose_file: Path, *args: str) -> int:
    """Run ``<runtime> compose`` on *compose_file* with *args*, showing the command; its exit code."""
    command = compose_argv(runtime, compose_file, *args)
    console.print(f"[dim]Running: {' '.join(command)}[/dim]")
    return subprocess.run(command, check=False).returncode
