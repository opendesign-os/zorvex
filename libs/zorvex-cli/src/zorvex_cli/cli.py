"""The ``zorvex`` command."""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from zorvex_api import __version__ as api_version
from zorvex_cli import __version__
from zorvex_cli.commands.db import db
from zorvex_cli.commands.new import new
from zorvex_cli.project import Project, config_option, load_project, project_options
from zorvex_cli.templates import UI_BUNDLER, docker_files
from zorvex_cli.utils.container_runtime import compose_argv, run_compose, select_runtime

console = Console()

# The server commands hand the arguments they do not know to uvicorn.
# Where a container's server listens: every interface.
_ALL_INTERFACES = "0.0.0.0"  # nosec B104
_UVICORN_PASSTHROUGH = {"ignore_unknown_options": True}


@click.group()
@click.version_option(
    version=__version__,
    prog_name="zorvex-cli",
    message=f"%(prog)s %(version)s (zorvex-api {api_version})",
)
def cli() -> None:
    """The zorvex command: create, run and deploy LangGraph agents on Zorvex.

    Zorvex is an open-source, self-hosted alternative to LangSmith Deployments. Every command
    but new works on a project: the zorvex.json in the current directory, or the one --config
    or ZORVEX_CONFIG names.
    """


def _uvicorn_argv(host: str, port: int, *args: str) -> list[str]:
    """``python -m uvicorn`` arguments serving Zorvex on *host*:*port*, then *args*."""
    argv = ["-m", "uvicorn", "zorvex_api.main:app", "--host", host, "--port", str(port)]
    if sys.platform == "win32":
        # The default Proactor loop breaks the psycopg pool LangGraph uses.
        argv += ["--loop", "zorvex_api.utils.event_loop:selector_loop_factory"]
    return [*argv, *args]


def _panel(title: str, heading: str, project: Project, host: str, port: int, *notes: str) -> None:
    lines = [
        f"[bold green]{heading}[/bold green]\n",
        f"[cyan]Host:[/cyan] {host}",
        f"[cyan]Port:[/cyan] {port}",
        f"[cyan]Config:[/cyan] {project.config_path}",
    ]
    if project.env_file is not None:
        lines.append(f"[cyan]Env:[/cyan] {project.env_file}")
    console.print(Panel("\n".join([*lines, *notes]), title=f"[bold]{title}[/bold]", border_style="green"))


def ensure_docker_files(project: Project) -> Path:
    """The project's docker-compose.yml, written with the Dockerfile and .dockerignore where missing."""
    for name, content in docker_files(ui=bool(project.config.get("ui"))).items():
        path = project.root / name
        if not path.exists():
            console.print(f"[cyan]Creating[/cyan] {path}")
            path.write_text(content, encoding="utf-8")
    return project.root / "docker-compose.yml"


def _compose(runtime: str, compose_file: Path, *args: str) -> None:
    """Run Compose; when it fails, exit with its code."""
    if returncode := run_compose(runtime, compose_file, *args):
        console.print(f"\n[bold red]Error:[/bold red] {runtime} compose exited with code {returncode}")
        sys.exit(returncode)


@cli.command(context_settings=_UVICORN_PASSTHROUGH)
@project_options
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind the server to.")
@click.option("--port", default=2026, type=int, show_default=True, help="Port to bind the server to.")
@click.option("--reload/--no-reload", default=True, show_default=True, help="Restart the server when code changes.")
@click.option(
    "--debug",
    metavar="[HOST:]PORT",
    help="Run under debugpy, listening on [HOST:]PORT (host 127.0.0.1 by default), and wait for a debugger "
    "to attach. Any other host exposes the unauthenticated debugger to the network.",
)
@click.option("--no-db-check", is_flag=True, help="Do not start PostgreSQL.")
@click.option(
    "--file",
    "-f",
    "compose_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Compose file with the postgres service (default: docker-compose.yml beside zorvex.json).",
)
@click.argument("uvicorn_args", nargs=-1, type=click.UNPROCESSED)
def dev(
    config_path: Path,
    env_file: Path | None,
    host: str,
    port: int,
    reload: bool,
    debug: str | None,
    no_db_check: bool,
    compose_file: Path | None,
    uvicorn_args: tuple[str, ...],
) -> None:
    """Run the development server, reloading on code changes.

    PostgreSQL is the postgres service of the project's docker-compose.yml (created, with the
    Dockerfile and .dockerignore, when missing), started unless it runs. ZORVEX_CONTAINER_RUNTIME
    (docker or podman) chooses the runtime; unset, it is Docker when its daemon runs, else
    Podman, else Docker. Use --no-db-check for a database of your own (DATABASE_URI).

    Other arguments go to uvicorn; put them after -- when they look like zorvex's own.

    Examples:

    \b
        zorvex dev
        zorvex dev -c path/to/zorvex.json -e .env.local
        zorvex dev --debug 5678
        zorvex dev --ssl-certfile cert.pem --ssl-keyfile key.pem
    """
    project = load_project(config_path, env_file, create_env=True)
    if debug and importlib.util.find_spec("debugpy") is None:
        raise click.ClickException("--debug needs debugpy: pip install 'zorvex-cli[debug]'")
    if not no_db_check:
        hint = "Use --no-db-check to run without starting PostgreSQL."
        compose_file = compose_file or ensure_docker_files(project)
        try:
            runtime = select_runtime()
        except click.ClickException as exc:
            raise click.ClickException(f"{exc.message}\n{hint}") from None
        if run_compose(runtime, compose_file, "up", "-d", "--wait", "postgres"):
            raise click.ClickException(f"PostgreSQL did not start. {hint}")

    reload_args = ["--reload", "--reload-dir", str(project.root)] if reload else []
    argv = _uvicorn_argv(host, port, *reload_args, *uvicorn_args)
    notes: list[str] = []
    if debug:
        argv = ["-m", "debugpy", "--listen", debug, "--wait-for-client", *argv]
        notes.append(f"[cyan]Debugger:[/cyan] waiting for a client on {debug}")
    if not reload:
        notes.append("\n[dim]Auto-reload is disabled[/dim]")
    notes.append("\n[dim]Press Ctrl+C to stop the server[/dim]")
    _panel("Zorvex Dev Server", "Starting Zorvex development server", project, host, port, *notes)

    bundler = _start_ui_bundler(project)
    server = subprocess.Popen([sys.executable, *argv])

    def stop(signum: int, frame: object) -> None:
        for child in (server, bundler):
            _stop(child)
        console.print("\n[yellow]Server stopped by user.[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    returncode = server.wait()
    _stop(bundler)
    sys.exit(returncode)


def _start_ui_bundler(project: Project) -> subprocess.Popen[bytes] | None:
    """zorvex.json's ``ui`` components, bundled by @langchain/langgraph-ui into .zorvex/ui
    and again as they change.

    Nothing without ``ui`` (or with ``http.disable_ui``); the bundler runs on Node.js through npx.
    """
    ui = project.config.get("ui")
    if not ui or (project.config.get("http") or {}).get("disable_ui"):
        return None
    npx = shutil.which("npx")
    if npx is None:
        console.print(
            "[yellow]Warning:[/yellow] zorvex.json sets ui, and its bundler needs Node.js with npx "
            "(https://nodejs.org/). Serving without the UI."
        )
        return None
    env = {
        **os.environ,
        "LANGGRAPH_UI": json.dumps(ui),
        "LANGGRAPH_UI_CONFIG": json.dumps(project.config.get("ui_config") or {}),
    }
    output = project.root / ".zorvex" / "ui"
    console.print(f"[cyan]UI:[/cyan] bundling {', '.join(ui)} into {output}")
    command = [npx, "-y", UI_BUNDLER, "watch", "-o", str(output)]
    if sys.platform == "win32":
        return subprocess.Popen(command, cwd=project.root, env=env)
    # Its own process group, so stopping it also stops the node process npx starts.
    return subprocess.Popen(command, cwd=project.root, env=env, start_new_session=True)


def _stop(child: subprocess.Popen[bytes] | None) -> None:
    """End a child process and those it started, killing it when it does not stop in five seconds.

    Terminating npx or uvicorn alone would leave the node or worker process it
    started running: on Windows taskkill ends the tree, elsewhere a child that
    leads its own process group (the bundler) gets the signal with its group.
    """
    if child is None or child.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(child.pid)], capture_output=True, check=False)
    elif os.getpgid(child.pid) == child.pid:
        os.killpg(child.pid, signal.SIGTERM)
    else:
        child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()


@cli.command(context_settings=_UVICORN_PASSTHROUGH)
@project_options
@click.option(
    "--host",
    default=_ALL_INTERFACES,
    show_default=True,
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=2026,
    type=int,
    envvar="PORT",
    show_envvar=True,
    show_default=True,
    help="Port to bind the server to.",
)
@click.argument("uvicorn_args", nargs=-1, type=click.UNPROCESSED)
def serve(config_path: Path, env_file: Path | None, host: str, port: int, uvicorn_args: tuple[str, ...]) -> None:
    """Run the production server, as a container's command.

    On Linux and macOS the command replaces itself with uvicorn, so the
    server is the container's PID 1: SIGTERM and SIGINT reach it directly,
    in-flight runs drain, and the exit code is uvicorn's own. Windows has
    no exec, so uvicorn runs there as a child process.

    Other arguments go to uvicorn; put them after -- when they look like zorvex's own.

    Examples:

    \b
        zorvex serve
        zorvex serve --port 8080 --proxy-headers
    """
    project = load_project(config_path, env_file)
    _panel("Zorvex Server", "Starting Zorvex production server", project, host, port)
    argv = [sys.executable, *_uvicorn_argv(host, port, *uvicorn_args)]
    if sys.platform == "win32":
        # os.exec* on Windows spawns a new process and ends this one at once,
        # so the server runs as a child there.
        try:
            sys.exit(subprocess.run(argv, check=False).returncode)
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped.[/yellow]")
            sys.exit(0)
    os.execv(sys.executable, argv)  # nosec B606  # the argv built above, no shell


@cli.command()
@config_option
@click.option(
    "--file",
    "-f",
    "compose_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Compose file (default: docker-compose.yml beside zorvex.json, created with the Dockerfile when missing).",
)
@click.option("--build/--no-build", default=True, show_default=True, help="Build the image before starting.")
@click.option("--wait", is_flag=True, help="Wait until the services are running and healthy.")
@click.option("--recreate/--no-recreate", default=False, help="Recreate the containers even if unchanged.")
@click.option(
    "--pull/--no-pull",
    default=None,
    help="Pull the images first, or never; the runtime decides when unset.",
)
@click.option("--port", "-p", type=int, default=None, help="Port the server listens on and is published on (PORT).")
@click.argument("services", nargs=-1)
def up(
    config_path: Path,
    compose_file: Path | None,
    build: bool,
    wait: bool,
    recreate: bool,
    pull: bool | None,
    port: int | None,
    services: tuple[str, ...],
) -> None:
    """Start PostgreSQL, Redis and the server with Docker or Podman Compose.

    ZORVEX_CONTAINER_RUNTIME (docker or podman) chooses the runtime; unset, it is Docker when
    its daemon runs, else Podman, else Docker.

    Examples:

    \b
        zorvex up                    # Build and start all services
        zorvex up --wait             # ... and wait until they are healthy
        zorvex up --no-build         # Start without rebuilding
        zorvex up postgres           # Start only postgres
        zorvex up -f ./custom.yml    # Use another compose file
    """
    # The server's container reads .env (env_file), so it is created here as zorvex dev creates it.
    project = load_project(config_path, create_env=True)
    compose_file = compose_file or ensure_docker_files(project)
    if port is not None:
        os.environ["PORT"] = str(port)
    runtime = select_runtime()
    args = ["up", "-d"]
    if build:
        args.append("--build")
    if wait:
        args.append("--wait")
    if recreate:
        args.append("--force-recreate")
    if pull is not None:
        args += ["--pull", "always" if pull else "never"]
    _compose(runtime, compose_file, *args, *services)
    console.print("\n[bold green]Services started.[/bold green]")
    console.print(f"[dim]View logs:    {' '.join(compose_argv(runtime, compose_file, 'logs', '-f'))}[/dim]")
    console.print("[dim]Stop:         zorvex down[/dim]")


@cli.command()
@config_option
@click.option(
    "--file",
    "-f",
    "compose_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Compose file (default: docker-compose.yml beside zorvex.json).",
)
@click.option("--volumes", "-v", is_flag=True, help="Also remove the volumes, and the data in them.")
def down(config_path: Path, compose_file: Path | None, volumes: bool) -> None:
    """Stop the services, with the runtime zorvex up chooses.

    Examples:

    \b
        zorvex down                  # Stop services
        zorvex down -v               # Stop and remove volumes
        zorvex down -f ./custom.yml  # Stop specific compose file
    """
    project = load_project(config_path)
    compose_file = compose_file or project.root / "docker-compose.yml"
    if not compose_file.is_file():
        console.print(f"[yellow]No {compose_file.name} in {project.root}. Nothing to stop.[/yellow]")
        return
    _compose(select_runtime(), compose_file, "down", *(["--volumes"] if volumes else []))
    console.print("\n[bold green]Services stopped.[/bold green]")


@cli.command()
@project_options
def validate(config_path: Path, env_file: Path | None) -> None:
    """Check zorvex.json against the server's configuration model, in the project's environment."""
    project = load_project(config_path, env_file)
    from pydantic import ValidationError

    try:
        # Imported after the environment is applied: zorvex_api's settings read it on import.
        from zorvex_api.config import ZorvexConfig

        config = ZorvexConfig.model_validate(project.config)
    except ValidationError as exc:
        raise click.ClickException(f"{project.config_path.name} is invalid\n{exc}") from None
    console.print(f"[bold green]{project.config_path.name} is valid[/bold green]: {len(config.graphs)} graph(s)")


cli.add_command(new)
cli.add_command(db)
