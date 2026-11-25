"""Zorvex CLI - Command-line interface for managing self-hosted agent deployments."""

import importlib.util
import ipaddress
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

from zorvex_cli import __version__
from zorvex_cli.commands import new
from zorvex_cli.commands.db import db
from zorvex_cli.env import apply_env, config_env, load_env_file
from zorvex_cli.templates import (
    UI_BUNDLER,
    get_docker_compose,
    get_dockerfile,
    slugify,
)
from zorvex_cli.utils.container_runtime import (
    ENV_CONTAINER_RUNTIME,
    ContainerRuntime,
    ProjectOwnership,
    RuntimeSelectionError,
    compose_argv,
    ensure_compose_plugin,
    ensure_postgres_running,
    ensure_runtime_ready,
    missing_binary_message,
    resolve_container_runtime,
    runtimes_owning_project,
)

console = Console()

# Default values for server options (single source of truth)
_DEFAULT_DEV_HOST = "127.0.0.1"
_DEFAULT_SERVE_HOST = "0.0.0.0"  # nosec B104 - intentional for Docker
_DEFAULT_PORT = 2026


def _windows_loop_args() -> list[str]:
    """Extra uvicorn args forcing a psycopg-compatible event loop on Windows.

    The default Proactor loop breaks the LangGraph psycopg pool.
    """
    if sys.platform != "win32":
        return []
    return ["--loop", "zorvex_api.utils.event_loop:selector_loop_factory"]


def _resolve_server_option[T: (str, int)](
    ctx: click.Context, param_name: str, cli_value: T, *, env_var: str, default: T
) -> T:
    """The command line's value, else *env_var* (the project .env is loaded), else *default*."""
    source = ctx.get_parameter_source(param_name)
    if source == click.core.ParameterSource.COMMANDLINE:
        return cli_value
    env_val = os.environ.get(env_var)
    if env_val:
        try:
            return type(default)(env_val)
        except (ValueError, TypeError):
            msg = f"Invalid value for {env_var}: {env_val!r} (expected {type(default).__name__})"
            raise click.ClickException(msg) from None
    return default


# Attempt to get zorvex-api version
try:
    from zorvex_api import __version__ as api_version
except ImportError:
    api_version = "not installed"


@click.group()
@click.version_option(
    version=__version__,
    prog_name="zorvex-cli",
    message=f"%(prog)s %(version)s (zorvex-api {api_version})",
)
def cli():
    """Zorvex CLI - Manage your self-hosted agent deployments.

    Zorvex is an open-source, self-hosted alternative to LangSmith Deployments.
    Use this CLI to run development servers, manage container services, and more.
    """


def find_config_file() -> Path | None:
    """Find the manifest via ZORVEX_CONFIG, else zorvex.json in cwd.

    Returns:
        Path to config file if found, None otherwise
    """
    # A pre-set ZORVEX_CONFIG is a deliberate choice (docker-compose, systemd unit,
    # CI matrix). Discovery must not silently outrank it.
    #
    # Only honored when it resolves inside the current directory, so a stale
    # ZORVEX_CONFIG exported in a shell cannot make `zorvex dev` silently boot a
    # different project from the one you are standing in. Point elsewhere with -c.
    env_config = os.environ.get("ZORVEX_CONFIG", "").strip()
    if env_config:
        resolved = Path(env_config).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        resolved = resolved.resolve()
        if resolved.is_relative_to(Path.cwd().resolve()) and resolved.is_file():
            return resolved

    zorvex_config = Path.cwd() / "zorvex.json"
    return zorvex_config if zorvex_config.exists() else None


def get_project_slug(config_path: Path | None) -> str:
    """Get project slug from config file or directory name.

    Args:
        config_path: Path to zorvex.json config file

    Returns:
        Slugified project name
    """
    if config_path and config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                if "name" in config:
                    return slugify(config["name"])
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback to directory name
    return slugify(Path.cwd().name)


def ensure_docker_files(project_path: Path, slug: str) -> Path:
    """Ensure docker-compose.yml and Dockerfile exist.

    Args:
        project_path: Project directory path
        slug: Project slug for naming

    Returns:
        Path to docker-compose.yml
    """
    compose_path = project_path / "docker-compose.yml"
    if not compose_path.exists():
        console.print(f"[cyan]Creating[/cyan] {compose_path}")
        compose_path.write_text(get_docker_compose(slug), encoding="utf-8")

    dockerfile_path = project_path / "Dockerfile"
    if not dockerfile_path.exists():
        console.print(f"[cyan]Creating[/cyan] {dockerfile_path}")
        config = find_config_file()
        ui = config is not None and bool(json.loads(config.read_text(encoding="utf-8")).get("ui"))
        dockerfile_path.write_text(get_dockerfile(ui=ui), encoding="utf-8")

    return compose_path


@cli.command()
@click.option(
    "--host",
    default=_DEFAULT_DEV_HOST,
    help="Host to bind the server to.",
    show_default=True,
)
@click.option(
    "--port",
    default=_DEFAULT_PORT,
    type=int,
    help="Port to bind the server to.",
    show_default=True,
)
@click.option(
    "--app",
    default="zorvex_api.main:app",
    help="Application import path.",
    show_default=True,
)
@click.option(
    "--config",
    "-c",
    "config_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to zorvex.json config file (auto-discovered if not specified).",
)
@click.option(
    "--env-file",
    "-e",
    "env_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env file (default: .env in project directory).",
)
@click.option(
    "--no-db-check",
    is_flag=True,
    default=False,
    help="Skip automatic PostgreSQL/Docker check.",
)
@click.option(
    "--file",
    "-f",
    "compose_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to docker-compose.yml file for PostgreSQL.",
)
@click.option(
    "--debug-port",
    default=None,
    type=int,
    help="Port for debugger to listen on (no debugger if not specified).",
    show_default=True,
)
@click.option(
    "--debug-host",
    default=None,
    help="Host for debugger to bind to (127.0.0.1 if not specified).",
    show_default=True,
)
@click.option(
    "--wait-for-client",
    is_flag=True,
    default=False,
    help="Break and wait for a debugger client to attach before starting.",
)
@click.option(
    "--no-reload",
    is_flag=True,
    default=None,
    help="Disable auto-reload",
)
@click.option(
    "--n-jobs-per-worker",
    type=int,
    default=None,
    help="Runs a worker executes at once (N_JOBS_PER_WORKER).",
)
@click.option(
    "--server-log-level", default=None, help="The server's log level (LOG_LEVEL), e.g. DEBUG."
)
@click.option(
    "--ssl-certfile",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Serve HTTPS with this certificate.",
)
@click.option(
    "--ssl-keyfile",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="The certificate's private key.",
)
@click.pass_context
def dev(
    ctx: click.Context,
    host: str,
    port: int,
    app: str,
    config_file: Path | None,
    env_file: Path | None,
    no_db_check: bool,
    compose_file: Path | None,
    debug_port: int | None,
    debug_host: str | None,
    wait_for_client: bool,
    no_reload: bool | None,
    n_jobs_per_worker: int | None,
    server_log_level: str | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
) -> None:
    """Run the development server with optional hot reload.

    Starts uvicorn with auto-reload enabled by default for development.
    The server will automatically restart when code changes are detected unless
    --no-reload is specified.

    Zorvex auto-discovers zorvex.json in the current directory, so you
    should run 'zorvex dev' from your project root.

    By default, Zorvex checks Docker or Podman and starts PostgreSQL if needed.
    With ZORVEX_CONTAINER_RUNTIME unset, it uses a running Docker daemon, then
    installed Podman, then installed Docker. Set it to docker or podman to choose.
    Use --no-db-check to skip this behavior.

    Examples:

        zorvex dev                        # Auto-discover config, start server

        zorvex dev -c /path/to/zorvex.json # Use specific config file

        zorvex dev -e /path/to/.env       # Use specific .env file

        zorvex dev --no-db-check          # Start without database check
    """
    # Validate debug options: --wait-for-client only makes sense with --debug-port
    if wait_for_client and debug_port is None:
        raise click.UsageError("--wait-for-client requires --debug-port to be set")
    if debug_host is not None and debug_port is None:
        raise click.UsageError("--debug-host requires --debug-port to be set")
    if (ssl_certfile is None) != (ssl_keyfile is None):
        raise click.UsageError("--ssl-certfile and --ssl-keyfile go together")
    if n_jobs_per_worker is not None:
        os.environ["N_JOBS_PER_WORKER"] = str(n_jobs_per_worker)
    if server_log_level is not None:
        os.environ["LOG_LEVEL"] = server_log_level.upper()

    # Discover or validate config file
    if config_file is not None:
        # User specified a config file explicitly
        resolved_config = config_file.resolve()
    else:
        # Auto-discover config file in current directory
        resolved_config = find_config_file()

    if resolved_config is None:
        console.print(
            "[bold red]Error:[/bold red] Could not find zorvex.json.\n"
            "Run [cyan]zorvex new[/cyan] to create a new project, or specify "
            "[cyan]--config[/cyan] to point to your config file."
        )
        sys.exit(1)

    console.print(f"[dim]Using config: {resolved_config}[/dim]")

    # Set ZORVEX_CONFIG env var so zorvex-api resolves paths relative to config location
    os.environ["ZORVEX_CONFIG"] = str(resolved_config)

    # Load the environment: --env-file, else zorvex.json's env, else the .env beside the config.
    configured = config_env(resolved_config)
    if env_file is None and isinstance(configured, Path):
        env_file = configured
    if env_file is None:
        # Try config directory first
        config_dir_env = resolved_config.parent / ".env"
        if config_dir_env.exists():
            env_file = config_dir_env

    # Auto-copy .env.example to .env if .env doesn't exist
    if env_file is None:
        dot_env = resolved_config.parent / ".env"
        dot_env_example = resolved_config.parent / ".env.example"
        if not dot_env.exists() and dot_env_example.exists():
            shutil.copy2(dot_env_example, dot_env)
            console.print(f"[cyan]Created[/cyan] {dot_env} [dim](copied from .env.example)[/dim]")

    loaded_env = load_env_file(env_file)
    if isinstance(configured, dict):
        apply_env(configured)
    if loaded_env:
        console.print(f"[dim]Loaded environment from: {loaded_env}[/dim]")
    elif env_file is not None:
        # User specified a file but it doesn't exist (shouldn't happen due to click validation)
        console.print(f"[yellow]Warning: .env file not found: {env_file}[/yellow]")

    # Resolve host/port with precedence: CLI flag > env var > default
    host = _resolve_server_option(ctx, "host", host, env_var="HOST", default=_DEFAULT_DEV_HOST)
    port = _resolve_server_option(ctx, "port", port, env_var="PORT", default=_DEFAULT_PORT)

    # Check and start PostgreSQL unless disabled
    if not no_db_check:
        console.print()

        # Auto-generate docker-compose.yml if not specified and doesn't exist
        if compose_file is None:
            project_path = resolved_config.parent
            default_compose = project_path / "docker-compose.yml"
            if default_compose.exists():
                compose_file = default_compose
            else:
                slug = get_project_slug(resolved_config)
                compose_file = ensure_docker_files(project_path, slug)

        if not ensure_postgres_running(compose_file):
            console.print(
                "\n[bold red]Cannot start server without PostgreSQL.[/bold red]\n"
                "[dim]Use --no-db-check to skip this check.[/dim]"
            )
            sys.exit(1)
        console.print()

    # Build info panel content
    info_lines = [
        "[bold green]Starting Zorvex development server[/bold green]\n",
        f"[cyan]Host:[/cyan] {host}",
        f"[cyan]Port:[/cyan] {port}",
        f"[cyan]App:[/cyan] {app}",
        f"[cyan]Config:[/cyan] {resolved_config}",
    ]
    if loaded_env:
        info_lines.append(f"[cyan]Env:[/cyan] {loaded_env}")
    if debug_port is not None:
        info_lines.append(f"[cyan]Debug port:[/cyan] {debug_port}")
        if debug_host is not None:
            info_lines.append(f"[cyan]Debug host:[/cyan] {debug_host}")
        if wait_for_client:
            info_lines.append("[cyan]Wait for client to attach:[/cyan] True")
    if no_reload:
        info_lines.append("\n[dim]Auto-reload is disabled[/dim]")
    info_lines.append("\n[dim]Press Ctrl+C to stop the server[/dim]")

    console.print(
        Panel(
            "\n".join(info_lines),
            title="[bold]Zorvex Dev Server[/bold]",
            border_style="green",
        )
    )

    # Build command. If debug_port is provided, wrap uvicorn with debugpy.
    cmd_uvicorn = ["-m", "uvicorn", app, "--host", host, "--port", str(port)]
    if ssl_certfile and ssl_keyfile:
        cmd_uvicorn += ["--ssl-certfile", ssl_certfile, "--ssl-keyfile", ssl_keyfile]
    if not no_reload:
        cmd_uvicorn.append("--reload")
    cmd_uvicorn.extend(_windows_loop_args())

    if debug_port is not None:
        if importlib.util.find_spec("debugpy") is None:
            console.print(
                "[bold red]Error:[/bold red] debugpy is not installed.\n"
                "Install it with: [cyan]pip install 'zorvex-cli[debug]'[/cyan]"
            )
            sys.exit(1)

        if not no_reload:
            console.print(
                "[yellow]Note:[/yellow] debugpy is active. Hot-reload will disconnect the debugger "
                "on each file change — reattach after every reload."
            )

        # Always bind debugpy to the debug host (default loopback) regardless of --host
        listen = str(debug_port)
        is_loopback = True

        # If debug_host is specified, warn if it's not loopback. Also handle ip v4 and v6 addresses
        if debug_host is not None:
            try:
                address = ipaddress.ip_address(debug_host)
                if address.version == 4:
                    listen = f"{debug_host}:{debug_port}"
                else:
                    listen = f"[{debug_host}]:{debug_port}"
                is_loopback = address.is_loopback
            except ValueError:
                if debug_host == "localhost":
                    listen = f"{debug_host}:{debug_port}"
                else:
                    console.print(
                        "[yellow]Warning:[/yellow] Invalid debug host specified. "
                        "Falling back to loopback binding for debugpy."
                    )

        if not is_loopback:
            console.print(
                Panel(
                    "[bold red]Warning:[/bold red] Debug host is non-loopback and exposes an "
                    "unauthenticated debug server!\n"
                    "[dim]Binding debugpy to a non-loopback address can allow remote code "
                    "execution. Use only if you understand the risk.[/dim]",
                    title="[bold red]Debug Exposure Warning[/bold red]",
                    border_style="red",
                )
            )
        cmd = [sys.executable, "-m", "debugpy", "--listen", listen]
        if wait_for_client:
            cmd.append("--wait-for-client")
        cmd.extend(cmd_uvicorn)
    else:
        cmd = [sys.executable] + cmd_uvicorn

    process = None
    bundler = _start_ui_bundler(resolved_config)
    try:
        # Use Popen for better signal handling across platforms
        process = subprocess.Popen(cmd)

        # Set up signal handler to forward signals to child process
        def signal_handler(signum, frame):
            for child in (process, bundler):
                _stop(child)
            console.print("\n[yellow]Server stopped by user.[/yellow]")
            sys.exit(0)

        # Register signal handlers (SIGTERM not available on Windows)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)

        # Wait for the process to complete
        returncode = process.wait()
        _stop(bundler)
        sys.exit(returncode)

    except FileNotFoundError:
        console.print(
            "[bold red]Error:[/bold red] uvicorn is not installed.\n"
            "Install it with: [cyan]pip install uvicorn[/cyan]"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        # Fallback handler if signal handler didn't catch it
        for child in (process, bundler):
            _stop(child)
        console.print("\n[yellow]Server stopped by user.[/yellow]")
        sys.exit(0)


def _start_ui_bundler(config_path: Path) -> subprocess.Popen[bytes] | None:
    """zorvex.json's ``ui`` components, bundled by @langchain/langgraph-ui into .zorvex/ui
    and again as they change.

    Nothing without ``ui`` (or with ``http.disable_ui``); the bundler runs on Node.js through npx.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ui = config.get("ui")
    if not ui or (config.get("http") or {}).get("disable_ui"):
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
        "LANGGRAPH_UI_CONFIG": json.dumps(config.get("ui_config") or {}),
    }
    output = config_path.parent / ".zorvex" / "ui"
    console.print(f"[cyan]UI:[/cyan] bundling {', '.join(ui)} into {output}")
    command = [npx, "-y", UI_BUNDLER, "watch", "-o", str(output)]
    if sys.platform == "win32":
        return subprocess.Popen(command, cwd=config_path.parent, env=env)
    # Its own process group, so stopping it also stops the node process npx starts.
    return subprocess.Popen(command, cwd=config_path.parent, env=env, start_new_session=True)


def _stop(child: subprocess.Popen[bytes] | None) -> None:
    """End a child process and those it started, killing it when it does not stop in five seconds.

    Terminating npx or uvicorn alone would leave the node or worker process it
    started running: on Windows taskkill ends the tree, elsewhere a child that
    leads its own process group (the bundler) gets the signal with its group.
    """
    if child is None or child.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(child.pid)], capture_output=True, check=False
        )
    elif os.getpgid(child.pid) == child.pid:
        os.killpg(child.pid, signal.SIGTERM)
    else:
        child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()


@cli.command()
@click.option(
    "--host",
    default=_DEFAULT_SERVE_HOST,
    help="Host to bind the server to.",
    show_default=True,
)
@click.option(
    "--port",
    default=_DEFAULT_PORT,
    type=int,
    help="Port to bind the server to.",
    show_default=True,
)
@click.option(
    "--app",
    default="zorvex_api.main:app",
    help="Application import path.",
    show_default=True,
)
@click.option(
    "--config",
    "-c",
    "config_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to zorvex.json config file (auto-discovered if not specified).",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, app: str, config_file: Path | None) -> None:
    """Run the production server.

    Starts uvicorn without --reload for production use.
    This command is typically used inside Docker containers.

    On Linux and macOS the command replaces itself with uvicorn, so the
    server is the container's PID 1: SIGTERM and SIGINT reach it directly,
    in-flight runs drain, and the exit code is uvicorn's own. Windows has
    no exec, so uvicorn runs there as a child process.

    Examples:

        zorvex serve                         # Start production server

        zorvex serve --host 0.0.0.0 --port 8080

    """
    # Discover or validate config file
    if config_file is not None:
        resolved_config = config_file.resolve()
    else:
        resolved_config = find_config_file()

    if resolved_config is None:
        console.print(
            "[bold red]Error:[/bold red] Could not find zorvex.json.\n"
            "Run [cyan]zorvex new[/cyan] to create a new project, or specify "
            "[cyan]--config[/cyan] to point to your config file."
        )
        sys.exit(1)

    # Set ZORVEX_CONFIG env var
    os.environ["ZORVEX_CONFIG"] = str(resolved_config)

    # zorvex.json's env, else the .env beside the config (as the dev command loads them)
    configured = config_env(resolved_config)
    config_dir_env = configured if isinstance(configured, Path) else resolved_config.parent / ".env"
    loaded_env = load_env_file(config_dir_env if config_dir_env.exists() else None)
    if isinstance(configured, dict):
        apply_env(configured)

    # Resolve host/port with precedence: CLI flag > env var > default
    host = _resolve_server_option(ctx, "host", host, env_var="HOST", default=_DEFAULT_SERVE_HOST)
    port = _resolve_server_option(ctx, "port", port, env_var="PORT", default=_DEFAULT_PORT)

    info_lines = [
        "[bold green]Starting Zorvex production server[/bold green]\n",
        f"[cyan]Host:[/cyan] {host}",
        f"[cyan]Port:[/cyan] {port}",
        f"[cyan]Config:[/cyan] {resolved_config}",
    ]
    if loaded_env:
        info_lines.append(f"[cyan]Env:[/cyan] {loaded_env}")

    console.print(
        Panel(
            "\n".join(info_lines),
            title="[bold]Zorvex Server[/bold]",
            border_style="green",
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app,
        "--host",
        host,
        "--port",
        str(port),
        *_windows_loop_args(),
    ]

    try:
        if sys.platform == "win32":
            # os.exec* on Windows spawns a new process and ends this one at once,
            # so the server runs as a child there.
            result = subprocess.run(cmd, check=False)
            sys.exit(result.returncode)
        os.execvp(sys.executable, cmd)  # nosec B606 - the argv built above, no shell
    except FileNotFoundError:
        console.print(
            "[bold red]Error:[/bold red] uvicorn is not installed.\n"
            "Install it with: [cyan]pip install uvicorn[/cyan]"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
        sys.exit(0)


def _select_runtime() -> ContainerRuntime | None:
    """Pick a runtime and start it if needed. None means it is unusable, after printing why."""
    # up/down ignore .env unless we load it. Existing shell variables still win.
    load_env_file(None)
    try:
        runtime = resolve_container_runtime()
    except RuntimeSelectionError as exc:
        console.print(f"\n[bold red]{exc}[/bold red]")
        return None
    if not ensure_runtime_ready(runtime):
        return None
    if not ensure_compose_plugin(runtime):
        return None
    return runtime


def _down_targets(compose_file: Path) -> ProjectOwnership | None:
    """Runtimes for `down` to stop. None means nothing can be stopped, after printing why."""
    # An explicit runtime wins. Otherwise stop whichever runtime still has the project.
    load_env_file(None)
    if os.environ.get(ENV_CONTAINER_RUNTIME, "").strip():
        runtime = _select_runtime()
        if runtime is None:
            return None
        return ProjectOwnership([runtime], False)
    found = runtimes_owning_project(compose_file)
    if not found.runtimes:
        if found.listing_failed:
            console.print(
                "\n[bold red]Nothing was stopped: a runtime could not be checked.[/bold red]\n"
                f"Set {ENV_CONTAINER_RUNTIME} to docker or podman to choose one directly."
            )
            return None
        runtime = _select_runtime()
        if runtime is None:
            return None
        return ProjectOwnership([runtime], False)
    usable: list[ContainerRuntime] = []
    listing_failed = found.listing_failed
    for runtime in found.runtimes:
        if ensure_compose_plugin(runtime):
            usable.append(runtime)
        else:
            # Same outcome as a failed listing: stop the other owners, then exit non-zero.
            listing_failed = True
    if not usable:
        return None
    return ProjectOwnership(usable, listing_failed)


@cli.command()
@click.option(
    "--file",
    "-f",
    "compose_file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to docker-compose file.",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build images before starting containers.",
)
@click.option("--wait", is_flag=True, help="Wait until the services are running and healthy.")
@click.option(
    "--recreate/--no-recreate", default=False, help="Recreate the containers even if unchanged."
)
@click.option(
    "--pull/--no-pull",
    default=None,
    help="Pull the images first, or never; the runtime decides when unset.",
)
@click.option(
    "--port", "-p", type=int, default=None, help="Host port the API is published on (PORT)."
)
@click.argument("services", nargs=-1)
def up(
    compose_file: Path | None,
    build: bool,
    wait: bool,
    recreate: bool,
    pull: bool | None,
    port: int | None,
    services: tuple[str, ...],
) -> None:
    """Start services with Docker Compose or Podman Compose.

    Uses docker-compose.yml which contains both postgres and the API service.
    Auto-generates Docker files if they don't exist.
    With ZORVEX_CONTAINER_RUNTIME unset, it uses a running Docker daemon, then
    installed Podman, then installed Docker. Set it to docker or podman to choose.

    Examples:

        zorvex up                    # Build and start all services

        zorvex up --no-build         # Start without rebuilding

        zorvex up postgres           # Start only postgres

        zorvex up -f ./custom.yml    # Use custom compose file
    """
    # Determine which compose file to use
    project_path = Path.cwd()
    config_file = find_config_file()
    slug = get_project_slug(config_file)

    if compose_file is None:
        compose_file = ensure_docker_files(project_path, slug)
    elif not compose_file.exists():
        console.print(f"[bold red]Error:[/bold red] Compose file not found: {compose_file}")
        sys.exit(1)

    console.print(
        Panel(
            "[bold green]Starting Zorvex services[/bold green]\n\n"
            f"[cyan]Compose file:[/cyan] {compose_file}",
            title="[bold]Zorvex Up[/bold]",
            border_style="green",
        )
    )

    runtime = _select_runtime()
    if runtime is None:
        sys.exit(1)

    cmd = compose_argv(runtime, compose_file, "up", "-d")

    # Build unless --no-build is specified
    if build:
        cmd.append("--build")
    if wait:
        cmd.append("--wait")
    if recreate:
        cmd.append("--force-recreate")
    if pull is not None:
        cmd += ["--pull", "always" if pull else "never"]
    if port is not None:
        os.environ["PORT"] = str(port)

    if services:
        cmd.extend(services)
        console.print(f"[cyan]Services:[/cyan] {', '.join(services)}")
    else:
        console.print("[cyan]Services:[/cyan] all")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            console.print("\n[bold green]Services started successfully![/bold green]")
            console.print()
            logs_cmd = " ".join(compose_argv(runtime, compose_file, "logs", "-f"))
            console.print(f"[dim]View logs:    {logs_cmd}[/dim]")
            console.print("[dim]Stop:         zorvex down[/dim]")
        else:
            console.print(
                f"\n[bold red]Error:[/bold red] {runtime.value} compose exited with code "
                f"{result.returncode}"
            )
        sys.exit(result.returncode)
    except FileNotFoundError:
        console.print(missing_binary_message(runtime))
        sys.exit(1)


@cli.command()
@click.option(
    "--file",
    "-f",
    "compose_file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to docker-compose file.",
)
@click.option(
    "--volumes",
    "-v",
    is_flag=True,
    default=False,
    help="Remove named volumes declared in the compose file.",
)
def down(compose_file: Path | None, volumes: bool) -> None:
    """Stop services with Docker Compose or Podman Compose.

    Runs compose down for the runtime that still has this project's containers.
    ZORVEX_CONTAINER_RUNTIME selects docker or podman directly.

    Examples:

        zorvex down                  # Stop services

        zorvex down -v               # Stop and remove volumes

        zorvex down -f ./custom.yml  # Stop specific compose file
    """
    console.print(
        Panel(
            "[bold yellow]Stopping Zorvex services[/bold yellow]",
            title="[bold]Zorvex Down[/bold]",
            border_style="yellow",
        )
    )

    if volumes:
        console.print("[yellow]Warning:[/yellow] Removing volumes - data will be lost!")

    project_path = Path.cwd()

    if compose_file:
        if not compose_file.exists():
            console.print(f"[bold red]Error:[/bold red] Compose file not found: {compose_file}")
            sys.exit(1)
        target_compose = compose_file
    else:
        target_compose = project_path / "docker-compose.yml"
        if not target_compose.exists():
            console.print("[yellow]No docker-compose.yml found. Nothing to stop.[/yellow]")
            sys.exit(0)

    console.print(f"\n[cyan]Stopping:[/cyan] {target_compose}")

    planned = _down_targets(target_compose)
    if planned is None:
        sys.exit(1)

    failed = False
    for runtime in planned.runtimes:
        cmd = compose_argv(runtime, target_compose, "down")
        if volumes:
            cmd.append("-v")
        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError:
            console.print(missing_binary_message(runtime))
            failed = True
            continue
        if result.returncode != 0:
            failed = True
    if failed:
        console.print("\n[bold red]Some services failed to stop.[/bold red]")
        sys.exit(1)
    if planned.listing_failed:
        console.print(
            "\n[yellow]Stopped the containers it could reach. "
            "Another runtime could not be checked or used.[/yellow]"
        )
        sys.exit(1)
    console.print("\n[bold green]Services stopped successfully![/bold green]")
    sys.exit(0)


# Register command groups and commands from the commands package


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to zorvex.json (auto-discovered if not specified).",
)
def validate(config_file: Path | None) -> None:
    """Check zorvex.json against the server's configuration model."""
    from pydantic import ValidationError
    from zorvex_api.config import ZorvexConfig

    resolved = config_file.resolve() if config_file is not None else find_config_file()
    if resolved is None:
        console.print("[bold red]Error:[/bold red] Could not find zorvex.json.")
        sys.exit(1)
    try:
        config = ZorvexConfig.model_validate_json(resolved.read_text(encoding="utf-8"))
    except ValidationError as exc:
        console.print(f"[bold red]{resolved.name} is invalid[/bold red]\n{exc}")
        sys.exit(1)
    console.print(
        f"[bold green]{resolved.name} is valid[/bold green]: {len(config.graphs)} graph(s)"
    )


cli.add_command(new)
cli.add_command(db)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
