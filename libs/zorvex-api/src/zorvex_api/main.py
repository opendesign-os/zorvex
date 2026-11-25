"""FastAPI application for Zorvex (Agent Protocol Server)"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import structlog
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.dependencies.utils import get_parameterless_sub_dependant
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from langgraph.errors import InvalidUpdateError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import HTTPConnection
from starlette.routing import Match, Mount
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from zorvex_api import __version__
from zorvex_api.api.a2a import router as a2a_router
from zorvex_api.api.assistants import router as assistants_router
from zorvex_api.api.crons import router as crons_router
from zorvex_api.api.event_streaming import router as event_streaming_router
from zorvex_api.api.mcp import router as mcp_router
from zorvex_api.api.runs import router as runs_router
from zorvex_api.api.stateless_runs import router as stateless_runs_router
from zorvex_api.api.store import router as store_router
from zorvex_api.api.system import alive, liveness, ok, system_metrics
from zorvex_api.api.system import router as system_router
from zorvex_api.api.threads import router as threads_router
from zorvex_api.api.ui import router as ui_router
from zorvex_api.api.ui import ui_dir
from zorvex_api.config import CorsConfig, HttpConfig, load_config, mount_prefix
from zorvex_api.core import loopback
from zorvex_api.core.auth import get_auth_backend, require_auth
from zorvex_api.core.authorization import check_authorization
from zorvex_api.core.database import db_manager
from zorvex_api.core.migrations import run_migrations_async
from zorvex_api.core.redis_manager import redis_manager
from zorvex_api.core.route_merger import (
    merge_exception_handlers,
    merge_lifespans,
)
from zorvex_api.middleware import (
    ConfigurableHeadersMiddleware,
    ContentTypeFixMiddleware,
    StructLogMiddleware,
)
from zorvex_api.middleware.auth_first import AuthFirstMiddleware
from zorvex_api.models.errors import FORBIDDEN, UNAUTHORIZED, VALIDATION_ERROR, AgentProtocolError, get_error_type
from zorvex_api.observability import metrics
from zorvex_api.observability.tracing import setup_tracing, shutdown_tracing
from zorvex_api.services import priority
from zorvex_api.services.broker import broker_manager
from zorvex_api.services.cron_scheduler import cron_scheduler
from zorvex_api.services.executor import executor
from zorvex_api.services.langgraph_service import get_langgraph_service
from zorvex_api.services.lease_reaper import lease_reaper
from zorvex_api.services.mcp_server import mcp_transport
from zorvex_api.services.thread_ttl import thread_ttl_sweeper
from zorvex_api.services.webhook_delivery import webhook_dispatcher
from zorvex_api.settings import settings
from zorvex_api.utils.loader import load_object
from zorvex_api.utils.setup_logging import setup_logging

OPENAPI_TAGS: list[dict[str, Any]] = [
    {"name": "Assistants", "description": "A configured instance of a graph."},
    {"name": "Threads", "description": "Accumulated state and outputs from a group of runs."},
    {"name": "Thread Runs", "description": "Invoke a graph on a thread, updating its persistent state."},
    {"name": "Stateless Runs", "description": "Invoke a graph without state or memory persistence."},
    {"name": "Crons", "description": "Scheduled recurring runs on a cron schedule."},
    {"name": "Store", "description": "Persistent key-value and semantic storage available from any thread."},
    {"name": "A2A", "description": "Assistants served as agents over the Agent-to-Agent protocol (JSON-RPC 2.0)."},
    {"name": "MCP", "description": "Assistants served as tools over the Model Context Protocol."},
    {"name": "System", "description": "Health probe, server information and metrics."},
    {"name": "Streaming", "description": "Thread event streaming and commands over SSE and HTTP."},
]

setup_logging()
logger = structlog.getLogger(__name__)

# Default CORS headers required for LangGraph SDK stream reconnection
DEFAULT_EXPOSE_HEADERS = ["Content-Location", "Location"]


def _log_connection_help(error: Exception) -> None:
    """Log a helpful error message when database connection fails."""
    logger.error(
        "Could not connect to PostgreSQL",
        error=str(error),
        hint="Check your database configuration and ensure PostgreSQL is running.",
    )
    logger.error(
        "Troubleshooting tips:\n"
        "  - Local development?  Run 'zorvex dev' (starts PostgreSQL automatically)\n"
        "  - Docker deployment?  Run 'zorvex up' (starts PostgreSQL + app)\n"
        "  - External database?  Check DATABASE_URI in your .env"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager for startup/shutdown"""
    # A configured auth.path or priority.path that cannot load must stop the boot.
    get_auth_backend()
    priority.callback()

    # Multi-pod K8s: set RUN_MIGRATIONS_ON_STARTUP=false and run `zorvex db upgrade` out-of-band,
    # so replicas do not race each other through the migrations.
    if settings.app.RUN_MIGRATIONS_ON_STARTUP:
        try:
            await run_migrations_async()
        except (ConnectionRefusedError, OSError) as e:
            _log_connection_help(e)
            raise
    else:
        logger.info("skipping startup migrations (RUN_MIGRATIONS_ON_STARTUP=false)")

    # Startup: Initialize database and LangGraph components
    try:
        await db_manager.initialize()
    except (ConnectionRefusedError, OSError) as e:
        _log_connection_help(e)
        raise
    http = load_config().http

    # Each cleanup is registered once its step has started; they run in reverse start order
    # on shutdown (ttl sweeper → cron → MCP → reaper → executor drains jobs → webhooks → broker → Redis → DB).
    async with AsyncExitStack() as stack:
        stack.push_async_callback(db_manager.close)

        # Its shutdown, which exports the spans still batched, runs after the executor has stopped.
        stack.push_async_callback(asyncio.to_thread, shutdown_tracing)

        # Initialize LangGraph service
        langgraph_service = get_langgraph_service()
        await langgraph_service.initialize()

        # Initialize Redis broker (if enabled)
        if settings.redis.enabled:
            try:
                await redis_manager.initialize()
            except (ConnectionError, OSError) as e:
                logger.error(
                    "Cannot connect to Redis. Unset REDIS_URI for single-instance mode "
                    "without Redis, or make Redis reachable at REDIS_URI.",
                    error=str(e),
                )
                raise
            stack.push_async_callback(redis_manager.close)
        else:
            logger.warning(
                "Running without Redis broker. Background runs have no crash recovery "
                "or horizontal scaling. Set REDIS_URI for production use.",
            )

        # Start broker manager (cleanup task for in-memory, cancel listener for Redis)
        await broker_manager.start()
        stack.push_async_callback(broker_manager.stop)

        # Registered before the executor so it drains after it: runs finalized while draining queue webhooks.
        stack.push_async_callback(webhook_dispatcher.stop)

        # Start executor (spawns worker coroutines when Redis is enabled)
        await executor.start()
        stack.push_async_callback(executor.stop)

        # Start lease reaper (recovers crashed worker runs, Redis mode only)
        if settings.redis.enabled:
            await lease_reaper.start()
            stack.push_async_callback(lease_reaper.stop)

        # MCP transport: stateless Streamable HTTP at /mcp
        if not http.disable_mcp:
            await mcp_transport.start()
            stack.push_async_callback(mcp_transport.stop)

        # Start cron scheduler (fires due cron jobs)
        if settings.cron.FF_CRONS_ENABLED:
            await cron_scheduler.start()
            stack.push_async_callback(cron_scheduler.stop)

        # The thread TTL sweeper deletes or compacts expired threads. It runs with
        # or without server defaults: a thread may carry a ttl of its own.
        await thread_ttl_sweeper.start()
        stack.push_async_callback(thread_ttl_sweeper.stop)

        yield


# Every error answers in the Agent Protocol format, {"error": ..., "message": ...};
# the SDK reports `message` to its caller.
def _error_response(status_code: int, message: str, headers: Mapping[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=AgentProtocolError(error=get_error_type(status_code), message=message).model_dump(),
        headers=headers,
    )


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP errors, the router's own 404 and 405 included."""
    return _error_response(exc.status_code, str(exc.detail), exc.headers)


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """A request its route's parameters reject: 422, each problem as ``location: message``."""
    problems = (f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
    return _error_response(422, "; ".join(problems))


async def invalid_update_handler(_request: Request, exc: Exception) -> JSONResponse:
    """A state update the graph cannot apply is the caller's error, not the server's."""
    return _error_response(400, str(exc))


async def general_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    """Anything else is a server bug: a generic 500, the traceback goes to the log."""
    return _error_response(500, "An unexpected error occurred")


exception_handlers: dict[type[Exception], Callable[..., Any]] = {
    HTTPException: http_exception_handler,
    RequestValidationError: validation_exception_handler,
    InvalidUpdateError: invalid_update_handler,
    Exception: general_exception_handler,
}


async def _require_auth_on_http(connection: HTTPConnection) -> None:
    """``require_auth`` for custom routes; websocket routes under an included router pass through."""
    # An included router's dependencies reach its websockets too, and a Request-typed
    # dependency there fails every connection with a TypeError.
    if isinstance(connection, Request):
        await require_auth(connection)


CUSTOM_ROUTE_AUTH = [Depends(_require_auth_on_http)]


def _apply_auth_to_custom_routes(app: FastAPI, auth_deps: list[Any]) -> int:
    """Prepend ``auth_deps`` to every FastAPI route the custom app declares.

    Must run before the core routers are included: health checks and the root
    stay public, and protocol routes carry their own auth. Returns the number
    of routes protected.
    """
    fastapi_pages = {app.openapi_url, app.docs_url, app.redoc_url, app.swagger_ui_oauth2_redirect_url}
    protected = 0
    uncovered: list[str] = []

    def walk(routes: list[Any], context_changed: bool | None) -> None:
        """``context_changed`` is None outside an included router, else whether its context got ``auth_deps``."""
        nonlocal protected
        for route in routes:
            if isinstance(route, APIRoute):
                if context_changed is None:
                    protected += _prepend_dependencies(route, auth_deps)
                else:
                    protected += int(context_changed)
                continue
            # FastAPI keeps an included router whole, as the wrapper's `original_router`.
            nested = getattr(route, "original_router", None)
            if nested is not None:
                changed = _prepend_to_include_context(route, auth_deps) if context_changed is None else context_changed
                walk(list(nested.routes), changed)
                # Effective routes are cached until the wrapped router reports a change; a
                # schema built at import would otherwise keep serving them unauthenticated.
                nested._mark_routes_changed()
            elif getattr(route, "routes", None):
                # A mounted app's routes never inherit an include context.
                walk(list(route.routes), None)
            elif getattr(route, "path", None) not in fastapi_pages:
                uncovered.append(getattr(route, "path", repr(route)))

    walk(list(app.routes), None)
    logger.info("Applied authentication dependency to custom routes", route_count=protected)
    if uncovered:
        # Plain Starlette routes and mounted non-FastAPI apps take no dependency, and
        # websockets are skipped by `_require_auth_on_http`; say so rather than imply cover.
        logger.warning("enable_custom_route_auth does not cover these custom routes", paths=uncovered)
    return protected


def _prepend_to_include_context(included: Any, deps: list[Any]) -> bool:
    """Put ``deps`` ahead of an included router's app- and include-level dependencies, once."""
    # On FastAPI >= 0.137 those run before every route's own list, so prepending
    # to the route alone would let them execute for unauthenticated requests.
    context = included.include_context
    if any(dep in context.dependencies for dep in deps):
        return False
    context.dependencies = [*deps, *context.dependencies]
    return True


def _prepend_dependencies(route: APIRoute, deps: list[Any]) -> int:
    """Prepend ``deps`` to one route, once. Returns 1 if the route changed."""
    if any(dep in route.dependencies for dep in deps):
        return 0
    route.dependencies = [*deps, *route.dependencies]
    # `dependencies` alone is only read at construction time; mirror what
    # APIRoute.__init__ does so the already-built dependant picks it up.
    for dep in reversed(deps):
        route.dependant.dependencies.insert(0, get_parameterless_sub_dependant(depends=dep, path=route.path_format))
    return 1


def _add_cors_middleware(app: FastAPI, cors: CorsConfig) -> None:
    """Add CORS middleware from ``http.cors``.

    ``allow_origins`` defaults to ``["*"]``, or to none when ``allow_origin_regex``
    is set so the regex decides. ``allow_credentials`` defaults to true only for an
    explicit origin list: never for the wildcard, and not with a regex.
    """
    has_regex = cors.allow_origin_regex is not None
    origins = cors.allow_origins if cors.allow_origins is not None else ([] if has_regex else ["*"])
    credentials = cors.allow_credentials if cors.allow_credentials is not None else not has_regex and origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=cors.allow_origin_regex,
        allow_credentials=credentials,
        allow_methods=cors.allow_methods,
        allow_headers=cors.allow_headers,
        expose_headers=cors.expose_headers if cors.expose_headers is not None else DEFAULT_EXPOSE_HEADERS,
        max_age=cors.max_age,
    )


def _add_common_middleware(app: FastAPI, cors: CorsConfig) -> None:
    """Add common middleware stack in correct order.

    Middleware runs in reverse registration order, so we register:
    1. ContentTypeFixMiddleware (outermost - fixes text/plain → application/json)
    2. CORSMiddleware (handles preflight early)
    3. CorrelationIdMiddleware (adds request ID)
    4. StructLogMiddleware (logs with correlation ID)
    5. ConfigurableHeadersMiddleware (innermost - the headers graphs see)

    """
    app.add_middleware(ConfigurableHeadersMiddleware)
    app.add_middleware(StructLogMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    _add_cors_middleware(app, cors)
    app.add_middleware(ContentTypeFixMiddleware)


def _protocol_routers(http: HttpConfig) -> list[APIRouter]:
    """The protocol routers *http* leaves on, as the Agent Server mounts them.

    Turning off runs also turns off the crons and the v2 event streaming, whose
    commands start runs. Every route in them declares its @auth.on event with
    Authorize; startup fails otherwise.
    """
    runs = not http.disable_runs
    groups: tuple[tuple[bool, tuple[APIRouter, ...]], ...] = (
        (not http.disable_assistants, (assistants_router,)),
        (not http.disable_threads, (threads_router,)),
        (runs, (runs_router, stateless_runs_router, crons_router)),
        (not http.disable_store, (store_router,)),
        (runs and not http.disable_event_streaming, (event_streaming_router,)),
        (not http.disable_a2a, (a2a_router,)),
    )
    return [router for enabled, routers in groups if enabled for router in routers]


# Errors every protocol route can answer, documented once for all of them.
_PROTOCOL_ERRORS: dict[int | str, dict[str, Any]] = {**UNAUTHORIZED, **FORBIDDEN, **VALIDATION_ERROR}
# The MCP route only authenticates; its operations answer their errors in JSON-RPC.
_MCP_ERRORS: dict[int | str, dict[str, Any]] = {**UNAUTHORIZED, **VALIDATION_ERROR}


def _include_core_routers(app: FastAPI, http: HttpConfig) -> None:
    """Include the system routes (public; a bare /ok with ``disable_meta``), the protocol routers and the
    enabled interfaces, then check events.

    A2A serves one assistant per request and authorizes reading it, as a protocol
    route. MCP only authenticates: each MCP operation authorizes as its REST
    counterpart (services/mcp_server.py), as the Agent Server's does.
    """
    app.include_router(liveness if http.disable_meta else system_router, responses=VALIDATION_ERROR)
    routers = _protocol_routers(http)
    for router in routers:
        app.include_router(router, responses=_PROTOCOL_ERRORS)
    if not http.disable_mcp:
        app.include_router(mcp_router, responses=_MCP_ERRORS)
    if load_config().ui and not http.disable_ui:
        # The route ahead of the mount, which serves the bundle's files to anyone.
        app.include_router(ui_router, responses=_MCP_ERRORS)
        app.mount("/ui", StaticFiles(directory=ui_dir() / "public", check_dir=False), name="ui")
    check_authorization(app, routers)
    if http.disable_webhooks:
        _hide_webhooks(app)


def _guarded(app: FastAPI, http: HttpConfig) -> Callable[[Scope], bool]:
    """Whether a request goes to a route that authenticates: a protocol or MCP route, or, with
    ``enable_custom_route_auth``, a custom one; never the system routes or the docs."""
    secured = {id(router) for router in [*_protocol_routers(http), mcp_router, ui_router]}
    public = {id(system_router), id(liveness)}
    pages = {app.openapi_url, app.docs_url, app.redoc_url, app.swagger_ui_oauth2_redirect_url}

    def guarded(scope: Scope) -> bool:
        for route in app.router.routes:
            if route.matches(scope)[0] is not Match.FULL:
                continue
            # FastAPI keeps an included router whole, as its original_router.
            router = getattr(route, "original_router", None)
            if router is not None:
                return id(router) in secured or (http.enable_custom_route_auth and id(router) not in public)
            return http.enable_custom_route_auth and isinstance(route, APIRoute) and route.path not in pages
        return False

    return guarded


def _hide_webhooks(app: FastAPI) -> None:
    """Leave ``webhook`` out of the documented run and cron bodies, as the Agent Server's docs do with webhooks off."""
    build = app.openapi

    def openapi() -> dict[str, Any]:
        schema = build()
        schemas = schema.get("components", {}).get("schemas", {})
        for name in ("RunCreate", "CronCreate", "ThreadCronCreate", "CronUpdate"):
            schemas.get(name, {}).get("properties", {}).pop("webhook", None)
        return schema

    app.openapi = openapi


def create_app() -> FastAPI | Starlette:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    config = load_config()
    http = config.http

    user_app = load_object(http.app, module_name="zorvex_http_app") if http.app else None
    if user_app is not None:
        if not isinstance(user_app, FastAPI):
            raise TypeError(f"http.app {http.app!r} names a {type(user_app).__name__}; custom apps are FastAPI apps")
        logger.info("Loaded custom app", app=http.app)
        application = user_app
        if http.enable_custom_route_auth:
            _apply_auth_to_custom_routes(application, CUSTOM_ROUTE_AUTH)
        if not application.openapi_tags:
            application.openapi_tags = OPENAPI_TAGS
        if settings.app.LANGGRAPH_API_URL:
            application.servers = [{"url": settings.app.LANGGRAPH_API_URL}]
        _include_core_routers(application, http)
        application = merge_lifespans(application, lifespan)
        application = merge_exception_handlers(application, exception_handlers)
        if http.middleware_order == "auth_first":
            # Outside the custom app's middleware, inside the common middleware (CORS, request ids, logs).
            guarded = _guarded(application, http)
            application.add_middleware(AuthFirstMiddleware, guarded=guarded, refuse=http_exception_handler)
        _add_common_middleware(application, http.cors)
    else:
        application = FastAPI(
            title=config.name,
            description="The LangGraph Agent Server API, self-hosted",
            version=__version__,
            docs_url=None if http.disable_meta else "/docs",
            redoc_url=None,
            openapi_url=None if http.disable_meta else "/openapi.json",
            lifespan=lifespan,
            openapi_tags=OPENAPI_TAGS,
            servers=[{"url": settings.app.LANGGRAPH_API_URL}] if settings.app.LANGGRAPH_API_URL else None,
        )

        _add_common_middleware(application, http.cors)
        _include_core_routers(application, http)

        for exc_type, handler in exception_handlers.items():
            application.exception_handler(exc_type)(handler)

    metrics.instrument(application)
    # Before the app first serves: its instrumentation wraps the middleware.
    setup_tracing(application)
    # Graphs' get_client() with no url reaches this app in process.
    loopback.serve(application)

    prefix = mount_prefix()
    return _mounted(application, prefix, meta=not http.disable_meta) if prefix else application


def _mounted(app: FastAPI, prefix: str, *, meta: bool) -> Starlette:
    """*app* served under *prefix*, as ``http.mount_prefix`` asks.

    The health probe and the metrics also answer at the root, as on the Agent
    Server, so probes need no prefix. *app* keeps its lifespan and its state,
    and its documents list the prefix as their server (FastAPI reads it from the
    mount's root_path).
    """
    root = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    if meta:
        root.add_api_route("/", ok, include_in_schema=False)
        root.add_api_route("/ok", ok)
        root.add_api_route("/metrics", system_metrics)
    else:
        root.add_api_route("/ok", alive)
    logger.info("Serving the API under a mount prefix", prefix=prefix)

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[Any]:
        async with app.router.lifespan_context(app) as state:
            yield state

    mounted = Starlette(routes=[Mount(prefix, app=app), Mount("", app=root)], lifespan=lifespan)
    mounted.state = app.state
    return mounted


# Create application instance
app = create_app()
