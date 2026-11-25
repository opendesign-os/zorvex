"""System routes as the Agent Server serves them: the health probe, server information and metrics."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import langgraph.version
import psycopg_pool
import structlog
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from sqlalchemy import exc, text

from zorvex_api import __version__
from zorvex_api.config import load_config
from zorvex_api.core.database import db_manager
from zorvex_api.core.redis_manager import redis_manager
from zorvex_api.observability import metrics
from zorvex_api.settings import settings

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["System"])

_TIMEOUT_SECONDS = 3.0
# A pool that hands out no connection in time is overloaded, not down: 503 instead of 500.
_OVERLOADED = (psycopg_pool.PoolTimeout, exc.TimeoutError)

Service = Literal["healthy", "unhealthy", "skipped"]


class Health(BaseModel):
    ok: bool = Field(description="Whether every checked service answered.")
    services: dict[str, Service] = Field(description="Each data service the server depends on, and how it answered.")


class Info(BaseModel):
    version: str = Field(description="The server's version.")
    langgraph_py_version: str = Field(description="The LangGraph library version the server runs.")
    flags: dict[str, bool] = Field(description="Enabled features.")
    host: dict[str, str | None] = Field(description="Where the server runs: kind and deployment ids.")


async def _postgres() -> None:
    async with db_manager.get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))
    async with db_manager.get_pool().connection() as conn:
        await conn.execute("SELECT 1")


async def _redis() -> None:
    await redis_manager.get_client().ping()


def _checks() -> dict[str, Callable[[], Awaitable[None]]]:
    return {"postgres": _postgres, **({"redis": _redis} if settings.redis.enabled else {})}


# All http.disable_meta leaves: /ok, answering without checks, so an operator still sees the server is up.
liveness = APIRouter(tags=["System"])


@liveness.get("/ok", include_in_schema=False)
async def alive() -> dict[str, bool]:
    return {"ok": True}


@router.get("/", include_in_schema=False)
@router.get("/ok", response_model=Health, responses={500: {"model": Health}, 503: {"model": Health}})
async def ok(
    check_db: int = Query(0, ge=0, le=1, description="1 also checks the data services: the readiness probe."),
) -> JSONResponse:
    """Liveness probe; with ``check_db=1`` readiness: 500 when a data service fails, 503 when one is overloaded."""
    checks = _checks()
    if not check_db:
        return JSONResponse({"ok": True, "services": dict.fromkeys(checks, "skipped")})
    results = await asyncio.gather(
        *(asyncio.wait_for(check(), _TIMEOUT_SECONDS) for check in checks.values()), return_exceptions=True
    )
    services: dict[str, Service] = {}
    for name, result in zip(checks, results, strict=True):
        services[name] = "unhealthy" if isinstance(result, BaseException) else "healthy"
        if isinstance(result, BaseException):
            logger.warning("Health check failed", service=name, exc_info=result)
    failed = any(isinstance(result, BaseException) for result in results)
    overloaded = any(isinstance(result, _OVERLOADED) for result in results)
    return JSONResponse(
        {"ok": not failed, "services": services}, status_code=503 if overloaded else 500 if failed else 200
    )


@router.get("/info", response_model=Info)
async def info() -> Info:
    """The server's version, the LangGraph version it runs, its enabled features and where it runs."""
    return Info(
        version=__version__,
        langgraph_py_version=langgraph.version.__version__,
        flags={
            "assistants": True,
            "crons": settings.cron.FF_CRONS_ENABLED,
            "langsmith": False,
            "langsmith_tracing_replicas": True,
            "langsmith_tracing_session_on_runs": True,
            "mcp": not load_config().http.disable_mcp,
        },
        host={
            "kind": "self-hosted",
            "project_id": None,
            "host_revision_id": None,
            "revision_id": None,
            "tenant_id": None,
        },
    )


@router.get(
    "/metrics",
    response_class=Response,
    responses={200: {"content": {CONTENT_TYPE_LATEST: {}, "application/json": {}}}},
)
async def system_metrics(
    format: str = Query("prometheus", description="prometheus (the default) or json."),
) -> Response:
    """HTTP, run queue and job slot metrics, in Prometheus' text format or as JSON; an unknown format is Prometheus."""
    if format != "json":
        return Response(await metrics.exposition(), media_type=CONTENT_TYPE_LATEST)
    body: dict[str, Any] = {"queue": await metrics.queue(), "api": {"http_requests_total": metrics.requests()}}
    if (workers := metrics.workers()) is not None:
        body["workers"] = workers
    return JSONResponse(body)
