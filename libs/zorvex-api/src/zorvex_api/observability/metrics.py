"""Prometheus metrics: HTTP requests, the run queue and job slots under the Agent Server's names, plus Zorvex's own."""

from collections.abc import Iterator
from datetime import UTC, datetime
from statistics import median
from typing import Any

import prometheus_client
from fastapi import FastAPI
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info
from sqlalchemy import select

from zorvex_api.core.active_runs import executing_runs
from zorvex_api.core.orm import Run, get_session_maker
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES
from zorvex_api.settings import settings

REQUESTS = prometheus_client.Counter(
    "lg_api_http_requests_total", "Total number of HTTP requests.", ["method", "path", "status"]
)
LATENCY = prometheus_client.Histogram(
    "lg_api_http_requests_latency_seconds", "HTTP request latency in seconds.", ["method", "path"]
)
_QUEUE = {
    "n_pending": prometheus_client.Gauge("lg_api_num_pending_runs", "Runs waiting to start."),
    "n_running": prometheus_client.Gauge("lg_api_num_running_runs", "Runs executing."),
    "pending_runs_wait_time_max_secs": prometheus_client.Gauge(
        "lg_api_pending_runs_wait_time_max", "Longest wait of a pending run, in seconds."
    ),
    "pending_runs_wait_time_med_secs": prometheus_client.Gauge(
        "lg_api_pending_runs_wait_time_med", "Median wait of the pending runs, in seconds."
    ),
    "pending_unblocked_runs_wait_time_max_secs": prometheus_client.Gauge(
        "lg_api_pending_unblocked_runs_wait_time_max",
        "Longest wait of a pending run that no running run on its thread holds back, in seconds.",
    ),
}

REAPER_RECOVERED_RUNS = prometheus_client.Counter(
    "zorvex_reaper_recovered_runs_total",
    "Runs recovered by the lease reaper, by outcome: crashed_retried (expired lease, re-enqueued), "
    "crashed_exhausted (max retries exceeded, marked failed), stuck_pending (missing from the queue, "
    "re-enqueued). Recovery left to the workers' Postgres poll during a Redis outage is not counted.",
    labelnames=["outcome"],
)
THREAD_TTL_SWEPT = prometheus_client.Counter(
    "zorvex_thread_ttl_swept_threads_total",
    "Threads processed by the TTL sweep, by outcome: deleted (strategy delete), pruned "
    "(strategy keep_latest), error (retried next tick).",
    labelnames=["outcome"],
)
# Every outcome renders from the start: a series that appears late breaks rate() alerts.
for _outcome in ("crashed_retried", "crashed_exhausted", "stuck_pending"):
    REAPER_RECOVERED_RUNS.labels(outcome=_outcome)
for _outcome in ("deleted", "pruned", "error"):
    THREAD_TTL_SWEPT.labels(outcome=_outcome)


def workers() -> dict[str, int] | None:
    """This instance's job slots, when it runs workers (Redis mode)."""
    if not settings.redis.enabled:
        return None
    total = settings.worker.WORKER_COUNT * settings.worker.N_JOBS_PER_WORKER
    active = len(executing_runs)
    return {"max": total, "active": active, "available": max(total - active, 0)}


class _Workers(Collector):
    def collect(self) -> Iterator[GaugeMetricFamily]:
        for key, value in (workers() or {}).items():
            yield GaugeMetricFamily(f"lg_api_workers_{key}", f"Job slots on this instance: {key}.", value=value)


prometheus_client.REGISTRY.register(_Workers())


async def queue() -> dict[str, Any]:
    """The run queue as the Agent Server reports it: counts, and how long the pending runs have waited."""
    async with get_session_maker()() as session:
        rows = (
            await session.execute(
                select(Run.status, Run.created_at, Run.thread_id).where(Run.status.in_(ACTIVE_RUN_STATUSES))
            )
        ).all()
    now = datetime.now(UTC)
    pending = [(created, thread) for status, created, thread in rows if status == "pending"]
    busy = {thread for status, _, thread in rows if status == "running"}
    waits = [(now - created).total_seconds() for created, _ in pending]
    unblocked = [(now - created).total_seconds() for created, thread in pending if thread not in busy and created < now]
    return {
        "n_pending": len(pending),
        "n_running": len(rows) - len(pending),
        "pending_runs_wait_time_max_secs": max(waits, default=None),
        "pending_runs_wait_time_med_secs": median(waits) if waits else None,
        "pending_unblocked_runs_wait_time_max_secs": max(unblocked, default=None),
    }


async def exposition() -> bytes:
    """Every metric in Prometheus' text format, the queue read now."""
    for key, value in (await queue()).items():
        _QUEUE[key].set(value or 0)
    return prometheus_client.generate_latest()


def requests() -> list[dict[str, Any]]:
    """HTTP request counts by method, route and status."""
    return [
        {
            "method": sample.labels["method"],
            "path": sample.labels["path"],
            "status": int(sample.labels["status"]),
            "count": int(sample.value),
        }
        for metric in REQUESTS.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    ]


def _record(info: Info) -> None:
    REQUESTS.labels(info.method, info.modified_handler, info.modified_status).inc()
    LATENCY.labels(info.method, info.modified_handler).observe(info.modified_duration)


def instrument(app: FastAPI) -> None:
    """Count and time the app's requests by route template; probes, metrics and docs are left out."""
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=[r"^/$", r"^/ok$", r"^/metrics$", r"^/docs$", r"^/openapi\.json$"],
    ).add(_record).instrument(app)
