"""Graph run execution logic.

Single source of truth for executing a graph run. Both LocalExecutor
(asyncio.create_task) and WorkerExecutor (Redis job queue) call
`execute_run`; it records the run's status (run_status) and streams its
events to the broker.
"""

import asyncio
from datetime import UTC, datetime
from functools import cache
from typing import Any

import structlog
from asgi_correlation_id import correlation_id
from langchain_core.tracers import LangChainTracer
from langgraph.pregel import Pregel
from langsmith import Client
from langsmith import utils as ls_utils

from zorvex_api.core.active_runs import active_runs, executing_runs
from zorvex_api.models.enums import RunStatus, StreamMode, ThreadStatus
from zorvex_api.models.run_job import RunJob
from zorvex_api.observability.tracing import run_span, start_run_trace
from zorvex_api.services import streaming_service
from zorvex_api.services.base_broker import cancelled_error
from zorvex_api.services.broker import broker_manager
from zorvex_api.services.event_streaming.native_stream import stream_native_v3_events
from zorvex_api.services.graph_streaming import stream_graph_events
from zorvex_api.services.langgraph_service import create_run_config, create_thread_config, get_langgraph_service
from zorvex_api.services.run_cancel import end_cancelled_run
from zorvex_api.services.run_status import Ending, finalize_run, start_run
from zorvex_api.services.streaming_service import error_event
from zorvex_api.services.thread_state_service import snapshot_values
from zorvex_api.services.webhook_delivery import finish_executing
from zorvex_api.utils.run_utils import (
    CONTEXT_PROVIDED,
    EVENT_STREAMING_V2,
    LANGSMITH_EXAMPLE_ID,
    LANGSMITH_PROJECT,
    map_command_to_langgraph,
    resolve_graph_context,
    stream_modes,
)

logger = structlog.getLogger(__name__)

_DEFAULT_STREAM_MODES = ["values"]
# The v2 stream's own mode; a v1 run that asked for it streams the rest (prepare_run refuses lifecycle).
_V2_MODES = frozenset({"tools"})

# Cancellation provenance prevents infrastructure stops from looking user-initiated.
_lease_loss_cancellations: set[str] = set()
_shutdown_cancellations: set[str] = set()
_timeout_cancellations: set[str] = set()

_TIMEOUT_ERROR = "Job exceeded maximum execution time"


async def execute_run(job: RunJob) -> None:
    """Execute a graph run, stream its events to the broker, and record how it ended.

    The run's status is the Agent Server's: success when the graph returns,
    pausing on an interrupt included; interrupted when a user cancels it;
    timeout or error when it fails. The thread's follows from the run and the
    state it stopped at (``run_status.thread_status_after``).
    """
    _bind_run_context(job)
    with run_span(job):
        await _execute(job)


async def _execute(job: RunJob) -> None:
    run_id = job.identity.run_id
    thread_id = job.identity.thread_id
    user_id = job.user.identity
    resumes_elsewhere = False
    finalized = False
    final = _FinalState()
    started_at: datetime | None = None
    executing_runs.add(run_id)

    try:
        # A run with after_seconds is not handed over before it is due (created_at): see run_queue.
        if not await start_run(run_id, user_id=user_id):
            logger.info("Run became terminal before execution started", run_id=run_id)
            return
        started_at = datetime.now(UTC)

        await _stream_graph(job, final)
        finalized = await finalize_run(run_id, thread_id, final.ending("success"), user_id=user_id, webhook_job=job)

    except asyncio.CancelledError:
        if run_id in _lease_loss_cancellations:
            resumes_elsewhere = True
            logger.info("Lease-loss cancel, skipping finalize", run_id=run_id)
        elif run_id in _shutdown_cancellations:
            # Drain cancel: the run goes back to the queue, so finalizing here
            # would destroy work a plain crash (SIGKILL) recovers.
            resumes_elsewhere = True
            logger.info("Shutdown drain cancel, skipping finalize for requeue", run_id=run_id)
        elif run_id in _timeout_cancellations:
            error = {"error": "TimeoutError", "message": _TIMEOUT_ERROR}
            finalized = await finalize_run(
                run_id, thread_id, final.ending("timeout", error), user_id=user_id, webhook_job=job
            )
            if finalized:
                await _best_effort_signal(streaming_service.end_run, run_id, "timeout", error)
        else:
            # A user's cancel: interrupted, or rolled back, as the cancel recorded on the run asks.
            finalized, error = await end_cancelled_run(
                run_id, thread_id, final.ending("interrupted"), user_id=user_id, webhook_job=job
            )
            if finalized:
                await _best_effort_signal(streaming_service.end_run, run_id, "interrupted", error or cancelled_error())
        raise
    except Exception as exc:
        logger.exception("Run failed", run_id=run_id)
        error = error_event(exc)
        finalized = await finalize_run(
            run_id, thread_id, final.ending("error", error), user_id=user_id, webhook_job=job
        )
        if finalized:
            await _best_effort_signal(streaming_service.end_run, run_id, "error", error)
    else:
        if finalized:
            await _best_effort_signal(streaming_service.end_run, run_id, "success")
    finally:
        _lease_loss_cancellations.discard(run_id)
        _shutdown_cancellations.discard(run_id)
        _timeout_cancellations.discard(run_id)
        active_runs.pop(run_id, None)
        finish_executing(job, values=final.values, run_started_at=started_at)
        if not resumes_elsewhere:
            await streaming_service.cleanup_run(run_id, resumable=job.kwargs.resumable)
            await _signal_run_done(run_id)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _bind_run_context(job: RunJob) -> None:
    """Tag the logs and spans of this task with the run; the task that executes a run calls it first."""
    if job.correlation_id:
        correlation_id.set(job.correlation_id)
    identity = job.identity
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        run_id=identity.run_id,
        thread_id=identity.thread_id,
        graph_id=identity.graph_id,
        user_id=job.user.identity,
        **({"request_id": job.correlation_id} if job.correlation_id else {}),
    )
    start_run_trace(job)


async def _best_effort_signal(fn: Any, *args: Any) -> None:
    """Call a signaling function, logging but not raising on failure.

    Signaling (end events, cancel/error notifications) must not override
    an already-committed DB status if it fails.
    """
    try:
        await fn(*args)
    except Exception:
        logger.warning("Signal failed (best-effort, DB status already committed)", fn=fn.__name__)


class _FinalState:
    """The thread's state where the run stopped: what the thread's status and the run's webhook report."""

    __slots__ = ("has_next", "interrupts", "values")

    def __init__(self) -> None:
        # None while unknown: the run stopped before its state could be read.
        self.values: dict[str, Any] | None = None
        self.interrupts: dict[str, Any] | None = None
        self.has_next: bool | None = None

    def ending(self, status: RunStatus, error: dict[str, str] | None = None) -> Ending:
        """The run's ending at *status*, by the Agent Server's rule for its thread.

        The thread is in error after a failure, interrupted when the graph has
        steps left (an interrupt it paused on, a breakpoint, a cancel mid-run),
        else idle; it keeps the values and interrupts the run left, when known.
        """
        failed = status in ("error", "timeout")
        thread_status: ThreadStatus = "error" if failed else "interrupted" if self.has_next else "idle"
        state = None if self.values is None else {"values": self.values, "interrupts": self.interrupts or {}}
        return Ending(status, thread_status, error, state)


async def _read_final_state(graph: Pregel, job: RunJob, final: _FinalState) -> None:
    """Read where the run stopped, however it stopped; left unknown when the state cannot be read."""
    try:
        snapshot = await graph.aget_state(create_thread_config(job.identity.thread_id, job.user))
    except Exception:
        logger.warning("Could not read the run's final state", run_id=job.identity.run_id, exc_info=True)
        return
    final.values, final.interrupts = snapshot_values(snapshot)
    final.has_next = bool(snapshot.next)


async def _stream_graph(job: RunJob, final: _FinalState) -> None:
    """Build the run's graph and stream its events to the broker; *final* gets the state it stops at."""
    run_config = create_run_config(job)
    # The server's markers are for the server; the graph sees none of them.
    configurable = run_config.setdefault("configurable", {})
    mirrors_context = configurable.pop(CONTEXT_PROVIDED, False)
    events_v2 = configurable.pop(EVENT_STREAMING_V2, False)
    if tracer := _replica(configurable.pop(LANGSMITH_PROJECT, None), configurable.pop(LANGSMITH_EXAMPLE_ID, None)):
        run_config["callbacks"] = [tracer]
    execution_input = _resolve_input(job)
    async with get_langgraph_service().get_graph(
        job.identity.graph_id,
        run_config,
        access_context="threads.create_run",
        user=job.user,
        context=job.kwargs.context,
    ) as graph:
        try:
            # Trimmed here rather than inside one producer: both reach LangGraph, and
            # a key the graph never declared is a TypeError on either of them.
            context = await resolve_graph_context(graph, job.kwargs.context)
            if mirrors_context:
                # What the context schema left out leaves the mirror too.
                given = job.kwargs.context or {}
                for key in given.keys() - context.keys():
                    if configurable.get(key) == given[key]:
                        configurable.pop(key, None)
            options = _options(job, context)
            if events_v2:
                events = stream_native_v3_events(graph, execution_input, run_config, options)
            else:
                events = stream_graph_events(
                    graph,
                    execution_input,
                    run_config,
                    stream_mode=_resolve_stream_modes(job.kwargs.stream_mode),
                    subgraphs=job.kwargs.subgraphs,
                    options=options,
                )
            run_id = job.identity.run_id
            async for event in events:
                await broker_manager.publish(run_id, event)
            # The graph has returned: the stream's last event before its end, as on the Agent Server.
            keys = job.kwargs.feedback_keys
            if keys and not events_v2 and (urls := await _feedback_urls(run_id, keys)):
                await broker_manager.publish(run_id, ("feedback", urls))
        finally:
            await _read_final_state(graph, job, final)


def _options(job: RunJob, context: dict[str, Any]) -> dict[str, Any]:
    """The astream arguments the run set; unset ones are left out, so LangGraph keeps its defaults."""
    options = {
        "context": context,
        "durability": job.kwargs.durability,
        "interrupt_before": job.kwargs.interrupt_before,
        "interrupt_after": job.kwargs.interrupt_after,
    }
    return {name: value for name, value in options.items() if value is not None}


@cache
def _langsmith_client() -> Client:
    return Client()


async def _feedback_urls(run_id: str, keys: list[str]) -> dict[str, str] | None:
    """Presigned LangSmith URLs to post the run's feedback to, one per key; None when LangSmith cannot make them.

    The run is traced under its own id (``create_run_config``), so the feedback
    lands on its trace. Without LangSmith the run still succeeds, without the event.
    """
    try:
        tokens = await asyncio.to_thread(_langsmith_client().create_presigned_feedback_tokens, run_id, keys)
    except Exception:
        logger.warning("Could not create the run's feedback URLs", run_id=run_id, exc_info=True)
        return None
    return {key: token.url for key, token in zip(keys, tokens, strict=False)}


def _replica(project: str | None, example_id: str | None) -> LangChainTracer | None:
    """A tracer to the LangSmith project the run named, besides the default one, while tracing is on."""
    if not project or not ls_utils.tracing_is_enabled() or project == ls_utils.get_tracer_project():
        return None
    return LangChainTracer(project_name=project, example_id=example_id)


def _resolve_input(job: RunJob) -> Any:
    """The graph's input: the run's command as LangGraph's, else its input."""
    if job.kwargs.command is not None:
        return map_command_to_langgraph(job.kwargs.command)
    return job.kwargs.input


def _resolve_stream_modes(stream_mode: StreamMode | list[StreamMode] | None) -> list[str]:
    """The modes a v1 stream carries: values when the run named none, and never the v2 stream's own."""
    if stream_mode is None:
        return _DEFAULT_STREAM_MODES.copy()
    return [mode for mode in stream_modes(stream_mode) if mode not in _V2_MODES]


async def _signal_run_done(run_id: str) -> None:
    """Tell the run's waiters it is done; best effort, as they also watch its status."""
    try:
        await broker_manager.publish_done(run_id)
    except Exception:
        logger.debug("Could not publish that the run is done", run_id=run_id, exc_info=True)
