"""A run's events as its SSE streams send them, and the ends of its stream: completed, failed, cancelled."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import structlog
from langgraph.errors import EmptyInputError, GraphBubbleUp, InvalidUpdateError, TaskNotFound
from sse_starlette import ServerSentEvent
from starlette.exceptions import HTTPException

from zorvex_api.core import sse
from zorvex_api.models import Run
from zorvex_api.models.enums import TERMINAL_RUN_STATUSES
from zorvex_api.services.base_broker import UNRESUMABLE_RETENTION_SECONDS, event_sequence, is_end
from zorvex_api.services.broker import broker_manager, live_events

logger = structlog.getLogger(__name__)

# Exceptions whose message an error event carries, as the Agent Server's does: Python's
# and LangGraph's own, whose text describes the graph rather than the server, and HTTP errors.
_SAFE_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    RecursionError,
    TimeoutError,
    InvalidUpdateError,
    GraphBubbleUp,
    EmptyInputError,
    TaskNotFound,
    HTTPException,
)

# Sent whatever modes a join asks for: the run's start, its end, its error, and
# the feedback URLs its creator asked for with feedback_keys.
_CONTROL_MODES = frozenset({"metadata", "end", "error", "feedback"})


def error_event(exc: BaseException) -> dict[str, str]:
    """The ``error`` event a run that raised *exc* ends with: its type, and its message when that is safe to show."""
    message = str(exc) if isinstance(exc, _SAFE_ERRORS) else "An internal error occurred"
    return {"error": type(exc).__name__, "message": message}


def to_sse(event_id: str, event: tuple[str, Any]) -> ServerSentEvent:
    """The SSE event for a broker event."""
    mode, payload = event
    return sse.event(mode, payload, event_id)


def matches(event: tuple[str, Any], modes: frozenset[str] | None) -> bool:
    """Whether a join asking for *modes* (None: all) carries *event*, as the Agent Server filters joins.

    A subgraph's ``mode|namespace`` event goes with its mode; ``messages`` and
    ``messages-tuple`` take every ``messages*`` event.
    """
    mode = event[0]
    if modes is None or mode in _CONTROL_MODES:
        return True
    base = mode.partition("|")[0]
    return base in modes or (base.startswith("messages") and not modes.isdisjoint({"messages", "messages-tuple"}))


async def stream_run(
    run: Run, last_event_id: str | None = None, stream_mode: Sequence[str] | None = None
) -> AsyncGenerator[ServerSentEvent]:
    """The run's events after *last_event_id*, stored then live, in *stream_mode* (all without one), to its end.

    A client that goes cancels the run, or not, at the transport (``sse.response``'s close handler).
    """
    run_id = run.run_id
    allowed = frozenset(stream_mode) if stream_mode else None
    try:
        last_sent = event_sequence(last_event_id) if last_event_id else 0
        # Stored events first, read without registering a broker for a run that may be long gone.
        for event_id, event in await broker_manager.replay(run_id, last_event_id):
            last_sent = max(last_sent, event_sequence(event_id))
            if matches(event, allowed):
                yield to_sse(event_id, event)
            if is_end(event):
                return

        # A run recorded as ended has its end stored, or on its way for a moment.
        ended = run.status in TERMINAL_RUN_STATUSES
        async for event_id, event in live_events(run_id, ended_at=run.updated_at if ended else None):
            if event_sequence(event_id) <= last_sent:
                continue
            last_sent = event_sequence(event_id)
            if matches(event, allowed):
                yield to_sse(event_id, event)
            if is_end(event):
                return

        # An ended run whose end did not come has no events left to send: they are
        # no longer kept, or never were (stream_resumable). Its end is all there is.
        if ended:
            yield sse.event("end", {"status": run.status})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # A 500 cannot go out mid-stream; the stream ends with an error event instead,
        # without the exception's text, which can name paths and internal ids.
        logger.exception("stream execution failed", run_id=run_id)
        yield sse.event("error", {"error": type(e).__name__, "message": "execution failed"})


async def end_run(run_id: str, status: str, error: dict[str, str] | None = None) -> None:
    """End the run's stream (its *error* first, for a failed or cancelled run) and release its broker here."""
    await broker_manager.end_stream(run_id, status, error)
    broker_manager.cleanup_broker(run_id)


async def cancel_run(run_id: str, *, end_error: dict[str, str] | None) -> None:
    """Stop the run's task wherever it executes; with *end_error* (``cancelled_error``), end its stream too.

    A broker that cannot be reached leaves the run to its worker's own cancel check.
    """
    try:
        await broker_manager.request_cancel(run_id, end_error=end_error)
    except Exception:
        logger.exception("Could not cancel the run's task", run_id=run_id)


async def cleanup_run(run_id: str, *, resumable: bool) -> None:
    """Release the run's broker once it has ended; a run without stream_resumable keeps its events briefly."""
    if not resumable:
        await broker_manager.retain(run_id, UNRESUMABLE_RETENTION_SECONDS)
    broker_manager.cleanup_broker(run_id)
