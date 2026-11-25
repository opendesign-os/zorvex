"""Streaming service for orchestrating SSE streaming."""

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
from zorvex_api.services.base_broker import UNRESUMABLE_RETENTION_SECONDS, is_end
from zorvex_api.services.broker import broker_manager, live_events
from zorvex_api.services.event_converter import matches, to_sse
from zorvex_api.utils import extract_event_sequence

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


def error_event(exc: BaseException) -> dict[str, str]:
    """The ``error`` event a run that raised *exc* ends with: its type, and its message when that is safe to show."""
    message = str(exc) if isinstance(exc, _SAFE_ERRORS) else "An internal error occurred"
    return {"error": type(exc).__name__, "message": message}


class StreamingService:
    """Service to handle SSE streaming orchestration.

    Events flow through the broker which handles both live broadcast
    and replay storage. No separate event store needed.
    """

    async def put_to_broker(
        self,
        run_id: str,
        event_id: str,
        raw_event: Any,
    ) -> None:
        """Put an event into the run's broker for live consumers and replay storage."""
        broker = broker_manager.get_or_create_broker(run_id)
        await broker.put(event_id, raw_event)

    async def signal_run_cancelled(self, run_id: str, error: dict[str, str] | None = None) -> None:
        """End a cancelled run's stream with *error* (``cancelled_error``); works from any instance."""
        await broker_manager.end_cancelled(run_id, error)
        broker_manager.cleanup_broker(run_id)

    async def signal_run_error(self, run_id: str, error: dict[str, str], status: str = "error") -> None:
        """End a failed run's stream with *error* (see :func:`error_event`), then its end with *status*.

        Uses get_or_create_broker so this works cross-instance.
        """
        broker = broker_manager.get_or_create_broker(run_id)
        if broker.is_finished():
            return

        error_event_id = await broker_manager.allocate_event_id(run_id)
        await broker.put(error_event_id, ("error", error))

        end_event_id = await broker_manager.allocate_event_id(run_id)
        await broker.put(end_event_id, ("end", {"status": status}))
        broker_manager.cleanup_broker(run_id)

    async def stream_run_execution(
        self,
        run: Run,
        last_event_id: str | None = None,
        stream_mode: Sequence[str] | None = None,
    ) -> AsyncGenerator[ServerSentEvent]:
        """Stream run execution with unified producer-consumer pattern.

        Cancellation on client disconnect is handled at the transport layer
        via ``EventSourceResponse(client_close_handler_callable=...)`` — the
        generator only propagates ``CancelledError`` cleanup without side
        effects on the broker.
        """
        run_id = run.run_id
        allowed = frozenset(stream_mode) if stream_mode else None
        try:
            last_sent_sequence = extract_event_sequence(last_event_id) if last_event_id else 0

            # Stored events first, read without registering a broker for a run that may be long gone.
            for event_id, raw_event in await broker_manager.replay(run_id, last_event_id):
                last_sent_sequence = max(last_sent_sequence, extract_event_sequence(event_id))
                if matches(raw_event, allowed):
                    yield to_sse(event_id, raw_event)
                if is_end(raw_event):
                    return

            # A run recorded as ended has its end stored, or on its way for a moment.
            ended = run.status in TERMINAL_RUN_STATUSES
            async for event_id, raw_event in live_events(run_id, ended_at=run.updated_at if ended else None):
                sequence = extract_event_sequence(event_id)
                if sequence <= last_sent_sequence:
                    continue  # already replayed
                last_sent_sequence = sequence
                if matches(raw_event, allowed):
                    yield to_sse(event_id, raw_event)
                if is_end(raw_event):
                    return

            # An ended run whose end did not come has no events left to send: they are
            # no longer kept, or never were (stream_resumable). Its end is all there is.
            if ended:
                yield sse.event("end", {"status": run.status})

        except asyncio.CancelledError:
            logger.debug(f"Stream cancelled for run {run_id}")
            raise
        except Exception as e:
            # Swallow rather than re-raise: this is the SSE producer for
            # an active connection. Re-raising would surface a 500 mid-
            # stream, which clients can't reconnect to. We instead emit a
            # structured error event so consumers can react cleanly.
            # str(e) is suppressed on the wire because Python exception
            # messages can leak file paths, internal IDs, and frame hints.
            logger.exception("stream execution failed", run_id=run_id)
            yield sse.event("error", {"error": type(e).__name__, "message": "execution failed"})

    async def cancel_run(
        self, run_id: str, *, emit_end_event: bool = True, error: dict[str, str] | None = None
    ) -> bool:
        """Cancel a pending or running execution; its stream ends with *error* (``cancelled_error``).

        Delegates to broker_manager for cross-instance support via Redis pub/sub.
        """
        try:
            await broker_manager.request_cancel(run_id, emit_end_event=emit_end_event, error=error)
            return True
        except Exception as e:
            logger.error(f"Error cancelling run {run_id}: {e}")
            return False

    async def cleanup_run(self, run_id: str, *, resumable: bool = True) -> None:
        """Release the run's broker once it has ended; a run without stream_resumable keeps its events briefly."""
        if not resumable:
            await broker_manager.retain(run_id, UNRESUMABLE_RETENTION_SECONDS)
        broker_manager.cleanup_broker(run_id)


# Global streaming service instance
streaming_service = StreamingService()
