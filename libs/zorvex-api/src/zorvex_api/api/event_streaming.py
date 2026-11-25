"""Agent Protocol v2 event streaming endpoints.

* ``POST /threads/{thread_id}/stream/events``: the thread's run events over
  SSE, filtered by channel; the body is an ``EventStreamRequest``.
* ``POST /threads/{thread_id}/commands``: one thread command (``run.start``,
  ``input.respond``, ``agent.getTree``), answered with its envelope.
* ``/threads/{thread_id}/stream/events`` over a WebSocket: both, on one socket.

Each authorizes reading the thread first, and checks it is the caller's.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import anyio
import structlog
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sse_starlette import EventSourceResponse, ServerSentEvent

from zorvex_api.core import sse
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.core.orm import Assistant as AssistantORM
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import SessionDep, get_session_maker
from zorvex_api.models import User
from zorvex_api.models.event_streaming import EventStreamRequest, ThreadCommand
from zorvex_api.services.event_streaming import ws
from zorvex_api.services.event_streaming.commands import INVALID_COMMAND, handle_command
from zorvex_api.services.event_streaming.protocol import build_error
from zorvex_api.services.event_streaming.session import RunLister, ThreadEventSession, validate_channels
from zorvex_api.services.ownership import existing_thread, owned_thread

logger = structlog.getLogger(__name__)

# Each route authenticates itself: a router-level Request dependency would fail the WebSocket route.
router = APIRouter(tags=["Streaming"])


def _thread_run_lister(thread_id: str, user: User) -> RunLister:
    """Async callable returning the thread's (run_id, status, graph_name) rows (oldest first), user-scoped.

    Called repeatedly while a stream is live so a run started after the stream
    opened is picked up. Status lets the session drain historical runs whose
    broker events expired instead of tailing them forever; graph_name feeds the
    run's root lifecycle events. Each tick uses a short-lived session so a
    connection is not held for the whole SSE lifetime.
    """

    async def list_run_ids() -> list[tuple[str, str | None, str | None]]:
        maker = get_session_maker()
        rows: list[tuple[str, str | None, str | None]] = []
        # sse-starlette's anyio cancel punches through asyncio.shield.
        with anyio.CancelScope(shield=True):
            async with maker() as session:
                result = await session.execute(
                    select(RunORM.run_id, RunORM.status, AssistantORM.graph_id)
                    .outerjoin(AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id)
                    .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
                    .order_by(RunORM.created_at.asc())
                )
                rows = [(run_id, status, graph_id) for run_id, status, graph_id in result.all()]
        return rows

    return list_run_ids


# Error responses both v2 routes can emit, for an accurate OpenAPI contract.
_V2_ERROR_RESPONSES: dict[int | str, dict[str, str]] = {
    400: {"description": "Invalid request (unsupported channels, malformed command)"},
    404: {"description": "Thread owned by another user, or one an @auth.on filter excludes"},
}


@router.post(
    "/threads/{thread_id}/stream/events",
    response_class=EventSourceResponse,
    responses={
        200: {"description": "SSE stream of protocol event envelopes", "content": {"text/event-stream": {}}},
        **_V2_ERROR_RESPONSES,
    },
)
async def stream_thread_events(
    thread_id: str,
    body: EventStreamRequest,
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "read"),
) -> EventSourceResponse:
    """Open a channel-filtered SSE stream of the thread's run events.

    Thread-scoped: events for any run on the thread flow through, so a
    client can open the stream then issue ``run.start``. Each SSE frame's
    ``data:`` is a protocol event envelope; ``id:`` is the ``seq`` a client
    echoes back as ``since`` on resume.
    """
    user = authorize.user
    channels, invalid = validate_channels(body.channels)
    if invalid:
        raise HTTPException(400, f"Unsupported channels: {', '.join(invalid)}")

    # A thread that does not exist yet is a 404, as on the Agent Server: the SDK
    # opens the stream again once its run.start has created the thread.
    filters = await authorize({"thread_id": thread_id})
    await owned_thread(session, thread_id, user, filters=filters)

    session_stream = ThreadEventSession(
        channels=channels,
        list_run_ids=_thread_run_lister(thread_id, user),
        since=body.since,
        namespaces=body.namespaces,
        depth=body.depth,
    )
    return sse.response(_frame_events(session_stream))


@router.post(
    "/threads/{thread_id}/commands",
    responses={
        200: {"description": "Protocol response envelope (success or error)"},
        400: {"description": "A body that is no command: an invalid_argument envelope"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": ThreadCommand.model_json_schema()}},
        }
    },
)
async def post_thread_command(
    thread_id: str,
    request: Request,
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "read"),
) -> JSONResponse:
    """Run one v2 command on the thread and answer its response envelope.

    Every outcome is an envelope on HTTP 200, as on the Agent Server, except a
    body that is no command (HTTP 400). Every command reads the thread; one that
    starts a run is also a ``threads.create_run``, and run.start may create it.
    """
    filters = await authorize({"thread_id": thread_id})
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "The command must be a JSON object") from None
    command_id = body.get("id") if isinstance(body, dict) else None
    if not isinstance(command_id, int) or isinstance(command_id, bool) or not isinstance(body.get("method"), str):
        return JSONResponse(build_error(None, "invalid_argument", INVALID_COMMAND), status_code=400)
    try:
        await existing_thread(session, thread_id, authorize.user, filters=filters)
    except HTTPException as exc:
        return JSONResponse(build_error(command_id, "no_such_run", str(exc.detail)))
    return JSONResponse(await handle_command(body, session=session, thread_id=thread_id, authorize=authorize))


@router.websocket("/threads/{thread_id}/stream/events")
async def thread_events_socket(websocket: WebSocket, thread_id: str) -> None:
    """The thread's events and commands over a WebSocket: subscription.subscribe/reconnect/unsubscribe and the
    commands POST /commands takes."""
    await ws.serve(websocket, thread_id, lambda user: _thread_run_lister(thread_id, user))


async def _frame_events(session_stream: ThreadEventSession) -> AsyncGenerator[ServerSentEvent]:
    """Frame v2 event envelopes as SSE messages (event=method, data=envelope, id=seq); a failure ends with an error."""
    try:
        async for envelope in session_stream.stream():
            yield sse.event(envelope["method"], envelope, str(envelope["seq"]))
    except Exception:
        logger.exception("Thread event stream failed")
        failure = {"type": "error", "id": None, "error": "unknown_error", "message": "Thread event stream failed."}
        yield sse.event("error", failure)
