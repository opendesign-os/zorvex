"""Run endpoints for Agent Protocol"""

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, MutableMapping
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import load_only
from sse_starlette import EventSourceResponse, ServerSentEvent

from zorvex_api.core import sse
from zorvex_api.core.auth import auth_dependency
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import SessionDep
from zorvex_api.models import Run, RunCreate, RunsCancel
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES, RunCancellationAction, RunStatus
from zorvex_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from zorvex_api.models.runs import RUN_EXECUTION_PARAMS_FIELDS, RunSelectField, run_fields_from_execution_params
from zorvex_api.services import checkpoints, run_waiters, streaming_service
from zorvex_api.services.ownership import on_visible_threads, owned_run, owned_thread
from zorvex_api.services.run_cleanup import (
    cancel_on_disconnect,
    request_run_interruption,
    stop_run_for_delete,
)
from zorvex_api.services.run_preparation import prepare_run
from zorvex_api.settings import settings

CloseHandler = Callable[[MutableMapping[str, Any]], Awaitable[None]]

router = APIRouter(tags=["Thread Runs"], dependencies=auth_dependency)

logger = structlog.getLogger(__name__)

_CANCEL_ACTION_DEFAULT: RunCancellationAction = "interrupt"
_CANCEL_ACTION_DOC = (
    "How to cancel: 'interrupt' stops the run, which ends interrupted; 'rollback' stops it and deletes it "
    "with the checkpoints it wrote, leaving the thread as the runs before it left it."
)
_NOTHING_TO_CANCEL = (
    "No matching runs to cancel. Please verify the thread ID and run IDs are correct, "
    "and the runs haven't been deleted or completed."
)


def _run_links(thread_id: str, run_id: str, attach: Literal["join", "stream"]) -> dict[str, str]:
    """``Content-Location`` names the run; ``Location`` is where a client attaches to it again."""
    run_url = f"/threads/{thread_id}/runs/{run_id}"
    return {"Location": f"{run_url}/{attach}", "Content-Location": run_url}


def _cancel_on_disconnect_handler(run_id: str) -> CloseHandler:
    async def handler(_msg: MutableMapping[str, Any]) -> None:
        await cancel_on_disconnect(run_id)

    return handler


def stream_response(
    run: Run,
    *,
    cancel_on_disconnect: bool,
    last_event_id: str | None = None,
    stream_mode: list[str] | None = None,
) -> EventSourceResponse:
    """The run's SSE stream after *last_event_id*, in *stream_mode*; a client that leaves cancels it on request.

    Only on request: other clients may follow the same run through join or stream.
    """
    return sse.response(
        streaming_service.stream_run(run, last_event_id, stream_mode),
        close_handler=_cancel_on_disconnect_handler(run.run_id) if cancel_on_disconnect else None,
        headers=_run_links(run.thread_id, run.run_id, "stream"),
    )


def wait_response(thread_id: str, run_id: str, body: AsyncIterator[bytes]) -> StreamingResponse:
    """The wait for a run as chunked JSON: *body*, ``run_waiters.wait_body``'s heartbeats then the result."""
    return StreamingResponse(body, media_type="application/json", headers=_run_links(thread_id, run_id, "join"))


def run_location(run: Run) -> dict[str, str]:
    """The Content-Location of a run just created, as the Agent Server sends it."""
    return {"Content-Location": f"/threads/{run.thread_id}/runs/{run.run_id}"}


@router.post("/threads/{thread_id}/runs", response_model=Run, responses={**NOT_FOUND, **CONFLICT})
async def create_run(
    thread_id: str,
    request: RunCreate,
    session: SessionDep,
    response: Response,
    authorize: Authorization = Authorize("threads", "create_run"),
) -> Run:
    """Create and execute a new run.

    Starts graph execution asynchronously and returns the run record
    immediately with status `pending`. Poll the run or use the stream
    endpoint to follow progress. Provide either `input` or `command` (for
    human-in-the-loop resumption) but not both.
    """
    user = authorize.user
    run = await prepare_run(session, thread_id, request, user, authorize=authorize)
    response.headers.update(run_location(run))
    return run


@router.post("/threads/{thread_id}/runs/stream", responses={**SSE_RESPONSE, **NOT_FOUND, **CONFLICT})
async def create_and_stream_run(
    thread_id: str,
    request: RunCreate,
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "create_run"),
) -> EventSourceResponse:
    """Create a new run and stream its execution via SSE.

    Returns a `text/event-stream` response with Server-Sent Events. Each
    event has a `type` field (e.g. `values`, `updates`, `messages`,
    `metadata`, `end`) and a JSON `data` payload.

    The run goes on when the client disconnects, unless `on_disconnect` is
    `"cancel"`. Use `stream_mode` to control which event types are emitted.

    A periodic SSE keepalive comment is sent every
    ``KEEPALIVE_INTERVAL_SECS`` so idle proxies don't drop long-running
    silent nodes (e.g. agents holding an upstream WebSocket).
    """
    user = authorize.user
    run = await prepare_run(session, thread_id, request, user, authorize=authorize)
    return stream_response(run, cancel_on_disconnect=request.on_disconnect == "cancel")


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=Run, responses={**NOT_FOUND})
async def get_run(
    thread_id: str,
    run_id: str,
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "read"),
) -> Run:
    """Get a run by its ID.

    Returns the current state of the run including its status, input, output,
    and error information.
    """
    user = authorize.user
    # Reading a run authorizes as a thread read (Agent Protocol has no `runs`
    # resource); @auth.on.threads.read covers it.
    filters = await authorize({"thread_id": thread_id, "run_id": run_id})
    return Run.model_validate(await owned_run(session, thread_id, run_id, user, filters=filters))


@router.get("/threads/{thread_id}/runs", response_model=list[Run], responses={**NOT_FOUND})
async def list_runs(
    thread_id: str,
    session: SessionDep,
    limit: int = Query(10, ge=1, le=settings.app.MAX_SEARCH_LIMIT, description="Maximum number of runs to return"),
    offset: int = Query(0, ge=0, description="Number of runs to skip for pagination"),
    status: RunStatus | None = Query(None, description="Only runs in this status."),
    fields: Annotated[
        list[RunSelectField] | None,
        Query(
            alias="select",
            description="Run fields to return. Omit to return every field. "
            "Leave out `output` to skip loading each run's final state.",
        ),
    ] = None,
    authorize: Authorization = Authorize("threads", "search"),
) -> list[Run] | JSONResponse:
    """List runs for a thread.

    Returns runs ordered by creation time (newest first). Use `status` to
    filter, `limit`/`offset` to paginate and `select` to return only some fields.
    """
    user = authorize.user
    filters = await authorize({"thread_id": thread_id, "status": status, "limit": limit, "offset": offset})
    stmt = (
        select(RunORM)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
            *([RunORM.status == status] if status else []),
        )
        .limit(limit)
        .offset(offset)
        .order_by(RunORM.created_at.desc())
    )
    stmt = on_visible_threads(stmt, filters)
    names: list[str] | None = list(dict.fromkeys(fields)) if fields else None
    if names:
        # Unselected columns stay out of the query: one run's output can be the whole thread history.
        columns = [name for name in names if name not in RUN_EXECUTION_PARAMS_FIELDS]
        if len(columns) < len(names):
            columns.append("execution_params")
        stmt = stmt.options(load_only(*(getattr(RunORM, name) for name in columns)))
    rows = (await session.scalars(stmt)).all()
    if names:
        return JSONResponse(jsonable_encoder([_selected_run_fields(row, names) for row in rows]))
    return [Run.model_validate(r) for r in rows]


def _selected_run_fields(row: RunORM, names: list[str]) -> dict[str, Any]:
    """The *names* fields of *row*, reading only the columns ``list_runs`` loaded."""
    derived = (
        run_fields_from_execution_params(row.execution_params)
        if RUN_EXECUTION_PARAMS_FIELDS.intersection(names)
        else {}
    )
    return {name: derived[name] if name in derived else getattr(row, name) for name in names}


@router.get("/threads/{thread_id}/runs/{run_id}/join", responses={**NOT_FOUND})
async def join_run(
    thread_id: str,
    run_id: str,
    request: Request,
    session: SessionDep,
    cancel_on_disconnect: bool = Query(False, description="Cancel the run when this client disconnects."),
    authorize: Authorization = Authorize("threads", "read"),
) -> Response:
    """Wait for a run to complete and return its result.

    Returns a chunked ``application/json`` response. While the run is still
    executing, the server sends periodic ``\\n`` heartbeat bytes to keep the
    connection alive through proxies and load balancers (AWS ALB, Cloudflare,
    etc.). The final chunk is the JSON result. Leading whitespace is ignored
    by JSON parsers, so clients can parse the concatenated body normally.

    The result is the run's last values, with ``__interrupt__`` listing the
    interrupts it paused on. A run that failed, was cancelled, or that the
    wait gave up on returns ``{"__error__": {"error": ..., "message": ...}}``.
    ``langgraph_sdk``'s ``runs.join`` hands that body back as-is, unlike
    ``runs.wait``, which inspects the key and raises.
    """
    user = authorize.user
    filters = await authorize({"thread_id": thread_id, "run_id": run_id})
    run = await owned_run(session, thread_id, run_id, user, filters=filters)
    if run.status in TERMINAL_RUN_STATUSES:
        result = await run_waiters.run_result(run_id, thread_id, user)
        return Response(run_waiters.encode_result(result), media_type="application/json")

    leaves = request.is_disconnected if cancel_on_disconnect else None
    return wait_response(thread_id, run_id, run_waiters.wait_body(run_id, thread_id, user, is_disconnected=leaves))


@router.post("/threads/{thread_id}/runs/wait", responses={**NOT_FOUND, **CONFLICT})
async def wait_for_run(
    thread_id: str,
    request: RunCreate,
    session: SessionDep,
    http: Request,
    authorize: Authorization = Authorize("threads", "create_run"),
) -> StreamingResponse:
    """Create a run, execute it, and wait for completion.

    Returns a chunked ``application/json`` response with periodic ``\\n``
    heartbeat bytes to keep the connection alive. The final chunk is the
    JSON result. Uses ``BG_JOB_TIMEOUT_SECS`` (default 1 hour) as the
    safety-net timeout.

    The result is the run's last values, with ``__interrupt__`` listing the
    interrupts it paused on. A run that failed, was cancelled, or that the wait
    gave up on returns ``{"__error__": {"error": ..., "message": ...}}``; the
    LangGraph SDK reads that key and raises.
    """
    user = authorize.user
    run = await prepare_run(session, thread_id, request, user, authorize=authorize)
    leaves = http.is_disconnected if request.on_disconnect == "cancel" else None
    body = run_waiters.wait_body(run.run_id, run.thread_id, user, is_disconnected=leaves)
    return wait_response(run.thread_id, run.run_id, body)


@router.get("/threads/{thread_id}/runs/{run_id}/stream", responses={**SSE_RESPONSE, **NOT_FOUND})
async def stream_run(
    thread_id: str,
    run_id: str,
    session: SessionDep,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    stream_mode: list[str] | None = Query(
        None, description="Narrow this connection to a subset of the modes the run was created with."
    ),
    cancel_on_disconnect: bool = Query(False, description="Cancel the run when this client disconnects."),
    authorize: Authorization = Authorize("threads", "read"),
) -> EventSourceResponse:
    """Stream an existing run's execution via SSE.

    Attach to a run that was created without streaming (e.g. via the create
    endpoint) to receive its events in real time. If the run has already
    finished, a single `end` event is emitted. Use the `Last-Event-ID`
    header to resume from a specific event after a disconnect.

    A periodic SSE keepalive comment is sent every
    ``KEEPALIVE_INTERVAL_SECS`` so idle proxies don't drop attached streams.
    """
    user = authorize.user
    filters = await authorize({"thread_id": thread_id, "run_id": run_id})
    run = Run.model_validate(await owned_run(session, thread_id, run_id, user, filters=filters))
    headers = _run_links(thread_id, run_id, "stream")

    # A finished run has nothing left to send but its end; with Last-Event-ID
    # the stream replays the events the client missed before ending.
    if run.status in TERMINAL_RUN_STATUSES and not last_event_id:

        async def end_only() -> AsyncGenerator[ServerSentEvent]:
            yield sse.event("end", {"status": run.status})

        return sse.response(end_only(), headers=headers)

    return stream_response(
        run, cancel_on_disconnect=cancel_on_disconnect, last_event_id=last_event_id, stream_mode=stream_mode
    )


@router.post(
    "/threads/{thread_id}/runs/{run_id}/cancel",
    status_code=202,
    response_model=None,
    responses={
        **NOT_FOUND,
        200: {"description": "With wait: the run's result once it has stopped, as join answers it."},
    },
)
async def cancel_run_endpoint(
    thread_id: str,
    run_id: str,
    session: SessionDep,
    wait: bool = Query(False, description="Answer once the run has stopped, with its result, as join does."),
    action: RunCancellationAction = Query(_CANCEL_ACTION_DEFAULT, description=_CANCEL_ACTION_DOC),
    authorize: Authorization = Authorize("threads", "update"),
) -> Response:
    """Cancel a pending or running run: interrupt it, or roll it back.

    Answers 202 at once; the run stops shortly after. With `wait`, the answer
    is the run's result once it has stopped, as join answers it: an
    interrupted run's `{"__error__": {"error": "UserInterrupt", "message": ...}}`,
    or a rolled-back run's `UserRollback`, the message saying why. A run that
    already finished is not found (404).
    """
    user = authorize.user
    filters = await authorize({"thread_id": thread_id, "run_id": run_id, "action": action})
    run = await owned_run(session, thread_id, run_id, user, filters=filters)
    if run.status in TERMINAL_RUN_STATUSES:
        raise HTTPException(404, _NOTHING_TO_CANCEL)
    logger.info("Cancelling run", run_id=run_id, thread_id=thread_id, action=action)
    reason = f"{action} requested via POST /threads/{{thread_id}}/runs/{{run_id}}/cancel"
    await request_run_interruption(session, run, action=action, reason=reason)
    if not wait:
        return Response(status_code=202)
    # Hold no pool connection while the run stops.
    await session.commit()
    return wait_response(thread_id, run_id, run_waiters.wait_body(run_id, thread_id, user))


@router.post("/runs/cancel", status_code=204, responses={**NOT_FOUND})
async def cancel_runs(
    request: RunsCancel,
    session: SessionDep,
    action: RunCancellationAction = Query(_CANCEL_ACTION_DEFAULT, description=_CANCEL_ACTION_DOC),
    authorize: Authorization = Authorize("threads", "update"),
) -> None:
    """Cancel several runs at once, by status or by thread_id plus run_ids: interrupt them, or roll them back.

    Runs the caller cannot see are left out, and runs that already finished
    are skipped. When no run is selected, or none of those selected is still
    pending or running, the answer is 404.
    """
    user = authorize.user
    filters = await authorize({**request.model_dump(), "action": action})
    stmt = select(RunORM).where(RunORM.user_id == user.identity)
    if request.thread_id is not None:
        thread = await owned_thread(session, request.thread_id, user, filters=filters)
        stmt = stmt.where(RunORM.thread_id == thread.thread_id, RunORM.run_id.in_(request.run_ids or []))
    else:
        statuses = ACTIVE_RUN_STATUSES if request.status == "all" else (request.status,)
        stmt = on_visible_threads(stmt.where(RunORM.status.in_(statuses)), filters)

    runs = (await session.scalars(stmt)).all()
    if not runs:
        raise HTTPException(404, "No runs found to cancel.")
    active = [run for run in runs if run.status in ACTIVE_RUN_STATUSES]
    if not active:
        raise HTTPException(404, _NOTHING_TO_CANCEL)
    logger.info("Cancelling runs", action=action, selector=request.model_dump(exclude_none=True), count=len(active))
    for run in active:
        await request_run_interruption(session, run, action=action, reason=f"{action} requested via POST /runs/cancel")


@router.delete("/threads/{thread_id}/runs/{run_id}", status_code=204, responses={**NOT_FOUND})
async def delete_run(
    thread_id: str,
    run_id: str,
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "delete"),
) -> None:
    """Delete a run and the checkpoints it wrote; an active run is stopped first."""
    user = authorize.user
    # Deleting a run authorizes as a thread delete (Agent Protocol has no `runs`
    # resource); @auth.on.threads.delete covers it.
    filters = await authorize({"thread_id": thread_id, "run_id": run_id})
    run = await owned_run(session, thread_id, run_id, user, filters=filters)
    if run.status in ACTIVE_RUN_STATUSES:
        logger.info("Cancelling run before deleting it", run_id=run_id, thread_id=thread_id)
        await stop_run_for_delete(run)
    await checkpoints.delete_run(thread_id, run_id)
    await session.delete(run)
    await session.commit()
