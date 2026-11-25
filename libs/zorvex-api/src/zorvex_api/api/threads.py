"""Thread endpoints for Agent Protocol"""

import re
from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.pregel import Pregel
from langgraph.types import StateSnapshot
from langgraph_sdk.runtime import AccessContext
from sqlalchemy import ColumnElement, case, exists, func, literal, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from zorvex_api.core import sse
from zorvex_api.core.auth_deps import auth_dependency, require_auth
from zorvex_api.core.auth_filters import build_metadata_filter
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.core.database import db_manager
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import SessionDep, get_session_maker
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import ThreadTTL as ThreadTTLORM
from zorvex_api.models import (
    Thread,
    ThreadCheckpoint,
    ThreadCheckpointPostRequest,
    ThreadCreate,
    ThreadHistoryRequest,
    ThreadPruneRequest,
    ThreadPruneResponse,
    ThreadSearchRequest,
    ThreadState,
    ThreadStateUpdate,
    ThreadStateUpdateResponse,
    ThreadUpdate,
    User,
)
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES
from zorvex_api.models.errors import BAD_REQUEST, CONFLICT, NOT_FOUND, SSE_RESPONSE, AgentProtocolError
from zorvex_api.models.threads import ThreadCountRequest, ThreadStorage, ThreadTTLInfo
from zorvex_api.services import checkpoints, thread_events, thread_stream
from zorvex_api.services.langgraph_service import create_thread_config, get_langgraph_service, thread_graph_config
from zorvex_api.services.ownership import owned_thread
from zorvex_api.services.thread_service import ThreadService
from zorvex_api.services.thread_state_service import snapshot_to_thread_state, snapshot_values
from zorvex_api.services.thread_ttl import prune_threads_by_id
from zorvex_api.settings import settings
from zorvex_api.utils.pagination import pagination_headers
from zorvex_api.utils.run_utils import strip_pinned_config_keys

router = APIRouter(tags=["Threads"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


async def _read_thread(thread_id: str, user: User, filters: dict[str, Any] | None) -> ThreadORM:
    """The thread, read in a short-lived session, so the caller's checkpoint ops hold no pool connection."""
    async with get_session_maker()() as session:
        return await owned_thread(session, thread_id, user, filters=filters)


async def _search_conditions(request: ThreadCountRequest, authorize: Authorization) -> list[ColumnElement[bool]]:
    """WHERE clauses of a thread search or count, after authorizing it."""
    filters = await authorize(request.model_dump())
    conditions: list[ColumnElement[bool]] = [ThreadORM.user_id == authorize.user.identity]
    # Compile the handler filter rather than merging only its "metadata" key:
    # the flat shape and the $eq/$contains/$or/$and operators were dropped here.
    auth_filter = build_metadata_filter(ThreadORM.metadata_dict, filters)
    if auth_filter is not None:
        conditions.append(auth_filter)
    if request.status:
        conditions.append(ThreadORM.status == request.status)
    if request.metadata:
        # JSONB containment: type-correct, deep-nested, GIN-indexable.
        conditions.append(ThreadORM.metadata_dict.op("@>")(request.metadata))
    if request.values:
        conditions.append(ThreadORM.values.op("@>")(request.values))
    excluded = request.exclude_metadata
    for metadata in [excluded] if isinstance(excluded, dict) else excluded or []:
        conditions.append(~ThreadORM.metadata_dict.op("@>")(metadata))
    for column, after, before in (
        (ThreadORM.created_at, request.created_after, request.created_before),
        (ThreadORM.state_updated_at, request.state_updated_after, request.state_updated_before),
    ):
        if after is not None:
            conditions.append(column >= after)
        if before is not None:
            conditions.append(column < before)
    return conditions


async def _record_state_write(thread_id: str, user: User, state: StateSnapshot) -> None:
    """A state written outside a run is the thread's: its values, its state time, its status unless a run is on it."""
    has_next = bool(state.next)
    values, interrupts = snapshot_values(state)
    now = datetime.now(UTC)
    active = exists(
        select(RunORM.run_id)
        .where(RunORM.thread_id == ThreadORM.thread_id, RunORM.status.in_(ACTIVE_RUN_STATUSES))
        .correlate(ThreadORM)
    )
    status = case((active, ThreadORM.status), else_=literal("interrupted" if has_next else "idle"))
    async with get_session_maker()() as session:
        await session.execute(
            update(ThreadORM)
            .where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
            .values(updated_at=now, state_updated_at=now, status=status, values=values, interrupts=interrupts)
        )
        await session.commit()


def _serialize_thread(thread: ThreadORM) -> Thread:
    """The API view of a thread row."""
    return Thread.model_validate(
        {
            "thread_id": thread.thread_id,
            "status": thread.status,
            "metadata": thread.metadata_dict or {},
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
            "state_updated_at": thread.state_updated_at,
            "config": thread.config or {},
            "values": thread.values or {},
            "interrupts": thread.interrupts or {},
            "run_priority": thread.run_priority,
            "fairness_group": thread.fairness_group,
        }
    )


async def _thread_ttl_info(session: AsyncSession, thread_id: str) -> ThreadTTLInfo | None:
    row = await session.scalar(select(ThreadTTLORM).where(ThreadTTLORM.thread_id == thread_id))
    if row is None:
        return None
    return ThreadTTLInfo(strategy=row.strategy, ttl_minutes=row.ttl_minutes, expires_at=row.expires_at)


def _extract(thread: dict[str, Any], path: str) -> Any:
    """The value at an ``extract`` path in a thread's JSON view, or None where the path leads nowhere."""
    node: Any = thread
    for key, index in _EXTRACT_STEP.findall(path):
        if key and isinstance(node, dict):
            node = node.get(key)
        elif index and isinstance(node, list) and -len(node) <= int(index) < len(node):
            node = node[int(index)]
        else:
            return None
    return node


# The steps of an extract path: a key (leading or after a dot) or an [index].
_EXTRACT_STEP = re.compile(r"\.?([^.\[\]]+)|\[(-?\d+)\]")


# --- Endpoints ---


@router.post("/threads", response_model=Thread, responses={**CONFLICT})
async def create_thread(
    request: ThreadCreate,
    session: SessionDep,
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "create"),
) -> Thread:
    """Create a new conversation thread.

    Threads hold conversation state and checkpoint history. Provide a
    `thread_id` for idempotent creation, or let the server generate one.
    Set `if_exists` to `"do_nothing"` to return the existing thread when the
    ID already exists instead of raising a 409 conflict.
    """
    thread = await ThreadService(session, user).create(authorize, request)
    return _serialize_thread(thread)


@router.get("/threads/{thread_id}", response_model=Thread, responses={**NOT_FOUND})
async def get_thread(
    thread_id: str,
    session: SessionDep,
    include: str | None = Query(
        None, description="Comma-separated list of additional fields to include. Supported values: ttl"
    ),
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "read"),
) -> Thread:
    """Get a thread by its ID, with its current values and interrupts.

    Returns 404 if the thread does not exist or does not belong to the
    authenticated user.
    """
    requested = {part.strip() for part in (include or "").split(",") if part.strip()}

    thread = await ThreadService(session, user).get(authorize, thread_id)
    result = _serialize_thread(thread)
    if "ttl" in requested:
        result.ttl = await _thread_ttl_info(session, thread_id)
    return result


@router.patch("/threads/{thread_id}", response_model=Thread, responses={**NOT_FOUND})
async def update_thread(
    thread_id: str,
    request: ThreadUpdate,
    session: SessionDep,
    prefer: str | None = Header(None, description="return=minimal answers 204 without the thread."),
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "update"),
) -> Thread | Response:
    """Update a thread's metadata (merged over the existing keys) or TTL."""
    thread = await ThreadService(session, user).update(authorize, thread_id, request)
    if prefer and "return=minimal" in prefer.lower():
        return Response(status_code=204)
    return _serialize_thread(thread)


_EMPTY_CHECKPOINT_NS = ""


@asynccontextmanager
async def _thread_graph(
    authorize: Authorization, thread_id: str, configurable: dict[str, Any] | None = None
) -> AsyncGenerator[tuple[Pregel, RunnableConfig] | None]:
    """Authorize, then the thread's graph bound to its config, or None before a run has bound a graph to the thread."""
    user = authorize.user
    filters = await authorize({"thread_id": thread_id})
    thread = await _read_thread(thread_id, user, filters)
    graph_id = (thread.metadata_dict or {}).get("graph_id")
    if not graph_id:
        yield None
        return
    access_context: AccessContext = "threads.update" if authorize.action == "update" else "threads.read"
    config = create_thread_config(thread_id, user)
    config["configurable"].update(configurable or {})
    async with get_langgraph_service().get_graph(
        graph_id, thread_graph_config(thread.config, config), access_context=access_context, user=user
    ) as graph:
        yield graph.with_config(config), config


def _empty_state(thread_id: str) -> ThreadState:
    return ThreadState(
        values={},
        checkpoint=ThreadCheckpoint(checkpoint_id=None, thread_id=thread_id, checkpoint_ns=_EMPTY_CHECKPOINT_NS),
    )


@router.get("/threads/{thread_id}/stream", responses={**SSE_RESPONSE, **NOT_FOUND})
async def stream_thread(
    thread_id: str,
    session: SessionDep,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    stream_modes: list[str] = Query(
        [],
        description="What to stream: run_modes (each run's events, and run_done as each ends; the default), "
        "lifecycle (each run's start and run_done), state_update (the thread's state updates).",
    ),
    stream_mode: list[str] = Query([], description="Alias of stream_modes, the name the SDKs send."),
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "read"),
) -> EventSourceResponse:
    """Stream the thread: each of its runs in turn, as they come, until the client goes.

    Without `Last-Event-ID` the stream begins with the runs not yet ended; `-`
    replays every run whose events are still kept, and a run event's id
    resumes after it.
    """
    modes = thread_stream.parse_modes(stream_modes or stream_mode)
    if unknown := [mode for mode in modes if mode not in thread_stream.MODES]:
        raise HTTPException(422, f"Invalid stream mode: {unknown[0]}")
    filters = await authorize({"thread_id": thread_id})
    await owned_thread(session, thread_id, user, filters=filters)
    await session.commit()
    events = thread_stream.join(thread_id, user, frozenset(modes), last_event_id)
    return sse.response(
        events,
        headers={"Location": f"/threads/{thread_id}/stream", "Content-Location": f"/threads/{thread_id}"},
    )


@router.get("/threads/{thread_id}/state", response_model=ThreadState, responses={**NOT_FOUND})
async def get_thread_state(
    thread_id: str,
    subgraphs: bool = Query(False, description="Include states from subgraphs"),
    authorize: Authorization = Authorize("threads", "read"),
) -> ThreadState:
    """Get the current state of a thread.

    Returns the latest checkpoint's values, pending next nodes, interrupt
    data, and metadata. A thread no run has bound to a graph yet has an
    empty state.
    """
    return await _state(authorize, thread_id, None, subgraphs)


@router.post(
    "/threads/{thread_id}/state", response_model=ThreadStateUpdateResponse, responses={**NOT_FOUND, **BAD_REQUEST}
)
async def update_thread_state(
    thread_id: str,
    request: ThreadStateUpdate,
    authorize: Authorization = Authorize("threads", "update"),
) -> ThreadStateUpdateResponse:
    """Write a new checkpoint with `values`, as if `as_node` had just run.

    `checkpoint_id` or `checkpoint` fork from that checkpoint instead of the
    latest. With null `values` and an `as_node` such as ``__copy__`` (what
    LangGraph Studio sends for "Re-run from here"), the new checkpoint copies
    that one, anchoring a later run as a fork of it.
    """
    configurable: dict[str, Any] = {}
    if request.checkpoint_id:
        configurable["checkpoint_id"] = request.checkpoint_id
    if request.checkpoint:
        configurable.update(strip_pinned_config_keys(request.checkpoint))
    async with _thread_graph(authorize, thread_id, configurable) as bound:
        if bound is None:
            raise HTTPException(400, f"Thread '{thread_id}' has no associated graph. Cannot update state.")
        graph, config = bound
        updated = await graph.aupdate_state(config, request.values, as_node=request.as_node)
        state = await graph.aget_state(updated)
    await _record_state_write(thread_id, authorize.user, state)
    await thread_events.publish(
        thread_id, "state_update", {"state": snapshot_to_thread_state(state, thread_id), "thread_id": thread_id}
    )
    written = updated["configurable"]
    return ThreadStateUpdateResponse(
        checkpoint={
            "checkpoint_id": written.get("checkpoint_id"),
            "thread_id": thread_id,
            "checkpoint_ns": written.get("checkpoint_ns", _EMPTY_CHECKPOINT_NS),
        }
    )


@router.get("/threads/{thread_id}/state/{checkpoint_id}", response_model=ThreadState, responses={**NOT_FOUND})
async def get_thread_state_at_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    subgraphs: bool = Query(False, description="Include states from subgraphs"),
    authorize: Authorization = Authorize("threads", "read"),
) -> ThreadState:
    """Get the thread state at a specific checkpoint. 404 when the checkpoint does not exist."""
    return await _state(authorize, thread_id, {"checkpoint_id": checkpoint_id}, subgraphs)


async def _state(
    authorize: Authorization, thread_id: str, checkpoint: dict[str, Any] | None, subgraphs: bool
) -> ThreadState:
    """The thread's state at *checkpoint*, its latest without one; 404 for a named checkpoint that does not exist."""
    async with _thread_graph(authorize, thread_id, checkpoint) as bound:
        snapshot = None
        if bound is not None:
            graph, config = bound
            # LangGraph only exposes subgraph checkpoints while the run is interrupted.
            snapshot = await graph.aget_state(config, subgraphs=subgraphs)
    named = (checkpoint or {}).get("checkpoint_id")
    # A checkpoint that does not exist comes back as an empty snapshot without a creation time.
    if named and (snapshot is None or snapshot.created_at is None):
        raise HTTPException(404, f"No checkpoint '{named}' on thread '{thread_id}'")
    return _empty_state(thread_id) if snapshot is None else snapshot_to_thread_state(snapshot, thread_id)


@router.post("/threads/{thread_id}/state/checkpoint", response_model=ThreadState, responses={**NOT_FOUND})
async def get_thread_state_at_checkpoint_post(
    thread_id: str,
    request: ThreadCheckpointPostRequest,
    authorize: Authorization = Authorize("threads", "read"),
) -> ThreadState:
    """Get the thread state at the checkpoint the body names (what the SDK sends); its latest without an id."""
    checkpoint = request.checkpoint.model_dump(include={"checkpoint_id", "checkpoint_ns", "checkpoint_map"})
    named = {key: value for key, value in checkpoint.items() if value}
    return await _state(authorize, thread_id, named, bool(request.subgraphs))


_STATE_FIELDS = frozenset(
    {"values", "next", "tasks", "checkpoint", "metadata", "created_at", "parent_checkpoint", "interrupts"}
)
_VALUE_FIELD = re.compile(r"values\.[A-Za-z0-9_-]+")


def _selected(states: list[ThreadState], select: list[str]) -> list[dict[str, Any]]:
    """Each state with only the selected fields; values.<key> narrows values to those keys unless all are selected."""
    if bad := [field for field in select if field not in _STATE_FIELDS and not _VALUE_FIELD.fullmatch(field)]:
        raise HTTPException(422, f"Invalid select columns: {bad}")
    keys = {field.removeprefix("values.") for field in select if field.startswith("values.")}
    fields = {field for field in select if field in _STATE_FIELDS} | ({"values"} if keys else set())
    out = []
    for state in states:
        item = state.model_dump(mode="json", include=fields)
        if keys and "values" not in select:
            item["values"] = {key: value for key, value in item["values"].items() if key in keys}
        out.append(item)
    return out


@router.get("/threads/{thread_id}/history", response_model=list[ThreadState], responses={**NOT_FOUND})
async def get_thread_history(
    thread_id: str,
    limit: int = Query(10, ge=1, le=1000, description="Number of states to return."),
    before: str | None = Query(None, description="Return states before this checkpoint id."),
    select: list[str] | None = Query(None, description="Fields of each state to return, as on POST."),
    authorize: Authorization = Authorize("threads", "read"),
) -> list[ThreadState] | JSONResponse:
    """Get the checkpoint history of a thread's root graph, newest first."""
    return await _history(authorize, thread_id, ThreadHistoryRequest(limit=limit, before=before, select=select))


@router.post("/threads/{thread_id}/history", response_model=list[ThreadState], responses={**NOT_FOUND})
async def get_thread_history_post(
    thread_id: str,
    request: ThreadHistoryRequest,
    authorize: Authorization = Authorize("threads", "read"),
) -> list[ThreadState] | JSONResponse:
    """Get the checkpoint history for a thread, newest first.

    `limit` caps the page and `before` (a checkpoint id, checkpoint or config)
    continues it; `metadata` filters on checkpoint metadata, `checkpoint` picks
    a subgraph's namespace and `select` the fields of each state.
    """
    return await _history(authorize, thread_id, request)


async def _history(
    authorize: Authorization, thread_id: str, request: ThreadHistoryRequest
) -> list[ThreadState] | JSONResponse:
    configurable = strip_pinned_config_keys(request.checkpoint) if request.checkpoint else None
    # aget_state_history reads only checkpoint_id from `before`; the thread comes from the pinned config.
    before: RunnableConfig | None = None
    if isinstance(request.before, str):
        before = {"configurable": {"checkpoint_id": request.before}}
    elif isinstance(request.before, dict):
        before = cast(
            RunnableConfig, request.before if "configurable" in request.before else {"configurable": request.before}
        )
    budget = settings.app.LANGGRAPH_THREAD_HISTORY_MAX_STATE_BYTES
    states: list[ThreadState] = []
    async with _thread_graph(authorize, thread_id, configurable) as bound:
        if bound is not None:
            graph, config = bound
            # An async generator, typed as an iterator: closed at once when the budget stops it.
            history = cast(
                AsyncGenerator[StateSnapshot],
                graph.aget_state_history(config, filter=request.metadata, before=before, limit=request.limit),
            )
            async with aclosing(history) as snapshots:
                size = 1  # the response array's opening bracket
                async for snapshot in snapshots:
                    state = snapshot_to_thread_state(snapshot, thread_id)
                    if budget:
                        # Each entry counts whole, as loaded, with the comma or bracket after it.
                        size += len(state.model_dump_json()) + 1
                        if states and size > budget:
                            break
                    states.append(state)
                    if budget and size >= budget:
                        break
    if request.select:
        return JSONResponse(_selected(states, request.select))
    return states


@router.get("/threads/{thread_id}/storage", response_model=ThreadStorage, responses={**NOT_FOUND})
async def get_thread_storage(
    thread_id: str,
    session: SessionDep,
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "read"),
) -> ThreadStorage:
    """The bytes the thread's history takes: its checkpoints, channel values and pending writes."""
    await ThreadService(session, user).get(authorize, thread_id)
    return ThreadStorage.model_validate(await checkpoints.storage(thread_id))


@router.delete(
    "/threads/{thread_id}",
    status_code=204,
    responses={
        **NOT_FOUND,
        500: {
            "model": AgentProtocolError,
            "description": "Checkpoint cleanup failed; the thread is preserved and the delete can be retried",
        },
    },
)
async def delete_thread(
    thread_id: str,
    session: SessionDep,
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "delete"),
) -> Response:
    """Delete a thread by its ID.

    Permanently removes the thread, its metadata, and its checkpoint history
    stored in the graph backend. Any active runs on the thread are
    automatically cancelled before deletion.
    """
    await ThreadService(session, user).delete(authorize, thread_id)
    return Response(status_code=204)


@router.post("/threads/prune", response_model=ThreadPruneResponse)
async def prune_threads(
    request: ThreadPruneRequest,
    session: SessionDep,
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "delete"),
) -> ThreadPruneResponse:
    """Delete the listed threads (strategy `delete`) or compact their checkpoint history (`keep_latest`).

    Threads with active runs are skipped; expired threads are swept by the TTL sweeper.
    """
    filters = await authorize({"thread_ids": request.thread_ids})
    auth_filter = build_metadata_filter(ThreadORM.metadata_dict, filters)

    count = await prune_threads_by_id(
        session,
        thread_ids=request.thread_ids,
        strategy=request.strategy,
        user_id=user.identity,
        auth_filter=auth_filter,
    )
    return ThreadPruneResponse(pruned_count=count)


@router.post("/threads/count")
async def count_threads(
    request: ThreadCountRequest,
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "search"),
) -> int:
    """Count threads matching the given filters."""
    stmt = select(func.count()).select_from(ThreadORM).where(*await _search_conditions(request, authorize))
    return await session.scalar(stmt) or 0


@router.post("/threads/{thread_id}/copy", response_model=Thread, responses={**NOT_FOUND})
async def copy_thread(
    thread_id: str,
    session: SessionDep,
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("threads", "create"),
) -> Thread:
    """Fork a thread: new id, same metadata and full checkpoint lineage."""
    filters = await authorize({"thread_id": thread_id})
    source = await owned_thread(session, thread_id, user, filters=filters)

    # Checkpoints first: until the thread row commits they belong to no visible thread,
    # so a failed copy leaves nothing behind instead of a thread with no history.
    target_id = str(uuid4())
    await checkpoints.copy(thread_id, target_id)

    now = datetime.now(UTC)
    target = ThreadORM(
        thread_id=target_id,
        user_id=user.identity,
        # A copy has no runs: its status follows from the state it copies.
        status="error" if source.status == "error" else "interrupted" if source.interrupts else "idle",
        metadata_dict=dict(source.metadata_dict or {}),
        config=dict(source.config or {}),
        values=dict(source.values or {}),
        interrupts=dict(source.interrupts or {}),
        created_at=now,
        updated_at=now,
        state_updated_at=now,
    )
    session.add(target)
    try:
        await session.commit()
    except Exception:
        await db_manager.get_checkpointer().adelete_thread(target_id)
        raise

    return _serialize_thread(target)


@router.post("/threads/search", response_model=list[Thread])  # select returns a projected subset
async def search_threads(
    request: ThreadSearchRequest,
    session: SessionDep,
    response: Response,
    authorize: Authorization = Authorize("threads", "search"),
) -> list[Thread] | JSONResponse:
    """Search threads with filters.

    Filter by status or metadata key-value pairs. Results are paginated via
    `limit` and `offset` and ordered by `sort_by` and `sort_order` (newest first by default).
    """
    conditions = await _search_conditions(request, authorize)
    if request.ids:
        conditions.append(ThreadORM.thread_id.in_(request.ids))
    column = getattr(ThreadORM, request.sort_by)
    stmt = (
        select(ThreadORM)
        .where(*conditions)
        # Tie-break on thread_id keeps offset pagination stable when the sort column has duplicates.
        .order_by(column.asc() if request.sort_order == "asc" else column.desc(), ThreadORM.thread_id.asc())
        .offset(request.offset)
        .limit(request.limit + 1)
    )
    rows = (await session.scalars(stmt)).all()
    response.headers.update(pagination_headers(len(rows), limit=request.limit, offset=request.offset))
    rows = rows[: request.limit]

    views = [_serialize_thread(row) for row in rows]
    if request.extract:
        for view in views:
            dumped = view.model_dump(mode="json")
            view.extracted = {alias: _extract(dumped, path) for alias, path in request.extract.items()}
    if request.select:
        chosen = {*request.select, *(["extracted"] if request.extract else [])}
        return JSONResponse(
            [view.model_dump(mode="json", include=chosen) for view in views], headers=dict(response.headers)
        )
    return views
