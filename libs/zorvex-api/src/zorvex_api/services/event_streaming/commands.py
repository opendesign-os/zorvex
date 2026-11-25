"""Dispatch Agent Protocol v2 thread commands.

Commands are JSON-RPC-style: ``{id, method, params}`` in, a success or
error envelope out. They re-front the existing run machinery — ``run.start``
and ``input.respond`` both build a ``RunCreate`` and go through the same
``prepare_run`` path the run endpoints use, so execution semantics are
identical; only the transport differs.
"""

from __future__ import annotations

import re
from typing import Any, get_args
from uuid import UUID

import structlog
from fastapi import HTTPException
from langchain_protocol import ErrorCode
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.models import User
from zorvex_api.models.enums import MultitaskStrategy
from zorvex_api.models.runs import RunCreate
from zorvex_api.services.event_streaming.protocol import Response, build_error, build_success
from zorvex_api.services.run_preparation import Launch, prepare_run

logger = structlog.getLogger(__name__)


def _status_to_error_code(status_code: int) -> ErrorCode:
    """Map an HTTP status from run preparation to a protocol error code."""
    if status_code == 404:
        return "no_such_run"
    if status_code == 403:
        return "permission_denied"
    return "invalid_argument"


async def handle_command(
    payload: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[Response, str | None]:
    """Dispatch one command. Returns ``(response_envelope, started_run_id)``.

    ``started_run_id`` is the run a ``run.start`` / ``input.respond`` created,
    so the caller can open a stream for it; ``None`` for other commands.
    """
    command_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params")

    if not isinstance(command_id, int) or not isinstance(method, str):
        return build_error(
            command_id if isinstance(command_id, int) else None,
            "invalid_argument",
            "Commands must include an integer id and string method.",
        ), None

    if not isinstance(params, dict):
        return build_error(command_id, "invalid_argument", "params must be an object."), None

    # Run preparation raises HTTPException (unknown assistant/graph, bad resume)
    # and RunCreate raises ValidationError on malformed params. Map both to a
    # protocol error envelope so the client never sees FastAPI's {detail: ...}.
    # Anything else becomes unknown_error — an RPC response must stay an envelope.
    try:
        if method == "run.start":
            return await _run_start(command_id, params, session=session, thread_id=thread_id, user=user)
        if method == "input.respond":
            return await _input_respond(command_id, params, session=session, thread_id=thread_id, user=user)
        if method == "agent.getTree":
            return await agent_get_tree(command_id, params, session=session, thread_id=thread_id, user=user)
        if method in _SUBSCRIPTION_COMMANDS:
            return subscription_not_supported(command_id, method)
    except HTTPException as exc:
        return build_error(command_id, _status_to_error_code(exc.status_code), str(exc.detail)), None
    except ValidationError as exc:
        return build_error(command_id, "invalid_argument", str(exc.errors()[0].get("msg", "invalid params"))), None
    except Exception:
        logger.exception("Unhandled error dispatching v2 command", method=method, thread_id=thread_id)
        return build_error(command_id, "unknown_error", f"Command {method!r} failed unexpectedly."), None

    return build_error(command_id, "unknown_command", f"Unknown command {method!r}."), None


_MULTITASK_STRATEGIES = frozenset(get_args(MultitaskStrategy))

# Protocol keys plus Zorvex extensions per handler. Unknown keys are logged, never
# silently dropped.
RUN_START_KEYS: frozenset[str] = frozenset(
    {
        "assistant_id",
        "input",
        "config",
        "metadata",
        "langsmith_tracer",
        "context",
        "interrupt_before",
        "interrupt_after",
        "multitaskStrategy",
        "multitask_strategy",
    }
)
INPUT_RESPOND_KEYS: frozenset[str] = frozenset(
    {
        "interrupt_id",
        "namespace",
        "response",
        "responses",
        "update",
        "goto",
        "config",
        "metadata",
        "context",
        "assistant_id",
    }
)


def _warn_unknown_keys(method: str, params: dict[str, Any], known: frozenset[str]) -> None:
    unknown = sorted(set(params) - known)
    if unknown:
        logger.warning("Ignoring unknown v2 command params", method=method, keys=unknown)


def _context_from_params(params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Returns ``(context, None)`` or ``(None, error_message)``."""
    context = params.get("context")
    if context is None:
        return {}, None
    if not isinstance(context, dict):
        return None, "context must be an object."
    return context, None


def _fork_checkpoint_id(params: dict[str, Any]) -> tuple[str | None, str | None]:
    """Returns ``(checkpoint_id, None)`` or ``(None, error_message)``."""
    config = params.get("config")
    configurable = config.get("configurable") if isinstance(config, dict) else None
    checkpoint_id = configurable.get("checkpoint_id") if isinstance(configurable, dict) else None
    if checkpoint_id is None:
        return None, None
    if not isinstance(checkpoint_id, str):
        return None, "config.configurable.checkpoint_id must be a UUID string."
    try:
        UUID(checkpoint_id)
    except ValueError:
        return None, f"Invalid config.configurable.checkpoint_id: {checkpoint_id!r}."
    return checkpoint_id, None


async def _run_start(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[Response, str | None]:
    """Start a run on the thread from ``RunStartParams``."""
    _warn_unknown_keys("run.start", params, RUN_START_KEYS)
    assistant_id = params.get("assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        return build_error(command_id, "invalid_argument", "run.start requires a string assistant_id."), None

    multitask = params.get("multitaskStrategy", params.get("multitask_strategy"))
    if multitask not in _MULTITASK_STRATEGIES:
        # An unknown strategy enqueues, as on the Agent Server.
        multitask = None

    context, error_message = _context_from_params(params)
    if context is None:
        return build_error(command_id, "invalid_argument", error_message or "invalid params"), None

    checkpoint_id, error_message = _fork_checkpoint_id(params)
    if error_message is not None:
        return build_error(command_id, "invalid_argument", error_message), None

    # Input to a thread waiting on interrupts answers them, unless the run forks: a resume, not a fresh
    # turn that would drop the pending tasks. Several pending interrupts need the input keyed by their ids.
    input_data = params.get("input")
    command: dict[str, Any] | None = None
    if input_data is not None and checkpoint_id is None:
        status, pending = await _thread_interrupts(session, thread_id, user)
        if status == "interrupted" or pending:
            keyed = isinstance(input_data, dict) and bool(input_data) and set(input_data) <= pending
            if len(pending) > 1 and not keyed:
                message = "Several interrupts are pending; answer them with input.respond."
                return build_error(command_id, "invalid_argument", message), None
            command = {"resume": input_data}
            input_data = None
    tracer = params.get("langsmith_tracer")

    # No stream_mode: v2 runs stream via the native v3 path, which selects
    # channels through transformers, not stream_mode. interrupt_before/after are
    # forwarded so v2 clients can set node-level HITL breakpoints like v1.
    # applied_through_seq is 0 on this stateless POST transport (no stream binding).
    request = RunCreate(
        assistant_id=assistant_id,
        input=input_data,
        command=command,
        if_not_exists="create",
        config=params.get("config") or {},
        # run.start has no checkpoint field: config.configurable.checkpoint_id is
        # the fork target, and it must count as a checkpoint for input-less runs.
        checkpoint={"checkpoint_id": checkpoint_id} if checkpoint_id else None,
        context=context,
        metadata=params.get("metadata"),
        interrupt_before=params.get("interrupt_before"),
        interrupt_after=params.get("interrupt_after"),
        multitask_strategy=multitask,
        langsmith_tracer=tracer if isinstance(tracer, dict) else None,
    )
    run_id = await _start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id}, applied_through_seq=0), run_id


async def _thread_interrupts(session: AsyncSession, thread_id: str, user: User) -> tuple[str | None, set[str]]:
    """The thread's status and the ids of the interrupts it waits on (none for a thread that does not exist)."""
    row = (
        await session.execute(
            select(ThreadORM.status, ThreadORM.interrupts).where(
                ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity
            )
        )
    ).first()
    if row is None:
        return None, set()
    status, interrupts = row
    ids = {
        item["id"]
        for items in (interrupts or {}).values()
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    return status, ids


# langgraph interrupt ids are xxh3-128 hexdigests; a resume map is only
# recognized as id-targeted when every key matches this shape.
_INTERRUPT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


async def _input_respond(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[Response, str | None]:
    """Resume an interrupted run by replaying a HITL response as a command.

    The stock SDK's ``input.respond`` sends only ``{interrupt_id, namespace,
    response}`` — no assistant. Recover the assistant from the thread's most
    recent run so a resume works without the client re-supplying it. A supplied
    ``interrupt_id`` targets that interrupt via an id-keyed resume map (the only
    form that works with multiple pending interrupts); the batch ``responses``
    form merges several targets into one resume.
    """
    _warn_unknown_keys("input.respond", params, INPUT_RESPOND_KEYS)
    resume, error = _build_resume(params)
    if error is not None:
        code, message = error
        return build_error(command_id, code, message), None
    # Each answer must name an interrupt the thread waits on; while a run is on it, a new one may yet come.
    status, pending = await _thread_interrupts(session, thread_id, user)
    if status != "busy" and (unknown := sorted(set(resume or {}) - pending)):
        return build_error(command_id, "no_such_interrupt", f"Unknown interrupt id {unknown[0]!r}."), None

    command, error_message = _build_resume_command(params, resume)
    if command is None:
        return build_error(command_id, "invalid_argument", error_message or "invalid params"), None

    context, error_message = _context_from_params(params)
    if context is None:
        return build_error(command_id, "invalid_argument", error_message or "invalid params"), None

    assistant_id = params.get("assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        assistant_id = await thread_assistant_id(session, thread_id, user)
    if not assistant_id:
        return build_error(command_id, "no_such_run", "No run on this thread to resume."), None

    request = RunCreate(
        assistant_id=assistant_id,
        config=params.get("config") or {},
        context=context,
        metadata=params.get("metadata"),
        command=command,
    )
    run_id = await _start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id}, applied_through_seq=0), run_id


def _build_resume_command(params: dict[str, Any], resume: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Fold ``update`` / ``goto`` into the resume so the run writes one checkpoint.

    Returns ``(command, None)`` or ``(None, error_message)``.
    """
    command: dict[str, Any] = {"resume": resume}
    update = params.get("update")
    if update is not None:
        if not isinstance(update, dict):
            return None, "update must be an object."
        command["update"] = update
    goto = params.get("goto")
    if goto is not None:
        targets = goto if isinstance(goto, list) else [goto]
        if not all(_is_goto_target(target) for target in targets):
            return None, "goto must be a node name, a {node, input} object, or a list of those."
        command["goto"] = goto
    return command, None


def _is_goto_target(target: Any) -> bool:
    if isinstance(target, str):
        return bool(target)
    return isinstance(target, dict) and isinstance(target.get("node"), str) and bool(target["node"])


def _build_resume(params: dict[str, Any]) -> tuple[dict[str, Any] | None, tuple[ErrorCode, str] | None]:
    """Build the resume value from single or batch input.respond params.

    Returns ``(resume, None)`` or ``(None, (error_code, message))``.
    """
    responses = params.get("responses")
    if isinstance(responses, list):
        resume_map: dict[str, Any] = {}
        for entry in responses:
            if not isinstance(entry, dict) or "response" not in entry:
                return None, ("invalid_argument", "Each responses entry requires interrupt_id and response.")
            interrupt_id = entry.get("interrupt_id")
            if not isinstance(interrupt_id, str) or not _INTERRUPT_ID_PATTERN.fullmatch(interrupt_id):
                return None, ("no_such_interrupt", f"Unknown interrupt id {interrupt_id!r}.")
            resume_map[interrupt_id] = entry["response"]
        if not resume_map:
            return None, ("invalid_argument", "responses must be a non-empty array.")
        return resume_map, None

    if "response" not in params:
        return None, ("invalid_argument", "input.respond requires a response value.")

    interrupt_id = params.get("interrupt_id")
    if interrupt_id is None:
        return None, ("invalid_argument", "input.respond requires an interrupt_id.")
    if not isinstance(interrupt_id, str) or not _INTERRUPT_ID_PATTERN.fullmatch(interrupt_id):
        return None, ("no_such_interrupt", f"Unknown interrupt id {interrupt_id!r}.")
    return {interrupt_id: params["response"]}, None


async def _start(session: AsyncSession, thread_id: str, request: RunCreate, user: User) -> str:
    """Persist + enqueue a run via the shared preparation path (native v3 stream).

    The route authorized reading the thread; starting a run on it is a
    ``threads.create_run`` of its own, as on the REST routes.
    """
    authorize = Authorization(user, "threads", "create_run")
    run = await prepare_run(session, thread_id, request, user, authorize=authorize, launch=Launch(protocol="v2"))
    return run.run_id


_RUN_STATUS_TO_AGENT_STATUS = {
    "pending": "started",
    "running": "running",
    "success": "completed",
    "error": "failed",
    "timeout": "failed",
    "interrupted": "interrupted",
}


async def _thread_graph(session: AsyncSession, thread_id: str, user: User) -> tuple[str | None, dict[str, Any]]:
    """The id of the graph bound to the thread, and the thread's config."""
    row = (
        await session.execute(
            select(ThreadORM.metadata_dict, ThreadORM.config).where(
                ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity
            )
        )
    ).first()
    if row is None:
        return None, {}
    metadata, config = row
    return (metadata or {}).get("graph_id"), config or {}


async def agent_get_tree(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[Response, str | None]:
    graph_id, _ = await _thread_graph(session, thread_id, user)
    if not graph_id:
        return build_error(command_id, "no_such_run", "Thread has no associated graph."), None

    run_id = params.get("run_id")
    stmt = select(RunORM.status, ThreadORM.status).join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
    stmt = stmt.where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
    if isinstance(run_id, str) and run_id:
        stmt = stmt.where(RunORM.run_id == run_id)
    row = (await session.execute(stmt.order_by(RunORM.created_at.desc()).limit(1))).first()
    run_status, thread_status = row if row is not None else (None, None)
    # A run that paused on an interrupt succeeded, and its thread waits: report it interrupted.
    if run_status == "success" and thread_status == "interrupted":
        run_status = "interrupted"

    tree = {
        "namespace": [],
        "status": _RUN_STATUS_TO_AGENT_STATUS.get(run_status or "", "completed"),
        "graph_name": graph_id,
    }
    return build_success(command_id, {"tree": tree}), None


_SUBSCRIPTION_COMMANDS = frozenset({"subscription.subscribe", "subscription.unsubscribe", "subscription.reconnect"})


def subscription_not_supported(command_id: int, method: str) -> tuple[Response, str | None]:
    """Channels are fixed per connection in the EventStreamRequest body."""
    return build_error(
        command_id,
        "not_supported",
        f"{method} is not supported; set channels in the /stream/events request body instead.",
    ), None


async def thread_assistant_id(session: AsyncSession, thread_id: str, user: User) -> str | None:
    """The assistant of the thread's most recent run, user-scoped: the one a resume continues."""
    return await session.scalar(
        select(RunORM.assistant_id)
        .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
        .order_by(RunORM.created_at.desc())
        .limit(1)
    )
