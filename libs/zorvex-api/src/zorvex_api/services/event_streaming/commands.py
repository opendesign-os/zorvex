"""Agent Protocol v2 thread commands: ``{id, method, params}`` in, a success or error envelope out.

``run.start`` and ``input.respond`` build a ``RunCreate`` and go through the
``prepare_run`` the run endpoints use, so runs behave the same; only the
transport differs.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
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
from zorvex_api.models.runs import LangSmithTracer, RunCreate
from zorvex_api.services.event_streaming.protocol import Response, build_error, build_success
from zorvex_api.services.run_preparation import Launch, prepare_run

logger = structlog.getLogger(__name__)

INVALID_COMMAND = "Commands must include an integer id and string method."


class _CommandError(Exception):
    """A command the server refuses: the error envelope's code and message."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message


def _status_to_error_code(status_code: int) -> ErrorCode:
    """The protocol's error code for an HTTP status from run preparation."""
    if status_code == 404:
        return "no_such_run"
    if status_code == 403:
        return "permission_denied"
    return "invalid_argument"


async def handle_command(
    payload: dict[str, Any], *, session: AsyncSession, thread_id: str, authorize: Authorization
) -> Response:
    """Run one command on the thread, as the caller *authorize* stands for, and answer its envelope.

    Every outcome is an envelope: run preparation's HTTP errors and RunCreate's
    validation errors map to protocol codes, and anything else to unknown_error.
    """
    command_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params")
    if not isinstance(command_id, int) or isinstance(command_id, bool) or not isinstance(method, str):
        return build_error(command_id if isinstance(command_id, int) else None, "invalid_argument", INVALID_COMMAND)
    if not isinstance(params, dict):
        return build_error(command_id, "invalid_argument", "params must be an object.")
    if method in _SUBSCRIPTION_COMMANDS:
        return build_error(
            command_id,
            "not_supported",
            f"{method} is not supported; set channels in the /stream/events request body instead.",
        )
    handler = _HANDLERS.get(method)
    if handler is None:
        return build_error(command_id, "unknown_command", f"Unknown command {method!r}.")
    try:
        return await handler(command_id, params, session, thread_id, authorize)
    except _CommandError as exc:
        return build_error(command_id, exc.code, exc.message)
    except HTTPException as exc:
        return build_error(command_id, _status_to_error_code(exc.status_code), str(exc.detail))
    except ValidationError as exc:
        return build_error(command_id, "invalid_argument", str(exc.errors()[0].get("msg", "invalid params")))
    except Exception:
        logger.exception("Unhandled error dispatching v2 command", method=method, thread_id=thread_id)
        return build_error(command_id, "unknown_error", f"Command {method!r} failed unexpectedly.")


_MULTITASK_STRATEGIES = frozenset(get_args(MultitaskStrategy))

# The keys each command takes: the protocol's, and Zorvex's. Others are logged, never silently dropped.
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

# langgraph interrupt ids are xxh3-128 hexdigests; a resume map is only
# recognized as id-targeted when every key matches this shape.
_INTERRUPT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def _warn_unknown_keys(method: str, params: dict[str, Any], known: frozenset[str]) -> None:
    if unknown := sorted(set(params) - known):
        logger.warning("Ignoring unknown v2 command params", method=method, keys=unknown)


def _context(params: dict[str, Any]) -> dict[str, Any]:
    context = params.get("context")
    if context is None:
        return {}
    if not isinstance(context, dict):
        raise _CommandError("invalid_argument", "context must be an object.")
    return context


def _fork_checkpoint_id(params: dict[str, Any]) -> str | None:
    """The checkpoint ``config.configurable.checkpoint_id`` forks the run from, if any."""
    config = params.get("config")
    configurable = config.get("configurable") if isinstance(config, dict) else None
    checkpoint_id = configurable.get("checkpoint_id") if isinstance(configurable, dict) else None
    if checkpoint_id is None:
        return None
    if not isinstance(checkpoint_id, str):
        raise _CommandError("invalid_argument", "config.configurable.checkpoint_id must be a UUID string.")
    try:
        UUID(checkpoint_id)
    except ValueError:
        raise _CommandError("invalid_argument", f"Invalid config.configurable.checkpoint_id: {checkpoint_id!r}.")
    return checkpoint_id


async def _run_start(
    command_id: int, params: dict[str, Any], session: AsyncSession, thread_id: str, authorize: Authorization
) -> Response:
    """Start a run on the thread from ``RunStartParams``."""
    _warn_unknown_keys("run.start", params, RUN_START_KEYS)
    assistant_id = params.get("assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        raise _CommandError("invalid_argument", "run.start requires a string assistant_id.")
    multitask = params.get("multitaskStrategy", params.get("multitask_strategy"))
    context = _context(params)
    checkpoint_id = _fork_checkpoint_id(params)

    # Input to a thread waiting on interrupts answers them, unless the run forks: a resume, not a fresh
    # turn that would drop the pending tasks. Several pending interrupts need the input keyed by their ids.
    input_data = params.get("input")
    command: dict[str, Any] | None = None
    if input_data is not None and checkpoint_id is None:
        status, pending = await _thread_interrupts(session, thread_id, authorize.user)
        if status == "interrupted" or pending:
            keyed = isinstance(input_data, dict) and bool(input_data) and set(input_data) <= pending
            if len(pending) > 1 and not keyed:
                raise _CommandError(
                    "invalid_argument", "Several interrupts are pending; answer them with input.respond."
                )
            command = {"resume": input_data}
            input_data = None
    tracer = params.get("langsmith_tracer")

    # No stream_mode: v2 runs stream through the native v3 path, whose channels transformers select.
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
        # An unknown strategy enqueues, as on the Agent Server.
        multitask_strategy=multitask if multitask in _MULTITASK_STRATEGIES else None,
        langsmith_tracer=LangSmithTracer.model_validate(tracer) if isinstance(tracer, dict) else None,
    )
    return build_success(
        command_id, {"run_id": await _start(session, thread_id, request, authorize)}, applied_through_seq=0
    )


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
        for items in interrupts.values()
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    return status, ids


async def _input_respond(
    command_id: int, params: dict[str, Any], session: AsyncSession, thread_id: str, authorize: Authorization
) -> Response:
    """Resume the thread's interrupted run with a human's response, as a command.

    The SDK's ``input.respond`` sends ``{interrupt_id, namespace, response}``
    and no assistant: the thread's latest run gives it. A response answers its
    interrupt by id (the only form several pending interrupts allow), and the
    batch ``responses`` form answers several at once.
    """
    _warn_unknown_keys("input.respond", params, INPUT_RESPOND_KEYS)
    user = authorize.user
    resume = _resume(params)
    # Each answer must name an interrupt the thread waits on; while a run is on it, a new one may yet come.
    status, pending = await _thread_interrupts(session, thread_id, user)
    if status != "busy" and (unknown := sorted(set(resume) - pending)):
        raise _CommandError("no_such_interrupt", f"Unknown interrupt id {unknown[0]!r}.")
    command = _resume_command(params, resume)
    context = _context(params)
    assistant_id = params.get("assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        assistant_id = await thread_assistant_id(session, thread_id, user)
    if not assistant_id:
        raise _CommandError("no_such_run", "No run on this thread to resume.")
    request = RunCreate(
        assistant_id=assistant_id,
        config=params.get("config") or {},
        context=context,
        metadata=params.get("metadata"),
        command=command,
    )
    return build_success(
        command_id, {"run_id": await _start(session, thread_id, request, authorize)}, applied_through_seq=0
    )


def _resume_command(params: dict[str, Any], resume: dict[str, Any]) -> dict[str, Any]:
    """The resume with ``update`` and ``goto`` folded in, so the run writes one checkpoint."""
    command: dict[str, Any] = {"resume": resume}
    update = params.get("update")
    if update is not None:
        if not isinstance(update, dict):
            raise _CommandError("invalid_argument", "update must be an object.")
        command["update"] = update
    goto = params.get("goto")
    if goto is not None:
        targets = goto if isinstance(goto, list) else [goto]
        if not all(_is_goto_target(target) for target in targets):
            raise _CommandError(
                "invalid_argument", "goto must be a node name, a {node, input} object, or a list of those."
            )
        command["goto"] = goto
    return command


def _is_goto_target(target: Any) -> bool:
    if isinstance(target, str):
        return bool(target)
    return isinstance(target, dict) and isinstance(target.get("node"), str) and bool(target["node"])


def _resume(params: dict[str, Any]) -> dict[str, Any]:
    """The resume, by interrupt id, from a single response or a batch of ``responses``."""
    responses = params.get("responses")
    if isinstance(responses, list):
        resume_map: dict[str, Any] = {}
        for entry in responses:
            if not isinstance(entry, dict) or "response" not in entry:
                raise _CommandError("invalid_argument", "Each responses entry requires interrupt_id and response.")
            resume_map[_interrupt_id(entry.get("interrupt_id"))] = entry["response"]
        if not resume_map:
            raise _CommandError("invalid_argument", "responses must be a non-empty array.")
        return resume_map
    if "response" not in params:
        raise _CommandError("invalid_argument", "input.respond requires a response value.")
    if params.get("interrupt_id") is None:
        raise _CommandError("invalid_argument", "input.respond requires an interrupt_id.")
    return {_interrupt_id(params["interrupt_id"]): params["response"]}


def _interrupt_id(value: Any) -> str:
    if not isinstance(value, str) or not _INTERRUPT_ID_PATTERN.fullmatch(value):
        raise _CommandError("no_such_interrupt", f"Unknown interrupt id {value!r}.")
    return value


async def _start(session: AsyncSession, thread_id: str, request: RunCreate, authorize: Authorization) -> str:
    """Create the run (streamed natively, for v2) as the caller's ``threads.create_run``, as on the REST routes."""
    run = await prepare_run(
        session,
        thread_id,
        request,
        authorize.user,
        authorize=authorize.to("threads", "create_run"),
        launch=Launch(protocol="v2"),
    )
    return run.run_id


_RUN_STATUS_TO_AGENT_STATUS = {
    "pending": "started",
    "running": "running",
    "success": "completed",
    "error": "failed",
    "timeout": "failed",
    "interrupted": "interrupted",
}


async def agent_get_tree(
    command_id: int, params: dict[str, Any], session: AsyncSession, thread_id: str, authorize: Authorization
) -> Response:
    """The agent tree of the thread's latest run (or of ``params.run_id``): its graph and status."""
    user = authorize.user
    metadata = await session.scalar(
        select(ThreadORM.metadata_dict).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    )
    graph_id = (metadata or {}).get("graph_id")
    if not graph_id:
        raise _CommandError("no_such_run", "Thread has no associated graph.")
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
    return build_success(command_id, {"tree": tree})


_HANDLERS: dict[str, Callable[[int, dict[str, Any], AsyncSession, str, Authorization], Awaitable[Response]]] = {
    "run.start": _run_start,
    "input.respond": _input_respond,
    "agent.getTree": agent_get_tree,
}
# Channels are fixed by the stream's request body; on a WebSocket, ws.py handles these.
_SUBSCRIPTION_COMMANDS = frozenset({"subscription.subscribe", "subscription.unsubscribe", "subscription.reconnect"})


async def thread_assistant_id(session: AsyncSession, thread_id: str, user: User) -> str | None:
    """The assistant of the thread's most recent run, user-scoped: the one a resume continues."""
    return await session.scalar(
        select(RunORM.assistant_id)
        .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
        .order_by(RunORM.created_at.desc())
        .limit(1)
    )
