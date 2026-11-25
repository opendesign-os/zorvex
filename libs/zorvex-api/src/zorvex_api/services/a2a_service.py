"""A2A on top of runs and threads, as the Agent Server serves it (api/a2a.py routes the JSON-RPC).

A task is the runs of one message. Its id is ``{contextId}:{runId}``, the
context being the thread and the run the one its first message started, whose
metadata records ``a2a.task_start_id``; each later message of the task starts
a run recording ``a2a.task_id``. A task's state is its latest run's: pending,
running, success, interrupted (a cancel), error or timeout are SUBMITTED,
WORKING, COMPLETED, CANCELED and FAILED, and a success on a thread that waits
on an interrupt, when it is the thread's latest run, is INPUT_REQUIRED.

Each method answers its params with its result, in the Agent Server's wire
format (a2a_messages.py), or raises ``RpcError``; SendStreamingMessage yields
the results its stream sends. A method operates as its caller, each step
authorized as its REST counterpart.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, cast

import structlog
from fastapi import HTTPException, Request
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.uuid import uuid7
from langgraph.pregel import Pregel
from langgraph.types import StateSnapshot
from pydantic import ValidationError
from sqlalchemy import select
from starlette.datastructures import Headers

from zorvex_api import __version__
from zorvex_api.core.authorization import Action, Authorization
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.models import Assistant, User
from zorvex_api.models.enums import StreamMode
from zorvex_api.models.runs import Run, RunCreate
from zorvex_api.services import a2a_messages as wire
from zorvex_api.services import a2ui
from zorvex_api.services.a2a_messages import (
    AUTH_REQUIRED,
    CANCELED,
    COMPLETED,
    FAILED,
    INPUT_REQUIRED,
    REJECTED,
    SUBMITTED,
    WORKING,
    HistoryScope,
    RunMessages,
    TaskRef,
)
from zorvex_api.services.a2ui import A2UIContext, A2UIValidationError
from zorvex_api.services.assistant_service import AssistantService
from zorvex_api.services.base_broker import event_sequence, is_end
from zorvex_api.services.broker import live_events
from zorvex_api.services.executor import executor
from zorvex_api.services.langgraph_service import get_langgraph_service
from zorvex_api.services.ownership import authorized_run, authorized_thread, owned_run, visible_threads
from zorvex_api.services.run_cleanup import request_run_interruption
from zorvex_api.services.run_preparation import create_run
from zorvex_api.services.run_waiters import (
    ERROR_KEY,
    INTERRUPT_KEY,
    Completed,
    Failed,
    Paused,
    run_outcome,
    wait_for_result,
)
from zorvex_api.services.thread_state_service import record_state
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

# The extension that lets a client ask for a task's own history instead of its context's.
HISTORY_SCOPE_EXTENSION = "https://langchain.com/a2a/extensions/history-scope/v1"
HISTORY_SCOPE_HEADER = "LangGraph-A2A-History-Scope"
IO_MODES = ("application/json", "text/plain")

# JSON-RPC's error codes, then A2A's.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005
INVALID_AGENT_RESPONSE = -32006

_STATES = {
    "pending": SUBMITTED,
    "running": WORKING,
    "success": COMPLETED,
    "interrupted": CANCELED,
    "error": FAILED,
    "timeout": FAILED,
}
_TERMINAL = frozenset({COMPLETED, FAILED, CANCELED, REJECTED})
_VALID_STATES = frozenset({*_STATES.values(), INPUT_REQUIRED, REJECTED, AUTH_REQUIRED})
_MAX_HISTORY = 10
_LIST_PAGE_SIZE = 50
_LIST_MAX_PAGE_SIZE = 100
# Bounds on what a task lookup scans, as on the Agent Server.
_RUN_SCAN_LIMIT = 1000
_CONTEXT_SEARCH_LIMIT = 1000
_SCHEMA_TTL_SECONDS = 60.0
# How long a cancel waits for the run to stop, and how far off the marker run is that ends a waiting task.
_CANCEL_WAIT_SECONDS = 10.0
_MARKER_DELAY_SECONDS = 3600
_INTERRUPTED_CHECKS = 3
_INTERRUPTED_CHECK_DELAY = 0.05


class RpcError(Exception):
    """A JSON-RPC error a method answers with."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data

    @property
    def error(self) -> dict[str, Any]:
        """The JSON-RPC error object."""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error

    def envelope(self, request_id: Any) -> dict[str, Any]:
        """The JSON-RPC response answering request *request_id* with this error."""
        return {"jsonrpc": "2.0", "id": request_id, "error": self.error}


def result_envelope(request_id: Any, result: Any) -> dict[str, Any]:
    """The JSON-RPC response answering request *request_id* with *result*."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


class _RunNotCreated(RpcError):
    """The run a message asks for could not be created; a stream answers it as an internal error."""


@dataclass(frozen=True)
class Call:
    """One JSON-RPC request to an assistant: its route's ``assistants.read``, which names the caller, and the
    request."""

    authorize: Authorization
    assistant_id: str
    request: Request

    @property
    def user(self) -> User:
        return self.authorize.user

    @property
    def headers(self) -> Headers:
        return self.request.headers

    def threads(self, action: Action) -> Authorization:
        """The caller's ``threads.{action}``, as the REST route doing the step authorizes it."""
        return self.authorize.to("threads", action)


@dataclass(frozen=True)
class Task:
    """A task, as its latest run on the assistant makes it."""

    ref: TaskRef
    run: Run
    state: str
    task_ids_by_start_id: dict[str, str]

    @property
    def status_timestamp(self) -> str:
        return _timestamp(self.run)


def _a2a(run: Run) -> dict[str, Any]:
    """What a task's run records under its ``a2a`` metadata: its task, and the A2UI it was started with."""
    recorded = run.metadata.get("a2a")
    return recorded if isinstance(recorded, dict) else {}


def _timestamp(run: Run) -> str:
    return run.updated_at.isoformat()


# --- methods


async def send_message(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """SendMessage: start the message's run (a new task, or the next of a task), then answer the task."""
    configuration = params.get("configuration")
    configuration = {} if configuration is None else configuration
    if not isinstance(configuration, dict):
        raise RpcError(INVALID_PARAMS, "Invalid params: 'configuration' must be an object")
    return_immediately = configuration.get("returnImmediately")
    if return_immediately is not None and not isinstance(return_immediately, bool):
        raise RpcError(INVALID_PARAMS, "Invalid params: 'returnImmediately' must be a boolean")
    view = wire.View(_history_length(configuration.get("historyLength")), _history_scope(configuration, call.headers))
    message = params["message"]
    _check_message(message, parts_array=True)
    started = await _start(call, message, params.get("context"))
    if return_immediately:
        return {"task": await _submitted(call, started, view)}
    return {"task": await _finished(call, started, view)}


async def _submitted(call: Call, started: _Started, view: wire.View) -> dict[str, Any]:
    """The task as its run was created, answering ``returnImmediately``."""
    status = {"state": _state_for_run(started.run), "timestamp": _timestamp(started.run)}
    history = None
    if view.length:
        run_messages = await _run_messages(call, started.ref.context_id, started.run.run_id)
        if not run_messages.messages:
            run_messages = RunMessages(list(started.input.get("messages", [])), 0)
        ids = {**started.task_ids}
        if started.start_id:
            ids.setdefault(started.start_id, started.ref.task_id)
        seen = replace(view, a2ui=started.a2ui, task_ids=ids)
        history = wire.history(run_messages.stamped(started.ref.task_id), started.ref, seen)
    return wire.task(started.ref, status=status, history=history)


async def _finished(call: Call, started: _Started, view: wire.View) -> dict[str, Any]:
    """The task once the message's run has ended."""
    ref = started.ref
    ids = dict(started.task_ids)
    if started.start_id:
        try:
            ids = _task_ids_by_start_id(await _context_runs(call, ref.context_id), ref.context_id, call.assistant_id)
        except Exception:
            logger.warning("Failed to resolve A2A history provenance", context_id=ref.context_id, exc_info=True)
        ids.setdefault(started.start_id, ref.task_id)

    result = await wait_for_result(started.run.run_id, ref.context_id, call.user)
    data_only = started.command is None and "messages" not in started.input
    turn = None
    if data_only or (view.scope == "task" and view.length != 0 and "messages" not in started.input):
        run_messages = await _run_messages(call, ref.context_id, started.run.run_id)
        result = {**result, "messages": run_messages.stamped(ref.task_id)}
        if data_only:
            turn = run_messages.added
    seen = replace(view, a2ui=started.a2ui, task_ids=ids)
    return _task_response(ref, result, seen, assistant_id=call.assistant_id, turn=turn)


async def stream_message(call: Call, params: dict[str, Any]) -> AsyncGenerator[dict[str, Any]]:
    """SendStreamingMessage: the results the stream sends, the task first and its final status last.

    Raises:
        RpcError: For a message whose run cannot start, or a reply no part can carry; the stream ends with it.
    """
    message = params["message"]
    _check_message(message, parts_array=False)
    try:
        started = await _start(call, message, params.get("context"), stream_mode=["messages", "values"])
    except _RunNotCreated as exc:
        # The Agent Server's stream fails as it starts the run, past its checks.
        raise RpcError(INTERNAL_ERROR, "Internal server error") from exc
    ref = started.ref
    history = [{"kind": "message", **message, "taskId": ref.task_id, "contextId": ref.context_id}]
    yield {"task": wire.task(ref, history=history, status={"state": SUBMITTED, "timestamp": wire.now_iso()})}

    followed = _Followed()
    async for event, data in _run_events(started.run.run_id):
        if (update := followed.absorb(event, data, ref)) is not None:
            yield update
    async for update in _stream_end(call, started, followed):
        yield update


@dataclass
class _Followed:
    """What a streamed run's events have said: its last values, its error, and whether it was retried."""

    result: Any = None
    error: Any = None
    started: bool = False
    retried: bool = False

    def absorb(self, event: str, data: Any, ref: TaskRef) -> dict[str, Any] | None:
        """Take in one event; the status update it has the stream send, if any."""
        try:
            if event == "metadata":
                # The first is the run's; another is a retry starting.
                if not self.started:
                    self.started = True
                elif isinstance(data, dict) and data.get("run_id") and not self.retried:
                    self.retried = True
                    return wire.status_update(ref, {"state": WORKING}, final=False)
            elif event == "error":
                self.error = data
            elif event == "values":
                self.error, self.result = None, data
            elif event.startswith("messages"):
                self.error = None
                if isinstance(data, list) and data:
                    update = wire.working_update(to_jsonable(data), ref)
                    if any(part.get("text") or part.get("data") for part in update["status"]["message"]["parts"]):
                        return update
        except Exception as exc:
            logger.exception("Failed to process message stream")
            self.error = {"error": type(exc).__name__, "message": str(exc)}
        return None


async def _stream_end(call: Call, started: _Started, followed: _Followed) -> AsyncGenerator[dict[str, Any]]:
    """The stream's last results: the reply or the interrupts as an artifact, then the final status."""
    ref = started.ref
    result = to_jsonable(followed.result)
    status: dict[str, Any]
    match run_outcome({ERROR_KEY: followed.error} if followed.error else result):
        case Failed(error):
            status = {"state": FAILED}
            if (text := _stream_error_text(error)) is not None:
                status["message"] = wire.agent_message(text, task_id=ref.task_id, context_id=ref.context_id)
        case Paused(interrupts):
            yield wire.artifact_update(ref, wire.interrupt_artifact(interrupts))
            status = {"state": INPUT_REQUIRED}
        case Completed():
            if isinstance(result, dict):
                turn = None
                if started.command is None and "messages" not in started.input:
                    turn = (await _run_messages(call, ref.context_id, started.run.run_id)).added
                parts = wire.response_parts(result, a2ui_context=started.a2ui, turn=turn)
                artifact = wire.response_artifact(parts, call.assistant_id)
                yield wire.artifact_update(ref, artifact, append=False, lastChunk=True)
            status = {"state": COMPLETED}
    status["timestamp"] = wire.now_iso()
    yield wire.status_update(ref, status, final=True)


def _stream_error_text(error: Any) -> str | None:
    """What a failed stream's final status says: its error's message, when the error has the shape of one."""
    if not isinstance(error, dict) or (ERROR_KEY not in error and "error" not in error):
        return None
    inner = error.get(ERROR_KEY)
    text = inner.get("error") if isinstance(inner, dict) else error.get("message")
    return str(text or "")


async def get_task(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """GetTask: the task, its history and, once it completed or waits, its artifacts."""
    task_id, context_id = _task_params(params)
    task = await _resolve_task(call, task_id, context_id)
    if task is None:
        location = f" in context '{context_id}'" if context_id else ""
        raise RpcError(TASK_NOT_FOUND, f"Task not found: {task_id}{location}")
    run = task.run
    view = wire.View(
        _history_length(params.get("historyLength")),
        _history_scope(params, call.headers),
        a2ui.run_context(run.metadata),
        task.task_ids_by_start_id,
    )

    if task.state in (COMPLETED, INPUT_REQUIRED):
        state = await _run_state(call, task.ref.context_id, run.run_id)
        result = dict(state.values or {})
        run_messages = state.messages
        if "messages" in result:
            result["messages"] = run_messages.stamped(task.ref.task_id)
        if task.state == INPUT_REQUIRED:
            result[INTERRUPT_KEY] = state.interrupts
        return _task_response(
            task.ref,
            result,
            view,
            assistant_id=call.assistant_id,
            turn=run_messages.added,
            timestamp=task.status_timestamp,
        )

    history = None
    if view.length != 0:
        history = []
        try:
            messages = (await _run_messages(call, task.ref.context_id, run.run_id)).stamped(task.ref.task_id)
            history = wire.history(messages, task.ref, view)
        except (A2UIValidationError, wire.InvalidAgentResponseError):
            raise
        except Exception:
            logger.exception("Failed to read a task's history for GetTask", task_id=task_id)
    status: dict[str, Any] = {"state": task.state, "timestamp": task.status_timestamp}
    if task.state == FAILED:
        status["message"] = wire.agent_message(f"Task failed with status: {run.status}", task_id=task.ref.task_id)
    return wire.task(task.ref, status=status, history=history)


async def cancel_task(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """CancelTask: stop the task's active run, or end its wait for input."""
    task_id, context_id = _task_params(params)
    try:
        task = await _resolve_task(call, task_id, context_id)
    except Exception as exc:
        logger.exception("Failed to resolve A2A task for cancellation")
        raise RpcError(INTERNAL_ERROR, "Failed to check task status") from exc
    if task is None:
        raise RpcError(TASK_NOT_FOUND, f"Task not found: {task_id}")
    if task.state in _TERMINAL:
        raise RpcError(TASK_NOT_CANCELABLE, "Task is not cancelable in its current state")

    context = task.ref.context_id
    marker: Run | None = None
    try:
        if task.state == INPUT_REQUIRED:
            # A marker run in the task, cancelled, makes the task's latest run interrupted; the
            # thread then stops waiting on its interrupt.
            canonical = _task_id_for_run(context, task.run)
            marker = await _create_run(
                call,
                context,
                RunCreate(
                    assistant_id=call.assistant_id,
                    metadata={"a2a": {**_a2a(task.run), "task_id": canonical}},
                    after_seconds=_MARKER_DELAY_SECONDS,
                ),
            )
            await _cancel_run(call, context, marker.run_id)
            await _end_thread_wait(call, context)
        elif task.run.status not in ("pending", "running"):
            raise RpcError(TASK_NOT_CANCELABLE, "Task has no active run to cancel")
        else:
            await _cancel_run(call, context, task.run.run_id)
    except RpcError:
        raise
    except Exception as exc:
        if marker is not None:
            await _delete_run(call, context, marker.run_id)
        logger.exception("Failed to cancel A2A task", task_id=task_id)
        raise RpcError(INTERNAL_ERROR, f"Failed to cancel task: {exc!s}") from exc

    status = {
        "state": CANCELED,
        "timestamp": wire.now_iso(),
        "message": wire.agent_message("Task was canceled", task_id=task_id),
    }
    return wire.task(TaskRef(task_id, context), status=status)


@dataclass(frozen=True)
class _ListQuery:
    """ListTasks' params, checked in the Agent Server's order."""

    page_size: int
    view: wire.View
    status: str | None
    after: str | None
    offset: int
    context_id: str | None
    include_artifacts: bool

    @classmethod
    def parse(cls, params: Mapping[str, Any], headers: Headers) -> _ListQuery:
        page_size = params.get("pageSize")
        if page_size is not None and (
            not isinstance(page_size, int) or isinstance(page_size, bool) or not 1 <= page_size <= _LIST_MAX_PAGE_SIZE
        ):
            raise RpcError(INVALID_PARAMS, f"pageSize must be an integer between 1 and {_LIST_MAX_PAGE_SIZE}.")
        view = wire.View(_history_length(params.get("historyLength")), _history_scope(params, headers))
        status = params.get("status")
        if status is not None and status not in _VALID_STATES:
            raise RpcError(INVALID_PARAMS, f"Invalid status value: '{status}'.")
        after = params.get("statusTimestampAfter")
        if after is not None and not _is_timestamp(after):
            raise RpcError(INVALID_PARAMS, "statusTimestampAfter must be a valid ISO 8601 timestamp.")
        page_token = params.get("pageToken")
        if page_token is not None and not isinstance(page_token, str):
            raise RpcError(INVALID_PARAMS, "pageToken must be a string.")
        try:
            offset = int(page_token) if page_token else 0
        except ValueError:
            raise RpcError(INVALID_PARAMS, "Invalid pageToken.") from None
        context_id = params.get("contextId")
        if context_id is not None and not isinstance(context_id, str):
            raise RpcError(INVALID_PARAMS, "contextId must be a string.")
        return cls(
            page_size=_LIST_PAGE_SIZE if page_size is None else page_size,
            view=view,
            status=status,
            after=after,
            offset=offset,
            context_id=context_id,
            include_artifacts=bool(params.get("includeArtifacts", False)),
        )


def _is_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


async def list_tasks(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """ListTasks: the assistant's tasks on the caller's threads, newest status first, a page at a time."""
    query = _ListQuery.parse(params, call.headers)
    tasks: dict[str, dict[str, Any]] = {}
    runs_by_task: dict[str, Run] = {}
    ids_by_context: dict[str, dict[str, str]] = {}
    for thread in await _context_threads(call, query.context_id):
        context = thread.thread_id
        runs = await _context_runs(call, context)
        latest_id = _latest_run_id(runs)
        ids_by_context[context] = _task_ids_by_start_id(runs, context, call.assistant_id)
        for run in _runs_for_assistant(runs, call.assistant_id):
            task_id = _task_id_for_run(context, run)
            if task_id in runs_by_task:
                continue
            runs_by_task[task_id] = run
            state = _state_for_run(run, thread if run.run_id == latest_id else None)
            timestamp = _timestamp(run)
            if (query.status and state != query.status) or (query.after and timestamp < query.after):
                continue
            artifacts = None if query.include_artifacts else []
            tasks[task_id] = wire.task(
                TaskRef(task_id, context), status={"state": state, "timestamp": timestamp}, artifacts=artifacts
            )

    ordered = sorted(tasks.values(), key=lambda task: task["status"].get("timestamp", ""), reverse=True)
    page = ordered[query.offset : query.offset + query.page_size]
    if query.view.length != 0:
        for task in page:
            ref, run = TaskRef(task["id"], task["contextId"]), runs_by_task[task["id"]]
            messages = (await _run_messages(call, ref.context_id, run.run_id)).stamped(ref.task_id)
            seen = replace(query.view, a2ui=a2ui.run_context(run.metadata), task_ids=ids_by_context.get(ref.context_id))
            task["history"] = wire.history(messages, ref, seen)
    next_offset = query.offset + len(page)
    return {
        "tasks": page,
        "totalSize": len(ordered),
        "pageSize": len(page),
        "nextPageToken": str(next_offset) if next_offset < len(ordered) else "",
    }


async def extended_card(call: Call, _params: dict[str, Any]) -> dict[str, Any]:
    """GetExtendedAgentCard: the public card; there is nothing more to tell an authenticated caller."""
    try:
        return await agent_card(call.request, call.authorize, call.assistant_id)
    except ValueError as exc:
        raise RpcError(INVALID_PARAMS, str(exc)) from exc


async def agent_card(request: Request, authorize: Authorization, assistant_id: str) -> dict[str, Any]:
    """The assistant's agent card, as the Agent Server builds it, read as *authorize* (``assistants.read``) lets.

    Raises:
        ValueError: For an assistant that cannot be read, or whose ``metadata.a2a`` modes are malformed.
    """
    assistant = await _assistant(authorize, assistant_id)
    schemas = await _schemas(authorize, assistant)
    input_schema = schemas.get("input_schema") or schemas.get("state_schema") or {}
    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = input_schema.get("required", []) if isinstance(input_schema, dict) else []
    description = assistant.description or f"{assistant.name} assistant"
    input_modes, output_modes = _card_modes(assistant.metadata)
    extensions: list[dict[str, Any]] = [
        {
            "uri": HISTORY_SCOPE_EXTENSION,
            "description": "Choose task-only or full-context response history",
            "required": False,
            "params": {
                "header": HISTORY_SCOPE_HEADER,
                "values": ["context", "task"],
                "default": "context",
                "methods": ["SendMessage", "GetTask", "ListTasks"],
            },
        }
    ]
    if a2ui.enabled_for(assistant.metadata):
        for modes in (input_modes, output_modes):
            if a2ui.MIME_TYPE not in modes:
                modes.append(a2ui.MIME_TYPE)
        extensions.append(
            {
                "uri": a2ui.EXTENSION_URI,
                "description": "Ability to render A2UI v0.9",
                "required": False,
                "params": {"v0.9": {"supportedCatalogIds": [], "acceptsInlineCatalogs": False}},
            }
        )
    url = rpc_url(request, assistant_id)
    return {
        "name": assistant.name,
        "description": description,
        "url": url,
        "supportedInterfaces": [{"url": url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False,
            "extensions": extensions,
        },
        "defaultInputModes": input_modes,
        "defaultOutputModes": output_modes,
        "skills": [
            {
                "id": f"{assistant_id}-main",
                "name": f"{assistant.name} Capabilities",
                "description": description,
                "tags": ["assistant", "langgraph"],
                "examples": [],
                "inputModes": input_modes,
                "outputModes": output_modes,
                "metadata": {
                    "inputSchema": {
                        "required": required,
                        "properties": sorted(properties),
                        "supportsA2A": "messages" in properties,
                    }
                },
            }
        ],
        "version": __version__,
    }


def rpc_url(request: Request, assistant_id: str) -> str:
    """Where clients send JSON-RPC to the assistant: under LANGGRAPH_API_URL, else this request's URL with the
    scheme a proxy forwarded."""
    if base := settings.app.LANGGRAPH_API_URL:
        return f"{base.rstrip('/')}/a2a/{assistant_id}"
    url = request.url_for("a2a_rpc", assistant_id=assistant_id)
    scheme = request.headers.get("x-forwarded-proto")
    return str(url.replace(scheme=scheme) if scheme else url)


def _card_modes(metadata: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    """The card's input and output modes: the defaults, or the assistant's ``metadata.a2a`` overrides."""
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Assistant metadata must be an object.")
    a2a = (metadata or {}).get("a2a")
    if a2a is None:
        return list(IO_MODES), list(IO_MODES)
    if not isinstance(a2a, dict):
        raise ValueError("Assistant metadata.a2a must be an object.")
    return _modes(a2a.get("input_modes"), "input_modes"), _modes(a2a.get("output_modes"), "output_modes")


def _modes(value: Any, name: str) -> list[str]:
    if value is None:
        return list(IO_MODES)
    if not isinstance(value, list) or not value or not all(isinstance(mode, str) and mode.strip() for mode in value):
        raise ValueError(f"Assistant metadata.a2a.{name} must be a non-empty array of strings.")
    return [mode.strip() for mode in value]


# --- starting a message's run


def _check_message(message: Mapping[str, Any], *, parts_array: bool) -> None:
    """Refuse a message without an id, an A2A 1.0 role or parts; with *parts_array*, parts that are no array."""
    if not message.get("messageId"):
        raise RpcError(INVALID_PARAMS, "Missing required 'messageId' in message")
    role = message.get("role")
    if not role:
        raise RpcError(INVALID_PARAMS, "Missing required 'role' in message")
    if not isinstance(role, str) or role not in wire.ROLES:
        raise RpcError(INVALID_PARAMS, "Invalid 'role' in message: must be 'ROLE_USER' or 'ROLE_AGENT'")
    parts = message.get("parts", [])
    if parts_array and not isinstance(parts, list):
        raise RpcError(INVALID_PARAMS, "Invalid params: 'parts' must be an array")
    if not parts:
        raise RpcError(INVALID_PARAMS, "Message must contain at least one part")


@dataclass(frozen=True)
class _Target:
    """Where a message goes: a new task in *context_id*, or the next message of *task*.

    *stamp* is what the message and its run record of their task: the task's
    ``task_id``, or a new task's ``task_start_id``.
    """

    context_id: str
    stamp: dict[str, str]
    task: Task | None = None

    @property
    def start_id(self) -> str | None:
        return self.stamp.get("task_start_id")

    @property
    def task_ids(self) -> dict[str, str]:
        return self.task.task_ids_by_start_id if self.task is not None else {}

    @property
    def mime_type(self) -> str:
        """The A2UI MIME type the task negotiated before, for a message that names none."""
        return a2ui.run_context(self.task.run.metadata).mime_type if self.task is not None else a2ui.MIME_TYPE


@dataclass(frozen=True)
class _Started:
    """A message's run, started: the task it is of, and what it was started with."""

    run: Run
    ref: TaskRef
    start_id: str | None
    task_ids: dict[str, str]
    input: dict[str, Any]
    command: dict[str, Any] | None
    a2ui: A2UIContext


async def _start(
    call: Call, message: Mapping[str, Any], run_context: Any, *, stream_mode: list[StreamMode] | None = None
) -> _Started:
    """Read the assistant and the task the message continues, then start the message's run."""
    try:
        assistant = await _assistant(call.authorize, call.assistant_id)
        await _require_messages_input(call.authorize, assistant, message.get("parts", []))
    except ValueError as exc:
        raise RpcError(INVALID_PARAMS, str(exc)) from exc
    target = await _target(call, message)
    try:
        a2ui_context = a2ui.message_context(
            message, assistant.metadata, call.headers, default_mime_type=target.mime_type
        )
        run_input = wire.message_input(message, a2ui_context=a2ui_context, stamp=target.stamp)
    except A2UIValidationError as exc:
        raise RpcError(INVALID_PARAMS, str(exc)) from exc
    except ValueError as exc:
        raise RpcError(CONTENT_TYPE_NOT_SUPPORTED, str(exc)) from exc
    command, run_input = await _command(call, message, target, run_input)
    try:
        request = RunCreate(
            assistant_id=call.assistant_id,
            input=run_input,
            command=command,
            context=run_context,
            metadata={"a2a": {**target.stamp, **a2ui_context.run_metadata()}},
            if_not_exists="create",
            stream_mode=stream_mode,
        )
    except ValidationError as exc:
        raise _RunNotCreated(INVALID_PARAMS, f"Failed to create run: {exc}") from exc
    run = await _create_run(call, target.context_id, request)
    task_id = target.task.ref.task_id if target.task is not None else f"{run.thread_id}:{run.run_id}"
    return _Started(
        run=run,
        ref=TaskRef(task_id, run.thread_id),
        start_id=target.start_id,
        task_ids=target.task_ids,
        input=run_input,
        command=command,
        a2ui=a2ui_context,
    )


async def _target(call: Call, message: Mapping[str, Any]) -> _Target:
    """The task the message names (a terminal one refused), or a new one in its context, or in a new context."""
    context_id, task_id = message.get("contextId"), message.get("taskId")
    if context_id is not None and not isinstance(context_id, str):
        raise RpcError(INVALID_PARAMS, "contextId must be a string")
    if task_id is not None and not isinstance(task_id, str):
        raise RpcError(INVALID_PARAMS, "taskId must be a string")
    if task_id is None:
        return _Target(context_id or str(uuid.uuid4()), {"task_start_id": str(uuid7())})
    task = await _resolve_task(call, task_id, context_id)
    if task is None:
        raise RpcError(TASK_NOT_FOUND, f"Task not found: {task_id}")
    if task.state in _TERMINAL:
        raise RpcError(UNSUPPORTED_OPERATION, "A terminal task cannot accept another message")
    return _Target(task.ref.context_id, {"task_id": task.ref.task_id}, task)


async def _command(
    call: Call, message: Mapping[str, Any], target: _Target, run_input: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The command the message's run goes with, its own or the resume it implies, and the input left with it."""
    command = message.get("command")
    if isinstance(command, dict) and command.get("resume"):
        if not message.get("contextId"):
            raise RpcError(INVALID_PARAMS, "contextId is required when resuming a task with command.resume")
        if target.task is None:
            raise RpcError(INVALID_PARAMS, "taskId is required when resuming a task")
    if command is None:
        command, run_input = await _implied_resume(
            call,
            message.get("parts", []),
            context_id=target.context_id if target.task is not None else message.get("contextId"),
            task_id=message.get("taskId"),
            run_input=run_input,
        )
    if target.task is not None and command is None:
        logger.warning("User requested to resume a task without specifying a Command.", task_id=message["taskId"])
    return command, run_input


async def _implied_resume(
    call: Call, parts: Sequence[Any], *, context_id: str | None, task_id: str | None, run_input: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The resume a message implies: a data part's ``resume``, or text to a task waiting for input."""
    values = wire.resume_values(parts)
    if len(values) > 1:
        raise RpcError(INVALID_PARAMS, "Only one resume value is allowed in data parts")
    if values and values[0] is not None:
        if not context_id or not task_id:
            raise RpcError(INVALID_PARAMS, "contextId and taskId are required when resuming a task")
        if not await _waits_for_input(call, context_id):
            raise RpcError(INVALID_PARAMS, "Task is not awaiting input")
        return {"resume": values[0]}, {key: value for key, value in run_input.items() if key != "resume"}
    text = wire.text_of_parts(parts)
    if text is None or not context_id or not task_id or not await _waits_for_input(call, context_id):
        return None, run_input
    return {"resume": text}, {key: value for key, value in run_input.items() if key != "messages"}


async def _waits_for_input(call: Call, context_id: str) -> bool:
    """Whether the thread waits on an interrupt, looked at a few times as the Agent Server does."""
    for attempt in range(_INTERRUPTED_CHECKS):
        try:
            thread = await _thread(call, context_id)
        except Exception:
            return False
        if thread is not None and thread.status == "interrupted":
            return True
        if attempt < _INTERRUPTED_CHECKS - 1:
            await asyncio.sleep(_INTERRUPTED_CHECK_DELAY)
    return False


async def _create_run(call: Call, context_id: str, request: RunCreate) -> Run:
    """The run started on the context, as ``POST /threads/{context_id}/runs`` starts it."""
    try:
        return await create_run(call.threads("create_run"), context_id, request)
    except HTTPException as exc:
        raise _run_create_error(exc, call.assistant_id, context_id) from exc


def _run_create_error(exc: HTTPException, assistant_id: str, thread_id: str) -> RpcError:
    detail = str(exc.detail)
    if exc.status_code == 404:
        if "thread" in detail.lower():
            return _RunNotCreated(
                INVALID_PARAMS,
                f"Thread '{thread_id}' not found. Please create the thread first before sending messages to it.",
                {"thread_id": thread_id, "error_type": "thread_not_found"},
            )
        return _RunNotCreated(INVALID_PARAMS, f"Assistant '{assistant_id}' not found")
    if exc.status_code == 400:
        return _RunNotCreated(INVALID_PARAMS, f"Invalid request: {detail}")
    if exc.status_code == 403:
        return _RunNotCreated(INVALID_PARAMS, "Access denied to assistant or thread")
    return _RunNotCreated(INVALID_PARAMS, f"Failed to create run: {detail}")


# --- answering with a task


def _task_response(
    ref: TaskRef,
    result: Mapping[str, Any],
    view: wire.View,
    *,
    assistant_id: str,
    turn: Sequence[Any] | None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """A finished run's task, its history as *view* shows it: its error, the interrupts it waits on, or its reply.

    Its status is timed *timestamp*, else now.
    """
    result = to_jsonable(dict(result))
    history = wire.history(result.get("messages", []) or [], ref, view) if view.length != 0 else None
    match run_outcome(result):
        case Failed(error):
            name = error.get("error") if isinstance(error, dict) else error
            message = wire.agent_message(
                f"Error executing assistant: {name}", task_id=ref.task_id, context_id=ref.context_id
            )
            return wire.task(ref, history=history, status={"state": FAILED, "message": message})
        case Paused(interrupts):
            status = {"state": INPUT_REQUIRED, "timestamp": timestamp or wire.now_iso()}
            return wire.task(ref, history=history, status=status, artifacts=[wire.interrupt_artifact(interrupts)])
        case Completed():
            status = {"state": COMPLETED, "timestamp": timestamp or wire.now_iso()}
            parts = wire.response_parts(result, a2ui_context=view.a2ui, turn=turn)
            artifacts = [wire.response_artifact(parts, assistant_id)]
            return wire.task(ref, history=history, status=status, artifacts=artifacts)


async def _run_events(run_id: str) -> AsyncGenerator[tuple[str, Any]]:
    """The run's ``(mode, data)`` events, those stored so far then live ones, until its stream ends."""
    last = 0
    async for event_id, event in live_events(run_id):
        sequence = event_sequence(event_id)
        if sequence <= last:
            continue  # stored and live events overlap
        last = sequence
        if is_end(event):
            return
        if isinstance(event, tuple) and len(event) == 2:
            yield event


# --- tasks


def _task_params(params: Mapping[str, Any]) -> tuple[str, str | None]:
    task_id, context_id = params.get("id"), params.get("contextId")
    if not task_id:
        raise RpcError(INVALID_PARAMS, "Missing required parameter: id (task_id)")
    if not isinstance(task_id, str) or (context_id is not None and not isinstance(context_id, str)):
        raise RpcError(INVALID_PARAMS, "Task id and contextId must be strings")
    return task_id, context_id


async def _resolve_task(call: Call, task_id: str, context_id: str | None) -> Task | None:
    """The task *task_id* names, as its latest run on the assistant makes it; None when there is no such task.

    The id is ``{context}:{run}``, or a run id alone with the context given apart.
    """
    parsed_context, separator, anchor_id = task_id.partition(":")
    if not separator:
        parsed_context, anchor_id = "", task_id
    if parsed_context and context_id and parsed_context != context_id:
        return None
    context = context_id or parsed_context
    if not context:
        return None
    try:
        thread = await _thread(call, context)
        anchor = await _run(call, context, anchor_id) if thread is not None else None
    except HTTPException as exc:
        if exc.status_code in (400, 404, 422):
            return None
        raise
    if thread is None or anchor is None or anchor.assistant_id != call.assistant_id:
        return None
    stored = _a2a(anchor).get("task_id")
    if isinstance(stored, str) and stored and stored != task_id:
        return None
    runs = await _context_runs(call, context)
    latest_id = _latest_run_id(runs)
    latest, newest = anchor, False
    for run in _runs_for_assistant(runs, call.assistant_id):
        if run.run_id == anchor_id or _a2a(run).get("task_id") == task_id:
            latest, newest = run, run.run_id == latest_id
            break
    return Task(
        ref=TaskRef(task_id, context),
        run=latest,
        state=_state_for_run(latest, thread if newest else None),
        task_ids_by_start_id=_task_ids_by_start_id(runs, context, call.assistant_id),
    )


def _state_for_run(run: Run, thread: ThreadORM | None = None) -> str:
    if run.status == "success" and thread is not None and thread.status == "interrupted":
        return INPUT_REQUIRED
    return _STATES.get(run.status, SUBMITTED)


def _task_id_for_run(context_id: str, run: Run) -> str:
    stored = _a2a(run).get("task_id")
    return stored if isinstance(stored, str) and stored else f"{context_id}:{run.run_id}"


def _task_ids_by_start_id(runs: Sequence[Run], context_id: str, assistant_id: str) -> dict[str, str]:
    """The task each new task's start stamp stands for, on the assistant's runs."""
    ids: dict[str, str] = {}
    for run in runs:
        start = _a2a(run).get("task_start_id")
        if run.assistant_id == assistant_id and isinstance(start, str) and start:
            ids[start] = _task_id_for_run(context_id, run)
    return ids


def _sort_key(run: Run) -> tuple[datetime, str]:
    return run.created_at, run.run_id


def _latest_run_id(runs: Sequence[Run]) -> str | None:
    return max(runs, key=_sort_key).run_id if runs else None


def _runs_for_assistant(runs: Sequence[Run], assistant_id: str) -> list[Run]:
    return sorted((run for run in runs if run.assistant_id == assistant_id), key=_sort_key, reverse=True)


def _history_length(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RpcError(INVALID_PARAMS, "historyLength must be a non-negative integer")
    if value > _MAX_HISTORY:
        raise RpcError(INVALID_PARAMS, f"historyLength cannot exceed {_MAX_HISTORY}")
    return value


def _history_scope(options: Mapping[str, Any], headers: Headers) -> HistoryScope:
    scope = options.get("historyScope")
    if scope is None:
        scope = headers.get(HISTORY_SCOPE_HEADER, "context")
    if scope not in ("context", "task"):
        raise RpcError(INVALID_PARAMS, "historyScope must be 'context' or 'task'")
    return cast(HistoryScope, scope)


# --- reading as the caller


async def _assistant(authorize: Authorization, assistant_id: str) -> Assistant:
    """The assistant, read as *authorize* (its caller's ``assistants.read``) lets them.

    Raises:
        ValueError: When the caller cannot read it, worded as the Agent Server words it.
    """
    try:
        uuid.UUID(assistant_id)
    except ValueError:
        raise ValueError(f"Failed to get assistant '{assistant_id}': Invalid assistant ID: must be a UUID") from None
    try:
        async with get_session_maker()() as session:
            service = AssistantService(session, authorize.user, get_langgraph_service())
            return await service.get_assistant(authorize, assistant_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise ValueError(f"Assistant '{assistant_id}' not found") from exc
        raise ValueError(f"Failed to get assistant '{assistant_id}': {exc.detail}") from exc


_schema_cache: dict[tuple[str, Any, str], tuple[float, dict[str, Any]]] = {}


async def _schemas(authorize: Authorization, assistant: Assistant) -> dict[str, Any]:
    """The assistant's schemas, read as *authorize* (``assistants.read``) lets, cached a minute as the Agent
    Server caches them."""
    key = (assistant.assistant_id, assistant.version, authorize.user.identity)
    now = asyncio.get_running_loop().time()
    cached = _schema_cache.get(key)
    if cached is not None and now - cached[0] < _SCHEMA_TTL_SECONDS:
        return cached[1]
    async with get_session_maker()() as session:
        service = AssistantService(session, authorize.user, get_langgraph_service())
        schemas = await service.get_assistant_schemas(authorize, assistant.assistant_id)
    _schema_cache[key] = (now, schemas)
    return schemas


async def _require_messages_input(authorize: Authorization, assistant: Assistant, parts: Sequence[Any]) -> None:
    """Refuse text, files and A2UI for an assistant whose input takes no ``messages``.

    Raises:
        ValueError: Worded as the Agent Server words it.
    """
    try:
        schemas = await _schemas(authorize, assistant)
    except Exception as exc:
        raise ValueError(f"Failed to get schemas for assistant '{assistant.assistant_id}': {exc}") from exc
    carries_messages = any(
        isinstance(part, dict) and ("text" in part or "file" in part or wire.is_a2ui_part(part)) for part in parts
    )
    if not carries_messages:
        return
    input_schema = schemas.get("input_schema") or schemas.get("state_schema")
    if not input_schema:
        raise ValueError(
            f"Assistant '{assistant.assistant_id}' has no input schema defined. A2A conversational agents using "
            "text or file parts must have an input schema with a 'messages' field."
        )
    properties = input_schema.get("properties", {})
    if "messages" not in properties:
        raise ValueError(
            f"Assistant '{assistant.assistant_id}' (graph '{assistant.graph_id}') does not support A2A "
            "conversational messages. Graph input schema must include a 'messages' field to accept text or file "
            f"parts. Available input fields: {list(properties)}"
        )


async def _thread(call: Call, thread_id: str) -> ThreadORM | None:
    """The caller's thread, read as ``threads.read``; None when there is none."""
    try:
        return await authorized_thread(call.threads("read"), thread_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


async def _run(call: Call, thread_id: str, run_id: str) -> Run | None:
    """The caller's run on the thread, read as ``threads.read``; None when there is none."""
    try:
        return await authorized_run(call.threads("read"), thread_id, run_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


async def _context_runs(call: Call, thread_id: str) -> list[Run]:
    """The thread's runs, newest first, as many as a task lookup scans."""
    async with get_session_maker()() as session:
        rows = (
            await session.scalars(
                select(RunORM)
                .where(RunORM.thread_id == thread_id, RunORM.user_id == call.user.identity)
                .order_by(RunORM.created_at.desc())
                .limit(_RUN_SCAN_LIMIT)
            )
        ).all()
    if len(rows) == _RUN_SCAN_LIMIT:
        logger.warning(
            "A2A context run scan reached its limit; task state or history may be truncated", context_id=thread_id
        )
    return [Run.model_validate(row) for row in rows]


async def _context_threads(call: Call, context_id: str | None) -> list[ThreadORM]:
    """The threads ListTasks looks in: the context named, or the caller's threads ``threads.search`` lets through."""
    if context_id is not None:
        try:
            thread = await _thread(call, context_id)
        except HTTPException as exc:
            if exc.status_code not in (400, 404, 422):
                raise
            thread = None
        return [thread] if thread is not None else []
    filters = await call.threads("search")({"metadata": {}, "limit": _CONTEXT_SEARCH_LIMIT, "offset": 0})
    async with get_session_maker()() as session:
        return list(
            (
                await session.scalars(
                    select(ThreadORM)
                    .where(visible_threads(call.user, filters))
                    .order_by(ThreadORM.updated_at.desc())
                    .limit(_CONTEXT_SEARCH_LIMIT)
                )
            ).all()
        )


@dataclass(frozen=True)
class _RunState:
    """What a run's checkpoints hold: the values and interrupts it ended with (None when it wrote none), and the
    messages it began with (None when that cannot be told)."""

    values: dict[str, Any] | None = None
    interrupts: list[Any] = field(default_factory=list)
    began_with: list[Any] | None = None

    @property
    def messages(self) -> RunMessages:
        """The messages the run ended with, and which of them it added."""
        if self.values is None:
            return RunMessages([], None)
        messages = self.values.get("messages") or []
        return (
            RunMessages(messages, None) if self.began_with is None else RunMessages.between(self.began_with, messages)
        )


async def _run_state(call: Call, context_id: str, run_id: str) -> _RunState:
    """The run's last checkpoint and its input one, read as ``threads.read``; empty without a thread to read."""
    thread = await _thread(call, context_id)
    service = get_langgraph_service()
    if thread is None or thread.metadata_dict.get("graph_id") not in service.list_graphs():
        return _RunState()
    async with service.thread_graph(thread, call.user, access_context="threads.read") as bound:
        if bound is None:
            return _RunState()
        graph, config = bound
        ended = await _latest_checkpoint(graph, config, {"run_id": run_id})
        began = await _latest_checkpoint(graph, config, {"run_id": run_id, "source": "input"})
    if ended is None:
        return _RunState()
    return _RunState(
        values=_values(ended),
        interrupts=to_jsonable(list(ended.interrupts or ())),
        began_with=None if began is None else _values(began).get("messages") or [],
    )


async def _run_messages(call: Call, context_id: str, run_id: str) -> RunMessages:
    """The messages a run ended with, and which of them it added."""
    return (await _run_state(call, context_id, run_id)).messages


async def _latest_checkpoint(graph: Pregel, config: RunnableConfig, metadata: dict[str, Any]) -> StateSnapshot | None:
    """The thread's latest checkpoint whose metadata matches, or None."""
    history = cast(AsyncGenerator[StateSnapshot], graph.aget_state_history(config, filter=metadata, limit=1))
    async with aclosing(history) as snapshots:
        async for snapshot in snapshots:
            return snapshot
    return None


def _values(snapshot: StateSnapshot) -> dict[str, Any]:
    return to_jsonable(snapshot.values) if isinstance(snapshot.values, dict) else {}


# --- cancelling


async def _cancel_run(call: Call, thread_id: str, run_id: str) -> None:
    """Interrupt the run, as ``threads.update``, and wait for it to stop."""
    filters = await call.threads("update")({"thread_id": thread_id, "run_id": run_id, "action": "interrupt"})
    async with get_session_maker()() as session:
        run = await owned_run(session, thread_id, run_id, call.user, filters=filters)
        # The Agent Server cancels through its REST route, whose reason the run then carries.
        reason = "interrupt requested via POST /threads/{thread_id}/runs/{run_id}/cancel"
        await request_run_interruption(session, run, action="interrupt", reason=reason)
    try:
        await executor.wait_for_completion(run_id, timeout=_CANCEL_WAIT_SECONDS)
    except TimeoutError:
        logger.info("A2A cancel stopped waiting before the run did", run_id=run_id)


async def _end_thread_wait(call: Call, thread_id: str) -> None:
    """End the thread's wait on an interrupt as ``POST /threads/{thread_id}/state`` would: a state update as
    ``__end__``, authorized as ``threads.update``, then kept as the thread's state."""
    thread = await authorized_thread(call.threads("update"), thread_id)
    async with get_langgraph_service().thread_graph(thread, call.user, access_context="threads.update") as bound:
        if bound is None:
            return
        graph, config = bound
        state = await graph.aget_state(await graph.aupdate_state(config, None, as_node="__end__"))
    await record_state(thread_id, call.user, state)


async def _delete_run(call: Call, thread_id: str, run_id: str) -> None:
    """Remove a marker run a failed cancel left; best effort."""
    try:
        async with get_session_maker()() as session:
            run = await session.get(RunORM, run_id)
            if run is not None and run.thread_id == thread_id and run.user_id == call.user.identity:
                await session.delete(run)
                await session.commit()
    except Exception:
        logger.exception("Failed to roll back A2A cancellation marker", marker_run_id=run_id)
