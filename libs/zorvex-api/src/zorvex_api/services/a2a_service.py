"""A2A on top of runs and threads, as the Agent Server serves it (api/a2a.py routes the JSON-RPC).

A task is the runs of one message. Its id is ``{contextId}:{runId}``, the
context being the thread and the run the one its first message started, whose
metadata records ``a2a.task_start_id``; each later message of the task starts
a run recording ``a2a.task_id``. A task's state is its latest run's: pending,
running, success, interrupted (a cancel), error or timeout are SUBMITTED,
WORKING, COMPLETED, CANCELED and FAILED, and a success on a thread that waits
on an interrupt, when it is the thread's latest run, is INPUT_REQUIRED.

Each handler answers a method's params with ``{"result": ...}`` or
``{"error": {...}}``, in the Agent Server's wire format (a2a_messages.py), and
operates as its caller, each step authorized as its REST counterpart.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, cast

import structlog
from fastapi import HTTPException, Request
from langchain_core.utils.uuid import uuid7
from pydantic import ValidationError
from sqlalchemy import select
from starlette.datastructures import Headers

from zorvex_api import __version__
from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.models import Assistant, User
from zorvex_api.models.runs import Run, RunCreate
from zorvex_api.services import a2a_messages as wire
from zorvex_api.services import a2ui
from zorvex_api.services.a2a_messages import HistoryScope, InvalidAgentResponseError, RunMessages
from zorvex_api.services.a2ui import A2UIContext, A2UIValidationError
from zorvex_api.services.assistant_service import AssistantService
from zorvex_api.services.broker import broker_manager
from zorvex_api.services.executor import executor
from zorvex_api.services.langgraph_service import create_thread_config, get_langgraph_service, thread_graph_config
from zorvex_api.services.ownership import owned_run, owned_thread, visible_threads
from zorvex_api.services.run_cleanup import request_run_interruption
from zorvex_api.services.run_preparation import prepare_run
from zorvex_api.services.run_waiters import run_result
from zorvex_api.services.thread_service import ThreadService
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

# The extension that lets a client ask for a task's own history instead of its context's.
HISTORY_SCOPE_EXTENSION = "https://langchain.com/a2a/extensions/history-scope/v1"
HISTORY_SCOPE_HEADER = "LangGraph-A2A-History-Scope"
# The methods the Agent Server serves: A2A 1.0's, less subscriptions and push
# notifications, and four v0.3 names it answers in v0.3's spelling.
METHODS = frozenset(
    {"SendMessage", "SendStreamingMessage", "GetTask", "ListTasks", "CancelTask", "GetExtendedAgentCard"}
)
V03_METHODS = {
    "message/send": "SendMessage",
    "message/stream": "SendStreamingMessage",
    "tasks/get": "GetTask",
    "tasks/cancel": "CancelTask",
}
IO_MODES = ("application/json", "text/plain")

INTERNAL_ERROR = -32603
INVALID_PARAMS = -32602
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005
INVALID_AGENT_RESPONSE = -32006

_STATES = {
    "pending": "TASK_STATE_SUBMITTED",
    "running": "TASK_STATE_WORKING",
    "success": "TASK_STATE_COMPLETED",
    "interrupted": "TASK_STATE_CANCELED",
    "error": "TASK_STATE_FAILED",
    "timeout": "TASK_STATE_FAILED",
}
_TERMINAL = frozenset({"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"})
_VALID_STATES = frozenset(
    {*_STATES.values(), "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_REJECTED", "TASK_STATE_AUTH_REQUIRED"}
)
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
    """A JSON-RPC error a handler answers with."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data

    def body(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return {"error": error}


class _RunNotCreated(RpcError):
    """The run a message asks for could not be created; a stream answers it as an internal error."""


@dataclass(frozen=True)
class Call:
    """Who asks which assistant, and with which request."""

    user: User
    assistant_id: str
    request: Request

    @property
    def headers(self) -> Headers:
        return self.request.headers


@dataclass(frozen=True)
class RunView:
    """A run as tasks read it."""

    run_id: str
    thread_id: str
    assistant_id: str
    status: str
    created_at: datetime
    updated_at: datetime | None
    metadata: dict[str, Any]

    @classmethod
    def of(cls, run: RunORM) -> RunView:
        model = Run.model_validate(run)
        return cls(
            run_id=model.run_id,
            thread_id=model.thread_id,
            assistant_id=model.assistant_id,
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata,
        )

    @property
    def a2a(self) -> dict[str, Any]:
        recorded = self.metadata.get("a2a")
        return recorded if isinstance(recorded, dict) else {}

    @property
    def timestamp(self) -> str:
        return (self.updated_at or self.created_at).isoformat()

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.created_at.isoformat(), self.run_id


@dataclass(frozen=True)
class Task:
    task_id: str
    context_id: str
    run: RunView
    state: str
    status_timestamp: str
    task_ids_by_start_id: dict[str, str]


# --- methods


async def send_message(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """SendMessage: start the message's run (a new task, or the next of a task), then answer the task."""
    try:
        return await _send_message(call, params)
    except RpcError as error:
        return error.body()
    except InvalidAgentResponseError as exc:
        return RpcError(INVALID_AGENT_RESPONSE, str(exc)).body()
    except Exception:
        logger.exception("A2A SendMessage failed", assistant_id=call.assistant_id)
        return RpcError(INTERNAL_ERROR, "Internal server error").body()


async def _send_message(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    configuration = params.get("configuration")
    configuration = {} if configuration is None else configuration
    if not isinstance(configuration, dict):
        raise RpcError(INVALID_PARAMS, "Invalid params: 'configuration' must be an object")
    for name in ("returnImmediately", "blocking"):
        if configuration.get(name) is not None and not isinstance(configuration[name], bool):
            raise RpcError(INVALID_PARAMS, f"Invalid params: '{name}' must be a boolean")
    return_immediately = configuration.get("returnImmediately")
    if return_immediately is None:
        return_immediately = configuration.get("blocking") is False
    view = wire.View(_history_length(configuration.get("historyLength")), _history_scope(configuration, call.headers))

    message = params.get("message")
    if not message:
        raise RpcError(INVALID_PARAMS, "Missing 'message' in params")
    started = await _start(call, message, params.get("context"), stream=False)

    if return_immediately:
        immediate: dict[str, Any] = {
            "kind": "task",
            "id": started.task_id,
            "contextId": started.context_id,
            "status": {"state": _state_for_run(started.run), "timestamp": started.run.timestamp},
        }
        if view.length:
            run_messages = await _run_messages(call, started.context_id, started.run.run_id)
            if not run_messages.messages:
                run_messages = RunMessages(list(started.input.get("messages", [])), 0)
            ids = {**started.task_ids_by_start_id}
            if started.task_start_id:
                ids.setdefault(started.task_start_id, started.task_id)
            immediate["history"] = wire.history(
                run_messages.stamped(started.task_id),
                started.task_id,
                started.context_id,
                replace(view, a2ui=started.a2ui, task_ids=ids),
            )
        return {"result": {"task": immediate}}

    ids = started.task_ids_by_start_id
    if started.task_start_id:
        try:
            ids = _task_ids_by_start_id(
                await _context_runs(call, started.context_id), started.context_id, call.assistant_id
            )
        except Exception:
            logger.warning("Failed to resolve A2A history provenance", context_id=started.context_id, exc_info=True)
        ids.setdefault(started.task_start_id, started.task_id)

    result = await _join(call, started.context_id, started.run.run_id)
    data_only = started.command is None and "messages" not in started.input
    turn = None
    if data_only or (view.scope == "task" and view.length != 0 and "messages" not in started.input):
        run_messages = await _run_messages(call, started.context_id, started.run.run_id)
        result = {**result, "messages": run_messages.stamped(started.task_id)}
        if data_only:
            turn = run_messages.added
    return _task_response(
        started.task_id,
        started.context_id,
        result,
        replace(view, a2ui=started.a2ui, task_ids=ids),
        assistant_id=call.assistant_id,
        turn=turn,
    )


async def stream_message(call: Call, params: dict[str, Any], rpc_id: Any) -> AsyncGenerator[dict[str, Any]]:
    """SendStreamingMessage: each JSON-RPC response of the stream, the task first and its final status last."""

    def response(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

    def failure(error: RpcError) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, **error.body()}

    try:
        message = params.get("message")
        if not message:
            yield failure(RpcError(INVALID_PARAMS, "Missing 'message' in params"))
            return
        try:
            started = await _start(call, message, params.get("context"), stream=True)
        except _RunNotCreated:
            # The Agent Server's stream fails as it starts the run, past its checks.
            yield failure(RpcError(INTERNAL_ERROR, "Internal server error"))
            return
        except RpcError as error:
            yield failure(error)
            return
        task_id, context_id = started.task_id, started.context_id
        initial = {
            "kind": "task",
            "id": task_id,
            "contextId": context_id,
            "history": [{"kind": "message", **message, "taskId": task_id, "contextId": context_id}],
            "status": {"state": "TASK_STATE_SUBMITTED", "timestamp": wire.now_iso()},
        }
        yield response({"task": initial})

        result: Any = None
        error: Any = None
        working = False
        first_metadata = True
        async for event, data in _run_events(started.run.run_id):
            try:
                if event == "metadata":
                    # The first is the run's; another is a retry starting.
                    if first_metadata:
                        first_metadata = False
                    elif isinstance(data, dict) and data.get("run_id") and not working:
                        working = True
                        yield response(
                            {
                                "taskId": task_id,
                                "contextId": context_id,
                                "kind": "status-update",
                                "status": {"state": "TASK_STATE_WORKING"},
                                "final": False,
                            }
                        )
                elif event == "error":
                    error = data
                elif event == "values":
                    error, result = None, data
                elif event.startswith("messages"):
                    error = None
                    if isinstance(data, list) and data:
                        update = wire.working_update(to_jsonable(data), task_id=task_id, context_id=context_id)
                        parts = update["status"]["message"]["parts"]
                        if any(part.get("text") or part.get("data") for part in parts):
                            yield response(update)
            except Exception as exc:
                logger.exception("Failed to process message stream")
                error = {"error": type(exc).__name__, "message": str(exc)}

        result = to_jsonable(result)
        final_message = None
        if isinstance(error, dict) and ("__error__" in error or "error" in error):
            inner = error.get("__error__")
            text = inner.get("error") if isinstance(inner, dict) else error.get("message")
            final_message = wire.agent_message(str(text or ""), task_id=task_id, context_id=context_id)
        if error:
            state = "TASK_STATE_FAILED"
        elif isinstance(result, dict) and "__interrupt__" in result:
            state = "TASK_STATE_INPUT_REQUIRED"
        else:
            state = "TASK_STATE_COMPLETED"
        if state == "TASK_STATE_COMPLETED" and isinstance(result, dict):
            turn = None
            if started.command is None and "messages" not in started.input:
                turn = (await _run_messages(call, context_id, started.run.run_id)).added
            try:
                parts = wire.response_parts(result, a2ui_context=started.a2ui, turn=turn)
            except A2UIValidationError:
                logger.exception("Agent emitted an invalid A2UI payload")
                yield failure(RpcError(INTERNAL_ERROR, "Internal server error"))
                return
            except InvalidAgentResponseError as exc:
                yield failure(RpcError(INVALID_AGENT_RESPONSE, str(exc)))
                return
            yield response(
                wire.response_artifact_update(
                    task_id=task_id, context_id=context_id, parts=parts, assistant_id=call.assistant_id
                )
            )
        if isinstance(result, dict) and "__interrupt__" in result:
            yield response(
                wire.interrupt_artifact_update(
                    task_id=task_id, context_id=context_id, interrupts=result["__interrupt__"]
                )
            )
        status: dict[str, Any] = {"state": state}
        if final_message is not None:
            status["message"] = final_message
        status["timestamp"] = wire.now_iso()
        yield response(
            {"taskId": task_id, "contextId": context_id, "kind": "status-update", "status": status, "final": True}
        )
    except Exception:
        logger.exception("A2A SendStreamingMessage failed", assistant_id=call.assistant_id)
        yield failure(RpcError(INTERNAL_ERROR, "Internal server error"))


async def get_task(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """GetTask: the task, its history and, once it completed or waits, its artifacts."""
    try:
        return await _get_task(call, params)
    except RpcError as error:
        return error.body()
    except InvalidAgentResponseError as exc:
        return RpcError(INVALID_AGENT_RESPONSE, str(exc)).body()
    except Exception:
        logger.exception("A2A GetTask failed", task_id=params.get("id"))
        return RpcError(INTERNAL_ERROR, "Internal server error").body()


async def _get_task(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    task_id, context_id = _task_params(params)
    history_length_value = params.get("historyLength")
    task = await _resolve_task(call, task_id, context_id)
    if task is None:
        location = f" in context '{context_id}'" if context_id else ""
        raise RpcError(TASK_NOT_FOUND, f"Task not found: {task_id}{location}")
    run = task.run
    view = wire.View(
        _history_length(history_length_value),
        _history_scope(params, call.headers),
        a2ui.run_context(run.metadata),
        task.task_ids_by_start_id,
    )

    if task.state in ("TASK_STATE_COMPLETED", "TASK_STATE_INPUT_REQUIRED"):
        values, interrupts = await _run_checkpoint(call, task.context_id, run.run_id)
        before = await _latest_messages(call, task.context_id, {"run_id": run.run_id, "source": "input"})
        result = dict(values or {})
        messages = result.get("messages", []) or []
        run_messages = RunMessages.between(before, messages) if before is not None else RunMessages(messages, None)
        if "messages" in result:
            result["messages"] = run_messages.stamped(task.task_id)
        if task.state == "TASK_STATE_INPUT_REQUIRED":
            result["__interrupt__"] = interrupts
        response = _task_response(
            task.task_id, task.context_id, result, view, assistant_id=call.assistant_id, turn=run_messages.added
        )
        body = response["result"]["task"]
        body["status"]["timestamp"] = task.status_timestamp
        return {"result": body}

    history: list[dict[str, Any]] = []
    if view.length != 0:
        try:
            messages = (await _run_messages(call, task.context_id, run.run_id)).stamped(task.task_id)
            history = wire.history(messages, task.task_id, task.context_id, view)
        except (A2UIValidationError, InvalidAgentResponseError):
            raise
        except Exception:
            logger.exception("Failed to get thread state for tasks/get")
    status: dict[str, Any] = {"state": task.state, "timestamp": task.status_timestamp}
    if task.state == "TASK_STATE_FAILED":
        status["message"] = wire.agent_message(f"Task failed with status: {run.status}", task_id=task.task_id)
    body: dict[str, Any] = {"kind": "task", "id": task.task_id, "contextId": task.context_id, "status": status}
    if view.length != 0:
        body["history"] = history
    return {"result": body}


async def cancel_task(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """CancelTask: stop the task's active run, or end its wait for input."""
    try:
        return await _cancel_task(call, params)
    except RpcError as error:
        return error.body()
    except Exception:
        logger.exception("A2A CancelTask failed", task_id=params.get("id"))
        return RpcError(INTERNAL_ERROR, "Internal server error").body()


async def _cancel_task(call: Call, params: dict[str, Any]) -> dict[str, Any]:
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

    marker: RunView | None = None
    try:
        if task.state == "TASK_STATE_INPUT_REQUIRED":
            # A marker run in the task, cancelled, makes the task's latest run interrupted; the
            # thread then stops waiting on its interrupt.
            canonical = _task_id_for_run(task.context_id, task.run)
            marker = await _create_run(
                call,
                task.context_id,
                RunCreate(
                    assistant_id=call.assistant_id,
                    metadata={"a2a": {**task.run.a2a, "task_id": canonical}},
                    after_seconds=_MARKER_DELAY_SECONDS,
                ),
            )
            await _cancel_run(call, task.context_id, marker.run_id)
            await _end_thread_wait(call, task.context_id)
        elif task.run.status not in ("pending", "running"):
            raise RpcError(TASK_NOT_CANCELABLE, "Task has no active run to cancel")
        else:
            await _cancel_run(call, task.context_id, task.run.run_id)
    except RpcError:
        raise
    except Exception as exc:
        if marker is not None:
            await _delete_run(call, task.context_id, marker.run_id)
        logger.exception("Failed to cancel A2A task", task_id=task_id)
        raise RpcError(INTERNAL_ERROR, f"Failed to cancel task: {exc!s}") from exc

    status = {
        "state": "TASK_STATE_CANCELED",
        "timestamp": wire.now_iso(),
        "message": wire.agent_message("Task was canceled", task_id=task_id),
    }
    return {"result": {"kind": "task", "id": task_id, "contextId": task.context_id, "status": status}}


async def list_tasks(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    """ListTasks: the assistant's tasks on the caller's threads, newest status first, a page at a time."""
    try:
        return await _list_tasks(call, params)
    except RpcError as error:
        return error.body()
    except InvalidAgentResponseError as exc:
        return RpcError(INVALID_AGENT_RESPONSE, str(exc)).body()
    except Exception:
        logger.exception("A2A ListTasks failed")
        return RpcError(INTERNAL_ERROR, "Internal server error").body()


async def _list_tasks(call: Call, params: dict[str, Any]) -> dict[str, Any]:
    page_size = params.get("pageSize")
    if page_size is not None and (
        not isinstance(page_size, int) or isinstance(page_size, bool) or not 1 <= page_size <= _LIST_MAX_PAGE_SIZE
    ):
        raise RpcError(INVALID_PARAMS, f"pageSize must be an integer between 1 and {_LIST_MAX_PAGE_SIZE}.")
    page_size = _LIST_PAGE_SIZE if page_size is None else page_size
    view = wire.View(_history_length(params.get("historyLength")), _history_scope(params, call.headers))
    status = params.get("status")
    status = wire.STATES_FROM_V03.get(status, status) if isinstance(status, str) else status
    if status is not None and status not in _VALID_STATES:
        raise RpcError(INVALID_PARAMS, f"Invalid status value: '{status}'.")
    after = params.get("statusTimestampAfter")
    if after is not None:
        try:
            if not isinstance(after, str):
                raise ValueError("Not a string")
            datetime.fromisoformat(after.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise RpcError(INVALID_PARAMS, "statusTimestampAfter must be a valid ISO 8601 timestamp.") from None
    page_token = params.get("pageToken")
    if page_token is not None:
        if not isinstance(page_token, str):
            raise RpcError(INVALID_PARAMS, "pageToken must be a string.")
        if page_token != "":  # nosec B105 - a page cursor, not a password
            try:
                int(page_token)
            except ValueError:
                raise RpcError(INVALID_PARAMS, "Invalid pageToken.") from None
    include_artifacts = params.get("includeArtifacts", False)
    context_id = params.get("contextId")
    if context_id is not None and not isinstance(context_id, str):
        raise RpcError(INVALID_PARAMS, "contextId must be a string.")

    threads = await _context_threads(call, context_id)
    tasks: dict[str, dict[str, Any]] = {}
    runs_by_task: dict[str, RunView] = {}
    ids_by_context: dict[str, dict[str, str]] = {}
    for thread in threads:
        runs = await _context_runs(call, thread.thread_id)
        latest_id = _latest_run_id(runs)
        ids_by_context[thread.thread_id] = _task_ids_by_start_id(runs, thread.thread_id, call.assistant_id)
        for run in _runs_for_assistant(runs, call.assistant_id):
            task_id = _task_id_for_run(thread.thread_id, run)
            if task_id in tasks or task_id in runs_by_task:
                continue
            runs_by_task[task_id] = run
            state = _state_for_run(run, thread if run.run_id == latest_id else None)
            if status and state != status:
                continue
            timestamp = run.timestamp
            if after and timestamp < after:
                continue
            task: dict[str, Any] = {
                "kind": "task",
                "id": task_id,
                "contextId": thread.thread_id,
                "status": {"state": state, "timestamp": timestamp},
            }
            if not include_artifacts:
                task["artifacts"] = []
            tasks[task_id] = task

    ordered = sorted(tasks.values(), key=lambda task: task["status"].get("timestamp", ""), reverse=True)
    offset = int(page_token) if page_token else 0
    page = ordered[offset : offset + page_size]
    if view.length != 0:
        for task in page:
            run = runs_by_task[task["id"]]
            messages = (await _run_messages(call, task["contextId"], run.run_id)).stamped(task["id"])
            seen = replace(view, a2ui=a2ui.run_context(run.metadata), task_ids=ids_by_context.get(task["contextId"]))
            task["history"] = wire.history(messages, task["id"], task["contextId"], seen)
    next_offset = offset + len(page)
    return {
        "result": {
            "tasks": page,
            "totalSize": len(ordered),
            "pageSize": len(page),
            "nextPageToken": str(next_offset) if next_offset < len(ordered) else "",
        }
    }


async def extended_card(call: Call) -> dict[str, Any]:
    """GetExtendedAgentCard: the public card; there is nothing more to tell an authenticated caller."""
    try:
        return {"result": await agent_card(call.request, call.user, call.assistant_id)}
    except ValueError as exc:
        return RpcError(INVALID_PARAMS, str(exc)).body()
    except Exception:
        logger.exception("Error generating extended agent card", assistant_id=call.assistant_id)
        return RpcError(INTERNAL_ERROR, "Internal server error").body()


async def agent_card(request: Request, user: User, assistant_id: str) -> dict[str, Any]:
    """The assistant's agent card, as the Agent Server builds it.

    Raises:
        ValueError: For an assistant that cannot be read, or whose ``metadata.a2a`` modes are malformed.
    """
    assistant = await _assistant(user, assistant_id)
    schemas = await _schemas(user, assistant)
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


@dataclass(frozen=True)
class _Started:
    run: RunView
    task_id: str
    context_id: str
    task_start_id: str | None
    task_ids_by_start_id: dict[str, str]
    input: dict[str, Any]
    command: dict[str, Any] | None
    a2ui: A2UIContext


async def _start(call: Call, message: Any, run_context: Any, *, stream: bool) -> _Started:
    """Check the message, read the assistant and the task it continues, then start its run."""
    message_id = message.get("messageId")
    if not message_id:
        raise RpcError(INVALID_PARAMS, "Missing required 'messageId' in message")
    if not message.get("role"):
        raise RpcError(INVALID_PARAMS, "Missing required 'role' in message")
    parts = message.get("parts", [])
    if not stream and not isinstance(parts, list):
        raise RpcError(INVALID_PARAMS, "Invalid params: 'parts' must be an array")
    if not parts:
        raise RpcError(INVALID_PARAMS, "Message must contain at least one part")
    try:
        assistant = await _assistant(call.user, call.assistant_id)
        await _require_messages_input(call.user, assistant, parts)
    except ValueError as exc:
        raise RpcError(INVALID_PARAMS, str(exc)) from exc

    requested_context = message.get("contextId")
    existing_task = message.get("taskId")
    if requested_context is not None and not isinstance(requested_context, str):
        raise RpcError(INVALID_PARAMS, "contextId must be a string")
    if existing_task is not None and not isinstance(existing_task, str):
        raise RpcError(INVALID_PARAMS, "taskId must be a string")

    task: Task | None = None
    if existing_task is None:
        context_id = requested_context or str(uuid.uuid4())
        task_id: str | None = None
        task_start_id: str | None = str(uuid7())
        ids: dict[str, str] = {}
        mime_type = a2ui.MIME_TYPE
    else:
        task = await _resolve_task(call, existing_task, requested_context)
        if task is None:
            raise RpcError(TASK_NOT_FOUND, f"Task not found: {existing_task}")
        if task.state in _TERMINAL:
            raise RpcError(UNSUPPORTED_OPERATION, "A terminal task cannot accept another message")
        context_id, task_id, task_start_id = task.context_id, task.task_id, None
        ids = task.task_ids_by_start_id
        mime_type = a2ui.run_context(task.run.metadata).mime_type

    try:
        a2ui_context = a2ui.message_context(message, assistant.metadata, call.headers, default_mime_type=mime_type)
        run_input = wire.message_input(
            parts,
            wire.normalize_role(message.get("role", "ROLE_USER")),
            message_id,
            a2ui_context=a2ui_context,
            task_id=task_id,
            task_start_id=task_start_id,
        )
    except A2UIValidationError as exc:
        raise RpcError(INVALID_PARAMS, str(exc)) from exc
    except ValueError as exc:
        raise RpcError(CONTENT_TYPE_NOT_SUPPORTED, str(exc)) from exc

    command = message.get("command")
    if command is not None and isinstance(command, dict) and command.get("resume"):
        if not requested_context:
            raise RpcError(INVALID_PARAMS, "contextId is required when resuming a task with command.resume")
        if existing_task is None:
            raise RpcError(INVALID_PARAMS, "taskId is required when resuming a task")
    if command is None:
        command, run_input = await _implied_resume(
            call,
            parts,
            context_id=context_id if existing_task else requested_context,
            task_id=existing_task,
            run_input=run_input,
        )
    if existing_task is not None and command is None:
        logger.warning("User requested to resume a task without specifying a Command.", task_id=existing_task)

    try:
        request = RunCreate(
            assistant_id=call.assistant_id,
            input=run_input,
            command=command,
            context=run_context,
            metadata=a2ui.run_metadata(assistant.metadata, a2ui_context, task_id=task_id, task_start_id=task_start_id),
            if_not_exists="create",
            stream_mode=["messages", "values"] if stream else None,
        )
    except ValidationError as exc:
        raise _RunNotCreated(INVALID_PARAMS, f"Failed to create run: {exc}") from exc
    run = await _create_run(call, context_id, request)
    context_id = run.thread_id
    return _Started(
        run=run,
        task_id=task_id or f"{context_id}:{run.run_id}",
        context_id=context_id,
        task_start_id=task_start_id,
        task_ids_by_start_id=ids,
        input=run_input,
        command=command,
        a2ui=a2ui_context,
    )


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


async def _create_run(call: Call, context_id: str, request: RunCreate) -> RunView:
    try:
        async with get_session_maker()() as session:
            run = await prepare_run(
                session, context_id, request, call.user, authorize=Authorization(call.user, "threads", "create_run")
            )
    except HTTPException as exc:
        raise _run_create_error(exc, call.assistant_id, context_id) from exc
    return RunView(
        run_id=run.run_id,
        thread_id=run.thread_id,
        assistant_id=run.assistant_id,
        status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        metadata=run.metadata,
    )


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
    task_id: str,
    context_id: str,
    result: Mapping[str, Any],
    view: wire.View,
    *,
    assistant_id: str,
    turn: Sequence[Any] | None,
) -> dict[str, Any]:
    """A finished run's task, its history as *view* shows it: its error, the interrupts it waits on, or its reply."""
    result = to_jsonable(dict(result))
    task: dict[str, Any] = {"kind": "task", "id": task_id, "contextId": context_id}
    if view.length != 0:
        task["history"] = wire.history(result.get("messages", []) or [], task_id, context_id, view)
    if "__error__" in result:
        error = result["__error__"]
        name = error.get("error") if isinstance(error, dict) else error
        task["status"] = {
            "state": "TASK_STATE_FAILED",
            "message": wire.agent_message(f"Error executing assistant: {name}", task_id=task_id, context_id=context_id),
        }
    elif "__interrupt__" in result:
        task["status"] = {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": wire.now_iso()}
        task["artifacts"] = [wire.interrupt_artifact(result["__interrupt__"])]
    else:
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": wire.now_iso()}
        parts = wire.response_parts(result, a2ui_context=view.a2ui, turn=turn)
        task["artifacts"] = [wire.response_artifact(parts, assistant_id)]
    return {"result": {"task": task}}


async def _join(call: Call, context_id: str, run_id: str) -> dict[str, Any]:
    """The run's result once it ends, as a join answers it."""
    timed_out = False
    try:
        await executor.wait_for_completion(run_id, timeout=settings.worker.BG_JOB_TIMEOUT_SECS)
    except TimeoutError:
        timed_out = True
    return await run_result(run_id, context_id, call.user, timed_out=timed_out)


async def _run_events(run_id: str) -> AsyncGenerator[tuple[str, Any]]:
    """The run's ``(event, data)`` pairs: those stored so far, then live ones, until its stream ends."""
    broker = broker_manager.get_or_create_broker(run_id)
    seen: set[str] = set()
    for event_id, payload in await broker.replay(None):
        seen.add(event_id)
        if isinstance(payload, (tuple, list)) and len(payload) == 2:
            if payload[0] == "end":
                return
            yield payload[0], payload[1]
    async for event_id, payload in broker.aiter():
        if event_id in seen or not isinstance(payload, (tuple, list)) or len(payload) != 2:
            continue
        if payload[0] == "end":
            return
        yield payload[0], payload[1]


# --- tasks


def _task_params(params: dict[str, Any]) -> tuple[str, str | None]:
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
    if ":" in task_id:
        parsed_context, anchor_id = task_id.split(":", 1)
    else:
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
    stored = anchor.a2a.get("task_id")
    if isinstance(stored, str) and stored and stored != task_id:
        return None
    runs = await _context_runs(call, context)
    latest_id = _latest_run_id(runs)
    latest, newest = anchor, False
    for run in _runs_for_assistant(runs, call.assistant_id):
        if run.run_id == anchor_id or run.a2a.get("task_id") == task_id:
            latest, newest = run, run.run_id == latest_id
            break
    return Task(
        task_id=task_id,
        context_id=context,
        run=latest,
        state=_state_for_run(latest, thread if newest else None),
        status_timestamp=latest.timestamp,
        task_ids_by_start_id=_task_ids_by_start_id(runs, context, call.assistant_id),
    )


def _state_for_run(run: RunView, thread: ThreadORM | None = None) -> str:
    if run.status == "success" and thread is not None and thread.status == "interrupted":
        return "TASK_STATE_INPUT_REQUIRED"
    return _STATES.get(run.status, "TASK_STATE_SUBMITTED")


def _task_id_for_run(context_id: str, run: RunView) -> str:
    stored = run.a2a.get("task_id")
    return stored if isinstance(stored, str) and stored else f"{context_id}:{run.run_id}"


def _task_ids_by_start_id(runs: Sequence[RunView], context_id: str, assistant_id: str) -> dict[str, str]:
    """The task each new task's start stamp stands for, on the assistant's runs."""
    ids: dict[str, str] = {}
    for run in runs:
        start = run.a2a.get("task_start_id")
        if run.assistant_id == assistant_id and isinstance(start, str) and start:
            ids[start] = _task_id_for_run(context_id, run)
    return ids


def _latest_run_id(runs: Sequence[RunView]) -> str | None:
    return max(runs, key=lambda run: run.sort_key).run_id if runs else None


def _runs_for_assistant(runs: Sequence[RunView], assistant_id: str) -> list[RunView]:
    return sorted((run for run in runs if run.assistant_id == assistant_id), key=lambda run: run.sort_key, reverse=True)


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


async def _assistant(user: User, assistant_id: str) -> Assistant:
    """The assistant, read as ``assistants.read``.

    Raises:
        ValueError: When the caller cannot read it, worded as the Agent Server words it.
    """
    try:
        uuid.UUID(assistant_id)
    except ValueError:
        raise ValueError(f"Failed to get assistant '{assistant_id}': Invalid assistant ID: must be a UUID") from None
    try:
        async with get_session_maker()() as session:
            service = AssistantService(session, user, get_langgraph_service())
            return await service.get_assistant(Authorization(user, "assistants", "read"), assistant_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise ValueError(f"Assistant '{assistant_id}' not found") from exc
        raise ValueError(f"Failed to get assistant '{assistant_id}': {exc.detail}") from exc


_schema_cache: dict[tuple[str, Any, str], tuple[float, dict[str, Any]]] = {}


async def _schemas(user: User, assistant: Assistant) -> dict[str, Any]:
    """The assistant's schemas, cached a minute as the Agent Server caches them."""
    key = (assistant.assistant_id, assistant.version, user.identity)
    now = asyncio.get_running_loop().time()
    cached = _schema_cache.get(key)
    if cached is not None and now - cached[0] < _SCHEMA_TTL_SECONDS:
        return cached[1]
    async with get_session_maker()() as session:
        service = AssistantService(session, user, get_langgraph_service())
        schemas = await service.get_assistant_schemas(Authorization(user, "assistants", "read"), assistant.assistant_id)
    _schema_cache[key] = (now, schemas)
    return schemas


async def _require_messages_input(user: User, assistant: Assistant, parts: Sequence[Any]) -> None:
    """Refuse text, files and A2UI for an assistant whose input takes no ``messages``.

    Raises:
        ValueError: Worded as the Agent Server words it.
    """
    try:
        schemas = await _schemas(user, assistant)
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
        async with get_session_maker()() as session:
            return await ThreadService(session, call.user).get(Authorization(call.user, "threads", "read"), thread_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


async def _run(call: Call, thread_id: str, run_id: str) -> RunView | None:
    authorize = Authorization(call.user, "threads", "read")
    try:
        async with get_session_maker()() as session:
            filters = await authorize({"thread_id": thread_id, "run_id": run_id})
            return RunView.of(await owned_run(session, thread_id, run_id, call.user, filters=filters))
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


async def _context_runs(call: Call, thread_id: str) -> list[RunView]:
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
    return [RunView.of(row) for row in rows]


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
    filters = await Authorization(call.user, "threads", "search")(
        {"metadata": {}, "limit": _CONTEXT_SEARCH_LIMIT, "offset": 0}
    )
    async with get_session_maker()() as session:
        return list(
            (
                await session.scalars(
                    select(ThreadORM)
                    .where(ThreadORM.user_id == call.user.identity, *visible_threads(filters))
                    .order_by(ThreadORM.updated_at.desc())
                    .limit(_CONTEXT_SEARCH_LIMIT)
                )
            ).all()
        )


async def _run_messages(call: Call, context_id: str, run_id: str) -> RunMessages:
    """The messages a run ended with, and which of them it added."""
    messages, before = await asyncio.gather(
        _latest_messages(call, context_id, {"run_id": run_id}),
        _latest_messages(call, context_id, {"run_id": run_id, "source": "input"}),
    )
    if messages is None:
        return RunMessages([], None)
    if before is None:
        return RunMessages(messages, None)
    return RunMessages.between(before, messages)


async def _latest_messages(call: Call, context_id: str, metadata: dict[str, Any]) -> list[Any] | None:
    """The messages of the latest checkpoint whose metadata matches, or None when there is none."""
    snapshot = await _latest_checkpoint(call, context_id, metadata)
    if snapshot is None:
        return None
    values = to_jsonable(snapshot.values) if isinstance(snapshot.values, dict) else {}
    return values.get("messages") or []


async def _run_checkpoint(call: Call, context_id: str, run_id: str) -> tuple[dict[str, Any] | None, list[Any]]:
    """The values and interrupts of the last checkpoint the run wrote."""
    snapshot = await _latest_checkpoint(call, context_id, {"run_id": run_id})
    if snapshot is None:
        return None, []
    values = to_jsonable(snapshot.values) if isinstance(snapshot.values, dict) else {}
    return values, to_jsonable(list(snapshot.interrupts or ()))


async def _latest_checkpoint(call: Call, context_id: str, metadata: dict[str, Any]) -> Any:
    thread = await _thread(call, context_id)
    graph_id = (thread.metadata_dict or {}).get("graph_id") if thread is not None else None
    if thread is None or not graph_id or graph_id not in get_langgraph_service().list_graphs():
        return None
    config = create_thread_config(context_id, call.user)
    async with get_langgraph_service().get_graph(
        graph_id, thread_graph_config(thread.config, config), access_context="threads.read", user=call.user
    ) as graph:
        async for snapshot in graph.aget_state_history(config, filter=metadata, limit=1):
            return snapshot
    return None


# --- cancelling


async def _cancel_run(call: Call, thread_id: str, run_id: str) -> None:
    """Interrupt the run, as ``threads.update``, and wait for it to stop."""
    filters = await Authorization(call.user, "threads", "update")(
        {"thread_id": thread_id, "run_id": run_id, "action": "interrupt"}
    )
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
    """End the thread's wait on an interrupt: a state update as ``__end__``, authorized as ``threads.update``."""
    filters = await Authorization(call.user, "threads", "update")({"thread_id": thread_id})
    async with get_session_maker()() as session:
        thread = await owned_thread(session, thread_id, call.user, filters=filters)
    graph_id = (thread.metadata_dict or {}).get("graph_id")
    if not graph_id:
        return
    config = create_thread_config(thread_id, call.user)
    async with get_langgraph_service().get_graph(
        graph_id, thread_graph_config(thread.config, config), access_context="threads.update", user=call.user
    ) as graph:
        await graph.aupdate_state(config, None, as_node="__end__")


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
