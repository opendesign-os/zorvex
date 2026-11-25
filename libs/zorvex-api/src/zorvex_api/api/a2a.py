"""Agent-to-Agent protocol, as the Agent Server serves it: JSON-RPC at ``/a2a/{assistant_id}`` and agent cards.

services/a2a_service.py answers each method from runs and threads, in the
Agent Server's wire format. This module checks the JSON-RPC envelope in the
Agent Server's order, routes the method (answering a v0.3 method name in v0.3's
spelling, and SendStreamingMessage as SSE), and serves the cards: at
``/a2a/{assistant_id}``, its ``.well-known`` paths, and the domain root, which
describes the ``?assistant_id`` or ``DEFAULT_A2A_ASSISTANT_ID``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette import ServerSentEvent

from zorvex_api.core import sse
from zorvex_api.core.auth_deps import auth_dependency, require_auth
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.models import User
from zorvex_api.services import a2a_service
from zorvex_api.services.a2a_messages import to_spec_format
from zorvex_api.services.a2a_service import METHODS, V03_METHODS, Call
from zorvex_api.settings import settings

router = APIRouter(tags=["A2A"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)

# The dispatcher reads the body itself; the schema is declared here for the docs.
_REQUEST_SCHEMA: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["jsonrpc", "id", "method"],
                    "properties": {
                        "jsonrpc": {"type": "string", "enum": ["2.0"]},
                        "id": {"type": ["string", "integer"]},
                        "method": {"type": "string", "enum": [*sorted(METHODS), *sorted(V03_METHODS)]},
                        "params": {
                            "type": "object",
                            "description": "Method parameters, as the A2A 1.0 (or 0.3) spec defines them. SendMessage "
                            "and SendStreamingMessage also take `context`, the run's context, and the message a "
                            "`command` (resume, update or goto), the LangGraph Command it runs with.",
                        },
                    },
                }
            }
        },
    },
}

_UNREADABLE = object()
_STREAMING = frozenset({"SendStreamingMessage", "message/stream"})
_SENDING = _STREAMING | {"SendMessage", "message/send"}
_HANDLERS: dict[str, Callable[[Call, dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    "SendMessage": a2a_service.send_message,
    "GetTask": a2a_service.get_task,
    "CancelTask": a2a_service.cancel_task,
    "ListTasks": a2a_service.list_tasks,
}


async def _body(request: Request) -> Any:
    try:
        return json.loads(await request.body())
    except ValueError:
        return _UNREADABLE


def _rpc_error(request_id: str | int | None, code: int, message: str) -> JSONResponse:
    """A JSON-RPC error; HTTP 200, as JSON-RPC answers."""
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _accepts(request: Request, media_type: str) -> bool:
    """Whether the Accept header admits *media_type*: by name, as ``type/*``, or as ``*/*``."""
    admitting = {"*/*", media_type, media_type.split("/", 1)[0] + "/*"}
    return any(
        item.split(";", 1)[0].strip().lower() in admitting for item in request.headers.get("accept", "").split(",")
    )


def _rejection(request: Request, body: Any) -> Response | None:
    """The answer to a request turned away before its method runs, checked in the Agent Server's order; None to run it.

    A float or boolean id is turned away too: the Agent Server would take it,
    but JSON-RPC asks for integers.
    """
    if body is _UNREADABLE:
        return _rpc_error(None, -32700, "Invalid JSON payload")
    if not isinstance(body, dict):
        return _rpc_error(None, -32600, "Invalid message format: expected object")
    request_id = body.get("id")
    id_ok = request_id is None or (isinstance(request_id, (str, int)) and not isinstance(request_id, bool))
    if body.get("jsonrpc") != "2.0":
        return _rpc_error(
            request_id if id_ok else None, -32600, "Invalid JSON-RPC message: missing or invalid jsonrpc version"
        )
    if not id_ok:
        return _rpc_error(None, -32600, "Invalid JSON-RPC request: 'id' must be a string, integer, or null")
    method = body.get("method")
    if method is not None and not isinstance(method, str):
        return _rpc_error(request_id, -32601, "Method not found: method must be a string")
    if method is not None and method not in METHODS and method not in V03_METHODS:
        return _rpc_error(request_id, -32601, f"Method not found: {method}")
    if method in _SENDING:
        params = body.get("params", {})
        if not isinstance(params, dict):
            return _rpc_error(request_id, -32602, "Invalid params: must be an object")
        if not params.get("message") or not isinstance(params["message"], dict):
            return _rpc_error(request_id, -32602, "Missing or invalid 'message' in params")
    if method in _STREAMING and not _accepts(request, "text/event-stream"):
        return JSONResponse({"error": "Accept header must include text/event-stream for streaming"}, status_code=400)
    if method not in _STREAMING and not _accepts(request, "application/json"):
        return JSONResponse({"error": "Accept header must include application/json"}, status_code=400)
    if request_id is None:
        # Notifications too: every A2A method has an answer to give.
        return _rpc_error(None, -32600, "Invalid message format: must be a JSON-RPC request or notification")
    if method is None:
        if "result" in body or "error" in body:
            return Response(status_code=202)  # a response, of which the server expects none
        return _rpc_error(request_id, -32600, "Invalid JSON-RPC request: missing 'method' field")
    return None


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _sse_data(payload: Any) -> str:
    """A payload as one SSE data line: compact JSON, with the line separators SSE readers split on escaped."""
    text = json.dumps(to_jsonable(payload), separators=(",", ":"), ensure_ascii=False)
    return text.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


async def _sse_frames(payloads: AsyncIterator[dict[str, Any]], *, spec_format: bool) -> AsyncIterator[ServerSentEvent]:
    async for payload in payloads:
        data = to_spec_format(payload) if spec_format else payload
        yield ServerSentEvent(_sse_data(data), event="message")


@router.post("/a2a/{assistant_id}", openapi_extra=_REQUEST_SCHEMA)
@router.post("/a2a/{assistant_id}/", include_in_schema=False, name="a2a_rpc_slash")
async def a2a_rpc(
    assistant_id: str,
    request: Request,
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("assistants", "read"),
) -> Response:
    """Serve one A2A JSON-RPC request against an assistant; SendStreamingMessage answers with SSE.

    Talking to an assistant authorizes as ``assistants.read``. A message also
    creates a run, authorized as ``threads.create_run``; reading a task
    authorizes ``threads.read``, listing them ``threads.search``, and
    cancelling one ``threads.update``.
    """
    body = await _body(request)
    rejection = _rejection(request, body)
    if rejection is None and not _is_uuid(assistant_id):
        rejection = _rpc_error(body["id"], -32602, "Invalid assistant ID: must be a UUID")
    await authorize({"assistant_id": assistant_id})
    if rejection is not None:
        return rejection

    method: str = body["method"]
    spec_format = method in V03_METHODS
    method = V03_METHODS.get(method, method)
    params = body.get("params", {})
    call = Call(user=user, assistant_id=str(uuid.UUID(assistant_id)), request=request)
    if method == "SendStreamingMessage":
        frames = _sse_frames(a2a_service.stream_message(call, params, body["id"]), spec_format=spec_format)
        return sse.response(frames, headers={"Content-Type": "text/event-stream"})
    if method == "GetExtendedAgentCard":
        result = await a2a_service.extended_card(call)
    else:
        result = await _HANDLERS[method](call, params)
    payload = {"jsonrpc": "2.0", "id": body["id"], **result}
    return JSONResponse(to_spec_format(payload) if spec_format else payload)


async def _card(request: Request, user: User, assistant_id: str) -> Response:
    """The agent card, or the Agent Server's error for an assistant it cannot describe."""
    try:
        return JSONResponse(await a2a_service.agent_card(request, user, assistant_id))
    except ValueError as exc:
        return JSONResponse({"error": {"code": -32602, "message": str(exc)}}, status_code=400)
    except Exception:
        logger.exception("Failed to generate agent card", assistant_id=assistant_id)
        return JSONResponse({"error": {"code": -32603, "message": "Internal server error"}}, status_code=500)


@router.get("/a2a/{assistant_id}")
@router.get("/a2a/{assistant_id}/", include_in_schema=False, name="a2a_agent_card_slash")
@router.get("/a2a/{assistant_id}/.well-known/agent-card.json")
@router.get("/a2a/{assistant_id}/.well-known/agent.json", include_in_schema=False)
async def a2a_agent_card(
    assistant_id: str,
    request: Request,
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("assistants", "read"),
) -> Response:
    """The assistant's A2A agent card; 400 with a JSON-RPC error for an assistant it cannot describe."""
    await authorize({"assistant_id": assistant_id})
    return await _card(request, user, assistant_id)


@router.get("/.well-known/agent-card.json")
async def default_agent_card(
    request: Request,
    assistant_id: str | None = Query(
        None, description="The assistant to describe; DEFAULT_A2A_ASSISTANT_ID when omitted."
    ),
    user: User = Depends(require_auth),
    authorize: Authorization = Authorize("assistants", "read"),
) -> Response:
    """The agent card of the assistant the query names, or of the deployment's default one."""
    assistant_id = assistant_id or settings.a2a.DEFAULT_A2A_ASSISTANT_ID
    await authorize({"assistant_id": assistant_id})
    if not assistant_id:
        message = "Missing required query parameter: assistant_id"
        return JSONResponse({"error": {"code": -32602, "message": message}}, status_code=400)
    return await _card(request, user, assistant_id)
