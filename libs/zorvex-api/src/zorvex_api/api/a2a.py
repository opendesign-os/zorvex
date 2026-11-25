"""Agent-to-Agent protocol 1.0, as the Agent Server serves it: JSON-RPC at ``/a2a/{assistant_id}`` and agent cards.

services/a2a_service.py answers each method from runs and threads, in the
Agent Server's wire format. This module checks the JSON-RPC envelope in the
Agent Server's order, routes the method (SendStreamingMessage as SSE), answers
what a method raises as its JSON-RPC error, and serves the cards: at
``/a2a/{assistant_id}``, its ``.well-known/agent-card.json``, and the domain
root's, which describes the ``?assistant_id`` or ``DEFAULT_A2A_ASSISTANT_ID``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import structlog
from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette import ServerSentEvent

from zorvex_api.core import sse
from zorvex_api.core.auth import auth_dependency
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.services import a2a_service
from zorvex_api.services.a2a_messages import InvalidAgentResponseError
from zorvex_api.services.a2a_service import (
    INTERNAL_ERROR,
    INVALID_AGENT_RESPONSE,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    Call,
    RpcError,
    result_envelope,
)
from zorvex_api.settings import settings

router = APIRouter(tags=["A2A"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)

STREAMING = "SendStreamingMessage"
_HANDLERS: dict[str, Callable[[Call, dict[str, Any]], Awaitable[Any]]] = {
    "SendMessage": a2a_service.send_message,
    "GetTask": a2a_service.get_task,
    "ListTasks": a2a_service.list_tasks,
    "CancelTask": a2a_service.cancel_task,
    "GetExtendedAgentCard": a2a_service.extended_card,
}
# A2A 1.0's methods, less subscriptions and push notifications.
METHODS = frozenset({*_HANDLERS, STREAMING})
_SENDING = frozenset({"SendMessage", STREAMING})

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
                        "method": {"type": "string", "enum": sorted(METHODS)},
                        "params": {
                            "type": "object",
                            "description": "Method parameters, as the A2A 1.0 spec defines them. SendMessage and "
                            "SendStreamingMessage also take `context`, the run's context, and the message a "
                            "`command` (resume, update or goto), the LangGraph Command it runs with.",
                        },
                    },
                }
            }
        },
    },
}

_UNREADABLE = object()


async def _body(request: Request) -> Any:
    try:
        return json.loads(await request.body())
    except ValueError:
        return _UNREADABLE


def _rpc_error(request_id: str | int | None, code: int, message: str) -> JSONResponse:
    """A JSON-RPC error; HTTP 200, as JSON-RPC answers."""
    return JSONResponse(RpcError(code, message).envelope(request_id))


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
        return _rpc_error(None, PARSE_ERROR, "Invalid JSON payload")
    if not isinstance(body, dict):
        return _rpc_error(None, INVALID_REQUEST, "Invalid message format: expected object")
    request_id = body.get("id")
    id_ok = request_id is None or (isinstance(request_id, (str, int)) and not isinstance(request_id, bool))
    if body.get("jsonrpc") != "2.0":
        return _rpc_error(
            request_id if id_ok else None,
            INVALID_REQUEST,
            "Invalid JSON-RPC message: missing or invalid jsonrpc version",
        )
    if not id_ok:
        return _rpc_error(None, INVALID_REQUEST, "Invalid JSON-RPC request: 'id' must be a string, integer, or null")
    method = body.get("method")
    if method is not None and not isinstance(method, str):
        return _rpc_error(request_id, METHOD_NOT_FOUND, "Method not found: method must be a string")
    if method is not None and method not in METHODS:
        return _rpc_error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")
    if method in _SENDING:
        params = body.get("params", {})
        if not isinstance(params, dict):
            return _rpc_error(request_id, INVALID_PARAMS, "Invalid params: must be an object")
        if not params.get("message") or not isinstance(params["message"], dict):
            return _rpc_error(request_id, INVALID_PARAMS, "Missing or invalid 'message' in params")
    if method == STREAMING and not _accepts(request, "text/event-stream"):
        return JSONResponse({"error": "Accept header must include text/event-stream for streaming"}, status_code=400)
    if method != STREAMING and not _accepts(request, "application/json"):
        return JSONResponse({"error": "Accept header must include application/json"}, status_code=400)
    if request_id is None:
        # Notifications too: every A2A method has an answer to give.
        return _rpc_error(None, INVALID_REQUEST, "Invalid message format: must be a JSON-RPC request or notification")
    if method is None:
        if "result" in body or "error" in body:
            return Response(status_code=202)  # a response, of which the server expects none
        return _rpc_error(request_id, INVALID_REQUEST, "Invalid JSON-RPC request: missing 'method' field")
    return None


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _failure(error: Exception, call: Call, method: str, request_id: Any) -> dict[str, Any]:
    """The JSON-RPC error answering a method that raised *error*, called as it is handled; anything unforeseen
    is an internal error."""
    if isinstance(error, RpcError):
        return error.envelope(request_id)
    if isinstance(error, InvalidAgentResponseError):
        return RpcError(INVALID_AGENT_RESPONSE, str(error)).envelope(request_id)
    logger.exception("A2A method failed", method=method, assistant_id=call.assistant_id)
    return RpcError(INTERNAL_ERROR, "Internal server error").envelope(request_id)


async def _answer(call: Call, method: str, params: Any, request_id: Any) -> dict[str, Any]:
    try:
        return result_envelope(request_id, await _HANDLERS[method](call, params))
    except Exception as exc:
        return _failure(exc, call, method, request_id)


async def _stream(call: Call, params: Any, request_id: Any) -> AsyncIterator[ServerSentEvent]:
    """SendStreamingMessage's answers as SSE ``message`` events; the stream ends with a method's error."""
    try:
        async for result in a2a_service.stream_message(call, params):
            yield sse.event("message", result_envelope(request_id, result))
    except Exception as exc:
        yield sse.event("message", _failure(exc, call, STREAMING, request_id))


@router.post("/a2a/{assistant_id}", openapi_extra=_REQUEST_SCHEMA)
@router.post("/a2a/{assistant_id}/", include_in_schema=False, name="a2a_rpc_slash")
async def a2a_rpc(
    assistant_id: str,
    request: Request,
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
        rejection = _rpc_error(body["id"], INVALID_PARAMS, "Invalid assistant ID: must be a UUID")
    await authorize({"assistant_id": assistant_id})
    if rejection is not None:
        return rejection

    method, request_id = body["method"], body["id"]
    params = body.get("params", {})
    call = Call(authorize=authorize, assistant_id=str(uuid.UUID(assistant_id)), request=request)
    if method == STREAMING:
        return sse.response(_stream(call, params, request_id), headers={"Content-Type": "text/event-stream"})
    return JSONResponse(await _answer(call, method, params, request_id))


async def _card(request: Request, authorize: Authorization, assistant_id: str) -> Response:
    """The agent card, or the Agent Server's error for an assistant it cannot describe."""
    try:
        return JSONResponse(await a2a_service.agent_card(request, authorize, assistant_id))
    except ValueError as exc:
        return JSONResponse({"error": RpcError(INVALID_PARAMS, str(exc)).error}, status_code=400)
    except Exception:
        logger.exception("Failed to generate agent card", assistant_id=assistant_id)
        return JSONResponse({"error": RpcError(INTERNAL_ERROR, "Internal server error").error}, status_code=500)


@router.get("/a2a/{assistant_id}")
@router.get("/a2a/{assistant_id}/", include_in_schema=False, name="a2a_agent_card_slash")
@router.get("/a2a/{assistant_id}/.well-known/agent-card.json")
async def a2a_agent_card(
    assistant_id: str,
    request: Request,
    authorize: Authorization = Authorize("assistants", "read"),
) -> Response:
    """The assistant's A2A agent card; 400 with a JSON-RPC error for an assistant it cannot describe."""
    await authorize({"assistant_id": assistant_id})
    return await _card(request, authorize, assistant_id)


@router.get("/.well-known/agent-card.json")
async def default_agent_card(
    request: Request,
    assistant_id: str | None = Query(
        None, description="The assistant to describe; DEFAULT_A2A_ASSISTANT_ID when omitted."
    ),
    authorize: Authorization = Authorize("assistants", "read"),
) -> Response:
    """The agent card of the assistant the query names, or of the deployment's default one."""
    assistant_id = assistant_id or settings.a2a.DEFAULT_A2A_ASSISTANT_ID
    await authorize({"assistant_id": assistant_id})
    if not assistant_id:
        message = "Missing required query parameter: assistant_id"
        return JSONResponse({"error": RpcError(INVALID_PARAMS, message).error}, status_code=400)
    return await _card(request, authorize, assistant_id)
