"""MCP for the deployment's assistants, as the Agent Server serves it.

Each assistant the caller can see is a tool, named after the assistant; calling
it runs the assistant and answers with its final values, or, when the run
pauses for human input, asks the client for it as mcp_calls.py describes. Every
step authorizes as its REST counterpart: listing as ``assistants.search``, each
tool's schema as ``assistants.read``, and a call as ``assistants.search`` (to
find the assistant), then as the thread and run operations it makes. The
transport is stateless Streamable HTTP answering JSON (api/mcp.py); a handler
finds its caller as the endpoint authenticated them, on the request's own ASGI
scope.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import re
from collections.abc import Generator
from typing import Any

import mcp_types
import structlog
from fastapi import HTTPException
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import MCPError
from mcp.shared.inbound import X_MCP_HEADER_KEY

from zorvex_api import __version__
from zorvex_api.config import load_config
from zorvex_api.core.auth import require_auth
from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models import Assistant, AssistantSearchRequest, User
from zorvex_api.services.assistant_service import AssistantService
from zorvex_api.services.langgraph_service import get_langgraph_service
from zorvex_api.services.mcp_calls import CallToolParams, ToolResult, call_tool

logger = structlog.getLogger(__name__)

_PAGE_SIZE = 100
# A call finds its tool among this many assistants, the Agent Server's bound.
_MAX_ASSISTANTS = 1000
# The largest request body: tool arguments may carry files.
_MAX_BODY_BYTES = 300 * 1024 * 1024

_NAME_MAX_LENGTH = 128
_NAME_INVALID = re.compile(r"[^A-Za-z0-9_.-]")
_WHITESPACE = re.compile(r"\s+")
_CURSOR = re.compile(r"\d{1,9}")

_INTERNAL_ERROR = "Internal server error"
# A tool runs an agent: it may change state outside the server, and calling it twice is two runs.
_ANNOTATIONS = mcp_types.ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True
)

# The LangChain message classes a graph's input schema names when its state holds messages.
_MESSAGE_DEFS = frozenset(
    f"{kind}{suffix}"
    for kind in ("AI", "Human", "Chat", "System", "Function", "Tool")
    for suffix in ("Message", "MessageChunk")
)
_LANGCHAIN_DEFS = _MESSAGE_DEFS | {
    "ToolCall",
    "ToolCallChunk",
    "InvalidToolCall",
    "UsageMetadata",
    "InputTokenDetails",
    "OutputTokenDetails",
}


def tool_name(assistant_name: str) -> str:
    """An assistant's name as an MCP tool name: whitespace runs as ``_``, other invalid characters dropped."""
    name = _WHITESPACE.sub("_", assistant_name.strip())
    return _NAME_INVALID.sub("", name)[:_NAME_MAX_LENGTH]


def _input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """An assistant's input schema as a tool's: a message list becomes the user's message as a string.

    An LLM fills tool arguments, and LangChain's message classes are no shape
    for it to fill; a graph accepts the plain string in their place. The
    ``x-mcp-header`` keys go too: no argument is taken from a header.
    """
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs")
    if isinstance(defs, dict) and _MESSAGE_DEFS & set(defs):
        for name in _LANGCHAIN_DEFS:
            defs.pop(name, None)
        if not defs:
            del schema["$defs"]
        properties = schema.get("properties") or {}
        for key, prop in properties.items():
            if _is_message_list(prop):
                properties[key] = {
                    "type": "string",
                    "title": prop.get("title", key),
                    "description": "The user message to send to the agent.",
                }
    _drop_header_keys(schema)
    schema.setdefault("type", "object")
    return schema


def _is_message_list(prop: Any) -> bool:
    if not isinstance(prop, dict) or prop.get("type") != "array":
        return False
    items = prop.get("items") or {}
    variants = items.get("oneOf") or items.get("anyOf") or []
    return any(
        isinstance(variant, dict) and variant.get("$ref", "").removeprefix("#/$defs/") in _MESSAGE_DEFS
        for variant in variants
    )


def _drop_header_keys(schema: Any) -> None:
    """Remove ``x-mcp-header`` from a schema and every schema nested in it."""
    if isinstance(schema, dict):
        schema.pop(X_MCP_HEADER_KEY, None)
        for value in schema.values():
            _drop_header_keys(value)
    elif isinstance(schema, list):
        for item in schema:
            _drop_header_keys(item)


@contextlib.contextmanager
def _mcp_errors() -> Generator[None]:
    """Answer a request error (4xx) as invalid params with its detail, and anything else as an internal error."""
    try:
        yield
    except MCPError:
        raise
    except HTTPException as exc:
        if exc.status_code < 500:
            raise MCPError(mcp_types.INVALID_PARAMS, str(exc.detail)) from exc
        logger.exception("MCP request failed")
        raise MCPError(mcp_types.INTERNAL_ERROR, _INTERNAL_ERROR) from exc
    except Exception as exc:
        logger.exception("MCP request failed")
        raise MCPError(mcp_types.INTERNAL_ERROR, _INTERNAL_ERROR) from exc


async def _caller(context: ServerRequestContext[Any]) -> User:
    """The request's caller, as the endpoint authenticated them: their ASGI scope keeps the user."""
    if context.request is None:
        raise RuntimeError("MCP request served without its HTTP request")
    return await require_auth(context.request)


async def _assistants(user: User, *, offset: int, limit: int) -> list[Assistant]:
    """The assistants the caller sees, as ``POST /assistants/search`` lists them."""
    async with get_session_maker()() as session:
        service = AssistantService(session, user, get_langgraph_service())
        assistants, _ = await service.search_assistants(
            Authorization(user, "assistants", "search"), AssistantSearchRequest(limit=limit, offset=offset)
        )
        return assistants


async def _tool(user: User, assistant: Assistant, name: str) -> mcp_types.Tool | None:
    """The assistant as a tool, or None when its input schema cannot be had."""
    try:
        async with get_session_maker()() as session:
            service = AssistantService(session, user, get_langgraph_service())
            schemas = await service.get_assistant_schemas(
                Authorization(user, "assistants", "read"), assistant.assistant_id
            )
    except Exception:
        logger.warning("Assistant left out of the MCP tools: no input schema", assistant_id=assistant.assistant_id)
        return None
    input_schema = schemas.get("input_schema")
    if not isinstance(input_schema, dict):
        logger.warning("Assistant left out of the MCP tools: no input schema", assistant_id=assistant.assistant_id)
        return None
    return mcp_types.Tool(
        name=name,
        title=assistant.name,
        description=assistant.description or f"Tool based on the {assistant.name} assistant",
        input_schema=_input_schema(input_schema),
        annotations=_ANNOTATIONS,
    )


async def _list_tools(
    context: ServerRequestContext[Any], params: mcp_types.PaginatedRequestParams | None
) -> mcp_types.ListToolsResult:
    cursor = params.cursor if params is not None else None
    if cursor is not None and not _CURSOR.fullmatch(cursor):
        raise MCPError(mcp_types.INVALID_PARAMS, "Invalid cursor")
    offset = int(cursor or 0)
    with _mcp_errors():
        user = await _caller(context)
        assistants = await _assistants(user, offset=offset, limit=_PAGE_SIZE)
        named: dict[str, Assistant] = {}
        for assistant in assistants:
            name = tool_name(assistant.name)
            if name in named:
                logger.warning("Duplicate assistant name in the MCP tools", name=assistant.name, tool=name)
            else:
                named[name] = assistant
        tools = await asyncio.gather(*(_tool(user, assistant, name) for name, assistant in named.items()))
        return mcp_types.ListToolsResult(
            tools=[tool for tool in tools if tool is not None],
            next_cursor=str(offset + _PAGE_SIZE) if len(assistants) == _PAGE_SIZE else None,
        )


async def _call_tool(context: ServerRequestContext[Any], params: CallToolParams) -> ToolResult:
    with _mcp_errors():
        user = await _caller(context)
        return await call_tool(context, user, await _assistant_named(user, params.name), params)


async def _assistant_named(user: User, name: str) -> str:
    """The id of the one visible assistant whose tool name is *name*; invalid params otherwise."""
    matches = [
        assistant.assistant_id
        for assistant in await _assistants(user, offset=0, limit=_MAX_ASSISTANTS)
        if tool_name(assistant.name) == name
    ]
    if not matches:
        raise MCPError(mcp_types.INVALID_PARAMS, f"Unknown tool: {name}")
    if len(matches) > 1:
        raise MCPError(mcp_types.INVALID_PARAMS, "Multiple tools found with the same name.")
    return matches[0]


def build_server() -> Server[Any]:
    server: Server[Any] = Server(
        load_config().name,
        version=__version__,
        instructions="Each tool runs one assistant of this deployment and returns its final values.",
        on_list_tools=_list_tools,
        # Tools take no argument from an Mcp-Param header (their schemas drop x-mcp-header), so there is
        # nothing to check one against; without this, a call would first list, and build, every tool.
        get_tool_input_schema=lambda _name: None,
    )
    server.add_request_handler("tools/call", CallToolParams, _call_tool)
    return server


class McpTransport:
    """The stateless Streamable HTTP session manager the endpoint hands its requests to."""

    def __init__(self) -> None:
        self._manager: StreamableHTTPSessionManager | None = None
        self._exit: contextlib.AsyncExitStack | None = None

    async def start(self) -> None:
        # Stateless and JSON: the documented contract answers each request with JSON, and keeps no session.
        manager = StreamableHTTPSessionManager(
            app=build_server(), stateless=True, json_response=True, max_request_body_size=_MAX_BODY_BYTES
        )
        stack = contextlib.AsyncExitStack()
        await stack.enter_async_context(manager.run())
        self._manager, self._exit = manager, stack
        logger.info("MCP transport started at /mcp")

    async def stop(self) -> None:
        if self._exit is not None:
            await self._exit.aclose()
        self._manager, self._exit = None, None

    async def handle(self, scope: Any, receive: Any, send: Any) -> None:
        if self._manager is None:
            raise RuntimeError("MCP transport is not running")
        await self._manager.handle_request(scope, receive, send)


mcp_transport = McpTransport()
