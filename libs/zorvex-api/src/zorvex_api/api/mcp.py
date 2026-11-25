"""The MCP endpoint: stateless Streamable HTTP at ``/mcp``, as the Agent Server serves it.

POST carries a JSON-RPC message and is answered with JSON. There are no
sessions, so GET and DELETE answer 405. A browser page may reach it only from
an origin the CORS settings name, or from a loopback one: a wildcard opens no
origin here. The route only authenticates; each MCP operation authorizes as its
REST counterpart (services/mcp_server.py).
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from zorvex_api.config import load_config
from zorvex_api.constants import LOOPBACK_HOSTS
from zorvex_api.core.auth_deps import auth_dependency, require_auth
from zorvex_api.models import User
from zorvex_api.services.mcp_server import mcp_transport

router = APIRouter(tags=["MCP"], dependencies=auth_dependency)


@cache
def _allowed_origins() -> tuple[frozenset[str], re.Pattern[str] | None]:
    cors = load_config().http.cors
    origins = frozenset(origin for origin in cors.allow_origins or () if origin != "*")
    pattern = re.compile(cors.allow_origin_regex) if cors.allow_origin_regex else None
    return origins, pattern


def origin_allowed(origin: str | None) -> bool:
    """Whether a request from *origin* may call the endpoint.

    A request without an Origin is not a browser's. A browser's must come from
    an origin ``http.cors`` names (by list or regex), or from a bare loopback
    origin, such as a local inspector's.
    """
    if origin is None:
        return True
    origins, pattern = _allowed_origins()
    if origin in origins or (pattern is not None and pattern.fullmatch(origin)):
        return True
    try:
        parts = urlsplit(origin)
        parts.port  # noqa: B018 - raises on a malformed port
    except ValueError:
        return False
    bare = not (parts.username or parts.password or parts.path or parts.query or parts.fragment)
    return bare and parts.scheme in ("http", "https") and parts.hostname in LOOPBACK_HOSTS


class _TransportResponse(Response):
    """Hands the live ASGI channels to the MCP session manager.

    A Starlette response is itself an ASGI app, which is the only place a route
    handler can reach ``send``. FastAPI has already drained ``receive`` by the
    time we get here, so the body is replayed rather than re-read.
    """

    def __init__(self, user: User, body: bytes) -> None:
        super().__init__()
        self.user = user
        self.body_bytes = body

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": self.body_bytes, "more_body": False}
            return await receive()

        async with mcp_transport.bound_to(self.user):
            await mcp_transport.handle(scope, replay, send)


# One operation per method, under the Agent Server's operation ids.
@router.post("/mcp", response_model=None, operation_id="post_mcp")
@router.get("/mcp", response_model=None, operation_id="get_mcp")
@router.delete("/mcp", response_model=None, operation_id="delete_mcp")
@router.api_route("/mcp/", methods=["GET", "POST", "DELETE"], response_model=None, include_in_schema=False)
async def mcp_endpoint(request: Request, user: User = Depends(require_auth)) -> Response:
    """Send one JSON-RPC 2.0 message to the MCP server; the answer is JSON.

    Tools are the assistants the caller can see. A call whose run pauses for
    human input answers ``input_required`` to a 2026-07-28 client that declares
    elicitation, which calls again with the answers and the ``requestState``;
    other clients get the pause reported. ``GET`` and ``DELETE`` answer 405: a
    stateless server has no stream to open and no session to end.
    """
    if not origin_allowed(request.headers.get("origin")):
        return JSONResponse({"error": "Origin is not allowed"}, status_code=403)
    if request.method != "POST":
        return Response(status_code=405, headers={"Allow": "POST"})
    return _TransportResponse(user, await request.body())
