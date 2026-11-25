"""``http.middleware_order: "auth_first"``: a request to an authenticated route is authenticated before the
custom app's own middleware sees it, as the Agent Server orders them. The route's dependency then takes the
user found. By default (``middleware_first``) the custom middleware runs first and authentication with the route.
"""

from collections.abc import Awaitable, Callable

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from zorvex_api.core.auth import require_auth


class AuthFirstMiddleware:
    """Authenticates the requests *guarded* picks; a failure answers as *refuse* renders it."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        guarded: Callable[[Scope], bool],
        refuse: Callable[[Request, HTTPException], Awaitable[Response]],
    ) -> None:
        self.app = app
        self.guarded = guarded
        self.refuse = refuse

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.guarded(scope):
            request = Request(scope, receive)
            try:
                await require_auth(request)
            except HTTPException as exc:
                await (await self.refuse(request, exc))(scope, receive, send)
                return
        await self.app(scope, receive, send)
