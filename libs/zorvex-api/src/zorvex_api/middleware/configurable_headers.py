"""Records the request's configurable headers (utils/headers.py) for the graphs it builds."""

from starlette.types import ASGIApp, Receive, Scope, Send

from zorvex_api.utils.headers import configurable_headers, request_headers


class ConfigurableHeadersMiddleware:
    """Pure ASGI, so the request's handler runs in this task and sees what it set."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        found = configurable_headers(
            (name.decode("latin-1"), value.decode("latin-1")) for name, value in scope["headers"]
        )
        token = request_headers.set(found)
        try:
            await self.app(scope, receive, send)
        finally:
            request_headers.reset(token)
