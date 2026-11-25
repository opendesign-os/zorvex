"""Reads a ``text/plain`` request body as JSON.

Some clients (notably LangGraph Studio) send valid JSON labelled
``text/plain``, which FastAPI would reject with a 422. The header is rewritten
to ``application/json`` for requests that carry a body; the body itself is
never buffered, so large and streamed payloads pass untouched.
"""

from starlette.types import ASGIApp, Receive, Scope, Send

_METHODS_WITH_BODY = frozenset({"POST", "PUT", "PATCH"})


class ContentTypeFixMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in _METHODS_WITH_BODY:
            scope["headers"] = [
                (name, b"application/json" if name == b"content-type" and _is_plain_text(value) else value)
                for name, value in scope["headers"]
            ]
        await self.app(scope, receive, send)


def _is_plain_text(content_type: bytes) -> bool:
    return content_type.split(b";", 1)[0].strip().lower() == b"text/plain"
