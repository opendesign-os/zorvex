"""Access log: one structured line per HTTP request, tagged with its correlation id.

Uvicorn's own access log is turned down (see setup_logging), so this is the
only one. An unhandled error is not logged here: uvicorn logs its traceback,
with the same request id. The request and response headers ``http.logging_headers``
lets through go along as ``req_header`` and ``res_header``, as on the Agent Server.
"""

import logging
import time
from typing import Any

import structlog
from asgi_correlation_id import correlation_id
from starlette.datastructures import URL
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from zorvex_api.settings import settings
from zorvex_api.utils.headers import logged_headers

access_logger = structlog.stdlib.get_logger("app.access_logs")


class StructLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=correlation_id.get())
        response: dict[str, Any] = {"status": 500, "headers": []}  # what the client gets when the app raises first

        async def send_and_record(message: Message) -> None:
            if message["type"] == "http.response.start":
                response["status"] = message["status"]
                response["headers"] = message.get("headers") or []
            await send(message)

        start = time.perf_counter_ns()
        try:
            await self.app(scope, receive, send_and_record)
        finally:
            _log_access(scope, response, time.perf_counter_ns() - start)


def _log_access(scope: Scope, response: dict[str, Any], duration_ns: int) -> None:
    """Log the request at info, a 4xx at warning and a 5xx at error; LOG_EXCLUDE_PATHS skip successes."""
    path: str = scope["path"]
    status_code: int = response["status"]
    if status_code < 400 and any(path.startswith(prefix) for prefix in settings.app.log_exclude_paths):
        return
    client_host, client_port = scope.get("client") or ("unknown", 0)
    method, version = scope["method"], scope["http_version"]
    url = URL(scope=scope)
    level = logging.ERROR if status_code >= 500 else logging.WARNING if status_code >= 400 else logging.INFO
    headers: dict[str, Any] = {}
    if requested := logged_headers(_decoded(scope["headers"])):
        headers["req_header"] = requested
    if answered := logged_headers(_decoded(response["headers"])):
        headers["res_header"] = answered
    access_logger.log(
        level,
        f'{client_host}:{client_port} - "{method} {path} HTTP/{version}" {status_code}',
        http={
            "url": f"{url.path}?{url.query}" if url.query else url.path,
            "status_code": status_code,
            "method": method,
            "version": version,
        },
        network={"client": {"ip": client_host, "port": client_port}},
        duration=duration_ns,
        **headers,
    )


def _decoded(headers: list[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    """Raw ASGI header pairs as text, in the latin-1 HTTP headers are."""
    return [(name.decode("latin-1"), value.decode("latin-1")) for name, value in headers]
