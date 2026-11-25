"""Server-Sent Events for the run and thread streams, framed and sent by sse-starlette as the Agent Server's are."""

import json
import re
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, MutableMapping
from typing import Any

import structlog
from sse_starlette import EventSourceResponse, ServerSentEvent
from sse_starlette.sse import SendTimeoutError
from starlette.types import Receive, Scope, Send

from zorvex_api.core.serializers import to_jsonable
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

# The keepalive comment, ": heartbeat", as the Agent Server sends it; one event serves every tick.
_HEARTBEAT = ServerSentEvent(comment="heartbeat")
# A client that stops reading but keeps its connection would hold a send, and the stream, for good.
_SEND_TIMEOUT_SECS = 120

# Some LLMs stream tool_call_chunks.args with literal \uXXXX sequences instead of
# the characters, which json.dumps then double-escapes. They are decoded back in two
# passes: surrogate pairs first (a lone surrogate cannot be encoded to UTF-8), then
# the other non-ASCII code points; ASCII control escapes stay, keeping the JSON
# valid, and (?<!\\) skips an escaped backslash in the content.
_SURROGATE_PAIR_RE = re.compile(
    r"(?<!\\)\\\\u(D[89AB][0-9a-fA-F]{2})\\\\u(D[C-F][0-9a-fA-F]{2})",
    re.IGNORECASE,
)
_NONASCII_ESCAPE_RE = re.compile(r"(?<!\\)\\\\u([0-9a-fA-F]{4})")


def _decode_literal_unicode_escapes(data_str: str) -> str:
    """Decode double-escaped \\uXXXX sequences in an already-JSON-encoded string."""
    if "\\u" not in data_str:
        return data_str
    data_str = _SURROGATE_PAIR_RE.sub(
        lambda m: chr(((int(m.group(1), 16) - 0xD800) << 10) + (int(m.group(2), 16) - 0xDC00) + 0x10000),
        data_str,
    )
    return _NONASCII_ESCAPE_RE.sub(
        lambda m: chr(cp) if (cp := int(m.group(1), 16)) >= 0x80 and not (0xD800 <= cp <= 0xDFFF) else m.group(0),
        data_str,
    )


# SSE ends a line at CR and LF only, but readers that split lines as str.splitlines does (httpx's
# aiter_lines, and so httpx-sse) also break at these, which JSON leaves raw: they go escaped.
_LINE_BREAKS = str.maketrans({"\u0085": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"})


def event(name: str, data: Any, id: str | None = None) -> ServerSentEvent:
    """An SSE event: *data* as compact JSON (None is ``null``), with the id a client resumes after."""
    text = json.dumps(data, default=to_jsonable, separators=(",", ":"), ensure_ascii=False)
    return ServerSentEvent(data=_decode_literal_unicode_escapes(text).translate(_LINE_BREAKS), event=name, id=id)


class _Stream(EventSourceResponse):
    """An SSE response that ends quietly when its client stops reading."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except* SendTimeoutError:
            logger.warning("Closed a stream whose client stopped reading")


def response(
    body: AsyncIterable[ServerSentEvent],
    *,
    headers: Mapping[str, str] | None = None,
    close_handler: Callable[[MutableMapping[str, Any]], Awaitable[None]] | None = None,
) -> EventSourceResponse:
    """The SSE response of a stream of events, with the heartbeat and send timeout every stream has.

    sse-starlette sets the stream's own headers (no-store caching, keep-alive, no
    proxy buffering) and closes *body* when the response stops early; CORS
    headers come from the CORS middleware, as for every response.
    """
    return _Stream(
        body,
        ping=settings.app.sse_ping_interval_secs,
        ping_message_factory=lambda: _HEARTBEAT,
        send_timeout=_SEND_TIMEOUT_SECS,
        client_close_handler_callable=close_handler,
        headers=headers,
    )
