"""Wire builders for the Agent Protocol v2 envelope, typed by the protocol's own package, ``langchain_protocol``.

Its types are plain ``TypedDict``s with no serialization behaviour: the wire
form is a plain dict serialized with ``json.dumps``. These helpers build those
dicts, so field names live in one place and stay exact (``event_id``, ``seq``,
``applied_through_seq``).
"""

from __future__ import annotations

import time
from typing import Any

from langchain_protocol import CommandResponse, ErrorCode, ErrorResponse

# What a command answers.
Response = CommandResponse | ErrorResponse


def build_event(
    method: str,
    data: dict[str, Any],
    *,
    namespace: list[str] | None = None,
    seq: int,
    event_id: str | None = None,
    node: str | None = None,
) -> dict[str, Any]:
    """Build a server-push event envelope; ``params.node`` names the node of an updates or tools event.

    ``params`` wraps the per-channel payload as ``{data, namespace,
    timestamp}`` — the shape the LangGraph SDK reads (``params.data`` is the
    payload, ``params.namespace`` the subgraph path, ``[]`` at the root,
    ``params.timestamp`` ms-epoch server time).
    """
    params: dict[str, Any] = {"data": data, "namespace": namespace or [], "timestamp": time.time_ns() // 1_000_000}
    if node is not None:
        params["node"] = node
    event: dict[str, Any] = {"type": "event", "seq": seq, "method": method, "params": params}
    if event_id is not None:
        event["event_id"] = event_id
    return event


def build_success(command_id: int, result: Any, *, applied_through_seq: int | None = None) -> CommandResponse:
    """Build a success command response."""
    response = CommandResponse(type="success", id=command_id, result=result)
    if applied_through_seq is not None:
        response["meta"] = {"applied_through_seq": applied_through_seq}
    return response


def build_error(command_id: int | None, error: ErrorCode, message: str) -> ErrorResponse:
    """Build an error command response."""
    return ErrorResponse(type="error", id=command_id, error=error, message=message)
