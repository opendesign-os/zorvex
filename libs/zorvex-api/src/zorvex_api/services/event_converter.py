"""A run's broker events, ``(mode, payload)``, as the SSE events its streams send."""

from typing import Any

from sse_starlette import ServerSentEvent

from zorvex_api.core import sse

# Sent whatever modes a join asks for: the run's start, its end, its error, and
# the feedback URLs its creator asked for with feedback_keys.
_CONTROL_MODES = frozenset({"metadata", "end", "error", "feedback"})


def to_sse(event_id: str, event: tuple[str, Any]) -> ServerSentEvent:
    """The SSE event for a broker event."""
    mode, payload = event
    return sse.event(mode, payload, event_id)


def matches(event: tuple[str, Any], modes: frozenset[str] | None) -> bool:
    """Whether a join asking for *modes* (None: all) carries *event*, as the Agent Server filters joins.

    A subgraph's ``mode|namespace`` event goes with its mode; ``messages`` and
    ``messages-tuple`` take every ``messages*`` event.
    """
    mode = event[0]
    if modes is None or mode in _CONTROL_MODES:
        return True
    base = mode.partition("|")[0]
    return base in modes or (base.startswith("messages") and not modes.isdisjoint({"messages", "messages-tuple"}))
