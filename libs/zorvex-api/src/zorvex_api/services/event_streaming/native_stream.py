"""Drive a run through langgraph's native v3 stream for Agent Protocol v2.

``astream_events(version="v3")`` emits events already in protocol shape —
``{type:"event", method, params:{data, namespace, ...}}`` — including the
content-block message lifecycle, tool-call blocks, token usage, and
``params.interrupts``. We forward those into the broker verbatim; the
session restamps seq/event_id and filters by channel. This is the v2
counterpart to the legacy ``stream_graph_events`` (v1) producer and does
not touch it.

The only reshaping here: ``messages`` events arrive as ``[event, metadata]``
tuples, so we unwrap element 0 into ``params.data``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig


@lru_cache(maxsize=1)
def _extra_transformers() -> list[Any]:
    """Transformers enabling the non-default v3 channels (updates/custom/checkpoints/tasks/tools).

    The default v3 mux ships only values/messages/lifecycle/subgraph; the tool
    call transformer asks for the ``tools`` mode and passes its events through.
    """
    # Optional-dep guard: a too-old langgraph should fail the capability probe,
    # but if it's bypassed surface a clean error, not a raw ImportError mid-run.
    try:
        from langgraph.prebuilt import ToolCallTransformer
        from langgraph.stream.transformers import (
            CheckpointsTransformer,
            CustomTransformer,
            TasksTransformer,
            UpdatesTransformer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "langgraph.stream.transformers is unavailable; langgraph is older than this package requires."
        ) from exc

    return [UpdatesTransformer, CustomTransformer, CheckpointsTransformer, TasksTransformer, ToolCallTransformer]


def unwrap_message_event(data: Any) -> dict[str, Any] | None:
    """Return the content-block event dict from a v3 ``messages`` payload.

    v3 ``messages`` data is ``[event_dict, metadata]`` where the event dict
    carries the ``event`` discriminator (``message-start`` etc.). Accepts a
    bare event dict too. Returns ``None`` for anything not v2-message-shaped.
    """
    if isinstance(data, dict) and isinstance(data.get("event"), str):
        return data
    if isinstance(data, (list, tuple)) and len(data) == 2:
        head = data[0]
        if isinstance(head, dict) and isinstance(head.get("event"), str):
            return head
    return None


async def stream_native_v3_events(
    graph: Any, input_data: Any, config: RunnableConfig, options: Mapping[str, Any]
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield ``(method, protocol_event)`` pairs from a native v3 run.

    Each pair goes into the broker as one raw event; the session re-envelopes
    it. ``messages`` payloads are unwrapped to their event dict; a message
    event we can't reconstruct is dropped rather than forwarded malformed.
    """
    # v3 forwards extra arguments to astream.
    run_stream = await graph.astream_events(
        input_data,
        config,
        version="v3",
        transformers=_extra_transformers(),
        **options,
    )
    async with run_stream as stream:
        async for event in stream:
            if not isinstance(event, dict) or event.get("type") != "event":
                continue
            method = event.get("method")
            if not isinstance(method, str):
                continue

            if method == "messages":
                unwrapped = unwrap_message_event(event.get("params", {}).get("data"))
                if unwrapped is None:
                    continue
                event = {**event, "params": {**event["params"], "data": unwrapped}}

            yield method, event
