"""A run driven through LangGraph's native v3 stream, for Agent Protocol v2.

``astream_events(version="v3")`` emits events already in protocol shape,
``{type: "event", method, params: {data, namespace, ...}}``, content-block
message lifecycle, tool-call blocks, token usage and ``params.interrupts``
included. They go into the broker as they are; the session restamps their
seq and event id and filters them by channel. This is the v2 counterpart of
graph_streaming.py, the producer of the SDK's run streams.

The one reshaping: a ``messages`` event's data is ``[event, metadata]``, and
its event becomes ``params.data``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolCallTransformer
from langgraph.stream.transformers import (
    CheckpointsTransformer,
    CustomTransformer,
    TasksTransformer,
    UpdatesTransformer,
)

# The channels the default v3 mux leaves out (it has values, messages, lifecycle and subgraph):
# updates, custom, checkpoints, tasks, and tools, whose transformer asks for the tools mode.
_TRANSFORMERS = [UpdatesTransformer, CustomTransformer, CheckpointsTransformer, TasksTransformer, ToolCallTransformer]


def _message_event(data: Any) -> dict[str, Any] | None:
    """The content-block event (``message-start`` and the rest) of a v3 ``messages`` payload, or None."""
    if isinstance(data, (list, tuple)) and len(data) == 2:
        head = data[0]
        if isinstance(head, dict) and isinstance(head.get("event"), str):
            return head
    return None


async def stream_native_v3_events(
    graph: Any, input_data: Any, config: RunnableConfig, options: Mapping[str, Any]
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """``(method, protocol_event)`` pairs from a native v3 run, each one broker event.

    A message event that cannot be read is dropped rather than forwarded malformed.
    """
    # v3 passes the remaining arguments on to astream.
    run_stream = await graph.astream_events(input_data, config, version="v3", transformers=_TRANSFORMERS, **options)
    async with run_stream as stream:
        async for event in stream:
            if not isinstance(event, dict) or event.get("type") != "event":
                continue
            method = event.get("method")
            if not isinstance(method, str):
                continue
            if method == "messages":
                message = _message_event(event.get("params", {}).get("data"))
                if message is None:
                    continue
                event = {**event, "params": {**event["params"], "data": message}}
            yield method, event
