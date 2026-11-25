"""A run's graph streamed in the Agent Server's run stream modes, as ``(mode, payload)`` events."""

from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    message_chunk_to_message,
)
from langchain_core.runnables import RunnableConfig


async def stream_graph_events(
    graph: Any,
    input_data: Any,
    config: RunnableConfig,
    *,
    stream_mode: list[str],
    subgraphs: bool,
    options: Mapping[str, Any],
) -> AsyncIterator[tuple[str, Any]]:
    """The run's events in the modes it asked for, after a ``metadata`` event.

    ``messages`` accumulates into ``messages/metadata``, ``messages/partial`` and
    ``messages/complete``; ``messages-tuple`` sends LangGraph's message tuples.
    Without ``updates``, an interrupt's update still comes, as ``values``.
    ``events`` adds astream_events' events. A subgraph's events carry
    ``|namespace`` after their mode. *options* go to astream as they are.
    """
    run_id = str(config["configurable"]["run_id"])
    requested = set(stream_mode)
    modes = requested - {"events", "messages-tuple"}
    if "messages-tuple" in requested:
        modes.add("messages")
    interrupts_only = "updates" not in modes
    modes.add("updates")
    seen: dict[Any, Any] = {}

    def emit(chunk: Any) -> list[tuple[str, Any]]:
        if subgraphs and len(chunk) == 3:
            namespace, mode, data = chunk
        else:
            namespace, (mode, data) = (), chunk
        suffix = "".join(f"|{part}" for part in namespace)
        if mode == "debug" and data.get("type") == "checkpoint":
            data["payload"] = _debug(data.get("payload"))
        if mode == "messages":
            return _messages(data, suffix, tuples="messages-tuple" in requested, seen=seen)
        if mode in requested:
            return [(mode + suffix, data)]
        if mode == "updates" and interrupts_only and isinstance(data, dict) and data.get("__interrupt__"):
            return [("values" + suffix, data)]
        return []

    yield "metadata", {"run_id": run_id, "attempt": config.get("metadata", {}).get("run_attempt", 1)}

    if "events" in requested:
        events = graph.astream_events(
            input_data, config, version="v2", stream_mode=list(modes), subgraphs=subgraphs, **options
        )
        async with aclosing(events) as stream:
            async for event in stream:
                if "langsmith:hidden" in (event.get("tags") or ()):
                    continue
                if event.get("event") == "on_chain_stream" and event.get("run_id") == run_id:
                    chunk = (event.get("data") or {}).get("chunk")
                    if chunk is None:
                        continue
                    for item in emit(chunk):
                        yield item
                yield "events", event
        return

    async with aclosing(
        graph.astream(input_data, config, stream_mode=list(modes), subgraphs=subgraphs, **options)
    ) as stream:
        async for chunk in stream:
            for item in emit(chunk):
                yield item


def _messages(chunk: Any, suffix: str, *, tuples: bool, seen: dict[Any, Any]) -> list[tuple[str, Any]]:
    """A ``messages`` chunk as the requested mode sends it."""
    if tuples:
        if isinstance(chunk, (tuple, list)) and len(chunk) == 2:
            message, metadata = chunk
            chunk = (_as_chunk(message), metadata)
        return [("messages" + suffix, chunk)]
    message, metadata = chunk
    events: list[tuple[str, Any]] = []
    if message.id in seen:
        seen[message.id] += message
    else:
        seen[message.id] = message
        events.append(("messages/metadata", {message.id: {"metadata": metadata}}))
    whole = message_chunk_to_message(seen[message.id])
    events.append(("messages/partial" if isinstance(message, BaseMessageChunk) else "messages/complete", [whole]))
    return events


def _as_chunk(message: BaseMessage) -> BaseMessage:
    """An AI or tool message as its chunk type, so ``messages-tuple`` always sends chunks."""
    if isinstance(message, (AIMessageChunk, ToolMessageChunk)):
        return message
    if isinstance(message, AIMessage):
        return AIMessageChunk(**message.model_dump(exclude={"type"}))
    if isinstance(message, ToolMessage):
        return ToolMessageChunk(**message.model_dump(exclude={"type"}))
    return message


def _debug(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """A debug checkpoint as the Agent Server sends it: with the checkpoint and parent checkpoint
    its configs point at, and tasks that name their subgraph's checkpoint instead of carrying its state."""
    if not payload:
        return None
    tasks = []
    for task in payload["tasks"]:
        state = task.get("state")
        configurable = state.get("configurable") if isinstance(state, dict) else None
        if configurable:
            task = {key: value for key, value in task.items() if key != "state"} | {"checkpoint": configurable}
        tasks.append(task)
    return {
        **payload,
        "checkpoint": _checkpoint(payload.get("config")),
        "parent_checkpoint": _checkpoint(payload.get("parent_config")),
        "tasks": tasks,
    }


def _checkpoint(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The checkpoint a config points at; None without its thread and checkpoint ids."""
    configurable = (config or {}).get("configurable") or {}
    if not configurable.get("thread_id") or not configurable.get("checkpoint_id"):
        return None
    checkpoint = {"checkpoint_id": configurable["checkpoint_id"], "thread_id": configurable["thread_id"]}
    if "checkpoint_ns" in configurable:
        checkpoint["checkpoint_ns"] = configurable["checkpoint_ns"] or ""
    if "checkpoint_map" in configurable:
        checkpoint["checkpoint_map"] = configurable["checkpoint_map"]
    return checkpoint
