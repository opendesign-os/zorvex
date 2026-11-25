"""Thread state: LangGraph snapshots in the Agent Protocol ThreadState shape, and a thread's current values."""

from collections.abc import Mapping
from typing import Any

import structlog
from langgraph.types import StateSnapshot

from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.models.auth import User
from zorvex_api.models.threads import ThreadCheckpoint, ThreadState
from zorvex_api.services.langgraph_service import create_thread_config, get_langgraph_service, thread_graph_config

logger = structlog.getLogger(__name__)


def _checkpoint(config: Mapping[str, Any] | None, thread_id: str) -> ThreadCheckpoint:
    configurable = (config or {}).get("configurable", {})
    return ThreadCheckpoint(
        checkpoint_id=configurable.get("checkpoint_id"),
        thread_id=thread_id,
        checkpoint_ns=configurable.get("checkpoint_ns", ""),
    )


def _task(task: Any, thread_id: str) -> dict[str, Any]:
    """One pending task. Its state is the subgraph's snapshot when subgraphs were requested, else its config."""
    state = task.state
    return {
        "id": task.id,
        "name": task.name,
        "error": str(task.error) if task.error else None,
        "interrupts": to_jsonable(task.interrupts),
        "checkpoint": state["configurable"] if isinstance(state, dict) else None,
        "state": snapshot_to_thread_state(state, thread_id) if isinstance(state, StateSnapshot) else None,
        "result": to_jsonable(getattr(task, "result", None)),
    }


def snapshot_to_thread_state(snapshot: StateSnapshot, thread_id: str) -> ThreadState:
    return ThreadState(
        values=snapshot.values if isinstance(snapshot.values, dict) else {},
        next=list(snapshot.next),
        tasks=[_task(task, thread_id) for task in snapshot.tasks],
        interrupts=to_jsonable(snapshot.interrupts),
        metadata=dict(snapshot.metadata or {}),
        created_at=snapshot.created_at,
        checkpoint=_checkpoint(snapshot.config, thread_id),
        parent_checkpoint=_checkpoint(snapshot.parent_config, thread_id) if snapshot.parent_config else None,
    )


async def read_thread_values(thread: ThreadORM, user: User) -> tuple[dict[str, Any], dict[str, Any]]:
    """The thread's current values and pending interrupts (by task id), or empties when no graph is bound.

    A thread whose graph zorvex.json no longer declares has no readable state,
    which reads as empty rather than failing every route that lists it. So
    does one whose pending writes its graph cannot apply, such as the input of
    a run that failed on a message it could not read: the Agent Server shows
    such a thread from what it stored, where Zorvex reads the state again.
    """
    graph_id = (thread.metadata_dict or {}).get("graph_id")
    if not graph_id or graph_id not in get_langgraph_service().list_graphs():
        return {}, {}
    config = create_thread_config(thread.thread_id, user)
    async with get_langgraph_service().get_graph(
        graph_id, thread_graph_config(thread.config, config), access_context="threads.read", user=user
    ) as agent:
        try:
            snapshot = await agent.aget_state(config)
        except ValueError:
            logger.warning("Thread state cannot be read; its values read as empty", thread_id=thread.thread_id)
            return {}, {}
    return snapshot_values(snapshot)


def snapshot_values(snapshot: StateSnapshot) -> tuple[dict[str, Any], dict[str, Any]]:
    """A snapshot's values and its pending interrupts by task id, as JSON, as a thread shows them."""
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    interrupts = {
        task.id: to_jsonable(task.interrupts) for task in (snapshot.tasks or []) if getattr(task, "interrupts", None)
    }
    return to_jsonable(values), interrupts
