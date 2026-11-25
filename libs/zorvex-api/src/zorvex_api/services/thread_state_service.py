"""Thread state: LangGraph snapshots in the Agent Protocol ThreadState shape, and the values a thread row keeps."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import structlog
from langgraph.types import StateSnapshot
from sqlalchemy import case, exists, literal, select, update

from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.models.auth import User
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES
from zorvex_api.models.threads import ThreadCheckpoint, ThreadState
from zorvex_api.services import thread_events
from zorvex_api.services.langgraph_service import get_langgraph_service

logger = structlog.getLogger(__name__)


def checkpoint(config: Mapping[str, Any] | None, thread_id: str) -> ThreadCheckpoint:
    """The checkpoint a snapshot's (or an update's) config points at."""
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
        "result": to_jsonable(task.result),
    }


def snapshot_to_thread_state(snapshot: StateSnapshot, thread_id: str) -> ThreadState:
    return ThreadState(
        values=snapshot.values if isinstance(snapshot.values, dict) else {},
        next=list(snapshot.next),
        tasks=[_task(task, thread_id) for task in snapshot.tasks],
        interrupts=to_jsonable(snapshot.interrupts),
        metadata=dict(snapshot.metadata or {}),
        created_at=datetime.fromisoformat(snapshot.created_at) if snapshot.created_at else None,
        checkpoint=checkpoint(snapshot.config, thread_id),
        parent_checkpoint=checkpoint(snapshot.parent_config, thread_id) if snapshot.parent_config else None,
    )


def empty_state(thread_id: str) -> ThreadState:
    """The state of a thread no run has bound to a graph yet."""
    return ThreadState(values={}, checkpoint=checkpoint(None, thread_id))


async def read_thread_values(thread: ThreadORM, user: User) -> tuple[dict[str, Any], dict[str, Any]]:
    """The thread's current values and pending interrupts (by task id), or empties when no graph is bound.

    A thread whose graph zorvex.json no longer declares has no readable state,
    which reads as empty rather than failing every route that lists it. So
    does one whose pending writes its graph cannot apply, such as the input of
    a run that failed on a message it could not read: the Agent Server shows
    such a thread from what it stored, where Zorvex reads the state again.
    """
    service = get_langgraph_service()
    if thread.metadata_dict.get("graph_id") not in service.list_graphs():
        return {}, {}
    async with service.thread_graph(thread, user, access_context="threads.read") as bound:
        if bound is None:
            return {}, {}
        graph, config = bound
        try:
            snapshot = await graph.aget_state(config)
        except ValueError:
            logger.warning("Thread state cannot be read; its values read as empty", thread_id=thread.thread_id)
            return {}, {}
    return snapshot_values(snapshot)


def snapshot_values(snapshot: StateSnapshot) -> tuple[dict[str, Any], dict[str, Any]]:
    """A snapshot's values and its pending interrupts by task id, as JSON, as a thread shows them."""
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    interrupts = {task.id: to_jsonable(task.interrupts) for task in snapshot.tasks if task.interrupts}
    return to_jsonable(values), interrupts


async def record_state(thread_id: str, user: User, state: StateSnapshot) -> None:
    """Keep a state written outside a run as the thread's, and tell the thread's followers.

    The thread row takes its values and state time, and its status unless a run is on it.
    """
    values, interrupts = snapshot_values(state)
    now = datetime.now(UTC)
    active = exists(
        select(RunORM.run_id)
        .where(RunORM.thread_id == ThreadORM.thread_id, RunORM.status.in_(ACTIVE_RUN_STATUSES))
        .correlate(ThreadORM)
    )
    status = case((active, ThreadORM.status), else_=literal("interrupted" if state.next else "idle"))
    async with get_session_maker()() as session:
        await session.execute(
            update(ThreadORM)
            .where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
            .values(updated_at=now, state_updated_at=now, status=status, values=values, interrupts=interrupts)
        )
        await session.commit()
    await thread_events.publish(
        thread_id, "state_update", {"state": snapshot_to_thread_state(state, thread_id), "thread_id": thread_id}
    )
