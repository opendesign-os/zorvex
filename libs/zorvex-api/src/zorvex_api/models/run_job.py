"""A run's job: who runs which graph, and the run's kwargs as the Agent Server keeps them.

The kwargs are the Agent Server's, key for key, so a new run option is one
field here; server markers (the v2 event stream, a LangSmith project) ride in
``config.configurable`` as they do there. The job is stored in the run row's
``execution_params`` and travels to workers whole.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zorvex_api.models.auth import User
from zorvex_api.models.enums import Durability, MultitaskStrategy, StreamMode

if TYPE_CHECKING:
    from zorvex_api.core.orm import Run as RunORM


class RunIdentity(BaseModel):
    """Locates a run within the system."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    thread_id: str
    graph_id: str
    # Run-row columns the completion webhook reports; carried here because stateless
    # cleanup can delete the row before the webhook is built.
    assistant_id: str
    created_at: datetime


class RunKwargs(BaseModel):
    """How a run executes: its ``kwargs``, as the Agent Server records and answers them."""

    model_config = ConfigDict(frozen=True)

    input: Any = None
    command: dict[str, Any] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    stream_mode: StreamMode | list[StreamMode] | None = None
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    webhook: str | None = None
    feedback_keys: list[str] | None = None
    temporary: bool = False
    subgraphs: bool = False
    resumable: bool = False
    checkpoint_during: bool = True
    # Resolved at creation (the run's, else the server's); None keeps LangGraph's default.
    durability: Durability | None = None


class RunJob(BaseModel):
    """A run as its executor needs it."""

    model_config = ConfigDict(frozen=True)

    identity: RunIdentity
    user: User
    kwargs: RunKwargs = RunKwargs()
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    multitask_strategy: MultitaskStrategy | None = None
    # Its thread's priority while a priority callback is set, else 0: the worker queue's order. A row column.
    priority: int = 0
    # The LangSmith project the run is traced to, while tracing is on.
    langsmith_session_name: str | None = None
    # The request that created the run, so the run's logs and spans link back to it.
    correlation_id: str | None = None

    def to_execution_params(self) -> dict[str, Any]:
        """The run row's ``execution_params``: the job but for the row's own columns (``graph_id`` is none)."""
        return {**self.model_dump(mode="json", exclude={"identity", "priority"}), "graph_id": self.identity.graph_id}

    @classmethod
    def from_run_orm(cls, run_orm: RunORM) -> RunJob:
        """The job a run row records: its ``execution_params`` and its own columns."""
        job = cls.from_execution_params(
            run_orm.execution_params,
            run_id=run_orm.run_id,
            thread_id=run_orm.thread_id,
            assistant_id=run_orm.assistant_id,
            created_at=run_orm.created_at,
        )
        return job.model_copy(update={"priority": run_orm.priority})

    @classmethod
    def from_execution_params(
        cls, params: dict[str, Any], *, run_id: str, thread_id: str, assistant_id: str, created_at: datetime
    ) -> RunJob:
        """The job from a run row's ``execution_params`` and the columns that are not in them."""
        identity = {
            "run_id": run_id,
            "thread_id": thread_id,
            "graph_id": params["graph_id"],
            "assistant_id": assistant_id,
            "created_at": created_at,
        }
        return cls.model_validate({**params, "identity": identity})
