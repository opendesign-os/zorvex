"""Run-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal, Self, get_args
from uuid import UUID

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from zorvex_api.models.enums import (
    BulkCancelRunsStatus,
    DisconnectMode,
    Durability,
    IfNotExists,
    MultitaskStrategy,
    OnCompletionBehavior,
    RunStatus,
    StreamMode,
)


class LangSmithTracer(BaseModel):
    project_name: str | None = Field(None, description="The LangSmith project to trace the run to.")
    example_id: str | None = Field(None, description="The dataset example the run is traced against.")


class RunCreate(BaseModel):
    """Request model for creating runs"""

    assistant_id: str = Field(..., description="Assistant to execute")
    input: Any = Field(
        None,
        description="Input to the graph, any JSON; none to continue from the thread's checkpoint or with a command.",
    )
    config: dict[str, Any] | None = Field(default_factory=dict, description="Execution config")
    context: dict[str, Any] | None = Field(default_factory=dict, description="Execution context")
    checkpoint: dict[str, Any] | None = Field(
        None,
        description="Checkpoint configuration (e.g., {'checkpoint_id': '...', 'checkpoint_ns': ''})",
    )
    checkpoint_id: UUID | None = Field(
        None,
        description="Checkpoint to run from. Short form of checkpoint={'checkpoint_id': ...}; 'checkpoint' wins if both set.",
    )
    durability: Durability | None = Field(
        None,
        description=(
            "When checkpoints are persisted: 'sync' before each next step starts, 'async' in the background "
            "while the next step runs (LangGraph default), 'exit' only once, when the run ends. "
            "Overrides the server default (checkpointer.durability)."
        ),
    )
    stream_mode: StreamMode | list[StreamMode] | None = Field(None, description="Requested stream mode(s)")
    on_disconnect: DisconnectMode = Field(
        "continue",
        description="Streaming and waiting routes: 'continue' (default) runs on when the client leaves, "
        "'cancel' cancels the run.",
    )
    on_completion: OnCompletionBehavior = Field(
        "delete",
        description="Stateless runs only: 'delete' (default) deletes the run's thread once the run has finished, "
        "'keep' keeps it.",
    )

    multitask_strategy: MultitaskStrategy | None = Field(
        None,
        description=(
            "How the run meets the runs already pending or running on its thread: 'enqueue' (default) waits its "
            "turn, as a thread runs one run at a time; 'reject' answers 409; 'interrupt' interrupts them and "
            "'rollback' deletes them with the checkpoints they wrote, this run then going on from what is left."
        ),
    )
    webhook: str | None = Field(
        None,
        description=(
            "Absolute http(s) URL that receives a POST with the run, its final values and any error once the run "
            "reaches a terminal status. Must pass the server's webhooks.url policy; a URL the server will not call "
            "returns 422."
        ),
    )

    if_not_exists: IfNotExists = Field(
        "reject", description="A missing thread: 'reject' (default) answers 404, 'create' creates it."
    )
    after_seconds: float | None = Field(
        None,
        ge=0,
        description=(
            "Seconds from now until the run is due: its created_at. It starts no sooner, and takes its turn on "
            "the thread from then."
        ),
    )

    # Human-in-the-loop fields (core HITL functionality)
    command: dict[str, Any] | None = Field(
        None,
        description="Command for resuming interrupted runs with state updates or navigation",
    )
    interrupt_before: Literal["*"] | list[str] | None = Field(
        None,
        description="Nodes to interrupt immediately before they get executed. Use '*' for all nodes.",
    )
    interrupt_after: Literal["*"] | list[str] | None = Field(
        None,
        description="Nodes to interrupt immediately after they get executed. Use '*' for all nodes.",
    )

    # Subgraph configuration
    stream_subgraphs: bool = Field(False, description="Include events from subgraphs in the stream.")
    stream_resumable: bool = Field(
        False,
        description=(
            "Whether the stream can be rejoined and replayed whole after the run ends: its events are kept 10 "
            "minutes, else only while it runs and briefly after."
        ),
    )
    feedback_keys: list[str] | None = Field(
        None,
        description=(
            "LangSmith feedback keys: the run's stream ends with a feedback event of presigned URLs, one per key, "
            "and its wait or join result carries them under __feedback__. Needs LangSmith."
        ),
    )
    checkpoint_during: bool | None = Field(
        None,
        deprecated=True,
        description="Deprecated, use durability: true is 'async', false is 'exit'; durability wins when both are set.",
    )
    langsmith_tracer: LangSmithTracer | None = Field(
        None, description="Also trace the run to this LangSmith project, when LangSmith tracing is on."
    )

    metadata: dict[str, Any] | None = Field(
        None,
        description="Metadata to assign to the run. Traces carry each key as langfuse.trace.metadata.<key>.",
    )

    @model_validator(mode="after")
    def _command_replaces_input(self) -> Self:
        """A command runs in place of the input, as the Agent Server runs it.

        Without either, the run continues from the thread's latest checkpoint
        (or the one ``checkpoint`` names): how a client resumes past an
        ``interrupt_before``/``interrupt_after`` breakpoint.
        """
        if self.command is not None:
            self.input = None
        return self


class RunsCancel(BaseModel):
    """Selector for ``POST /runs/cancel``: a status filter, or thread_id plus run_ids."""

    status: BulkCancelRunsStatus | None = Field(
        None, description="Cancel every active run of the caller in this status. 'all' means pending and running."
    )
    thread_id: str | None = Field(None, description="Thread that owns the runs listed in run_ids.")
    run_ids: list[str] | None = Field(None, description="Runs to cancel; requires thread_id.")

    @model_validator(mode="after")
    def validate_exactly_one_selector(self) -> Self:
        """Reject a half-filled selector: the handler reads one branch and would
        ignore the rest, so {status, thread_id} would cancel every matching run."""
        by_status = self.status is not None
        complete_ids = self.thread_id is not None and self.run_ids is not None
        any_ids = self.thread_id is not None or self.run_ids is not None
        if by_status == any_ids or (any_ids and not complete_ids):
            raise ValueError("Provide either 'status' or both 'thread_id' and 'run_ids'")
        return self


class Run(BaseModel):
    """Run entity model

    Status values: pending, running, error, success, timeout, interrupted
    """

    model_config = ConfigDict(from_attributes=True)

    run_id: str = Field(..., description="Unique identifier for the run.")
    thread_id: str = Field(..., description="Thread this run belongs to.")
    assistant_id: str = Field(..., description="Assistant that is executing this run.")
    status: RunStatus = Field(
        "pending", description="Current run status: pending, running, error, success, timeout, or interrupted."
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata assigned to the run at creation.")
    multitask_strategy: MultitaskStrategy | None = Field(
        None, description="Strategy applied for concurrent runs on the same thread."
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="What the run was created with: input (null for a checkpoint-only resume), config, context, "
        "stream and interrupt options, webhook and durability.",
    )
    created_at: datetime = Field(
        ..., description="When the run was created; for a run with after_seconds, when it is due to start."
    )
    updated_at: datetime = Field(..., description="Timestamp when the run was last updated.")
    cancel_reason: str | None = Field(None, description="Why the run was cancelled, if it was.")
    priority: int = Field(0, description="Priority of the run, -5 to 5, lower first; 0 without a priority callback.")
    langsmith_session_name: str | None = Field(
        None, description="The LangSmith project the run is traced to, when LangSmith tracing is on."
    )

    @model_validator(mode="before")
    @classmethod
    def derive_execution_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) or not hasattr(data, "execution_params"):
            return data
        values = {
            name: getattr(data, name, None) for name in cls.model_fields if name not in RUN_EXECUTION_PARAMS_FIELDS
        }
        values.update(run_fields_from_execution_params(getattr(data, "execution_params", None)))
        return values


# Run fields with no column of their own; they are read out of the row's execution_params.
RUN_EXECUTION_PARAMS_FIELDS: frozenset[str] = frozenset(
    {"metadata", "multitask_strategy", "kwargs", "langsmith_session_name"}
)


def redact_webhook_url(url: str | None) -> str | None:
    """Scheme, host and path only, or None for an invalid URL.

    A webhook's query string or userinfo often carries the receiver's token, so
    responses and logs must never show it.
    """
    if not url:
        return url
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return None
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.host}{port}{parsed.path}"


def run_fields_from_execution_params(params: Any) -> dict[str, Any]:
    """The ``RUN_EXECUTION_PARAMS_FIELDS`` values a run row's ``execution_params`` records."""
    params = params if isinstance(params, dict) else {}
    kwargs = dict(params.get("kwargs") or {})
    if kwargs:
        kwargs["webhook"] = redact_webhook_url(kwargs.get("webhook"))
    return {
        "metadata": params.get("run_metadata") or {},
        "multitask_strategy": params.get("multitask_strategy"),
        "langsmith_session_name": params.get("langsmith_session_name"),
        "kwargs": kwargs,
    }


# Every Run field; checked against the model at import so the two cannot drift.
RunSelectField = Literal[
    "run_id",
    "thread_id",
    "assistant_id",
    "status",
    "metadata",
    "multitask_strategy",
    "kwargs",
    "created_at",
    "updated_at",
    "cancel_reason",
    "priority",
    "langsmith_session_name",
]
if set(get_args(RunSelectField)) != set(Run.model_fields):
    raise RuntimeError("RunSelectField must list exactly the fields of Run")
