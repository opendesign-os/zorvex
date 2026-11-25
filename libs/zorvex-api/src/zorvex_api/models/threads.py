"""Thread-related Pydantic models for Agent Protocol"""

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from zorvex_api.models.entity_ids import ENTITY_ID_PATTERN, MAX_ENTITY_ID_LENGTH
from zorvex_api.models.enums import OnConflictBehavior, PruneStrategy, ThreadSortBy, ThreadStatus
from zorvex_api.models.search import SearchPage

# Upper bound keeping now + timedelta(minutes=ttl) finite and timedelta-safe
# (timedelta.max is ~1.44e9 minutes); rejects inf/1e308 at validation time.
MAX_TTL_MINUTES = 1_000_000_000


class ThreadTTLSpec(BaseModel):
    """Per-thread TTL, as the LangGraph SDK sends it: ``{"ttl": minutes, "strategy": ...}``."""

    ttl: float | None = Field(
        default=None, gt=0, le=MAX_TTL_MINUTES, description="Thread TTL in minutes; the server default when omitted"
    )
    strategy: PruneStrategy | None = Field(
        default=None,
        description="Expiry strategy: 'delete' removes the thread, 'keep_latest' prunes history",
    )


class ThreadCreate(BaseModel):
    """Request model for creating threads"""

    metadata: dict[str, Any] | None = Field(default=None, description="Thread metadata")
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ID_LENGTH,
        pattern=ENTITY_ID_PATTERN,
        description=(
            "Optional client-provided thread ID for idempotent creation. "
            "Omit or null to let the server generate a UUID. "
            f"When set, 1–{MAX_ENTITY_ID_LENGTH} characters and not blank "
            "(must fit PostgreSQL btree keys uncompressed)."
        ),
    )
    if_exists: OnConflictBehavior = Field(
        default="raise", description="Behavior when thread exists: 'raise' (default) or 'do_nothing'"
    )
    ttl: ThreadTTLSpec | None = Field(
        default=None,
        description="Per-thread TTL; without a server-side default it must carry ttl",
    )
    supersteps: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Checkpoint history to seed the thread with. Each superstep is "
            '{"updates": [{"values": ..., "as_node": ...}]}. Requires graph_id in metadata.'
        ),
    )

    @field_validator("supersteps")
    @classmethod
    def validate_supersteps(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v is None:
            return None
        for step in v:
            updates = step.get("updates")
            if not isinstance(updates, list) or not updates:
                raise ValueError("each superstep needs a non-empty 'updates' list")
            for update in updates:
                if not isinstance(update, dict):
                    raise ValueError("each update must be an object")
                if update.get("command") is not None:
                    raise ValueError("supersteps 'command' is not supported; use 'values' with 'as_node'")
        return v


class ThreadUpdate(BaseModel):
    """Request model for updating threads"""

    metadata: dict[str, Any] | None = Field(default=None, description="Thread metadata to update")
    ttl: ThreadTTLSpec | None = Field(default=None, description="Replace the thread's TTL")


class ThreadPruneRequest(BaseModel):
    """Request body for POST /threads/prune"""

    thread_ids: list[str] = Field(description="Threads to prune. Ids the caller cannot see are skipped.")
    strategy: PruneStrategy = Field(
        default="delete",
        description="'delete' removes the threads entirely; 'keep_latest' keeps each thread and its latest state.",
    )


class ThreadPruneResponse(BaseModel):
    """Response model for POST /threads/prune"""

    pruned_count: int = Field(default=0, description="Threads deleted or pruned by this request")


class ThreadTTLInfo(BaseModel):
    """A thread's TTL, as ``GET /threads/{thread_id}?include=ttl`` reports it."""

    strategy: PruneStrategy = Field(description="What expiry does: 'delete' the thread or keep only its latest state.")
    ttl_minutes: float = Field(description="The TTL in minutes.")
    expires_at: datetime = Field(description="When the thread expires.")


def _absent(value: Any) -> bool:
    return value is None


class Thread(BaseModel):
    """Thread entity model

    Status values: idle, busy, interrupted, error
    """

    model_config = ConfigDict(from_attributes=True)

    thread_id: str = Field(description="Unique identifier for the thread.")
    status: ThreadStatus = Field(
        default="idle", description="Current thread status: idle, busy, interrupted, or error."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata_dict", "metadata"),
        description="Arbitrary metadata attached to the thread.",
    )
    created_at: datetime = Field(description="Timestamp when the thread was created.")
    updated_at: datetime = Field(description="Timestamp when the thread was last updated.")
    state_updated_at: datetime | None = Field(default=None, description="Timestamp of the last state write.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The thread config: its last run's assistant config and run config, with only the assistant's "
            "configurable. The thread's graph is built with it to read and update the thread's state."
        ),
    )
    values: dict[str, Any] = Field(default_factory=dict, description="The current state of the thread.")
    interrupts: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="The current interrupts of the thread, by the id of the task that raised them.",
    )
    run_priority: int | None = Field(
        default=None,
        description="Priority of the thread's runs, -5 to 5, lower first; set by a priority callback, else null.",
    )
    fairness_group: str | None = Field(
        default=None,
        description="Group the thread's runs are scheduled fairly within; set by a priority callback, else null.",
    )
    ttl: ThreadTTLInfo | None = Field(
        default=None,
        exclude_if=_absent,
        description="The thread's TTL; only present with include=ttl, on a thread that has one.",
    )
    extracted: dict[str, Any] | None = Field(
        default=None, exclude_if=_absent, description="The values a search's `extract` asked for; only present then."
    )


ThreadSelectField = Literal[
    "thread_id",
    "created_at",
    "updated_at",
    "state_updated_at",
    "metadata",
    "config",
    "status",
    "values",
    "interrupts",
    "run_priority",
    "fairness_group",
]


# An extract path: a Thread field, then .key and [index] steps, as in values.messages[-1].content.
EXTRACT_PATH = re.compile(r"(?P<root>values|metadata|config|interrupts)(?:\.[^.\[\]]+|\[-?\d+\])*")


class ThreadCountRequest(BaseModel):
    """Filters of POST /threads/count; search takes them too."""

    metadata: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    exclude_metadata: dict[str, Any] | list[dict[str, Any]] | None = Field(
        default=None,
        description="Exclude threads whose metadata contains this object, or any object in this list. Uses the "
        "same matching as `metadata`; threads missing the keys are not excluded.",
    )
    created_after: datetime | None = Field(
        default=None, description="Only threads created at or after this time; without a timezone it is UTC."
    )
    created_before: datetime | None = Field(
        default=None, description="Only threads created before this time; without a timezone it is UTC."
    )
    state_updated_after: datetime | None = Field(
        default=None,
        description="Only threads whose state last changed at or after this time; without a timezone it is UTC.",
    )
    state_updated_before: datetime | None = Field(
        default=None,
        description="Only threads whose state last changed before this time; without a timezone it is UTC.",
    )
    values: dict[str, Any] | None = Field(default=None, description="Threads whose values contain these pairs.")
    status: ThreadStatus | None = Field(
        default=None, description="Thread status filter (idle, busy, interrupted, error)"
    )

    @field_validator("created_after", "created_before", "state_updated_after", "state_updated_before")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        """A time without a timezone, as UTC."""
        return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


class ThreadSearchRequest(ThreadCountRequest, SearchPage[ThreadSelectField]):
    """Request model for thread search"""

    sort_by: ThreadSortBy = Field(default="created_at", description="Field to sort by.")

    ids: list[str] | None = Field(default=None, description="Restrict results to these thread IDs")
    extract: dict[str, str] | None = Field(
        default=None,
        max_length=10,
        description="Values to return in each thread's `extracted`, by alias: a path into the thread's values, "
        "metadata, config or interrupts, with .key and [index] steps, e.g. values.messages[-1]. At most 10.",
    )

    @field_validator("extract")
    @classmethod
    def _check_extract(cls, extract: dict[str, str] | None) -> dict[str, str] | None:
        bad = [path for path in (extract or {}).values() if not EXTRACT_PATH.fullmatch(path)]
        if bad:
            raise ValueError(f"extract paths start with values, metadata, config or interrupts: {bad}")
        return extract


class ThreadCheckpoint(BaseModel):
    """Checkpoint identifier for thread history"""

    checkpoint_id: str | None = None
    thread_id: str | None = None
    checkpoint_ns: str | None = ""
    checkpoint_map: dict[str, Any] | None = None


class ThreadCheckpointPostRequest(BaseModel):
    """Request model for fetching thread checkpoint"""

    checkpoint: ThreadCheckpoint = Field(description="Checkpoint to fetch")
    subgraphs: bool | None = Field(default=False, description="Include subgraph states")


class ThreadState(BaseModel):
    """Thread state model for history endpoint"""

    values: dict[str, Any] = Field(description="Channel values (messages, etc.)")
    next: list[str] = Field(default_factory=list, description="Next nodes to execute")
    tasks: list[dict[str, Any]] = Field(default_factory=list, description="Tasks to execute")
    interrupts: list[dict[str, Any]] = Field(default_factory=list, description="Interrupt data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Checkpoint metadata")
    created_at: datetime | None = Field(default=None, description="Timestamp of state creation")
    checkpoint: ThreadCheckpoint = Field(description="Current checkpoint")
    parent_checkpoint: ThreadCheckpoint | None = Field(default=None, description="Parent checkpoint")


class ThreadStateUpdate(BaseModel):
    """Request model for updating thread state"""

    values: dict[str, Any] | list[dict[str, Any]] | None = Field(
        default=None, description="The values to update the state with"
    )
    checkpoint: dict[str, Any] | None = Field(default=None, description="The checkpoint to update the state of")
    checkpoint_id: str | None = Field(default=None, description="Optional checkpoint ID to update from")
    as_node: str | None = Field(default=None, description="Update the state as if this node had just executed")


class ThreadStateUpdateResponse(BaseModel):
    """Response model for thread state update"""

    checkpoint: dict[str, Any] = Field(description="The checkpoint that was created/updated")


class ThreadHistoryRequest(BaseModel):
    """Request model for thread history endpoint"""

    limit: int = Field(default=10, ge=1, le=1000, description="Number of states to return")
    before: dict[str, Any] | str | None = Field(
        default=None,
        description="Return states before this checkpoint (checkpoint ID string, raw checkpoint dict, or RunnableConfig with 'configurable' key)",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Filter by metadata")
    checkpoint: dict[str, Any] | None = Field(default=None, description="Checkpoint for subgraph filtering")
    select: list[str] | None = Field(
        default=None,
        description="Fields of each state to return: values, next, tasks, checkpoint, metadata, created_at, "
        "parent_checkpoint, interrupts, or values.<key> for one key of the values; all when omitted.",
    )


class StorageSize(BaseModel):
    count: int = Field(description="How many there are.")
    bytes: int = Field(description="The bytes they take.")


class ChannelSize(BaseModel):
    versions: int = Field(description="Versions of the channel's value stored.")
    bytes: int = Field(description="The bytes they take.")


class ChannelValuesSize(StorageSize):
    by_channel: dict[str, ChannelSize] = Field(description="The same, per channel.")


class ThreadStorage(BaseModel):
    """The bytes a thread's history takes in the checkpointer."""

    thread_id: str
    total_bytes: int = Field(description="The sum of the three parts.")
    checkpoints: StorageSize = Field(description="Checkpoints, values stored inline included.")
    channel_values: ChannelValuesSize = Field(description="Channel values stored apart, one per version.")
    pending_writes: StorageSize = Field(description="Writes of tasks not yet applied to a checkpoint.")
