"""Enumerated values of the Agent Protocol.

Taken from the LangGraph SDK wherever this server supports every value the SDK
names, so the two cannot drift; the server's narrower sets follow.
"""

from typing import Literal

from langgraph_sdk.schema import (
    AssistantSortBy,
    BulkCancelRunsStatus,
    CronSortBy,
    DisconnectMode,
    Durability,
    IfNotExists,
    MultitaskStrategy,
    OnCompletionBehavior,
    OnConflictBehavior,
    PruneStrategy,
    RunStatus,
    SortOrder,
    StreamMode,
    ThreadSortBy,
    ThreadStatus,
)

__all__ = [
    "ACTIVE_RUN_STATUSES",
    "TERMINAL_RUN_STATUSES",
    "AssistantSortBy",
    "BulkCancelRunsStatus",
    "CronSortBy",
    "DisconnectMode",
    "Durability",
    "IfNotExists",
    "MultitaskStrategy",
    "OnCompletionBehavior",
    "OnConflictBehavior",
    "PruneStrategy",
    "RunCancellationAction",
    "RunStatus",
    "SortOrder",
    "StreamMode",
    "ThreadSortBy",
    "ThreadStatus",
]

# How a cancel stops a run: interrupt it, or roll it back (delete it and its checkpoints).
RunCancellationAction = Literal["interrupt", "rollback"]


# A run in one of these states can still change; every other status is final.
ACTIVE_RUN_STATUSES: tuple[RunStatus, ...] = ("pending", "running")
TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset({"error", "success", "timeout", "interrupted"})
