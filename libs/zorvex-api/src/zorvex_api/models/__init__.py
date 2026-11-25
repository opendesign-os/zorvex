"""Agent Protocol Pydantic models"""

from zorvex_api.models.assistants import (
    AgentSchemas,
    Assistant,
    AssistantCountRequest,
    AssistantCreate,
    AssistantSearchRequest,
    AssistantUpdate,
)
from zorvex_api.models.auth import User
from zorvex_api.models.crons import (
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
)
from zorvex_api.models.errors import AgentProtocolError, get_error_type
from zorvex_api.models.runs import Run, RunCreate, RunsCancel
from zorvex_api.models.store import (
    StoreDeleteRequest,
    StoreItem,
    StoreListNamespacesRequest,
    StoreListNamespacesResponse,
    StorePutRequest,
    StoreSearchRequest,
    StoreSearchResponse,
)
from zorvex_api.models.threads import (
    Thread,
    ThreadCheckpoint,
    ThreadCheckpointPostRequest,
    ThreadCreate,
    ThreadHistoryRequest,
    ThreadPruneRequest,
    ThreadPruneResponse,
    ThreadSearchRequest,
    ThreadState,
    ThreadStateUpdate,
    ThreadStateUpdateResponse,
    ThreadTTLSpec,
    ThreadUpdate,
)

__all__ = [
    # Assistants
    "Assistant",
    "AssistantCountRequest",
    "AssistantCreate",
    "AssistantSearchRequest",
    "AssistantUpdate",
    "AgentSchemas",
    # Threads
    "Thread",
    "ThreadCreate",
    "ThreadSearchRequest",
    "ThreadState",
    "ThreadStateUpdate",
    "ThreadStateUpdateResponse",
    "ThreadCheckpoint",
    "ThreadCheckpointPostRequest",
    "ThreadHistoryRequest",
    "ThreadPruneRequest",
    "ThreadPruneResponse",
    "ThreadTTLSpec",
    # Runs
    "Run",
    "RunCreate",
    "RunsCancel",
    # Crons
    "CronCreate",
    "CronResponse",
    "CronUpdate",
    "CronSearchRequest",
    "CronCountRequest",
    # Store
    "StorePutRequest",
    "StoreSearchRequest",
    "StoreSearchResponse",
    "StoreItem",
    "StoreDeleteRequest",
    "StoreListNamespacesRequest",
    "StoreListNamespacesResponse",
    # Errors
    "AgentProtocolError",
    "get_error_type",
    # Auth
    "User",
    "ThreadUpdate",
]
