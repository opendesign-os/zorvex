"""Assistant endpoints; services/assistant_service.py does the work."""

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import JSONResponse

from zorvex_api.core import search
from zorvex_api.core.auth import auth_dependency
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.models import (
    AgentSchemas,
    Assistant,
    AssistantCountRequest,
    AssistantCreate,
    AssistantSearchRequest,
    AssistantUpdate,
)
from zorvex_api.models.assistants import AssistantVersionsRequest
from zorvex_api.models.errors import CONFLICT, NOT_FOUND
from zorvex_api.services.assistant_service import AssistantService, get_assistant_service

router = APIRouter(tags=["Assistants"], dependencies=auth_dependency)


@router.post(
    "/assistants",
    response_model=Assistant,
    responses={**NOT_FOUND, **CONFLICT},
)
async def create_assistant(
    request: AssistantCreate,
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "create"),
):
    """Create a new assistant.

    An assistant is a configured instance of a graph. Provide a `graph_id`
    referencing a graph defined in your `zorvex.json`. If `assistant_id` is
    omitted, one is auto-generated. Set `if_exists` to `"do_nothing"` for
    idempotent creation; otherwise a taken `assistant_id` or an identical
    graph/config pair answers 409.
    """
    return await service.create_assistant(authorize, request)


@router.post("/assistants/search", response_model=list[Assistant], responses={**NOT_FOUND})
async def search_assistants(
    request: AssistantSearchRequest,
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "search"),
) -> JSONResponse:
    """Search assistants with filters.

    Filter by name, graph ID, or metadata. Results are paginated via `limit`
    and `offset`, and `select` picks the fields returned.
    """
    items, headers = await service.search_assistants(authorize, request)
    return search.respond(items, headers, request.select)


@router.post("/assistants/count", response_model=int, responses={**NOT_FOUND})
async def count_assistants(
    request: AssistantCountRequest,
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "search"),
):
    """Count assistants matching the given filters.

    Accepts the same filter parameters as the search endpoint but returns only
    the total count.
    """
    return await service.count_assistants(authorize, request)


@router.get(
    "/assistants/{assistant_id}",
    response_model=Assistant,
    responses={**NOT_FOUND},
)
async def get_assistant(
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "read"),
):
    """Get an assistant by its ID.

    Returns the latest version of the assistant. Returns 404 if the assistant
    does not exist or does not belong to the authenticated user.
    """
    return await service.get_assistant(authorize, assistant_id)


@router.patch(
    "/assistants/{assistant_id}",
    response_model=Assistant,
    responses={**NOT_FOUND},
)
async def update_assistant(
    assistant_id: str,
    request: AssistantUpdate,
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "update"),
):
    """Update an assistant by its ID.

    Partial update: only fields included in the request body are changed.
    Creates a new version of the assistant.
    """
    return await service.update_assistant(authorize, assistant_id, request)


@router.delete("/assistants/{assistant_id}", responses={**NOT_FOUND})
async def delete_assistant(
    assistant_id: str,
    delete_threads: bool = Query(
        False, description="Also delete threads bound to this assistant, with their runs and checkpoints."
    ),
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "delete"),
):
    """Delete an assistant by its ID.

    Permanently removes the assistant and all of its versions. This action
    cannot be undone.
    """
    return await service.delete_assistant(authorize, assistant_id, delete_threads=delete_threads)


@router.post(
    "/assistants/{assistant_id}/latest",
    response_model=Assistant,
    responses={**NOT_FOUND},
)
async def set_assistant_latest(
    assistant_id: str,
    version: int = Body(..., embed=True, description="The version number to set as latest."),
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "update"),
):
    """Pin a specific version as the latest version of an assistant.

    After calling this endpoint, the assistant will use the specified version's
    configuration when executing runs.
    """
    return await service.set_assistant_latest(authorize, assistant_id, version)


@router.post(
    "/assistants/{assistant_id}/versions",
    response_model=list[Assistant],
    responses={**NOT_FOUND},
)
async def list_assistant_versions(
    assistant_id: str,
    request: AssistantVersionsRequest | None = None,
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "search"),
):
    """List the versions of an assistant, newest first.

    Each version captures the assistant's configuration at the time of its
    creation or update. `limit` and `offset` page them, and `metadata` keeps
    the versions whose metadata contains it.
    """
    return await service.list_assistant_versions(authorize, assistant_id, request or AssistantVersionsRequest())


@router.get(
    "/assistants/{assistant_id}/schemas",
    response_model=AgentSchemas,
    responses={**NOT_FOUND},
)
async def get_assistant_schemas(
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "read"),
):
    """Get the JSON schemas for an assistant's graph.

    Returns ``graph_id`` and the input, output, state, config and context
    schemas derived from the graph's type annotations. A schema the server
    could not derive is ``null``; the key is always present.
    """
    return await service.get_assistant_schemas(authorize, assistant_id)


@router.get("/assistants/{assistant_id}/graph", responses={**NOT_FOUND})
async def get_assistant_graph(
    assistant_id: str,
    xray: bool | int = Query(
        False, description="Expand subgraph nodes. Pass true or a depth integer to control nesting."
    ),
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "read"),
):
    """Get the graph structure for visualization.

    Returns a JSON representation of the graph's nodes and edges suitable for
    rendering in graph visualizers. Use `xray` to expand subgraph nodes into
    their internal structure.
    """
    return await service.get_assistant_graph(authorize, assistant_id, xray)


@router.get("/assistants/{assistant_id}/subgraphs", responses={**NOT_FOUND})
async def get_assistant_subgraphs(
    assistant_id: str,
    recurse: bool = Query(False, description="Recursively include nested subgraphs."),
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "read"),
):
    """Get subgraphs of an assistant.

    Returns the subgraph definitions used by this assistant's graph. Set
    `recurse=true` to include deeply nested subgraphs, or filter to a single
    namespace.
    """
    return await service.get_assistant_subgraphs(authorize, assistant_id, None, recurse)


@router.get("/assistants/{assistant_id}/subgraphs/{namespace}", responses={**NOT_FOUND})
async def get_assistant_subgraphs_by_namespace(
    assistant_id: str,
    namespace: str,
    recurse: bool = Query(False, description="Recursively include nested subgraphs."),
    service: AssistantService = Depends(get_assistant_service),
    authorize: Authorization = Authorize("assistants", "read"),
):
    """Get the subgraphs of an assistant under a single namespace."""
    return await service.get_assistant_subgraphs(authorize, assistant_id, namespace, recurse)
