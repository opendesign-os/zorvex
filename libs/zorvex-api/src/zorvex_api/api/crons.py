"""Cron job endpoints for Agent Protocol.

Implements the six endpoints consumed by the LangGraph SDK ``CronsClient``:

* ``POST  /runs/crons``                  → create (stateless, returns the Cron)
* ``POST  /threads/{thread_id}/runs/crons`` → create for thread (returns the Cron)
* ``PATCH /runs/crons/{cron_id}``         → update (returns Cron)
* ``DELETE /runs/crons/{cron_id}``        → delete (204)
* ``POST  /runs/crons/search``            → search (returns list[Cron])
* ``POST  /runs/crons/count``             → count (returns int)
"""

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.auth_deps import auth_dependency, require_auth
from zorvex_api.core.authorization import Authorization, Authorize, Filters
from zorvex_api.core.orm import SessionDep
from zorvex_api.models import User
from zorvex_api.models.crons import (
    CRON_RUN_FIELDS,
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
    ThreadCronCreate,
)
from zorvex_api.models.errors import NOT_FOUND
from zorvex_api.services.cron_service import (
    CronService,
    _cron_to_response,
    get_cron_service,
)
from zorvex_api.services.ownership import owned_thread
from zorvex_api.utils.pagination import pagination_headers

router = APIRouter(tags=["Crons"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


async def _authorize_cron_create[C: (CronCreate, ThreadCronCreate)](
    authorize: Authorization,
    request: C,
    session: AsyncSession,
    *,
    thread_id: str | None,
) -> tuple[C, Filters | None]:
    """Authorize creating the cron through the chain the LangGraph SDK reference fires.

    A cron create touches three resources, so a handler can deny at any of them:
    ``crons.create``, whose handler may change the payload of the cron's runs
    and its metadata; ``assistants.read``, whose filter applies to the cron's
    assistant; then ``threads.read`` for a thread-bound cron, whose filter
    applies to the thread, or ``threads.search`` for a stateless one.

    Returns the cron as the handler left it, and the assistant filter.
    """
    user = authorize.user
    value: dict[str, Any] = {
        "payload": request.model_dump(mode="json", include=set(CRON_RUN_FIELDS), exclude_none=True),
        "schedule": request.schedule,
        "thread_id": thread_id,
        "user_id": user.identity,
        "end_time": request.end_time,
        "metadata": dict(request.metadata or {}),
    }
    await authorize(value)
    request = type(request).model_validate(
        {
            **request.model_dump(exclude=set(CRON_RUN_FIELDS)),
            **value["payload"],
            **{key: value[key] for key in ("schedule", "end_time", "metadata")},
        }
    )

    assistant_filters = await Authorization(user, "assistants", "read")({"assistant_id": request.assistant_id})
    if thread_id is not None:
        thread_filters = await Authorization(user, "threads", "read")({"thread_id": thread_id})
        # Every firing runs on this thread, so it must exist and be the caller's:
        # unlike a run, a cron does not create the thread it names.
        await owned_thread(session, thread_id, user, filters=thread_filters)
    else:
        await Authorization(user, "threads", "search")({})
    return request, assistant_filters


# ---------------------------------------------------------------------------
# Create (stateless) – POST /runs/crons → returns the Cron
# ---------------------------------------------------------------------------


@router.post("/runs/crons", response_model=CronResponse, responses={**NOT_FOUND})
async def create_cron(
    request: CronCreate,
    session: SessionDep,
    user: User = Depends(require_auth),
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "create"),
) -> CronResponse:
    """Create a cron whose every run goes on a new thread; it first fires at the schedule's next time."""
    request, assistant_filters = await _authorize_cron_create(authorize, request, session, thread_id=None)
    cron = await service.create_cron(request, user.identity, assistant_filters=assistant_filters)
    return _cron_to_response(cron)


# ---------------------------------------------------------------------------
# Create for thread – POST /threads/{thread_id}/runs/crons → returns the Cron
# ---------------------------------------------------------------------------


@router.post("/threads/{thread_id}/runs/crons", response_model=CronResponse, responses={**NOT_FOUND})
async def create_cron_for_thread(
    thread_id: str,
    request: ThreadCronCreate,
    session: SessionDep,
    user: User = Depends(require_auth),
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "create"),
) -> CronResponse:
    """Create a cron whose runs all go on an existing thread; it first fires at the schedule's next time."""
    request, assistant_filters = await _authorize_cron_create(authorize, request, session, thread_id=thread_id)
    cron = await service.create_cron(request, user.identity, thread_id=thread_id, assistant_filters=assistant_filters)
    return _cron_to_response(cron)


# ---------------------------------------------------------------------------
# Update – PATCH /runs/crons/{cron_id} → returns Cron
# ---------------------------------------------------------------------------


@router.patch("/runs/crons/{cron_id}", response_model=CronResponse, responses={**NOT_FOUND})
async def update_cron(
    cron_id: str,
    request: CronUpdate,
    user: User = Depends(require_auth),
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "update"),
) -> CronResponse:
    """Update an existing cron job.

    Only provided fields are updated (partial patch). Returns the full
    ``Cron`` object after update.
    """
    if not request.model_fields_set:
        raise HTTPException(400, "Request body cannot be empty")
    # Only the fields the caller set, explicit nulls included: a null clears end_time or the webhook.
    supplied = request.model_dump(mode="json", exclude_unset=True)
    value: dict[str, Any] = {
        "cron_id": cron_id,
        "payload": {key: supplied.pop(key) for key in CRON_RUN_FIELDS if key in supplied},
        **supplied,
    }
    filters = await authorize(value)
    # The update as the handler left it; what it adds counts as set.
    changes = {key: field for key, field in value.items() if key not in ("cron_id", "payload")}
    request = CronUpdate.model_validate({**changes, **value["payload"]})
    return await service.update_cron(cron_id, request, user.identity, filters=filters)


# ---------------------------------------------------------------------------
# Delete – DELETE /runs/crons/{cron_id} → 204
# ---------------------------------------------------------------------------


@router.delete("/runs/crons/{cron_id}", status_code=204, responses={**NOT_FOUND})
async def delete_cron(
    cron_id: str,
    user: User = Depends(require_auth),
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "delete"),
) -> Response:
    """Delete a cron job."""
    filters = await authorize({"cron_id": cron_id})
    await service.delete_cron(cron_id, user.identity, filters=filters)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Search – POST /runs/crons/search → list[Cron]
# ---------------------------------------------------------------------------


@router.get("/runs/crons/{cron_id}", response_model=CronResponse, responses={**NOT_FOUND})
async def get_cron(
    cron_id: str,
    user: User = Depends(require_auth),
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "read"),
) -> CronResponse:
    """Get a cron job by id."""
    filters = await authorize({"cron_id": cron_id})
    return await service.get_cron(cron_id, user.identity, filters=filters)


@router.post("/runs/crons/search", response_model=list[CronResponse])
async def search_crons(
    request: CronSearchRequest,
    response: Response,
    user: User = Depends(require_auth),
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "search"),
) -> list[CronResponse] | JSONResponse:
    """Search cron jobs with filters and pagination."""
    filters = await authorize(request.model_dump(exclude_none=True))
    page = request.model_copy(update={"limit": request.limit + 1})
    results = await service.search_crons(page, user.identity, filters=filters)
    headers = pagination_headers(len(results), limit=request.limit, offset=request.offset)
    results = results[: request.limit]
    if request.select:
        chosen: set[str] = set(request.select)
        return JSONResponse([c.model_dump(mode="json", include=chosen) for c in results], headers=headers)
    response.headers.update(headers)
    return results


# ---------------------------------------------------------------------------
# Count – POST /runs/crons/count → int
# ---------------------------------------------------------------------------


@router.post("/runs/crons/count")
async def count_crons(
    request: CronCountRequest,
    user: User = Depends(require_auth),
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "search"),
) -> int:
    """Count cron jobs matching filters."""
    filters = await authorize(request.model_dump(exclude_none=True))
    return await service.count_crons(request, user.identity, filters=filters)
