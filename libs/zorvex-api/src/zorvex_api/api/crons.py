"""Cron endpoints, the ones the LangGraph SDK's ``CronsClient`` calls; services/cron_service.py does the work.

``POST /runs/crons`` and ``POST /threads/{thread_id}/runs/crons`` create a cron,
``GET``/``PATCH``/``DELETE /runs/crons/{cron_id}`` read, change and delete one,
and ``POST /runs/crons/search`` and ``/count`` list and count them.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core import search
from zorvex_api.core.auth import auth_dependency
from zorvex_api.core.authorization import Authorization, Authorize, Filters
from zorvex_api.core.orm import SessionDep
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
from zorvex_api.services.cron_service import CronService, get_cron_service
from zorvex_api.services.ownership import owned_thread

router = APIRouter(tags=["Crons"], dependencies=auth_dependency)


async def _authorize_create[C: (CronCreate, ThreadCronCreate)](
    authorize: Authorization, request: C, session: AsyncSession, *, thread_id: str | None
) -> tuple[C, Filters | None]:
    """Authorize creating the cron through the chain the LangGraph SDK reference fires.

    ``crons.create``, whose handler may change the payload of the cron's runs and
    its metadata; ``assistants.read``, whose filter applies to the cron's assistant;
    then ``threads.read`` for a thread's cron (the thread must exist and be the
    caller's: a cron does not create the thread it names), or ``threads.search``.

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
        await owned_thread(session, thread_id, user, filters=thread_filters)
    else:
        await Authorization(user, "threads", "search")({})
    return request, assistant_filters


@router.post("/runs/crons", response_model=CronResponse, responses={**NOT_FOUND})
async def create_cron(
    request: CronCreate,
    session: SessionDep,
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "create"),
) -> CronResponse:
    """Create a cron whose every run goes on a new thread; it first fires at the schedule's next time."""
    request, assistant_filters = await _authorize_create(authorize, request, session, thread_id=None)
    return await service.create_cron(request, assistant_filters=assistant_filters)


@router.post("/threads/{thread_id}/runs/crons", response_model=CronResponse, responses={**NOT_FOUND})
async def create_cron_for_thread(
    thread_id: str,
    request: ThreadCronCreate,
    session: SessionDep,
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "create"),
) -> CronResponse:
    """Create a cron whose runs all go on an existing thread; it first fires at the schedule's next time."""
    request, assistant_filters = await _authorize_create(authorize, request, session, thread_id=thread_id)
    return await service.create_cron(request, thread_id=thread_id, assistant_filters=assistant_filters)


@router.patch("/runs/crons/{cron_id}", response_model=CronResponse, responses={**NOT_FOUND})
async def update_cron(
    cron_id: str,
    request: CronUpdate,
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "update"),
) -> CronResponse:
    """Change the fields the request sets and return the whole cron; an explicit null clears end_time or the webhook."""
    if not request.model_fields_set:
        raise HTTPException(400, "Request body cannot be empty")
    supplied = request.model_dump(mode="json", exclude_unset=True)
    value: dict[str, Any] = {
        "cron_id": cron_id,
        "payload": {key: supplied.pop(key) for key in CRON_RUN_FIELDS if key in supplied},
        **supplied,
    }
    filters = await authorize(value)
    # The update as the handler left it; what it adds counts as set.
    changes = {key: field for key, field in value.items() if key not in ("cron_id", "payload")}
    return await service.update_cron(
        cron_id, CronUpdate.model_validate({**changes, **value["payload"]}), filters=filters
    )


@router.delete("/runs/crons/{cron_id}", status_code=204, responses={**NOT_FOUND})
async def delete_cron(
    cron_id: str,
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "delete"),
) -> Response:
    """Delete a cron job."""
    await service.delete_cron(cron_id, filters=await authorize({"cron_id": cron_id}))
    return Response(status_code=204)


@router.get("/runs/crons/{cron_id}", response_model=CronResponse, responses={**NOT_FOUND})
async def get_cron(
    cron_id: str,
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "read"),
) -> CronResponse:
    """Get a cron job by id."""
    return await service.get_cron(cron_id, filters=await authorize({"cron_id": cron_id}))


@router.post("/runs/crons/search", response_model=list[CronResponse])
async def search_crons(
    request: CronSearchRequest,
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "search"),
) -> JSONResponse:
    """Search cron jobs with filters and pagination; `select` picks the fields returned."""
    items, headers = await service.search_crons(request, filters=await authorize(request.model_dump()))
    return search.respond(items, headers, request.select)


@router.post("/runs/crons/count")
async def count_crons(
    request: CronCountRequest,
    service: CronService = Depends(get_cron_service),
    authorize: Authorization = Authorize("crons", "search"),
) -> int:
    """Count cron jobs matching filters."""
    return await service.count_crons(request, filters=await authorize(request.model_dump()))
