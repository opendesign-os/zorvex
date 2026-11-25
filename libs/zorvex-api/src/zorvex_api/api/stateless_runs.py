"""Stateless runs: each runs on a thread of its own.

The thread is deleted once the run has finished, by whichever instance ends the run, unless the
request sets ``on_completion="keep"``.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Request, Response
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse

from zorvex_api.api.runs import run_location, stream_response, wait_response
from zorvex_api.core.auth import auth_dependency
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.core.orm import SessionDep
from zorvex_api.models import Run, RunCreate
from zorvex_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from zorvex_api.services import run_waiters
from zorvex_api.services.run_preparation import prepare_stateless_run

router = APIRouter(tags=["Stateless Runs"], dependencies=auth_dependency)


@router.post("/runs/wait", responses={**NOT_FOUND, **CONFLICT})
async def stateless_wait_for_run(
    request: RunCreate,
    session: SessionDep,
    http: Request,
    authorize: Authorization = Authorize("threads", "create_run"),
) -> StreamingResponse:
    """Create a stateless run and wait for its output.

    Responds as `POST /threads/{thread_id}/runs/wait` does, the output read from the run's events.
    The run's thread is deleted once the run has finished, unless `on_completion` is `"keep"`.
    """
    user = authorize.user
    run = await prepare_stateless_run(session, request, user, authorize=authorize)
    leaves = http.is_disconnected if request.on_disconnect == "cancel" else None
    body = run_waiters.wait_body(run.run_id, run.thread_id, user, is_disconnected=leaves)
    return wait_response(run.thread_id, run.run_id, body)


@router.post("/runs/stream", responses={**SSE_RESPONSE, **NOT_FOUND, **CONFLICT})
async def stateless_stream_run(
    request: RunCreate,
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "create_run"),
) -> EventSourceResponse:
    """Create a stateless run and stream its execution.

    Streams as `POST /threads/{thread_id}/runs/stream` does. The run's thread
    is deleted once the run has finished, unless `on_completion` is `"keep"`.
    """
    run = await prepare_stateless_run(session, request, authorize.user, authorize=authorize)
    return stream_response(run, cancel_on_disconnect=request.on_disconnect == "cancel")


@router.post("/runs/batch", response_model=list[Run], responses={**NOT_FOUND, **CONFLICT})
async def stateless_create_run_batch(
    requests: Annotated[list[RunCreate], Body(min_length=1)],
    session: SessionDep,
    authorize: Authorization = Authorize("threads", "create_run"),
) -> list[Run]:
    """Create a batch of stateless background runs.

    Runs are created sequentially; a failure aborts the batch and the runs
    already created keep executing.
    """
    return [await prepare_stateless_run(session, request, authorize.user, authorize=authorize) for request in requests]


@router.post("/runs", response_model=Run, responses={**NOT_FOUND, **CONFLICT})
async def stateless_create_run(
    request: RunCreate,
    session: SessionDep,
    response: Response,
    authorize: Authorization = Authorize("threads", "create_run"),
) -> Run:
    """Create a stateless background run.

    The run's thread is deleted once the run has finished, unless
    `on_completion` is `"keep"`.
    """
    run = await prepare_stateless_run(session, request, authorize.user, authorize=authorize)
    response.headers.update(run_location(run))
    return run
