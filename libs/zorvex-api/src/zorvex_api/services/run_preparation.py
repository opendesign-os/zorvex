"""Creating runs: authorization, binding the thread, persisting the run and submitting it."""

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

import structlog
from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from langgraph_sdk import PriorityContext
from langsmith import utils as ls_utils
from sqlalchemy import ColumnElement, case, func, literal_column, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.config import load_config
from zorvex_api.core.authorization import Authorization, Filters
from zorvex_api.core.orm import Assistant as AssistantORM
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models import Run, RunCreate, User
from zorvex_api.models.auth import as_base_user
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES, Durability
from zorvex_api.models.run_job import RunIdentity, RunJob, RunKwargs
from zorvex_api.observability.tracing import trace_markers
from zorvex_api.services.executor import executor
from zorvex_api.services.langgraph_service import get_langgraph_service
from zorvex_api.services.ownership import existing_thread, owned_assistant
from zorvex_api.services.priority import ThreadPriority, assign, callback
from zorvex_api.services.run_cleanup import request_run_interruption
from zorvex_api.services.thread_deletion import delete_thread_by_id
from zorvex_api.services.webhook_policy import admit_webhook
from zorvex_api.utils.headers import request_headers
from zorvex_api.utils.jsonb import jsonb_patch, jsonb_shallow_merge
from zorvex_api.utils.run_utils import (
    CONTEXT_PROVIDED,
    CRON_ID,
    EVENT_STREAMING_V2,
    LANGSMITH_EXAMPLE_ID,
    LANGSMITH_PROJECT,
    SERVER_KEYS,
    merge_jsonb,
    merge_run_config,
    stream_modes,
    strip_pinned_config_keys,
)

logger = structlog.getLogger(__name__)


# The interrupt reaches the client (via the broker/SSE) before the run executor
# commits thread_status="interrupted" in finalize_run. A client that resumes the
# instant it sees the interrupt can beat that commit, so after the first read poll
# a few times on FRESH sessions (each a new snapshot that sees the commit) before
# rejecting — otherwise a valid resume races to a spurious 400.
_RESUME_SETTLE_ATTEMPTS = 10
_RESUME_SETTLE_INTERVAL_SECONDS = 0.1


_BOTH_CONFIGURABLE_AND_CONTEXT = (
    "Cannot specify both configurable and context. Prefer setting context alone. Context was introduced in "
    "LangGraph 0.6.0 and is the long term planned replacement for configurable."
)


async def _validate_resume_command(thread_id: str, thread: ThreadORM | None, command: dict[str, Any] | None) -> None:
    """Reject a resume against a thread that is not interrupted.

    Guard on the presence of the ``resume`` key, not a non-None value: ``None``
    is a valid resume payload, and a ``{"resume": None}`` command must still be
    rejected against a thread that is not interrupted.
    """
    if not command or "resume" not in command:
        return
    if thread is None:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    if thread.status == "interrupted":
        return

    # Not interrupted on the request session's snapshot — poll fresh sessions in
    # case finalize_run's commit is still in flight.
    status = select(ThreadORM.status).where(ThreadORM.thread_id == thread_id)
    maker = get_session_maker()
    for _ in range(_RESUME_SETTLE_ATTEMPTS):
        await asyncio.sleep(_RESUME_SETTLE_INTERVAL_SECONDS)
        async with maker() as fresh:
            if await fresh.scalar(status) == "interrupted":
                return

    raise HTTPException(400, "Cannot resume: thread is not in interrupted state")


_THREAD_NAME_MAX_LENGTH = 100
_HUMAN_ROLES = frozenset({"human", "user"})


def _content_text(content: Any) -> str:
    """The text of a message's content: a string, or the text blocks of a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return ""


def _thread_name(input_data: dict[str, Any] | None) -> str:
    """The first human message of the run input, shortened; empty when there is none.

    Messages name their sender in ``role`` (chat format) or ``type`` (LangChain
    format), and their content may be a string or a list of blocks.
    """
    messages = (input_data or {}).get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("type")
        if not isinstance(role, str) or role not in _HUMAN_ROLES:
            continue
        name = _content_text(message.get("content")).strip()
        if name:
            if len(name) > _THREAD_NAME_MAX_LENGTH:
                return name[:_THREAD_NAME_MAX_LENGTH].rsplit(" ", 1)[0] + "..."
            return name
    return ""


async def _bind_thread(
    session: AsyncSession,
    thread: ThreadORM | None,
    thread_id: str,
    assistant: AssistantORM,
    user: User,
    request: RunCreate,
) -> None:
    """Mark the thread busy running *assistant*, creating it when missing; does not commit.

    A thread the run creates starts with the run's metadata, so what an
    ``@auth.on.threads.create_run`` handler adds there reaches the thread too.
    A thread without a name takes the first human message of the input. The
    metadata update merges in the database, so a PATCH /threads/{id} racing
    this run cannot drop the keys written here, nor have its own dropped.

    The thread's config becomes the one its graph is built with to read and
    update its state, as the Agent Server sets it: the assistant's config and
    the run's, with only the assistant's configurable, the run's own staying
    with the run.
    """
    binding = {"assistant_id": assistant.assistant_id, "graph_id": assistant.graph_id}
    assistant_config = assistant.config or {}
    config = {
        **assistant_config,
        **(request.config or {}),
        "configurable": dict(assistant_config.get("configurable") or {}),
    }
    thread_name = _thread_name(request.input)
    if thread is None:
        session.add(
            ThreadORM(
                thread_id=thread_id,
                status="busy",
                metadata_dict={**(request.metadata or {}), **binding, "thread_name": thread_name},
                config=config,
                user_id=user.identity,
            )
        )
        # Inserted now: no relationship() tells the flush to write it before the run that references it.
        await session.flush()
        return

    patches: list[ColumnElement[Any]] = [jsonb_patch(binding, "metadata_patch")]
    if thread_name:
        # Only name a thread that has no name yet. The condition is evaluated at
        # write time against the row the UPDATE locks, not against an earlier read.
        patches.append(
            case(
                (
                    func.coalesce(ThreadORM.metadata_dict["thread_name"].astext, "") == "",
                    jsonb_patch({"thread_name": thread_name}, "thread_name_patch"),
                ),
                else_=literal_column("'{}'::jsonb"),
            )
        )
    await session.execute(
        update(ThreadORM)
        .where(ThreadORM.thread_id == thread_id)
        .values(
            metadata_dict=jsonb_shallow_merge(ThreadORM.metadata_dict, *patches),
            config=config,
            status="busy",
            updated_at=datetime.now(UTC),
        )
        .execution_options(synchronize_session=False)
    )


def _resolve_checkpoint(request: RunCreate) -> dict[str, Any] | None:
    """Fold the top-level ``checkpoint_id`` into ``checkpoint``; ``checkpoint`` keys win."""
    if request.checkpoint_id is None:
        return request.checkpoint
    return {"checkpoint_id": str(request.checkpoint_id), **(request.checkpoint or {})}


# RunCreate fields a run's ``kwargs`` carry; ``stream_subgraphs`` goes as ``subgraphs``, as Run.kwargs shows it.
_KWARGS_FIELDS = frozenset(
    {
        "input",
        "command",
        "config",
        "context",
        "checkpoint",
        "checkpoint_id",
        "stream_mode",
        "interrupt_before",
        "interrupt_after",
        "webhook",
        "durability",
    }
)
# Top-level RunsCreate keys a handler may change; the ids and the status only say what is authorized.
_OPTION_FIELDS = ("metadata", "multitask_strategy", "if_not_exists", "after_seconds")


async def _authorize_run(
    authorize: Authorization, thread_id: str, run_id: str, request: RunCreate
) -> tuple[RunCreate, Filters | None]:
    """Run the ``threads.create_run`` handler on the run, shaped as the SDK's ``RunsCreate``.

    Returns the run as the handler left its metadata, options and ``kwargs``,
    and the handler's filter.
    """
    value: dict[str, Any] = {
        "thread_id": thread_id,
        "assistant_id": request.assistant_id,
        "run_id": run_id,
        "status": "pending",
        "metadata": dict(request.metadata or {}),
        "multitask_strategy": request.multitask_strategy,
        "if_not_exists": request.if_not_exists,
        "after_seconds": request.after_seconds,
        "kwargs": {
            **request.model_dump(mode="json", include=set(_KWARGS_FIELDS)),
            "subgraphs": request.stream_subgraphs,
        },
    }
    filters = await authorize(value)
    kwargs = value["kwargs"]
    run = RunCreate.model_validate(
        {
            **request.model_dump(),
            **{key: kwargs[key] for key in _KWARGS_FIELDS if key in kwargs},
            "stream_subgraphs": kwargs.get("subgraphs", False),
            **{key: value[key] for key in _OPTION_FIELDS},
        }
    )
    return run, filters


@dataclass(frozen=True)
class Launch:
    """How the server launches a run, beyond what the request asks: options no client sets.

    *run_id* names the run, for a caller that must find it again; *temporary*
    marks a run whose thread goes once it has finished; *protocol* ``v2``
    streams the run as Agent Protocol v2 events instead of run stream modes;
    *cron_id* names the cron that started it, which its configurable carries.
    """

    run_id: str | None = None
    temporary: bool = False
    protocol: Literal["v1", "v2"] = "v1"
    cron_id: str | None = None


def _markers(request: RunCreate, launch: Launch) -> dict[str, Any]:
    """The server markers the run's configurable carries: its v2 stream, the LangSmith project it is traced to."""
    markers: dict[str, Any] = {EVENT_STREAMING_V2: True} if launch.protocol == "v2" else {}
    if launch.cron_id is not None:
        markers[CRON_ID] = launch.cron_id
    markers.update(trace_markers())
    tracer = request.langsmith_tracer
    if tracer and tracer.project_name:
        markers[LANGSMITH_PROJECT] = tracer.project_name
    if tracer and tracer.example_id:
        markers[LANGSMITH_EXAMPLE_ID] = tracer.example_id
    return markers


def _durability(request: RunCreate) -> Durability | None:
    """The run's durability, else the deprecated checkpoint_during's (true is async, false exit), else the server's."""
    if request.durability:
        return request.durability
    if request.checkpoint_during is not None:
        return "async" if request.checkpoint_during else "exit"
    return load_config().checkpointer.durability


def _langsmith_project(request: RunCreate) -> str | None:
    """The LangSmith project the run is traced to: its tracer's, else the default; None while tracing is off."""
    if not ls_utils.tracing_is_enabled():
        return None
    tracer = request.langsmith_tracer
    return (tracer.project_name if tracer else None) or ls_utils.get_tracer_project()


async def prepare_run(
    session: AsyncSession,
    thread_id: str,
    request: RunCreate,
    user: User,
    *,
    authorize: Authorization | None,
    launch: Launch = Launch(),
) -> Run:
    """Authorize *request*, persist its run on the thread and submit it; returns the pending run.

    *authorize* is the ``threads.create_run`` event the run answers to. Its
    filter applies to the thread, and creating the run also reads its assistant,
    so ``assistants.read`` applies to that. None is for a cron firing: the cron
    was authorized and its webhook vetted when it was saved, and the webhook is
    re-checked before each call, so a later policy change skips the call instead
    of blocking the run.

    A missing thread is created unless ``if_not_exists`` is ``"reject"``, and
    another user's thread reads as missing. A run of a *launch* ``run_id``
    already made answers 409.

    The runs already active on the thread are met as the Agent Server meets
    them, by ``multitask_strategy``: ``reject`` answers 409, ``interrupt`` and
    ``rollback`` cancel them once this run is in, and ``enqueue`` (the default)
    lets this run wait its turn. A run with ``after_seconds`` falls due then,
    its ``created_at``: it starts no sooner, and runs in line from there.
    """
    run_id = launch.run_id or str(uuid4())
    # Checked as the caller sent them, before a handler adds its own keys.
    if ((request.config or {}).get("configurable") or {}) and request.context:
        raise HTTPException(400, _BOTH_CONFIGURABLE_AND_CONTEXT)
    if launch.protocol != "v2" and "lifecycle" in stream_modes(request.stream_mode):
        raise HTTPException(
            422, "stream_mode entries ['lifecycle'] are only supported on the v2 event streaming endpoints."
        )
    filters: Filters | None = None
    if authorize is not None:
        request, filters = await _authorize_run(authorize, thread_id, run_id, request)
        request = await admit_webhook(request)

    thread = await existing_thread(session, thread_id, user, filters=filters)
    if thread is None and request.if_not_exists == "reject":
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    await _validate_resume_command(thread_id, thread, request.command)

    config = request.config or {}
    supplied = config.get("configurable", {})
    if not isinstance(supplied, dict):
        raise HTTPException(status_code=422, detail="`config.configurable` must be a mapping")
    # The server's markers are the server's to set (_markers), whatever the client sent.
    configurable = {key: value for key, value in supplied.items() if key not in SERVER_KEYS}
    # Each mirrors the other, as on the Agent Server, under what a handler added; the marker
    # lets the run drop from the mirror what the graph's context schema leaves out.
    if request.context:
        context = request.context
        mirrored = {key: value for key, value in context.items() if key not in SERVER_KEYS}
        configurable = {**mirrored, **configurable, CONTEXT_PROVIDED: True}
    else:
        context = configurable.copy()
    # The checkpoint to start from rides in configurable, as the Agent Server keeps it; the ids are the server's.
    checkpoint = strip_pinned_config_keys(_resolve_checkpoint(request) or {})
    configurable = {
        **configurable,
        **(request_headers.get() or {}),
        **{key: value for key, value in checkpoint.items() if value is not None},
        **_markers(request, launch),
    }
    config = {**config, "configurable": configurable}

    graphs = get_langgraph_service()
    assistant_id = graphs.assistant_id(request.assistant_id)
    assistant_filters: Filters | None = None
    if authorize is not None:
        assistant_filters = await authorize.to("assistants", "read")({"assistant_id": assistant_id})
    assistant = await owned_assistant(session, assistant_id, user, filters=assistant_filters)
    graphs.require_graph(assistant.graph_id)

    strategy = request.multitask_strategy or "enqueue"
    active: list[RunORM] = []
    if thread is not None:
        # The thread stays locked until the run is in, so two runs cannot both find it free.
        await session.execute(select(ThreadORM.thread_id).where(ThreadORM.thread_id == thread_id).with_for_update())
        active = list(
            (
                await session.scalars(
                    select(RunORM).where(RunORM.thread_id == thread_id, RunORM.status.in_(ACTIVE_RUN_STATUSES))
                )
            ).all()
        )
    if active and strategy == "reject":
        raise HTTPException(
            409, "Thread is already running a task. Wait for it to finish or choose a different multitask strategy."
        )

    # A thread gets its priority settings at its first run, from the priority callback, and its runs take them;
    # without a callback every run has priority 0. An existing thread is locked above, so it is asked once.
    priority = 0
    assigned: ThreadPriority | None = None
    if callback() is not None:
        if thread is not None and thread.run_priority is not None:
            priority = thread.run_priority
        else:
            binding = {"assistant_id": assistant.assistant_id, "graph_id": assistant.graph_id}
            assigned = await assign(
                PriorityContext(
                    user=as_base_user(user),
                    permissions=list(user.permissions),
                    thread_id=thread_id,
                    thread_metadata={**((thread.metadata_dict if thread else request.metadata) or {}), **binding},
                    assistant_id=assistant.assistant_id,
                    graph_id=assistant.graph_id,
                    assistant_metadata=assistant.metadata_dict or {},
                    run_metadata=request.metadata or {},
                    cron_id=launch.cron_id,
                )
            )
            priority = assigned.priority
    await _bind_thread(session, thread, thread_id, assistant, user, request)
    if assigned is not None:
        await session.execute(
            update(ThreadORM)
            .where(ThreadORM.thread_id == thread_id)
            .values(run_priority=assigned.priority, fairness_group=assigned.fairness_group)
        )

    now = datetime.now(UTC)
    due = now + timedelta(seconds=request.after_seconds or 0)
    job = RunJob(
        identity=RunIdentity(
            run_id=run_id,
            thread_id=thread_id,
            graph_id=assistant.graph_id,
            assistant_id=assistant_id,
            created_at=due,
        ),
        user=user,
        kwargs=RunKwargs(
            input=request.input,  # None continues from the checkpoint
            command=request.command,
            config=merge_run_config(assistant.config, config),
            context=merge_jsonb(assistant.config.get("configurable"), context),
            stream_mode=request.stream_mode,
            interrupt_before=request.interrupt_before,
            interrupt_after=request.interrupt_after,
            webhook=request.webhook or None,
            feedback_keys=request.feedback_keys,
            temporary=launch.temporary,
            subgraphs=request.stream_subgraphs,
            # A v2 run always is, as on the Agent Server: its thread's event stream replays it.
            resumable=request.stream_resumable or launch.protocol == "v2",
            checkpoint_during=request.checkpoint_during is not False,
            durability=_durability(request),
        ),
        multitask_strategy=strategy,
        priority=priority,
        langsmith_session_name=_langsmith_project(request),
        run_metadata=request.metadata or {},
        correlation_id=correlation_id.get(),
    )

    run_orm = RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=assistant_id,
        status="pending",
        user_id=user.identity,
        created_at=due,
        updated_at=now,
        priority=job.priority,
        execution_params=job.to_execution_params(),
    )
    session.add(run_orm)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        if await session.get(RunORM, run_id) is None:
            raise
        # The caller named a run that exists: it was made already, by an earlier attempt.
        raise HTTPException(409, f"Run '{run_id}' already exists") from None

    await executor.submit(job)
    logger.info("Submitted run", run_id=run_id, thread_id=thread_id, user_id=user.identity, multitask=strategy)
    if active and strategy in ("interrupt", "rollback"):
        reason = f"superseded by run {run_id} (multitask_strategy={strategy})"
        for superseded in active:
            await request_run_interruption(session, superseded, action=strategy, reason=reason)
    return Run.model_validate(run_orm)


async def prepare_stateless_run(
    session: AsyncSession,
    request: RunCreate,
    user: User,
    *,
    authorize: Authorization | None,
    launch: Launch = Launch(),
) -> Run:
    """Run *request* on a new thread of its own, which is deleted again when the run cannot be set up."""
    if request.command and request.command.get("resume"):
        raise HTTPException(400, "You must provide a thread_id when resuming.")
    thread_id = str(uuid4())
    request.if_not_exists = "create"
    try:
        launch = replace(launch, temporary=request.on_completion != "keep")
        return await prepare_run(session, thread_id, request, user, authorize=authorize, launch=launch)
    except Exception:
        await delete_thread_by_id(thread_id, user.identity)
        raise


async def create_run(
    authorize: Authorization, thread_id: str | None, request: RunCreate, *, launch: Launch | None = None
) -> Run:
    """Start a run as ``POST /threads/{thread_id}/runs`` does, or without a thread as ``POST /runs`` does.

    *authorize* is its caller's ``threads.create_run``; the run is pending when this returns.
    """
    launch = launch or Launch()
    async with get_session_maker()() as session:
        if thread_id is None:
            return await prepare_stateless_run(session, request, authorize.user, authorize=authorize, launch=launch)
        return await prepare_run(session, thread_id, request, authorize.user, authorize=authorize, launch=launch)
