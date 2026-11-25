"""Tool calls over MCP, as the Agent Server answers them: the run's values, or its pause for human input.

A call runs the assistant and answers its final values. How a pause is
answered depends on the client (``call_tool``):

- before protocol 2026-07-28 nothing can carry the questions: the call runs
  on a throwaway thread, and a pause is reported, with its interrupts, as an
  error result;
- a 2026-07-28 client without elicitation is told the tool needs it: error
  -32021, naming the modes the questions take;
- a 2026-07-28 client with elicitation is asked (``ElicitingCall``): the call
  runs on a thread of its own, and a pause answers ``input_required`` with the
  round's requests (mcp_elicitation.py) and a ``requestState``. The client
  calls again with the answers and that state; the answers resume the run,
  and the call goes on until the run finishes, pauses again, or the user
  declines.

A ``requestState`` is ``{thread_id}.{token}``. The thread holds the call's
binding under ``mcp_call`` in its metadata (a digest of the call, a verifier of
the token, a digest of the round asked), so any replica can check a retry
without a shared key. The run a round's answers start has an id derived from
the token, so retrying the same answers joins that run rather than starting
another, and each round's token is derived from the last. A paused call's
thread lives a day; once the call ends it is deleted, or, when the end came on
a retry, left to expire within ten minutes.

The first section, what a run came to and the threads and runs its caller
reads and starts, serves the A2A server (a2a_service.py) too.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import UUID, uuid5

import anyio
import mcp_types
import structlog
from fastapi import HTTPException
from mcp.server.context import ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.models import ThreadCreate, ThreadTTLSpec, ThreadUpdate, User
from zorvex_api.models.runs import RunCreate
from zorvex_api.services.mcp_elicitation import EndCall, Mode, Refusal, Round, UnaskableInterrupt, digest
from zorvex_api.services.ownership import authorized_run, authorized_thread
from zorvex_api.services.run_cleanup import delete_thread_after_run
from zorvex_api.services.run_preparation import Launch, create_run
from zorvex_api.services.run_waiters import (
    Completed,
    Failed,
    Paused,
    RunOutcome,
    public_interrupt,
    run_outcome,
    wait_for_result,
)
from zorvex_api.services.thread_service import ThreadService
from zorvex_api.services.thread_state_service import read_thread_values
from zorvex_api.services.thread_ttl import get_thread_ttl_config

logger = structlog.getLogger(__name__)

PAUSED_TOOL_MESSAGE = "This tool paused to ask for human input, and this client can't answer it."
UNRESUMABLE_PAUSE_MESSAGE = "This tool paused to ask for human input, and this call can't be resumed."
RUN_FAILED_MESSAGE = "The tool's run failed."
BREAKPOINT_MESSAGE = (
    "This tool stopped at a breakpoint (interrupt_before or interrupt_after), which can't be resumed over MCP."
)
INVALID_REQUEST_STATE = "Invalid or expired requestState"
PAUSED_CALL_TTL_MINUTES = 24 * 60
FINISHED_CALL_TTL_MINUTES = 10
BINDING_KEY = "mcp_call"

_TOKEN_BYTES = 32
_STORED_LABEL = b"stored"
_NEXT_ROUND_LABEL = b"next-round"


# --- a call's results


class CallToolParams(mcp_types.CallToolRequestParams):
    """``tools/call`` params, with the run's ``context`` the Agent Server also accepts."""

    context: dict[str, Any] | None = None

    @property
    def input(self) -> dict[str, Any]:
        return self.arguments or {}

    @property
    def answers(self) -> Mapping[str, mcp_types.InputResponse]:
        return self.input_responses or {}

    def digest(self, assistant_id: str) -> str:
        """What the call is: the assistant, the tool name and the arguments."""
        return digest(assistant_id, self.name, self.input)


ToolResult = mcp_types.CallToolResult | mcp_types.InputRequiredResult
# What a call's run came to: the questions its pause asks, or the call's result.
Step = Paused | mcp_types.CallToolResult


def _step(outcome: RunOutcome) -> Step:
    match outcome:
        case Paused():
            return outcome
        case Failed(error):
            return _error_result(_error_message(error))
        case Completed(values):
            return mcp_types.CallToolResult(content=[_text(_json(values))])


def _error_result(message: str) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(content=[_text(message)], is_error=True)


def _reported(step: Step, notice: str) -> mcp_types.CallToolResult:
    """The call's result, a pause reported as an error: *notice*, then the interrupts as JSON."""
    if isinstance(step, Paused):
        return mcp_types.CallToolResult(content=[_text(notice), _text(_json(list(step.interrupts)))], is_error=True)
    return step


def _error_message(error: Any) -> str:
    if isinstance(error, dict) and isinstance(error.get("error"), str):
        message = error.get("message")
        return f"{error['error']}: {message}" if isinstance(message, str) and message else error["error"]
    return RUN_FAILED_MESSAGE


def _json(value: Any) -> str:
    return json.dumps(to_jsonable(value), separators=(",", ":"), ensure_ascii=False)


def _text(text: str) -> mcp_types.TextContent:
    return mcp_types.TextContent(type="text", text=text)


async def run_stateless(user: User, assistant_id: str, params: CallToolParams) -> Step:
    """Run the assistant on a throwaway thread and answer what it came to, as ``POST /runs/wait`` does."""
    request = RunCreate(assistant_id=assistant_id, input=params.input, context=params.context)
    run = await create_run(Authorization(user, "threads", "create_run"), None, request)
    try:
        return _step(run_outcome(await wait_for_result(run.run_id, run.thread_id, user)))
    finally:
        # The result is read from the thread, so the thread goes only after it.
        delete_thread_after_run(run, user.identity)


# --- a call's state across rounds


class InvalidRequestState(MCPError):
    def __init__(self) -> None:
        super().__init__(mcp_types.INVALID_PARAMS, INVALID_REQUEST_STATE)


@dataclass(frozen=True, slots=True)
class RequestState:
    """A paused call's ``requestState``: its thread and the round's secret token."""

    thread_id: UUID
    token: str = field(repr=False)

    @classmethod
    def new(cls, thread_id: str) -> RequestState:
        return cls(UUID(thread_id), secrets.token_urlsafe(_TOKEN_BYTES))

    @classmethod
    def parse(cls, raw: str) -> RequestState:
        thread_id, _, token = raw.partition(".")
        try:
            parsed = UUID(thread_id)
        except ValueError:
            raise InvalidRequestState from None
        if not token:
            raise InvalidRequestState
        return cls(parsed, token)

    def __str__(self) -> str:
        return f"{self.thread_id}.{self.token}"

    @property
    def verifier(self) -> str:
        """What the thread keeps of the token: enough to check it, not to forge it."""
        return self._derive(_STORED_LABEL)

    @property
    def run_id(self) -> UUID:
        """The id of the run the round's answers start."""
        return uuid5(self.thread_id, self.verifier)

    def next(self) -> RequestState:
        """The state of the round after this one."""
        return replace(self, token=self._derive(_NEXT_ROUND_LABEL))

    def _derive(self, label: bytes) -> str:
        return hmac.new(self.token.encode(), label, hashlib.sha256).hexdigest()


class Binding(BaseModel):
    """What a call's thread records of it: which call, which round's token, which questions; or how it ended."""

    model_config = ConfigDict(frozen=True, strict=True)

    call: str
    state: str
    asked: str
    ended: Refusal | None = None

    @classmethod
    def of(cls, call: str, state: RequestState, asked: Round, ended: Refusal | None = None) -> Binding:
        return cls(call=call, state=state.verifier, asked=asked.digest, ended=ended)

    @classmethod
    def stored(cls, metadata: Mapping[str, Any] | None) -> Binding | None:
        try:
            return cls.model_validate(metadata.get(BINDING_KEY) if metadata else None)
        except PydanticValidationError:
            return None

    @classmethod
    def verified(cls, metadata: Mapping[str, Any] | None, call: str, state: RequestState) -> Binding:
        """The binding *state* answers to, as this round's or the last one's; InvalidRequestState otherwise."""
        binding = cls.stored(metadata)
        if binding is None or not (binding.admits(call, state) or binding.admits(call, state.next())):
            raise InvalidRequestState
        return binding

    def admits(self, call: str, state: RequestState) -> bool:
        return self.call == call and hmac.compare_digest(self.state.encode(), state.verifier.encode())

    @property
    def metadata(self) -> dict[str, Any]:
        return {BINDING_KEY: self.model_dump(exclude_none=True)}


def _within_operator_ttl(minutes: float) -> float:
    """*minutes*, or the server's default thread ttl when that is shorter."""
    config = get_thread_ttl_config()
    return minutes if config is None else min(minutes, config.default_ttl)


@contextlib.contextmanager
def _invalid_state_if_missing() -> Generator[None]:
    try:
        yield
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        raise InvalidRequestState from None


@dataclass(frozen=True, slots=True)
class CallThread:
    """The thread one paused call runs on, operated as its caller, as the REST API would."""

    user: User
    assistant_id: str
    thread_id: str

    @classmethod
    async def create(cls, user: User, assistant_id: str) -> CallThread:
        request = ThreadCreate(ttl=ThreadTTLSpec(ttl=_within_operator_ttl(PAUSED_CALL_TTL_MINUTES)))
        async with get_session_maker()() as session:
            thread = await ThreadService(session, user).create(Authorization(user, "threads", "create"), request)
        return cls(user, assistant_id, thread.thread_id)

    async def get(self) -> ThreadORM:
        with _invalid_state_if_missing():
            return await authorized_thread(Authorization(self.user, "threads", "read"), self.thread_id)

    async def pending(self, thread: ThreadORM) -> list[dict[str, Any]]:
        """The interrupts the thread waits on, in task order; none unless it is interrupted."""
        if thread.status != "interrupted":
            return []
        _, interrupts = await read_thread_values(thread, self.user)
        return [interrupt for task in interrupts.values() for interrupt in task]

    async def bind(self, binding: Binding, replacing: Binding | None) -> None:
        """Record *binding*, unless another retry already moved the call past *replacing*."""
        if binding == replacing:
            return
        if replacing is not None and await self._bound() not in (replacing, binding):
            return
        await self._update(binding.metadata, PAUSED_CALL_TTL_MINUTES)

    async def run(self, params: CallToolParams) -> Step:
        request = RunCreate(
            assistant_id=self.assistant_id, input=params.input, context=params.context, durability="exit"
        )
        run = await create_run(Authorization(self.user, "threads", "create_run"), self.thread_id, request)
        return await self._outcome(await wait_for_result(run.run_id, self.thread_id, self.user))

    async def resume(self, value: Any, state: RequestState, context: dict[str, Any] | None) -> Step:
        """Resume the run with the round's answers; a retry of the same answers joins the run they started."""
        request = RunCreate(
            assistant_id=self.assistant_id,
            command={"resume": value},
            context=context,
            multitask_strategy="reject",
            if_not_exists="reject",
            durability="exit",
        )
        authorize = Authorization(self.user, "threads", "create_run")
        try:
            await create_run(authorize, self.thread_id, request, launch=Launch(run_id=str(state.run_id)))
        except HTTPException as exc:
            if exc.status_code == 404:
                raise InvalidRequestState from None
            if exc.status_code != 409:
                raise
            return await self.join(state)
        return await self._wait_for(state)

    async def join(self, state: RequestState) -> Step:
        if not await self.resumed_with(state):
            raise InvalidRequestState
        return await self._wait_for(state)

    async def resumed_with(self, state: RequestState) -> bool:
        """Whether the round's answers already started their run."""
        try:
            await authorized_run(Authorization(self.user, "threads", "read"), self.thread_id, str(state.run_id))
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            return False
        return True

    async def delete(self) -> None:
        async with self._best_effort("Could not delete a finished MCP call's thread"), get_session_maker()() as session:
            await ThreadService(session, self.user).delete(
                Authorization(self.user, "threads", "delete"), self.thread_id
            )

    async def expire_soon(self) -> None:
        async with self._best_effort("Could not shorten a finished MCP call's TTL"):
            await self._update({}, FINISHED_CALL_TTL_MINUTES)

    async def _update(self, metadata: dict[str, Any], ttl_minutes: float) -> None:
        request = ThreadUpdate(metadata=metadata, ttl=ThreadTTLSpec(ttl=_within_operator_ttl(ttl_minutes)))
        async with get_session_maker()() as session:
            await ThreadService(session, self.user).update(
                Authorization(self.user, "threads", "update"), self.thread_id, request
            )

    async def _bound(self) -> Binding | None:
        return Binding.stored((await self.get()).metadata_dict)

    @contextlib.asynccontextmanager
    async def _best_effort(self, failure: str) -> AsyncGenerator[None]:
        try:
            yield
        except HTTPException as exc:
            if exc.status_code != 404:
                logger.warning(failure, thread_id=self.thread_id, status_code=exc.status_code)

    async def _wait_for(self, state: RequestState) -> Step:
        return await self._outcome(await wait_for_result(str(state.run_id), self.thread_id, self.user))

    async def _outcome(self, result: dict[str, Any]) -> Step:
        """What the run came to, read from the thread once it is done: a pause asks the thread's own interrupts."""
        outcome = run_outcome(result)
        if isinstance(outcome, Failed):
            return _step(outcome)
        thread = await self.get()
        if thread.status == "busy":
            raise InvalidRequestState
        if thread.status == "interrupted":
            if interrupts := await self.pending(thread):
                return Paused(tuple(public_interrupt(item) for item in interrupts))
            return _error_result(BREAKPOINT_MESSAGE)
        return _step(outcome)


# --- how a pause is answered


class ElicitationRequired(MCPError):
    """The tool asks questions in modes the client did not declare: -32021, naming them."""

    def __init__(self, modes: frozenset[Mode]) -> None:
        elicitation = mcp_types.ElicitationCapability(
            form=mcp_types.FormElicitationCapability() if "form" in modes else None,
            url=mcp_types.UrlElicitationCapability() if "url" in modes else None,
        )
        data = mcp_types.MissingRequiredClientCapabilityErrorData(
            required_capabilities=mcp_types.ClientCapabilities(elicitation=elicitation)
        )
        super().__init__(
            mcp_types.MISSING_REQUIRED_CLIENT_CAPABILITY,
            f"This tool asks for human input and needs {' and '.join(sorted(modes))} elicitation.",
            data.model_dump(by_alias=True, mode="json", exclude_none=True),
        )


def elicitation_modes(capabilities: mcp_types.ClientCapabilities | None) -> frozenset[Mode]:
    """The elicitation modes a client declared; an elicitation capability naming none is the older form-only one."""
    elicitation = capabilities.elicitation if capabilities else None
    if elicitation is None:
        return frozenset()
    declared: set[Mode] = set()
    if elicitation.form is not None:
        declared.add("form")
    if elicitation.url is not None:
        declared.add("url")
    return frozenset(declared) or frozenset({"form"})


async def call_tool(
    context: ServerRequestContext[Any], user: User, assistant_id: str, params: CallToolParams
) -> ToolResult:
    """Run the assistant as a tool for *user*; a pause is answered as the client can take it (module doc)."""
    if context.protocol_version not in MODERN_PROTOCOL_VERSIONS:
        # Nothing carries questions to a client before 2026-07-28, nor a retry from it.
        if params.request_state is not None:
            raise InvalidRequestState
        return _reported(await run_stateless(user, assistant_id, params), PAUSED_TOOL_MESSAGE)
    modes = elicitation_modes(context.session.client_capabilities)
    if params.request_state is not None:
        state = RequestState.parse(params.request_state)
        thread = CallThread(user, assistant_id, str(state.thread_id))
        return await ElicitingCall(thread, params, state, modes, release=thread.expire_soon).retry()
    if not modes:
        return await _require_elicitation(user, assistant_id, params)
    try:
        thread = await CallThread.create(user, assistant_id)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise
        # Not allowed a thread to pause on: the call can only report a pause.
        return _reported(await run_stateless(user, assistant_id, params), UNRESUMABLE_PAUSE_MESSAGE)
    state = RequestState.new(thread.thread_id)
    return await ElicitingCall(thread, params, state, modes, release=thread.delete).first()


async def _require_elicitation(user: User, assistant_id: str, params: CallToolParams) -> ToolResult:
    """Run once; a pause the client could answer with elicitation is -32021, naming the modes it needs."""
    step = await run_stateless(user, assistant_id, params)
    if not isinstance(step, Paused):
        return step
    try:
        needed = Round.of(step.interrupts).modes
    except UnaskableInterrupt as error:
        return _error_result(error.message)
    raise ElicitationRequired(needed)


@dataclass(frozen=True, slots=True)
class ElicitingCall:
    """One request of a call that asks: its first, or a retry carrying a round's answers."""

    thread: CallThread
    params: CallToolParams
    state: RequestState
    modes: frozenset[Mode]
    release: Callable[[], Awaitable[None]]
    verified: Binding | None = None
    # Whether the request's answers are for this round; a round the call moves on to has none yet.
    answering: bool = True

    @property
    def answers(self) -> Mapping[str, mcp_types.InputResponse]:
        return self.params.answers if self.answering else {}

    async def first(self) -> ToolResult:
        try:
            return await self._settle(await self.thread.run(self.params))
        except UnaskableInterrupt as error:
            return await self._finish(_error_result(error.message))
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self.release()
            raise

    async def retry(self) -> ToolResult:
        snapshot = await self.thread.get()
        binding = Binding.verified(snapshot.metadata_dict, self._call_digest, self.state)
        if binding.ended is not None:
            return _error_result(EndCall(binding.ended).message)
        call = replace(self, verified=binding)
        try:
            if binding.admits(self._call_digest, self.state):
                return await call._continue(snapshot, binding)
            # A retry of the last round, which the call has moved past: carry on from the round it is at.
            return await call._next()._continue(snapshot, binding)
        except ElicitationRequired:
            await self.release()
            raise

    async def _continue(self, snapshot: ThreadORM, binding: Binding) -> ToolResult:
        try:
            if interrupts := await self.thread.pending(snapshot):
                pending = Round.of(interrupts)
                if pending.digest == binding.asked:
                    return await self._answer(pending)
                if not await self.thread.resumed_with(self.state):
                    # The thread asks something this round did not: ask that instead.
                    return await replace(self, answering=False)._answer(pending)
            return await self._next()._settle(await self.thread.join(self.state))
        except UnaskableInterrupt as error:
            return await self._finish(_error_result(error.message))

    async def _settle(self, step: Step) -> ToolResult:
        if isinstance(step, Paused):
            return await self._answer(Round.of(step.interrupts))
        return await self._finish(step)

    async def _answer(self, pending: Round) -> ToolResult:
        if not pending.modes <= self.modes:
            raise ElicitationRequired(pending.modes)
        if not pending.answered_by(self.answers):
            return await self._ask(pending)
        reply = pending.reply(self.answers)
        if isinstance(reply, EndCall):
            ending = Binding.of(self._call_digest, self.state, pending, reply.refusal)
            await self.thread.bind(ending, self.verified)
            return await self._finish(_error_result(reply.message))
        step = await self.thread.resume(reply.value, self.state, self.params.context)
        return await self._next()._settle(step)

    async def _ask(self, pending: Round) -> ToolResult:
        try:
            await self.thread.bind(Binding.of(self._call_digest, self.state, pending), self.verified)
        except HTTPException as exc:
            if exc.status_code != 403:
                raise
            return await self._finish(_error_result(UNRESUMABLE_PAUSE_MESSAGE))
        return mcp_types.InputRequiredResult(input_requests=pending.input_requests, request_state=str(self.state))

    async def _finish(self, result: mcp_types.CallToolResult) -> mcp_types.CallToolResult:
        await self.release()
        return result

    @property
    def _call_digest(self) -> str:
        return self.params.digest(self.thread.assistant_id)

    def _next(self) -> ElicitingCall:
        return replace(self, state=self.state.next(), answering=False)
