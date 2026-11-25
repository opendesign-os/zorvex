"""What a wait or join answers once a run has ended, as the Agent Server answers it.

The result comes from the run's events while the broker keeps them: the run's
error, else its last values with the interrupts it paused on, else its last
values. Once they have expired, from what is stored: the run's status and
error, or its thread's current state.

While the run goes on, the response sends a ``\\n`` every
``KEEPALIVE_INTERVAL_SECS`` so proxies keep the connection open; JSON parsers
skip the leading whitespace.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import select

from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.core.serializers import to_jsonable
from zorvex_api.models.auth import User
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES
from zorvex_api.services.base_broker import cancelled_error
from zorvex_api.services.broker import broker_manager
from zorvex_api.services.executor import executor
from zorvex_api.services.run_cleanup import cancel_on_disconnect
from zorvex_api.services.thread_state_service import read_thread_values
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

# The keys of a result body the LangGraph SDK reads: `runs.wait()` raises on an error.
ERROR_KEY = "__error__"
INTERRUPT_KEY = "__interrupt__"
# The presigned feedback URLs of a run created with feedback_keys, as the Agent Server adds them.
FEEDBACK_KEY = "__feedback__"


def error_envelope(error: str, message: str) -> dict[str, Any]:
    """The body that reports a run produced no result."""
    return {ERROR_KEY: {"error": error, "message": message}}


class RunResult:
    """A run's result, gathered from its events."""

    __slots__ = ("error", "feedback", "interrupts", "values")

    def __init__(self) -> None:
        self.error: Any = None
        self.values: dict[str, Any] | None = None
        # By id: LangGraph sends an interrupt in both the values and the updates of its step.
        self.interrupts: dict[str, Any] = {}
        self.feedback: dict[str, Any] | None = None

    def absorb(self, mode: str, data: Any) -> None:
        """Take in one ``(mode, data)`` event; a subgraph's, and every other mode, carry no result."""
        if isinstance(data, dict) and data.get("method") == mode and "params" in data:
            return  # an Agent Protocol v2 event: its run is answered from what is stored
        if mode == "error":
            self.error = to_jsonable(data)
        elif mode == "feedback" and isinstance(data, dict):
            self.feedback = to_jsonable(data)
        elif mode in ("values", "updates") and isinstance(data, dict):
            data = to_jsonable(data)
            for interrupt in data.pop(INTERRUPT_KEY, None) or []:
                key = interrupt.get("id") if isinstance(interrupt, dict) else None
                self.interrupts[key or str(len(self.interrupts))] = interrupt
            if mode == "values":
                self.values = data

    def body(self) -> dict[str, Any] | None:
        """The result, or None when the events carried none."""
        if self.error is not None:
            return {ERROR_KEY: self.error}
        if self.interrupts:
            return {**(self.values or {}), INTERRUPT_KEY: list(self.interrupts.values())}
        return self.values


async def run_result(run_id: str, thread_id: str, user: User, *, timed_out: bool = False) -> dict[str, Any]:
    """What a wait or join on a run answers: from its events while they are kept, else from what is stored.

    *timed_out* means the wait gave up on a run that may still be going. The
    run's feedback URLs, when it made any, ride along under ``__feedback__``.
    """
    result = RunResult()
    for _, payload in await broker_manager.events(run_id):
        if isinstance(payload, (tuple, list)) and len(payload) == 2 and isinstance(payload[0], str):
            result.absorb(payload[0], payload[1])
    body = result.body()
    if body is None:
        body = await _stored_result(run_id, thread_id, user, timed_out=timed_out)
    return body if result.feedback is None else {**body, FEEDBACK_KEY: result.feedback}


async def _stored_result(run_id: str, thread_id: str, user: User, *, timed_out: bool) -> dict[str, Any]:
    """The result from what is stored, once the run's events are no longer kept (or carried none)."""
    async with get_session_maker()() as session:
        run = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == run_id, RunORM.thread_id == thread_id, RunORM.user_id == user.identity
            )
        )
        thread = await session.get(ThreadORM, thread_id) if run is None or run.status == "success" else None
    if run is None:
        # Gone with its thread, as a stateless run's goes once the run ends: nothing to read, as on the Agent Server.
        return {} if thread is None else error_envelope("RunNotFound", f"Run '{run_id}' no longer exists")
    if run.status in ACTIVE_RUN_STATUSES:
        if timed_out:
            return error_envelope("TimeoutError", f"Run '{run_id}' did not finish within the wait timeout")
        return error_envelope("IncompleteRun", f"Wait ended before run '{run_id}' completed (status: {run.status})")
    if run.status == "interrupted":
        return {ERROR_KEY: cancelled_error(run.cancel_action, run.cancel_reason)}
    if run.status in ("error", "timeout"):
        return {ERROR_KEY: run.error or {"error": "Error", "message": "Run failed"}}
    if thread is None:
        return {}
    values, interrupts = await read_thread_values(thread, user)
    pending = [item for task in interrupts.values() for item in task]
    return {**values, INTERRUPT_KEY: pending} if pending else values


async def wait_for_result(run_id: str, thread_id: str, user: User) -> dict[str, Any]:
    """The run's result once it ends, as ``wait_body`` sends it, for a caller that holds no connection open.

    Gives up waiting after ``BG_JOB_TIMEOUT_SECS`` and answers what the run has come to by then.
    """
    timed_out = False
    timeout = settings.worker.BG_JOB_TIMEOUT_SECS
    try:
        await executor.wait_for_completion(run_id, timeout=timeout)
    except TimeoutError:
        timed_out = True
        logger.warning("Wait timed out", run_id=run_id, timeout=timeout)
    except Exception:
        logger.exception("Wait failed", run_id=run_id)
    return await run_result(run_id, thread_id, user, timed_out=timed_out)


def encode_result(body: dict[str, Any]) -> bytes:
    """A result body as JSON bytes."""
    return json.dumps(to_jsonable(body)).encode()


async def wait_body(
    run_id: str,
    thread_id: str,
    user: User,
    *,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncGenerator[bytes]:
    """Keep the connection open until the run ends, then send its result.

    Gives up after ``BG_JOB_TIMEOUT_SECS``. With *is_disconnected*, a client
    that leaves cancels the run.
    """
    done = asyncio.Event()
    timed_out = False
    timeout = settings.worker.BG_JOB_TIMEOUT_SECS

    async def _wait_for_run() -> None:
        nonlocal timed_out
        try:
            await executor.wait_for_completion(run_id, timeout=timeout)
        except TimeoutError:
            timed_out = True
            logger.warning("Wait timed out", run_id=run_id, timeout=timeout)
        except Exception:
            logger.exception("Wait failed", run_id=run_id)
        finally:
            done.set()

    task = asyncio.create_task(_wait_for_run())
    interval = settings.app.KEEPALIVE_INTERVAL_SECS
    try:
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=interval)
            except TimeoutError:
                if is_disconnected is not None and await is_disconnected():
                    logger.info("Client disconnected, cancelling run", run_id=run_id)
                    await cancel_on_disconnect(run_id)
                    return
                yield b"\n"
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    yield encode_result(await run_result(run_id, thread_id, user, timed_out=timed_out))


@dataclass(frozen=True, slots=True)
class Completed:
    """A run that finished: its result, the values it ended with."""

    values: Any


@dataclass(frozen=True, slots=True)
class Failed:
    """A run that failed: its error, as its result reports it."""

    error: Any


@dataclass(frozen=True, slots=True)
class Paused:
    """A run that paused for input: the interrupts it waits on, as ``public_interrupt`` shows them."""

    interrupts: tuple[dict[str, Any], ...]


RunOutcome = Completed | Failed | Paused


def run_outcome(result: Any) -> RunOutcome:
    """What a run's result (a wait's, or a stream's last values) says it came to: its error, the interrupts it
    paused on, or its values."""
    if isinstance(result, Mapping):
        if ERROR_KEY in result:
            return Failed(result[ERROR_KEY])
        if INTERRUPT_KEY in result:
            return Paused(tuple(public_interrupt(item) for item in result[INTERRUPT_KEY] or ()))
    return Completed(result)


def public_interrupt(item: Any) -> dict[str, Any]:
    """An interrupt as the Agent Server shows one: its id and value, and its response schema when it has one."""
    if not isinstance(item, Mapping):
        return {"id": None, "value": item}
    shown = {"id": item.get("id"), "value": item.get("value")}
    if item.get("response_schema") is not None:
        shown["response_schema"] = item["response_schema"]
    return shown
