"""A thread's stream, as the Agent Server serves it: its runs one after another, and its state updates.

``stream_modes``:

- ``run_modes``: every event of each run (in the modes it was created with),
  and as it ends, a ``metadata`` event ``{"status": "run_done", "run_id": ...}``;
- ``lifecycle``: only each run's start (its first ``metadata`` event) and its ``run_done``;
- ``state_update``: the thread's state updates.

The stream stays open, taking up each run as it comes. Without a
``Last-Event-ID`` it begins with the runs not yet ended; ``-`` replays every
run whose events are still kept, and a run event's id resumes after it.
State updates are live only.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import or_, select
from sse_starlette import ServerSentEvent

from zorvex_api.core import sse
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.auth import User
from zorvex_api.models.enums import ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES
from zorvex_api.services import thread_events
from zorvex_api.services.base_broker import event_id as make_event_id
from zorvex_api.services.base_broker import event_sequence
from zorvex_api.services.broker import broker_manager, live_events
from zorvex_api.services.streaming_service import to_sse

MODES = ("run_modes", "lifecycle", "state_update")
_POLL_SECONDS = 0.5


def parse_modes(values: list[str]) -> list[str]:
    """``stream_modes`` as the SDKs send it: repeated, comma-separated, or one JSON array; ``run_modes`` by default."""
    modes: list[str] = []
    for value in values:
        items: Any = value
        if value.startswith("["):
            with contextlib.suppress(ValueError):
                items = json.loads(value)
        for item in items if isinstance(items, list) else [items]:
            modes.extend(part.strip() for part in str(item).split(",") if part.strip())
    return modes or ["run_modes"]


def kept(event: str, data: Any, modes: frozenset[str]) -> bool:
    """Whether the stream's modes carry an event."""
    if event == "state_update":
        return "state_update" in modes
    if "run_modes" in modes:
        return True
    if "lifecycle" in modes and event == "metadata" and isinstance(data, dict):
        return data.get("status") == "run_done" or ("attempt" in data and "run_id" in data)
    return False


async def join(
    thread_id: str, user: User, modes: frozenset[str], last_event_id: str | None
) -> AsyncGenerator[ServerSentEvent]:
    """The stream's events, until the client goes."""
    out: asyncio.Queue[ServerSentEvent] = asyncio.Queue()
    producers = [asyncio.create_task(_runs(thread_id, user, modes, last_event_id, out))]
    if "state_update" in modes:
        producers.append(asyncio.create_task(_updates(thread_id, out)))
    try:
        while True:
            yield await out.get()
    finally:
        for task in producers:
            task.cancel()
        await asyncio.gather(*producers, return_exceptions=True)


async def _updates(thread_id: str, out: asyncio.Queue[ServerSentEvent]) -> None:
    async with thread_events.follow(thread_id) as events:
        while True:
            event, data = await events.get()
            await out.put(sse.event(event, data))


async def _runs(
    thread_id: str,
    user: User,
    modes: frozenset[str],
    last_event_id: str | None,
    out: asyncio.Queue[ServerSentEvent],
) -> None:
    """Relay the thread's runs in turn: those the start point names, then each new one."""
    seen: set[str] = set()
    since = datetime.now(UTC)
    resume_run, resume_seq = _resume_point(last_event_id)
    replay_all = last_event_id == "-"
    if resume_run is not None:
        async with get_session_maker()() as session:
            anchor = await session.scalar(select(RunORM.created_at).where(RunORM.run_id == resume_run))
        since = anchor or since
    while True:
        condition = (
            RunORM.thread_id == thread_id
            if replay_all
            else or_(RunORM.status.in_(ACTIVE_RUN_STATUSES), RunORM.created_at >= since)
        )
        stmt = (
            select(RunORM.run_id, RunORM.status, RunORM.updated_at)
            .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity, condition)
            .order_by(RunORM.created_at, RunORM.run_id)
        )
        # Shielded: the stream stops this task when its client goes, and a query cut off mid-flight
        # closes its connection; it finishes instead, in a task of its own.
        runs = await asyncio.shield(_rows(stmt))
        for run_id, status, updated_at in runs:
            if run_id in seen:
                continue
            seen.add(run_id)
            after = resume_seq if run_id == resume_run else 0
            async for message in _run(run_id, status, updated_at, after, modes):
                await out.put(message)
        await asyncio.sleep(_POLL_SECONDS)


async def _rows(stmt: Any) -> list[Any]:
    async with get_session_maker()() as session:
        return list((await session.execute(stmt)).all())


async def _run(
    run_id: str, status: str, updated_at: datetime, after: int, modes: frozenset[str]
) -> AsyncGenerator[ServerSentEvent]:
    """One run's events after sequence *after* (stored, then live), its end as ``run_done``."""
    last = after
    # Read without registering a broker: the run may have ended long ago.
    for event_id, payload in await broker_manager.replay(run_id, make_event_id(run_id, after) if after else None):
        last = event_sequence(event_id)
        message, ended = _message(run_id, event_id, payload, modes)
        if message is not None:
            yield message
        if ended:
            return
    # An ended run whose end is not stored has its end on the way for a moment at
    # most; after that it has nothing more to send: its events are no longer kept,
    # or never were (stream_resumable), as on the Agent Server.
    ended_at = updated_at if status in TERMINAL_RUN_STATUSES else None
    async for event_id, payload in live_events(run_id, ended_at=ended_at):
        if event_sequence(event_id) <= last:
            continue
        message, ended = _message(run_id, event_id, payload, modes)
        if message is not None:
            yield message
        if ended:
            return


def _message(
    run_id: str, event_id: str, event: tuple[str, Any], modes: frozenset[str]
) -> tuple[ServerSentEvent | None, bool]:
    """A run event as the thread stream sends it (None when its modes drop it), and whether it ended the run."""
    mode, data = event
    if mode == "end":
        done = {"status": "run_done", "run_id": run_id}
        return (sse.event("metadata", done, event_id) if kept("metadata", done, modes) else None), True
    return (to_sse(event_id, event) if kept(mode, data, modes) else None), False


def _resume_point(last_event_id: str | None) -> tuple[str | None, int]:
    """The run and sequence a ``Last-Event-ID`` resumes after; none for ``-`` or no id."""
    if not last_event_id or last_event_id == "-" or "_event_" not in last_event_id:
        return None, 0
    run_id, _, _ = last_event_id.rpartition("_event_")
    return run_id, event_sequence(last_event_id)
