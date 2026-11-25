"""Loading the caller's threads and runs.

Another user's rows, and rows an ``@auth.on`` filter excludes, read as missing.
"""

from typing import Any

from fastapi import HTTPException
from sqlalchemy import ColumnElement, Select, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.auth_filters import build_metadata_filter
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.models import User


async def existing_thread(
    session: AsyncSession,
    thread_id: str,
    user: User,
    *,
    filters: dict[str, Any] | None = None,
) -> ThreadORM | None:
    """The thread, or None when no thread has the id; 404 when it is another user's or *filters* exclude it.

    For operations that create the thread they do not find, such as runs.
    """
    auth_filter = build_metadata_filter(ThreadORM.metadata_dict, filters)
    visible = true() if auth_filter is None else auth_filter
    row = (await session.execute(select(ThreadORM, visible).where(ThreadORM.thread_id == thread_id))).first()
    if row is None:
        return None
    thread, is_visible = row
    if thread.user_id != user.identity or not is_visible:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    return thread


async def owned_thread(
    session: AsyncSession,
    thread_id: str,
    user: User,
    *,
    filters: dict[str, Any] | None = None,
) -> ThreadORM:
    """The user's thread, within *filters*; 404 otherwise."""
    thread = await existing_thread(session, thread_id, user, filters=filters)
    if thread is None:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    return thread


def visible_threads(filters: dict[str, Any] | None) -> list[ColumnElement[bool]]:
    """The WHERE clauses that keep the threads *filters* lets through; none without filters."""
    auth_filter = build_metadata_filter(ThreadORM.metadata_dict, filters)
    return [] if auth_filter is None else [auth_filter]


def on_visible_threads(stmt: Select[Any], filters: dict[str, Any] | None) -> Select[Any]:
    """Narrow a query over runs to those on threads *filters* lets through."""
    auth_filter = build_metadata_filter(ThreadORM.metadata_dict, filters)
    if auth_filter is None:
        return stmt
    return stmt.join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id).where(auth_filter)


async def owned_run(
    session: AsyncSession,
    thread_id: str,
    run_id: str,
    user: User,
    *,
    filters: dict[str, Any] | None = None,
) -> RunORM:
    """The user's run on the thread, on a thread within *filters*; 404 otherwise."""
    stmt = select(RunORM).where(
        RunORM.run_id == run_id,
        RunORM.thread_id == thread_id,
        RunORM.user_id == user.identity,
    )
    run = await session.scalar(on_visible_threads(stmt, filters))
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run
