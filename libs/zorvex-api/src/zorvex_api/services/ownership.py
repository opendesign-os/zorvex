"""Loading the caller's assistants, threads and runs.

Another user's rows, and rows an ``@auth.on`` filter excludes, read as missing.
"""

from typing import Any

from fastapi import HTTPException
from sqlalchemy import ColumnElement, Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.auth_filters import build_metadata_filter
from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import Assistant as AssistantORM
from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models import User
from zorvex_api.models.runs import Run
from zorvex_api.services.langgraph_service import SYSTEM_OWNER


def readable_assistants(user: User, filters: dict[str, Any] | None) -> ColumnElement[bool]:
    """The assistants *user* can read: their own and the server's, within *filters*."""
    return AssistantORM.user_id.in_((user.identity, SYSTEM_OWNER)) & build_metadata_filter(
        AssistantORM.metadata_dict, filters
    )


async def owned_assistant(
    session: AsyncSession,
    assistant_id: str,
    user: User,
    *,
    filters: dict[str, Any] | None = None,
    writable: bool = False,
) -> AssistantORM:
    """The assistant *user* can read, or with *writable* change (only their own), within *filters*; 404 otherwise."""
    visible = (
        (AssistantORM.user_id == user.identity) & build_metadata_filter(AssistantORM.metadata_dict, filters)
        if writable
        else readable_assistants(user, filters)
    )
    assistant = await session.scalar(select(AssistantORM).where(AssistantORM.assistant_id == assistant_id, visible))
    if assistant is None:
        raise HTTPException(404, f"Assistant '{assistant_id}' not found")
    return assistant


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
    visible = build_metadata_filter(ThreadORM.metadata_dict, filters)
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


def visible_threads(user: User, filters: dict[str, Any] | None) -> ColumnElement[bool]:
    """The threads *user* owns within *filters*."""
    return (ThreadORM.user_id == user.identity) & build_metadata_filter(ThreadORM.metadata_dict, filters)


def on_visible_threads(stmt: Select[Any], filters: dict[str, Any] | None) -> Select[Any]:
    """Narrow a query over runs to those on threads *filters* lets through."""
    if not filters:
        return stmt
    return stmt.join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id).where(
        build_metadata_filter(ThreadORM.metadata_dict, filters)
    )


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


async def authorized_thread(authorize: Authorization, thread_id: str) -> ThreadORM:
    """The thread, as *authorize* (one of its caller's ``threads`` events) lets them have it; 404 otherwise."""
    filters = await authorize({"thread_id": thread_id})
    async with get_session_maker()() as session:
        return await owned_thread(session, thread_id, authorize.user, filters=filters)


async def authorized_run(authorize: Authorization, thread_id: str, run_id: str) -> Run:
    """The run on the thread, as *authorize* (its caller's ``threads.read``) lets them see it; 404 otherwise."""
    filters = await authorize({"thread_id": thread_id, "run_id": run_id})
    async with get_session_maker()() as session:
        return Run.model_validate(await owned_run(session, thread_id, run_id, authorize.user, filters=filters))
