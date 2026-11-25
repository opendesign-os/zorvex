"""Which pending run may start: each thread runs its runs one at a time, in line.

As on the Agent Server, a run waits while another run on its thread is
running, and a thread's pending runs start in the order they fall due
(``created_at``, which is later than creation for a run with ``after_seconds``).
How a new run treats the runs already on its thread is its
``multitask_strategy`` (services/run_preparation.py); this module only keeps
the line.
"""

from datetime import datetime

from sqlalchemy import ColumnElement, Row, and_, exists, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from zorvex_api.core.orm import Run as RunORM
from zorvex_api.core.orm import Thread as ThreadORM


def runnable(now: datetime) -> ColumnElement[bool]:
    """A pending run that may start now: due, with no running run on its thread and none due before it."""
    other = aliased(RunORM)
    ahead = or_(
        other.status == "running",
        and_(
            other.status == "pending",
            tuple_(other.created_at, other.run_id) < tuple_(RunORM.created_at, RunORM.run_id),
        ),
    )
    return and_(
        RunORM.status == "pending",
        RunORM.created_at <= now,
        ~exists(select(other.run_id).where(other.thread_id == RunORM.thread_id, other.run_id != RunORM.run_id, ahead)),
    )


async def lock_line(session: AsyncSession, run_id: str) -> None:
    """Lock the thread of *run_id*, so the runs on it start one at a time."""
    thread_id = select(RunORM.thread_id).where(RunORM.run_id == run_id).scalar_subquery()
    await session.execute(select(ThreadORM.thread_id).where(ThreadORM.thread_id == thread_id).with_for_update())


async def next_on(session: AsyncSession, thread_id: str, now: datetime) -> Row[str, int, datetime] | None:
    """The run on *thread_id* that may start now, if any: its ``run_id``, ``priority`` and ``created_at``."""
    result = await session.execute(
        select(RunORM.run_id, RunORM.priority, RunORM.created_at)
        .where(RunORM.thread_id == thread_id, RunORM.claimed_by.is_(None), runnable(now))
        .order_by(RunORM.created_at, RunORM.run_id)
        .limit(1)
    )
    return result.first()
