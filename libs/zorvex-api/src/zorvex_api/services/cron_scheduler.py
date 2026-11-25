"""Firing crons: claim the enabled crons whose next time has passed, start a run for each, move each on.

A claim holds a cron for CRON_CLAIM_DURATION_SECONDS (``claimed_until``), so
instances polling together fire it once and a slow firing does not race the
next tick. Claims are taken and judged by the database's clock, which every
instance shares. Firing is at least once: an instance that dies between starting
the run and moving the cron on leaves the claim to expire, and the cron fires again.
"""

import contextlib
from datetime import UTC, datetime, timedelta

import structlog
from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.orm import Cron as CronORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models import Run, RunCreate, User
from zorvex_api.services.cron_service import next_run_date
from zorvex_api.services.periodic import PeriodicTask
from zorvex_api.services.run_preparation import Launch, prepare_run, prepare_stateless_run
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)


async def start_cron_run(session: AsyncSession, cron: CronORM, user: User) -> Run:
    """Create one run from the cron's payload, on the cron's thread or on a new one.

    Saving the cron authorized its runs, so they are not authorized again. A new
    thread is deleted once the run has finished, unless the cron keeps it. The
    cron's own metadata is for searching crons; its runs do not carry it.
    """
    launch = Launch(cron_id=cron.cron_id)
    fields = {"assistant_id": cron.assistant_id, **cron.payload}
    if cron.thread_id is not None:
        return await prepare_run(
            session, cron.thread_id, RunCreate.model_validate(fields), user, authorize=None, launch=launch
        )
    request = RunCreate.model_validate({**fields, "on_completion": cron.on_run_completed or "delete"})
    return await prepare_stateless_run(session, request, user, authorize=None, launch=launch)


async def _claim_due(now: datetime) -> list[CronORM]:
    """Claim the enabled crons whose next time has passed and that nobody holds, the most overdue first.

    SKIP LOCKED gives instances polling together disjoint claims. At most
    CRON_SCHEDULER_BATCH_SIZE crons when that is positive.
    """
    batch = settings.cron.CRON_SCHEDULER_BATCH_SIZE
    async with get_session_maker()() as session:
        due = await session.scalars(
            select(CronORM.cron_id)
            .where(
                CronORM.enabled.is_(True),
                CronORM.next_run_date <= now,
                CronORM.claimed_until.is_(None) | (CronORM.claimed_until <= func.now()),
            )
            .order_by(CronORM.next_run_date.asc())
            .limit(batch if batch > 0 else None)
            .with_for_update(skip_locked=True)
        )
        cron_ids = list(due.all())
        claimed: list[CronORM] = []
        if cron_ids:
            claimed = list(
                await session.scalars(
                    update(CronORM)
                    .where(CronORM.cron_id.in_(cron_ids))
                    .values(claimed_until=func.now() + timedelta(seconds=settings.cron.CRON_CLAIM_DURATION_SECONDS))
                    .returning(CronORM)
                )
            )
        await session.commit()
        return claimed


async def _advance(session: AsyncSession, cron: CronORM) -> None:
    """Move the cron to its next time and release its claim; past its end time, disable it instead."""
    now = datetime.now(UTC)
    if cron.end_time and now >= cron.end_time:
        values: dict[str, object] = {"enabled": False, "updated_at": now}
    else:
        values = {"next_run_date": next_run_date(cron.schedule, cron.timezone, now)}
    await session.execute(update(CronORM).where(CronORM.cron_id == cron.cron_id).values(claimed_until=None, **values))
    await session.commit()


async def _fire(session: AsyncSession, cron: CronORM) -> None:
    """Start the cron's run and move the cron on; a run that cannot start leaves the cron to the next tick."""
    user = User(identity=cron.user_id, display_name="cron-scheduler")
    try:
        run = await start_cron_run(session, cron, user)
    except HTTPException as exc:
        logger.error("Cron run creation failed", cron_id=cron.cron_id, status_code=exc.status_code, detail=exc.detail)
    except Exception:
        logger.exception("Cron run creation failed unexpectedly", cron_id=cron.cron_id)
    else:
        logger.info("Cron fired run", cron_id=cron.cron_id, run_id=run.run_id, thread_id=run.thread_id)
        await _advance(session, cron)
        return
    # The claim goes and next_run_date stays, so the next tick tries again.
    await session.rollback()
    await session.execute(update(CronORM).where(CronORM.cron_id == cron.cron_id).values(claimed_until=None))
    await session.commit()


class CronScheduler(PeriodicTask):
    """Fires due crons every CRON_SCHEDULER_SLEEP_TIME seconds, and at once after a full batch."""

    name = "cron scheduler"

    def interval(self) -> float:
        return settings.cron.CRON_SCHEDULER_SLEEP_TIME

    async def tick(self) -> bool:
        due = await _claim_due(datetime.now(UTC))
        if due:
            logger.info("Firing due crons", count=len(due))
        for cron in due:
            # A session each, so one cron's failed SQL cannot abort the others' transaction.
            async with get_session_maker()() as session:
                try:
                    await _fire(session, cron)
                except Exception:
                    logger.exception("Failed to fire cron job", cron_id=cron.cron_id)
                    with contextlib.suppress(Exception):
                        await session.rollback()
        batch = settings.cron.CRON_SCHEDULER_BATCH_SIZE
        return 0 < batch <= len(due)


cron_scheduler = CronScheduler()
