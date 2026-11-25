"""Background scheduler that fires due cron jobs.

Wakes up every ``CRON_SCHEDULER_SLEEP_TIME`` seconds (default 5), or at once
after a full batch, claims any enabled crons whose ``next_run_date`` has
passed, and triggers a run for each one via the existing run-preparation
pipeline. Once a run is created
the cron is advanced to its next occurrence (or disabled when ``end_time``
has passed). The claim window itself is governed by
``CRON_CLAIM_DURATION_SECONDS`` so a slow ``_fire_cron`` cannot race the
next tick.

Follows the same ``start()/stop()`` lifecycle pattern used by
:class:`zorvex_api.services.lease_reaper.LeaseReaper`.
"""

import asyncio
import contextlib
from datetime import UTC, datetime

import structlog
from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.orm import Cron as CronORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models import Run, RunCreate, User
from zorvex_api.services.cron_service import CronService
from zorvex_api.services.run_cleanup import delete_thread_after_run
from zorvex_api.services.run_preparation import Launch, prepare_run, prepare_stateless_run
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)


def _build_run_create(cron: CronORM) -> RunCreate:
    """Build a ``RunCreate`` from the cron's stored payload."""
    # The cron's own metadata is for searching crons; its runs do not carry it.
    return RunCreate(assistant_id=cron.assistant_id, **(cron.payload or {}))


async def start_cron_run(session: AsyncSession, cron: CronORM, user: User) -> Run:
    """Create one run from the cron's payload, on the cron's thread or on a new one.

    Saving the cron authorized its runs, so they are not authorized again. A new
    thread is deleted once the run has finished, unless the cron keeps it.
    """
    request = _build_run_create(cron)
    launch = Launch(cron_id=cron.cron_id)
    if cron.thread_id is not None:
        return await prepare_run(session, cron.thread_id, request, user, authorize=None, launch=launch)
    run = await prepare_stateless_run(session, request, user, authorize=None, launch=launch)
    if cron.on_run_completed != "keep":
        delete_thread_after_run(run, user.identity)
    return run


async def _validate_cron_user(user_id: str) -> bool:
    """Liveness check called before constructing a forged ``User`` for a fire.

    Default implementation accepts any non-empty user id. Operators that
    integrate a real identity store can monkey-patch this symbol to
    deactivate or revoke firing for users that no longer exist. Returning
    False from this hook causes the scheduler to log a warning, disable the
    cron, and skip the firing.
    """
    return bool(user_id)


class CronScheduler:
    """Periodically fires due cron jobs by creating runs."""

    def __init__(self) -> None:
        """Initialize the scheduler state for the background polling loop."""
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background polling task if the scheduler is enabled."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Cron scheduler started",
            interval_seconds=settings.cron.CRON_SCHEDULER_SLEEP_TIME,
            claim_seconds=settings.cron.CRON_CLAIM_DURATION_SECONDS,
        )

    async def stop(self) -> None:
        """Stop the background polling task and wait for cancellation to finish."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Cron scheduler stopped")

    async def _loop(self) -> None:
        """Tick, then sleep unless the tick claimed a full batch, as the Agent Server's scheduler does.

        Ticking first claims the crons that fell due while the server was down
        without waiting a poll; a failing tick still sleeps, so a database
        outage cannot spin the loop.
        """
        batch = settings.cron.CRON_SCHEDULER_BATCH_SIZE
        while self._running:
            claimed = 0
            try:
                claimed = await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cron scheduler tick")
            if 0 < batch <= claimed:
                continue
            try:
                await asyncio.sleep(settings.cron.CRON_SCHEDULER_SLEEP_TIME)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> int:
        """Fire a run for each due cron it claims, and tell how many it claimed.

        Each cron runs in its own session so a SQL failure on one cron
        cannot put the shared transaction into ``InFailedSqlTransaction`` and
        silently kill the rest of the batch.
        """
        now = datetime.now(UTC)
        maker = get_session_maker()

        async with maker() as claim_session:
            due_crons = await self._find_due_crons(claim_session, now)

        if not due_crons:
            logger.debug("Cron tick: no jobs due")
            return 0

        logger.info("Cron tick: found due jobs", count=len(due_crons))

        for cron in due_crons:
            async with maker() as cron_session:
                try:
                    await self._fire_cron(cron_session, cron)
                except Exception:
                    logger.exception("Failed to fire cron job", cron_id=cron.cron_id)
                    with contextlib.suppress(Exception):
                        await cron_session.rollback()
        return len(due_crons)

    @staticmethod
    async def _find_due_crons(session: AsyncSession, now: datetime) -> list[CronORM]:
        """Return enabled crons whose next_run_date has passed."""
        return await CronService(session).get_due_crons(now)

    @staticmethod
    async def _fire_cron(session: AsyncSession, cron: CronORM) -> None:
        """Create a run from the cron's payload and advance next_run_date."""
        now = datetime.now(UTC)

        # Liveness check: refuse to forge a User for a deleted/revoked identity.
        if not await _validate_cron_user(cron.user_id):
            logger.warning(
                "Disabling cron because the owning user failed liveness check",
                cron_id=cron.cron_id,
                user_id=cron.user_id,
            )
            await session.execute(
                update(CronORM)
                .where(CronORM.cron_id == cron.cron_id)
                .values(enabled=False, claimed_until=None, updated_at=now)
            )
            await session.commit()
            return

        user = User(
            identity=cron.user_id,
            display_name="cron-scheduler",
            is_authenticated=True,
        )
        try:
            run = await start_cron_run(session, cron, user)
        except HTTPException as exc:
            logger.error(
                "Cron run creation failed",
                cron_id=cron.cron_id,
                status_code=exc.status_code,
                detail=exc.detail,
            )
        except Exception:
            logger.exception("Cron run creation failed unexpectedly", cron_id=cron.cron_id)
        else:
            logger.info("Cron fired run", cron_id=cron.cron_id, run_id=run.run_id, thread_id=run.thread_id)
            # Delegate advance/disable to CronService so the rule lives in one place.
            await CronService(session).advance_next_run(cron)
            return

        # Run setup failed: release the claim so the next tick retries, but
        # leave next_run_date unchanged.
        await session.rollback()
        await session.execute(update(CronORM).where(CronORM.cron_id == cron.cron_id).values(claimed_until=None))
        await session.commit()


# Module-level singleton (matches executor / lease_reaper pattern)
cron_scheduler = CronScheduler()
