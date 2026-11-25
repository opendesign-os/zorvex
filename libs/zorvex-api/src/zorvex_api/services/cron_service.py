"""Crons: saving, reading, searching and deleting them; services/cron_scheduler.py fires them."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog
from croniter import croniter
from fastapi import Depends, HTTPException
from sqlalchemy import ColumnElement, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core import search
from zorvex_api.core.auth import require_auth
from zorvex_api.core.auth_filters import build_metadata_filter
from zorvex_api.core.orm import Cron as CronORM
from zorvex_api.core.orm import SessionDep
from zorvex_api.models import User
from zorvex_api.models.crons import (
    CRON_RUN_FIELDS,
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
    ThreadCronCreate,
    is_seconds_schedule,
)
from zorvex_api.services.langgraph_service import LangGraphService, get_langgraph_service
from zorvex_api.services.ownership import owned_assistant
from zorvex_api.services.webhook_policy import admit_webhook
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)


def next_run_date(schedule: str, timezone: str | None, after: datetime) -> datetime:
    """The schedule's next time after *after*, in UTC; evaluated in *timezone* (``0 9 * * *`` is 09:00 there).

    A stored timezone that no longer resolves (a tzdata upgrade) falls back to
    UTC, so the cron keeps firing.
    """
    base = after
    if timezone is not None:
        try:
            base = after.astimezone(ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            logger.warning("Invalid stored timezone, falling back to UTC", timezone=timezone)
    return (
        croniter(schedule, base, second_at_beginning=is_seconds_schedule(schedule)).get_next(datetime).astimezone(UTC)
    )


def _payload(request: CronCreate | ThreadCronCreate | CronUpdate) -> dict[str, Any]:
    """The request's run fields: the payload each fired run is created from."""
    return request.model_dump(include=set(CRON_RUN_FIELDS), exclude_none=True)


class CronService:
    """One user's crons, over one database session."""

    def __init__(self, session: AsyncSession, user: User, langgraph_service: LangGraphService) -> None:
        self.session = session
        self.user = user
        self.langgraph_service = langgraph_service

    async def create_cron(
        self,
        request: CronCreate | ThreadCronCreate,
        *,
        thread_id: str | None = None,
        assistant_filters: dict[str, Any] | None = None,
    ) -> CronResponse:
        """Save a cron; the scheduler first fires it at the schedule's next time."""
        request = await admit_webhook(request)
        cap = settings.cron.CRON_MAX_PER_USER
        if cap > 0:
            owned = await search.count(self.session, CronORM, [CronORM.user_id == self.user.identity])
            if owned >= cap:
                raise HTTPException(429, f"Cron limit reached: a single user may own at most {cap} crons.")

        # The caller's assistant or a system one: a cron must not carry another user's config.
        assistant = await owned_assistant(
            self.session,
            self.langgraph_service.assistant_id(request.assistant_id),
            self.user,
            filters=assistant_filters,
        )
        self.langgraph_service.require_graph(assistant.graph_id)

        now = datetime.now(UTC)
        cron = CronORM(
            cron_id=str(uuid4()),
            assistant_id=assistant.assistant_id,
            thread_id=thread_id,
            user_id=self.user.identity,
            schedule=request.schedule,
            timezone=request.timezone,
            payload=_payload(request),
            metadata_dict=request.metadata or {},
            on_run_completed=request.on_run_completed if isinstance(request, CronCreate) else None,
            enabled=request.enabled if request.enabled is not None else True,
            end_time=request.end_time,
            next_run_date=next_run_date(request.schedule, request.timezone, now),
            created_at=now,
            updated_at=now,
        )
        self.session.add(cron)
        await self.session.commit()
        logger.info("Created cron job", cron_id=cron.cron_id, schedule=request.schedule)
        return CronResponse.model_validate(cron)

    async def update_cron(self, cron_id: str, request: CronUpdate, *, filters: dict[str, Any] | None) -> CronResponse:
        """Change the fields the request sets; an explicit null clears end_time, or the webhook."""
        # Before the row lock: the check can wait on DNS.
        request = await admit_webhook(request)
        cron = await self._owned(cron_id, filters, lock=True)
        now = datetime.now(UTC)
        values: dict[str, Any] = {"updated_at": now}
        for field in ("schedule", "timezone", "on_run_completed", "enabled"):
            if (value := getattr(request, field)) is not None:
                values[field] = value
        # The SDK sends an explicit null for end_time=None to clear the end time; omitting it keeps it.
        if "end_time" in request.model_fields_set:
            values["end_time"] = request.end_time
        # Metadata is replaced whole, as the SDK expects.
        if request.metadata is not None:
            values["metadata_dict"] = request.metadata
        payload = {**cron.payload, **_payload(request)}
        if "webhook" in request.model_fields_set and request.webhook is None:
            payload.pop("webhook", None)
        if payload != cron.payload:
            values["payload"] = payload
        # A new schedule or timezone moves the next time, and so does re-enabling a cron whose
        # next time has passed: it would otherwise fire on the next tick.
        if (
            request.schedule is not None
            or request.timezone is not None
            or (request.enabled and cron.next_run_date is not None and cron.next_run_date <= now)
        ):
            values["next_run_date"] = next_run_date(
                values.get("schedule", cron.schedule), values.get("timezone", cron.timezone), now
            )
        updated = (
            await self.session.execute(
                update(CronORM).where(CronORM.cron_id == cron_id).values(**values).returning(CronORM)
            )
        ).scalar_one()
        await self.session.commit()
        logger.info("Updated cron job", cron_id=cron_id)
        return CronResponse.model_validate(updated)

    async def delete_cron(self, cron_id: str, *, filters: dict[str, Any] | None) -> None:
        await self.session.delete(await self._owned(cron_id, filters, lock=True))
        await self.session.commit()
        logger.info("Deleted cron job", cron_id=cron_id)

    async def get_cron(self, cron_id: str, *, filters: dict[str, Any] | None) -> CronResponse:
        return CronResponse.model_validate(await self._owned(cron_id, filters))

    def _search_conditions(
        self, request: CronCountRequest, filters: dict[str, Any] | None
    ) -> list[ColumnElement[bool]]:
        """The caller's crons, narrowed by the request's filters and the handler's."""
        conditions = [CronORM.user_id == self.user.identity, build_metadata_filter(CronORM.metadata_dict, filters)]
        if request.assistant_id is not None:
            conditions.append(CronORM.assistant_id == self.langgraph_service.assistant_id(request.assistant_id))
        if request.thread_id is not None:
            conditions.append(CronORM.thread_id == request.thread_id)
        if request.metadata:
            conditions.append(CronORM.metadata_dict.op("@>")(request.metadata))
        return conditions

    async def search_crons(
        self, request: CronSearchRequest, *, filters: dict[str, Any] | None
    ) -> tuple[list[CronResponse], dict[str, str]]:
        """A page of the matching crons, and its pagination headers."""
        conditions = self._search_conditions(request, filters)
        if request.enabled is not None:
            conditions.append(CronORM.enabled == request.enabled)
        rows, headers = await search.page(self.session, CronORM.cron_id, conditions, request)
        return [CronResponse.model_validate(row) for row in rows], headers

    async def count_crons(self, request: CronCountRequest, *, filters: dict[str, Any] | None) -> int:
        return await search.count(self.session, CronORM, self._search_conditions(request, filters))

    async def _owned(self, cron_id: str, filters: dict[str, Any] | None, *, lock: bool = False) -> CronORM:
        """The caller's cron, within *filters*; 404 otherwise. Locked against the scheduler with *lock*."""
        stmt = select(CronORM).where(
            CronORM.cron_id == cron_id,
            CronORM.user_id == self.user.identity,
            build_metadata_filter(CronORM.metadata_dict, filters),
        )
        cron = await self.session.scalar(stmt.with_for_update() if lock else stmt)
        if cron is None:
            raise HTTPException(404, f"Cron '{cron_id}' not found")
        return cron


def get_cron_service(
    session: SessionDep,
    user: User = Depends(require_auth),
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
) -> CronService:
    return CronService(session, user, langgraph_service)
