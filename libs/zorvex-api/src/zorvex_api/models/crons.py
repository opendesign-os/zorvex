"""Pydantic models for cron job endpoints."""

from datetime import UTC, datetime
from typing import Any, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from zorvex_api.models.entity_ids import MAX_ENTITY_ID_LENGTH
from zorvex_api.models.enums import CronSortBy, Durability, MultitaskStrategy, OnCompletionBehavior, StreamMode
from zorvex_api.models.runs import RunCreate, redact_webhook_url
from zorvex_api.models.search import SearchPage
from zorvex_api.settings import settings

_SCHEDULE_MAX_LEN = 256
_TIMEZONE_MAX_LEN = 64


def is_seconds_schedule(schedule: str) -> bool:
    """Whether *schedule* has six fields, seconds first."""
    return len(schedule.split()) == 6


def _check_schedule(schedule: str | None) -> str | None:
    """A schedule croniter can step through, of six fields only when CRON_ALLOW_SECONDS_SCHEDULE is on."""
    if schedule is None:
        return None
    seconds = is_seconds_schedule(schedule)
    if seconds and not settings.cron.CRON_ALLOW_SECONDS_SCHEDULE:
        raise ValueError("6-field (seconds) cron schedules are disabled; set CRON_ALLOW_SECONDS_SCHEDULE to allow them")
    try:
        # Some invalid expressions only fail when stepped through.
        croniter(schedule, datetime.now(UTC), second_at_beginning=seconds).get_next(datetime)
    except (ValueError, KeyError):
        raise ValueError(f"Invalid cron schedule: {schedule}") from None
    return schedule


def _check_timezone(timezone: str | None) -> str | None:
    """An IANA timezone name."""
    if timezone is not None:
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            raise ValueError(f"Invalid timezone: {timezone}") from None
    return timezone


class _CronFields(BaseModel):
    """What creating and updating a cron share: the payload of its runs and the options of its schedule."""

    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    webhook: str | None = None
    end_time: datetime | None = None
    enabled: bool | None = None
    stream_mode: StreamMode | list[StreamMode] | None = None
    stream_subgraphs: bool | None = None
    stream_resumable: bool | None = None
    timezone: str | None = Field(None, max_length=_TIMEZONE_MAX_LEN)
    durability: Durability | None = None

    _timezone = field_validator("timezone")(_check_timezone)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.end_time is not None:
            end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=UTC)
            if end <= datetime.now(UTC):
                raise ValueError("end_time must be in the future")
        cap = settings.cron.CRON_MAX_PAYLOAD_BYTES
        if len(self.model_dump_json().encode("utf-8")) > cap:
            raise ValueError(f"cron payload exceeds {cap} bytes")
        return self


# The fields each run a cron fires is created with; they keep their RunCreate names in the cron's payload.
CRON_RUN_FIELDS: frozenset[str] = frozenset(
    {
        "input",
        "config",
        "context",
        "interrupt_before",
        "interrupt_after",
        "webhook",
        "multitask_strategy",
        "stream_mode",
        "stream_subgraphs",
        "stream_resumable",
        "durability",
    }
)


class _NewCron(_CronFields):
    assistant_id: str = Field(..., max_length=MAX_ENTITY_ID_LENGTH)
    schedule: str = Field(..., max_length=_SCHEDULE_MAX_LEN)
    input: Any = None

    _schedule = field_validator("schedule")(_check_schedule)


class CronCreate(_NewCron):
    """A cron whose every run goes on a new thread (POST /runs/crons)."""

    on_run_completed: OnCompletionBehavior = Field(
        "delete", description="What happens to a run's thread once the run completes: 'delete' (default) or 'keep'."
    )


class ThreadCronCreate(_NewCron):
    """A cron whose runs all go on one thread (POST /threads/{thread_id}/runs/crons)."""

    multitask_strategy: MultitaskStrategy | None = None


class CronUpdate(_CronFields):
    """Request body for updating an existing cron job; omitted fields keep their value."""

    schedule: str | None = Field(None, max_length=_SCHEDULE_MAX_LEN)
    input: Any = None
    on_run_completed: OnCompletionBehavior | None = None

    _schedule = field_validator("schedule")(_check_schedule)


class CronResponse(BaseModel):
    """Response model matching the SDK ``Cron`` TypedDict."""

    model_config = ConfigDict(from_attributes=True)

    cron_id: str
    assistant_id: str
    thread_id: str | None = None
    on_run_completed: OnCompletionBehavior | None = None
    end_time: datetime | None = None
    schedule: str
    timezone: str | None = None
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    next_run_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias=AliasChoices("metadata_dict", "metadata"))
    enabled: bool = True

    @field_validator("payload")
    @classmethod
    def _redact(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """The webhook without its credentials or query, which often carry the receiver's token."""
        if webhook := payload.get("webhook"):
            return {**payload, "webhook": redact_webhook_url(webhook)}
        return payload


CronSelectField = Literal[
    "cron_id",
    "assistant_id",
    "thread_id",
    "on_run_completed",
    "end_time",
    "schedule",
    "timezone",
    "created_at",
    "updated_at",
    "user_id",
    "payload",
    "metadata",
    "next_run_date",
    "enabled",
]


class CronCountRequest(BaseModel):
    """Filters of POST /runs/crons/count; search takes them too."""

    assistant_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] | None = None


class CronSearchRequest(CronCountRequest, SearchPage[CronSelectField]):
    """Request body for searching cron jobs."""

    sort_by: CronSortBy = Field("created_at", description="Field to sort by.")

    enabled: bool | None = None


if (
    not (set(CronCreate.model_fields) | set(ThreadCronCreate.model_fields)) & set(RunCreate.model_fields)
    >= CRON_RUN_FIELDS
):
    raise RuntimeError("CRON_RUN_FIELDS must be fields of RunCreate and of a cron create request")
