"""Pydantic models for cron job endpoints."""

from datetime import UTC, datetime
from typing import Any, Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from zorvex_api.models.enums import (
    CronSortBy,
    Durability,
    MultitaskStrategy,
    OnCompletionBehavior,
    SortOrder,
    StreamMode,
)
from zorvex_api.models.runs import RunCreate
from zorvex_api.models.search import SearchLimit, SearchOffset
from zorvex_api.settings import settings

# Field length caps. Keep these conservative; cron metadata is small by nature.
_SCHEDULE_MAX_LEN = 256
_TIMEZONE_MAX_LEN = 64
_WEBHOOK_MAX_LEN = 2048
_ASSISTANT_ID_MAX_LEN = 256


def _validate_webhook_url(value: str | None) -> str | None:
    """Reject malformed or non-http(s) webhook URLs at the API boundary.

    The cron record stores ``webhook`` verbatim and every fired run calls it.
    The server's webhooks.url policy is applied on top, in CronService.
    """
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("webhook must use http or https scheme")
    if not parsed.netloc:
        raise ValueError("webhook must include a host")
    return value


class _CronFields(BaseModel):
    """What creating and updating a cron share: the payload of its runs and the options of its schedule."""

    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    webhook: str | None = Field(None, max_length=_WEBHOOK_MAX_LEN)
    end_time: datetime | None = None
    enabled: bool | None = None
    stream_mode: StreamMode | list[StreamMode] | None = None
    stream_subgraphs: bool | None = None
    stream_resumable: bool | None = None
    timezone: str | None = Field(None, max_length=_TIMEZONE_MAX_LEN)
    durability: Durability | None = None

    # A field validator, unlike assigning in _check, keeps "webhook" out of model_fields_set when it
    # was omitted, so update_cron can tell an omitted webhook from an explicit null that removes it.
    _check_webhook = field_validator("webhook")(_validate_webhook_url)

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


class CronCreate(_CronFields):
    """A cron whose every run goes on a new thread (POST /runs/crons)."""

    assistant_id: str = Field(..., max_length=_ASSISTANT_ID_MAX_LEN)
    schedule: str = Field(..., max_length=_SCHEDULE_MAX_LEN)
    input: Any = None
    on_run_completed: OnCompletionBehavior = Field(
        "delete", description="What happens to a run's thread once the run completes: 'delete' (default) or 'keep'."
    )


class ThreadCronCreate(_CronFields):
    """A cron whose runs all go on one thread (POST /threads/{thread_id}/runs/crons)."""

    assistant_id: str = Field(..., max_length=_ASSISTANT_ID_MAX_LEN)
    schedule: str = Field(..., max_length=_SCHEDULE_MAX_LEN)
    input: Any = None
    multitask_strategy: MultitaskStrategy | None = None


class CronUpdate(_CronFields):
    """Request body for updating an existing cron job; omitted fields keep their value."""

    schedule: str | None = Field(None, max_length=_SCHEDULE_MAX_LEN)
    input: Any = None
    on_run_completed: OnCompletionBehavior | None = None


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
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


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


class CronSearchRequest(CronCountRequest):
    """Request body for searching cron jobs."""

    enabled: bool | None = None
    limit: SearchLimit = 10
    offset: SearchOffset = 0
    sort_by: CronSortBy = Field("created_at", description="Field to sort by.")
    sort_order: SortOrder = Field("desc", description="Sort direction.")
    select: list[CronSelectField] | None = Field(None, description="Fields to return; all when omitted")


if (
    not (set(CronCreate.model_fields) | set(ThreadCronCreate.model_fields)) & set(RunCreate.model_fields)
    >= CRON_RUN_FIELDS
):
    raise RuntimeError("CRON_RUN_FIELDS must be fields of RunCreate and of a cron create request")
