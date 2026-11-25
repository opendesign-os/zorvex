import base64
import binascii
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_upper(v: str) -> str:
    """Converts to uppercase and strips whitespace."""
    return v.strip().upper() if isinstance(v, str) else v


# Custom types for automatic formatting
UpperStr = Annotated[str, BeforeValidator(parse_upper)]


class EnvBase(BaseSettings):
    """Base settings model that ignores unknown environment variables."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )


class AppSettings(EnvBase):
    """General application settings."""

    # The zorvex.json to load; relative paths inside it resolve against its directory.
    ZORVEX_CONFIG: str = "zorvex.json"
    # The deployment's public base URL, under the Agent Server's name: agent cards send clients there and
    # the OpenAPI document lists it as the server. Unset, each request's own URL serves.
    LANGGRAPH_API_URL: str | None = None
    KEEPALIVE_INTERVAL_SECS: float = Field(default=5, gt=0)  # Heartbeat interval for join/wait endpoints
    # Default 1000 matches LangGraph Platform threads.search (Agent Server OpenAPI max).
    MAX_SEARCH_LIMIT: int = Field(default=1000, ge=1)
    # The path the API is served under, as on the Agent Server; it overrides http.mount_prefix.
    MOUNT_PREFIX: str | None = None
    # Bytes one thread history response may hold, as the Agent Server counts them; 0 for no limit.
    LANGGRAPH_THREAD_HISTORY_MAX_STATE_BYTES: int = Field(default=0, ge=0)

    # Run alembic upgrade head on startup. Default True (dev / single-pod).
    # Set False for multi-pod K8s to avoid advisory-lock probe timeouts;
    # run migrations out-of-band via `zorvex db upgrade`.
    RUN_MIGRATIONS_ON_STARTUP: bool = True

    # Logging: human-readable console output, or one JSON object per line.
    LOG_FORMAT: Literal["console", "json"] = "console"
    LOG_LEVEL: UpperStr = "INFO"
    LOG_EXCLUDE_PATHS: str = ""  # Comma-separated path prefixes to skip in access logs

    @computed_field
    @property
    def log_exclude_paths(self) -> tuple[str, ...]:
        """Parse LOG_EXCLUDE_PATHS into a tuple of path prefixes."""
        if not self.LOG_EXCLUDE_PATHS:
            return ()
        return tuple(part.strip() for part in self.LOG_EXCLUDE_PATHS.split(",") if part.strip())

    @computed_field
    @property
    def sse_ping_interval_secs(self) -> int:
        """Integer ping interval for ``EventSourceResponse``.

        sse-starlette accepts only ``int`` seconds; the underlying setting is
        ``float`` to support sub-second heartbeats in the JSON-wait
        endpoints. Clamp to ``>= 1`` so 0/negative floats can't
        produce a zero ping interval.
        """
        return max(1, int(self.KEEPALIVE_INTERVAL_SECS))


class DatabaseSettings(EnvBase):
    """Database connection.

    DATABASE_URI is a libpq connection URL (postgresql://user:password@host:5432/db,
    password URL-encoded). Both drivers parse it themselves, psycopg through libpq
    and asyncpg through its own DSN parser, so multiple hosts, sslmode, sslrootcert,
    target_session_attrs and the PG* environment variables work as in libpq.
    """

    DATABASE_URI: str = "postgresql://postgres:postgres@localhost:5432/zorvex"
    DB_ECHO_LOG: bool = False


class PoolSettings(EnvBase):
    """Connection pool settings for SQLAlchemy and LangGraph."""

    SQLALCHEMY_POOL_SIZE: int = 10
    SQLALCHEMY_MAX_OVERFLOW: int = 20

    LANGGRAPH_CHECKPOINTER_POSTGRES_POOL_MIN_SIZE: int = 5
    LANGGRAPH_POSTGRES_POOL_MAX_SIZE: int = 20


class ObservabilitySettings(EnvBase):
    """Zorvex's own observability switches.

    Trace export is configured with the standard OTEL_* variables (see
    observability.tracing), not here.
    """

    # Seed each run's trace id from its run_id (turned off for ratio samplers).
    OTEL_TRACE_ID_FROM_RUN_ID: bool = True


class RedisSettings(EnvBase):
    """Redis, which turns on the multi-instance mode.

    With REDIS_URI set, SSE streaming goes through Redis pub/sub and background
    runs through the worker job queue with lease-based crash recovery; unset,
    both stay in process. Connection options redis-py reads from the URL
    (max_connections, health_check_interval, socket_timeout, ...) override the
    defaults below.
    """

    REDIS_URI: str | None = None
    # Prefix of every Redis key and channel Zorvex uses.
    REDIS_KEY_PREFIX: str = "zorvex:"
    # Non-negative retries after the initial attempt (up to 4 calls total) on
    # connection and timeout errors, with exponential backoff 50ms..1s.
    REDIS_RETRY_ATTEMPTS: int = Field(default=3, ge=0)

    @property
    def enabled(self) -> bool:
        return self.REDIS_URI is not None

    @property
    def channel_prefix(self) -> str:
        """Prefix of run event channels and their replay, counter and done keys."""
        return f"{self.REDIS_KEY_PREFIX}run:"

    @property
    def queue_key(self) -> str:
        """The worker job queue: a Redis sorted set of run_ids (services/job_queue.py)."""
        return f"{self.REDIS_KEY_PREFIX}queue"


class WorkerSettings(EnvBase):
    """Worker configuration for background graph execution.

    In Redis mode, runs are dispatched to worker coroutines via a Redis sorted
    set, by priority, instead of local asyncio tasks. Each worker loop dequeues
    run_ids and spawns up to N_JOBS_PER_WORKER concurrent asyncio tasks.
    """

    WORKER_COUNT: int = Field(default=3, ge=1)
    N_JOBS_PER_WORKER: int = Field(default=10, ge=1)
    # How long a worker blocks on the empty queue before it looks again; 0 would block forever.
    QUEUE_WAIT_SECONDS: int = Field(default=5, gt=0)
    WORKER_DRAIN_TIMEOUT: float = 30.0
    BG_JOB_TIMEOUT_SECS: int = 3600
    BG_JOB_MAX_RETRIES: int = 3

    # Lease-based crash recovery. Heartbeats come every third of the lease, so a
    # worker survives two missed heartbeats. Worst-case recovery is about the
    # lease plus the reaper interval.
    LEASE_DURATION_SECONDS: int = Field(default=30, gt=0)
    REAPER_INTERVAL_SECONDS: int = 15
    STUCK_PENDING_THRESHOLD_SECONDS: int = 120
    POSTGRES_POLL_INTERVAL_SECONDS: int = 5

    @property
    def heartbeat_interval_seconds(self) -> float:
        return self.LEASE_DURATION_SECONDS / 3

    @property
    def blocking_socket_timeout_seconds(self) -> float:
        """Read timeout for the connections that block on the queue.

        It must outlast the block: a client that gives up first drops the reply of
        a job Redis already popped, stranding the run until the reaper.
        """
        return self.QUEUE_WAIT_SECONDS + 10


class CronSettings(EnvBase):
    """Cron scheduler configuration.

    Controls the background scheduler that fires cron jobs.
    """

    FF_CRONS_ENABLED: bool = True
    # Seconds the scheduler sleeps between polls for due crons.
    CRON_SCHEDULER_SLEEP_TIME: int = Field(default=5, gt=0)
    # Maximum lease duration for an in-flight cron firing. Once a cron is
    # claimed by ``get_due_crons`` its ``claimed_until`` is set to
    # ``now + CRON_CLAIM_DURATION_SECONDS`` so concurrent pollers and
    # subsequent ticks don't double-fire it. Should comfortably exceed the
    # worst-case ``_fire_cron`` duration. Defaults to 5 minutes.
    CRON_CLAIM_DURATION_SECONDS: int = Field(default=300, gt=0)
    # Cap on how many crons a single user may own. Set to 0 to disable.
    CRON_MAX_PER_USER: int = Field(default=100, ge=0)
    # Allow 6-field (seconds-first) cron schedules. Sub-minute schedules
    # multiply scheduler load and DB writes; off by default.
    CRON_ALLOW_SECONDS_SCHEDULE: bool = False
    # Most due crons one poll claims, no limit when <= 0; a full batch is followed by another poll at once.
    CRON_SCHEDULER_BATCH_SIZE: int = -1
    # Soft cap on JSONB payload size (input + config + context + checkpoint
    # + metadata combined) accepted on create/update.
    CRON_MAX_PAYLOAD_BYTES: int = Field(default=64 * 1024, gt=0)


WEBHOOK_SECRET_PREFIX = "whsec_"  # nosec B105 - the key format's prefix, not a key


def _tool_names(value: str) -> frozenset[str]:
    return frozenset(name.strip() for name in value.split(",") if name.strip())


class A2ASettings(EnvBase):
    """The A2A endpoint's settings, named as the Agent Server names them."""

    # The assistant /.well-known/agent-card.json describes when the request names none.
    DEFAULT_A2A_ASSISTANT_ID: str | None = None
    # A2UI for every assistant whose metadata.a2a.a2ui does not say otherwise.
    A2A_A2UI_ENABLED: bool = False
    # Comma-separated tools whose results are A2UI messages, published as A2UI parts where A2UI is negotiated.
    A2A_A2UI_TOOL_CALL_RESULTS: str = ""
    # Comma-separated tools whose results tasks show; unset shows every tool's, empty none.
    A2A_ALLOWED_TOOL_CALL_RESULTS: str | None = None

    @property
    def a2ui_tool_call_results(self) -> frozenset[str]:
        return _tool_names(self.A2A_A2UI_TOOL_CALL_RESULTS)

    @property
    def allowed_tool_call_results(self) -> frozenset[str] | None:
        """The tools whose results show, or None for all."""
        if self.A2A_ALLOWED_TOOL_CALL_RESULTS is None:
            return None
        return _tool_names(self.A2A_ALLOWED_TOOL_CALL_RESULTS)


class WebhookSettings(EnvBase):
    """Outbound webhook signing.

    ZORVEX_WEBHOOK_SECRET signs every delivery the Standard Webhooks way, so a
    receiver can tell a real callback from anyone who learned the URL. Unset
    leaves deliveries unsigned. Which URLs the server agrees to call is the
    ``webhooks`` block of zorvex.json (services.webhook_policy), not an env var.
    """

    ZORVEX_WEBHOOK_SECRET: SecretStr | None = None

    @model_validator(mode="after")
    def _check_secret(self) -> "WebhookSettings":
        # A whsec_ key that does not decode would fail every delivery, so it fails the boot instead.
        secret = self.ZORVEX_WEBHOOK_SECRET.get_secret_value() if self.ZORVEX_WEBHOOK_SECRET else None
        if secret and secret.startswith(WEBHOOK_SECRET_PREFIX):
            try:
                key = base64.b64decode(secret.removeprefix(WEBHOOK_SECRET_PREFIX))
            except binascii.Error:
                raise ValueError(
                    f"ZORVEX_WEBHOOK_SECRET is not valid base64 behind its {WEBHOOK_SECRET_PREFIX} prefix"
                ) from None
            if not key:
                raise ValueError(f"ZORVEX_WEBHOOK_SECRET has no key behind its {WEBHOOK_SECRET_PREFIX} prefix")
        return self


class Settings:
    """Container object that instantiates all application settings groups."""

    def __init__(self) -> None:
        """Build the settings tree from environment-backed settings models."""
        self.app = AppSettings()
        self.db = DatabaseSettings()
        self.pool = PoolSettings()
        self.observability = ObservabilitySettings()
        self.redis = RedisSettings()
        self.worker = WorkerSettings()
        self.cron = CronSettings()
        self.webhook = WebhookSettings()
        self.a2a = A2ASettings()


settings = Settings()
