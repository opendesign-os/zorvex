"""zorvex.json, the project configuration: one validated model, loaded once.

The file is ZORVEX_CONFIG, ./zorvex.json by default. Relative paths in it (graphs,
dependencies, http.app, auth.path) resolve against its directory. Every object
rejects unknown keys, so a misspelled key fails the boot instead of being ignored.
"""

import os
import re
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self

import structlog
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.base import TTLConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from zorvex_api.models.enums import Durability, PruneStrategy
from zorvex_api.models.threads import MAX_TTL_MINUTES
from zorvex_api.settings import settings

logger = structlog.get_logger(__name__)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphEntry(_Strict):
    """Object form of a ``graphs`` entry; the string form is just the path."""

    path: str
    description: str | None = None


class CorsConfig(_Strict):
    """``http.cors``. Unset origins and credentials take the defaults main.py derives."""

    allow_origins: list[str] | None = None
    allow_origin_regex: str | None = None
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool | None = None
    expose_headers: list[str] | None = None
    max_age: int = 600


class ConfigurableHeadersConfig(_Strict):
    """``http.configurable_headers``: which request headers runs and graphs see in ``config.configurable``.

    Names, or patterns with ``*``, matched against lower-cased header names; a
    header must match ``includes`` and no ``excludes``. Credentials
    (authorization, x-api-key, x-service-key, x-tenant-id) and the server's own
    configurable keys are never taken from a header.
    """

    includes: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)


class HttpConfig(_Strict):
    # Import path of a custom FastAPI app; the API routes are added to it.
    app: str | None = None
    # Require authentication on the custom app's routes too.
    enable_custom_route_auth: bool = False
    # The path every route is served under, such as /my-deployment/api (MOUNT_PREFIX overrides it).
    mount_prefix: str | None = None
    # Whether the custom app's middleware sees a request before authentication (the default) or after it.
    middleware_order: Literal["middleware_first", "auth_first"] = "middleware_first"
    cors: CorsConfig = Field(default_factory=CorsConfig)
    configurable_headers: ConfigurableHeadersConfig = Field(default_factory=ConfigurableHeadersConfig)
    # The request and response headers the access log shows, by the same patterns; none by default.
    logging_headers: ConfigurableHeadersConfig = Field(default_factory=ConfigurableHeadersConfig)
    # Interfaces turned off, as langgraph.json turns them off: runs takes the crons and the v2 event
    # streaming with it, meta leaves a bare /ok (and no docs), webhooks drops every request's webhook.
    disable_assistants: bool = False
    disable_threads: bool = False
    disable_runs: bool = False
    disable_store: bool = False
    disable_meta: bool = False
    disable_webhooks: bool = False
    disable_event_streaming: bool = False
    disable_ui: bool = False
    disable_mcp: bool = False
    disable_a2a: bool = False


class StoreIndexConfig(_Strict):
    """Vector index for semantic store search (pgvector)."""

    dims: int = Field(gt=0)
    # '<provider>:<model-id>', e.g. openai:text-embedding-3-small (1536 dims).
    embed: str
    # JSON paths to embed, e.g. ["text", "metadata.title"]; None embeds the whole document.
    fields: list[str] | None = None


class StoreConfig(_Strict):
    index: StoreIndexConfig | None = None
    # Item expiry as in langgraph.json, merged over Zorvex's defaults (core.database).
    ttl: TTLConfig | None = None
    # Namespace prefix -> User attributes that scope it (api.store).
    scopes: dict[str, Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]] = Field(
        default_factory=dict
    )


class ThreadTTLConfig(_Strict):
    """``checkpointer.ttl``: expiry of threads created without their own ttl."""

    strategy: PruneStrategy = "delete"
    default_ttl: float = Field(43200, gt=0, le=MAX_TTL_MINUTES)  # minutes; 30 days
    sweep_interval_minutes: float = Field(5, gt=0, le=MAX_TTL_MINUTES)
    # 10000 matches the documented LangGraph Platform default for this block.
    sweep_limit: int = Field(10000, gt=0)


class SerdeConfig(_Strict):
    """``checkpointer.serde``: what the checkpoint serializer may rebuild, as langgraph.json sets it."""

    # The module paths (["my_agent", "types", "Thing"]) custom objects may be rebuilt from; true allows any.
    allowed_json_modules: list[list[str]] | Literal[True] | None = None
    # Pickle what JSON cannot hold, as the Agent Server does unless told not to.
    pickle_fallback: bool = True

    def serializer(self) -> JsonPlusSerializer:
        modules = self.allowed_json_modules
        return JsonPlusSerializer(
            pickle_fallback=self.pickle_fallback,
            allowed_json_modules=True if modules is True else [tuple(path) for path in modules or ()] or None,
        )


class CheckpointerConfig(_Strict):
    # The built-in Postgres checkpointer, the only one: thread copies, storage and rollbacks read its tables.
    backend: Literal["default"] = "default"
    # Thread retention; off when unset.
    ttl: ThreadTTLConfig | None = None
    # The serializer's options, the Agent Server's defaults when unset.
    serde: SerdeConfig = Field(default_factory=SerdeConfig)
    # Durability for runs that set none; unset keeps LangGraph's default.
    durability: Durability | None = None


class UiConfig(_Strict):
    """``ui_config``: the UI bundler's options (@langchain/langgraph-ui), as langgraph.json has them."""

    # Packages the components take from the client page instead of bundling them, such as "@mui/material".
    shared: list[str] = Field(default_factory=list)


class PriorityConfig(_Strict):
    """``priority``: the deployment's priority callback (services/priority.py), as langgraph.json names it."""

    # './priority.py:priority' (relative to zorvex.json) or 'package.module:priority': a langgraph_sdk Priority.
    path: str


class AuthConfig(_Strict):
    # './auth.py:auth' (relative to zorvex.json) or 'package.module:auth'.
    path: str | None = None


# --- webhooks: the same keys as LangGraph's langgraph.json block, plus `enabled`.

DEFAULT_ENV_PREFIX = "LG_WEBHOOK_"

# Top-level payload keys; the same vocabulary as LangGraph's webhooks.allowed_fields.
WEBHOOK_PAYLOAD_FIELDS: frozenset[str] = frozenset(
    {
        "run_id",
        "thread_id",
        "assistant_id",
        "created_at",
        "updated_at",
        "metadata",
        "kwargs",
        "multitask_strategy",
        "status",
        "run_started_at",
        "run_ended_at",
        "webhook_sent_at",
        "values",
        "error",
    }
)

_ENV_TEMPLATE_RE = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


class WebhookUrlPolicy(_Strict):
    """``webhooks.url``: which user-supplied webhook URLs the server agrees to call."""

    require_https: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_ports: list[int] = Field(default_factory=list)
    max_url_length: int = Field(2048, gt=0)
    disable_loopback: bool = True
    # Stricter than LangGraph and upstream Zorvex (False): webhook URLs come from callers, so a target
    # on the deployment's private network is refused unless the operator opts in.
    disable_private_ips: bool = True

    @field_validator("allowed_domains")
    @classmethod
    def _normalize_domains(cls, domains: list[str]) -> list[str]:
        """Lowercase entries and only allow ``*.`` as a leading wildcard."""
        normalized: list[str] = []
        for domain in domains:
            entry = domain.strip().lower().rstrip(".")
            if not entry:
                raise ValueError("webhooks.url.allowed_domains entries must be non-empty strings")
            if "*" in entry and (not entry.startswith("*.") or "*" in entry[2:]):
                raise ValueError(f"webhooks.url.allowed_domains wildcards must be a '*.' prefix, got {domain!r}")
            normalized.append(entry)
        return normalized

    @field_validator("allowed_ports")
    @classmethod
    def _check_ports(cls, ports: list[int]) -> list[int]:
        """Reject port numbers no URL can carry."""
        for port in ports:
            if not 0 < port < 65536:
                raise ValueError(f"webhooks.url.allowed_ports entries must be between 1 and 65535, got {port}")
        return ports

    def allows_domain(self, host: str) -> bool:
        """True when *host* equals an entry or is a subdomain of a ``*.`` entry."""
        return any(
            host.endswith(entry[1:]) if entry.startswith("*.") else host == entry for entry in self.allowed_domains
        )

    def trusts(self, host: str) -> bool:
        """Exact allow-list entries are vouched for by the operator and skip the address checks."""
        return host in self.allowed_domains


def _unconfigured_url_policy() -> WebhookUrlPolicy:
    # Without a webhooks.url block LangGraph accepts plain http and URLs up to 4096 characters.
    return WebhookUrlPolicy(require_https=False, max_url_length=4096)


def _render_header(value: str, env_prefix: str) -> str:
    """Resolve ``${{ env.VAR }}`` templates; VAR must start with *env_prefix* and be set."""

    def _substitute(match: re.Match[str]) -> str:
        expression = match.group(1)
        if not expression.startswith("env."):
            raise ValueError(f"webhooks.headers only supports ${{{{ env.VAR }}}} templates, got {match.group(0)!r}")
        name = expression.removeprefix("env.")
        if not _ENV_NAME_RE.match(name):
            raise ValueError(f"webhooks.headers references an invalid environment variable name {name!r}")
        if not name.startswith(env_prefix):
            raise ValueError(f"webhooks.headers variable {name!r} must start with env_prefix {env_prefix!r}")
        resolved = os.environ.get(name)
        if resolved is None:
            raise ValueError(f"webhooks.headers references {name!r}, which is not set")
        return resolved

    return _ENV_TEMPLATE_RE.sub(_substitute, value)


class WebhooksConfig(_Strict):
    """``webhooks``, with header templates already resolved."""

    env_prefix: str = DEFAULT_ENV_PREFIX
    url: WebhookUrlPolicy = Field(default_factory=_unconfigured_url_policy)
    headers: dict[str, str] = Field(default_factory=dict)
    allowed_fields: frozenset[str] | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _default_url_policy(cls, value: object) -> object:
        """Treat an explicit null like a missing block."""
        return _unconfigured_url_policy() if value is None else value

    @field_validator("allowed_fields")
    @classmethod
    def _check_allowed_fields(cls, fields: frozenset[str] | None) -> frozenset[str] | None:
        """Reject unknown payload keys; an empty list means every key, as in LangGraph."""
        if not fields:
            return None
        unknown = sorted(fields - WEBHOOK_PAYLOAD_FIELDS)
        if unknown:
            raise ValueError(
                f"webhooks.allowed_fields has unknown fields {unknown}; valid fields are {sorted(WEBHOOK_PAYLOAD_FIELDS)}"
            )
        return fields

    @model_validator(mode="after")
    def _render_headers(self) -> Self:
        """Resolve header templates once, so a missing variable fails the boot instead of every call."""
        rendered: dict[str, str] = {}
        for name, value in self.headers.items():
            if not _HEADER_NAME_RE.match(name):
                raise ValueError(f"webhooks.headers has an invalid header name {name!r}")
            resolved = _render_header(value, self.env_prefix)
            if "\r" in resolved or "\n" in resolved:
                raise ValueError(f"webhooks.headers value for {name!r} contains a line break")
            rendered[name] = resolved
        self.headers = rendered
        return self


class ZorvexConfig(_Strict):
    name: str = "Zorvex"
    graphs: dict[str, str | GraphEntry] = Field(default_factory=dict)
    # The server's environment, as the CLI applies it: a .env file (relative to zorvex.json) or the variables.
    env: str | dict[str, str] | None = None
    # Directories added to sys.path before the graphs load; the first entry wins.
    dependencies: list[str] = Field(default_factory=list)
    http: HttpConfig = Field(default_factory=HttpConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    webhooks: WebhooksConfig = Field(default_factory=WebhooksConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    priority: PriorityConfig | None = None
    # Graph id -> its generative UI components file, bundled by @langchain/langgraph-ui (api/ui.py).
    ui: dict[str, str] | None = None
    ui_config: UiConfig | None = None


def config_path() -> Path:
    return Path(settings.app.ZORVEX_CONFIG)


@cache
def load_config() -> ZorvexConfig:
    """The validated zorvex.json. Raises when the file is missing or invalid."""
    path = config_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {path} not found; set ZORVEX_CONFIG or create ./zorvex.json") from None
    config = ZorvexConfig.model_validate_json(raw)
    logger.info("Loaded config", path=str(path))
    return config


def mount_prefix() -> str | None:
    """The path the API is served under: MOUNT_PREFIX, else ``http.mount_prefix``; None for the root.

    It starts with ``/``, ends without one, and is never ``/noauth``, the root of
    in-process loopback requests.
    """
    prefix = (settings.app.MOUNT_PREFIX or load_config().http.mount_prefix or "").rstrip("/")
    if not prefix:
        return None
    if not prefix.startswith("/"):
        raise ValueError(f"mount_prefix {prefix!r} must start with '/', as in '/my-api' or '/api/v1'")
    if prefix == "/noauth" or prefix.startswith("/noauth/"):
        raise ValueError(f"mount_prefix {prefix!r}: '/noauth' is reserved for in-process loopback requests")
    return prefix


def config_dir() -> Path:
    """The directory relative paths in zorvex.json resolve against."""
    return config_path().resolve().parent
