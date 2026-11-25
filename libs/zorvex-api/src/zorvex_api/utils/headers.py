"""Request headers runs see in ``config.configurable`` (``http.configurable_headers``) and the access log shows
(``http.logging_headers``).

As on the Agent Server: a header is let through when its lower-cased name
matches an ``includes`` pattern and no ``excludes`` pattern (patterns are names
with ``*`` wildcards), never when it is a credential (or, for runs, names a
configurable key the server sets itself). Without ``includes``, none is.
"""

import re
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from typing import Any

from zorvex_api.config import ConfigurableHeadersConfig, load_config
from zorvex_api.utils.run_utils import SERVER_KEYS, SERVER_PINNED_CONFIG_KEYS

_CREDENTIALS = frozenset({"authorization", "x-api-key", "x-service-key", "x-tenant-id"})
# Credentials no log shows, whatever http.logging_headers lets through.
_SECRETS = _CREDENTIALS | {"cookie", "set-cookie"}
# What a header never sets in a run's configurable: credentials, who the run acts for
# (langgraph_service.with_user), the run's ids and checkpoint, and the server's markers.
_USER_KEYS = frozenset({"langgraph_auth_user", "langgraph_auth_user_id", "langgraph_auth_permissions", "user_id"})
_NEVER = _CREDENTIALS | _USER_KEYS | SERVER_PINNED_CONFIG_KEYS | SERVER_KEYS | {"checkpoint_id", "checkpoint_ns"}

# The configurable headers of the request being served.
request_headers: ContextVar[dict[str, str] | None] = ContextVar("configurable_request_headers", default=None)


def configurable_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    """The headers, ``(name, value)`` pairs, that ``http.configurable_headers`` lets into the configurable."""
    return _filters()[0].pick(headers)


def logged_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    """The headers, ``(name, value)`` pairs, that ``http.logging_headers`` lets into the access log."""
    return _filters()[1].pick(headers)


def with_headers(config: dict[str, Any], headers: dict[str, str] | None) -> dict[str, Any]:
    """*config* with *headers* in its configurable, under the keys it already has."""
    if not headers:
        return config
    return {**config, "configurable": {**headers, **(config.get("configurable") or {})}}


@dataclass(frozen=True)
class _Filter:
    """Header name patterns: a name passes when it matches an include and no exclude, and is not one of *never*."""

    includes: tuple[re.Pattern[str], ...]
    excludes: tuple[re.Pattern[str], ...]
    never: frozenset[str]

    @classmethod
    def of(cls, config: ConfigurableHeadersConfig, never: frozenset[str]) -> "_Filter":
        return cls(tuple(map(_pattern, config.includes)), tuple(map(_pattern, config.excludes)), never)

    def pick(self, headers: Iterable[tuple[str, str]]) -> dict[str, str]:
        if not self.includes:
            return {}
        picked: dict[str, str] = {}
        for name, value in headers:
            key = name.lower()
            if key in self.never or any(pattern.fullmatch(key) for pattern in self.excludes):
                continue
            if any(pattern.fullmatch(key) for pattern in self.includes):
                picked[key] = value
        return picked


@cache
def _filters() -> tuple[_Filter, _Filter]:
    """The configurable headers' filter and the logged headers' filter, from zorvex.json."""
    http = load_config().http
    return _Filter.of(http.configurable_headers, _NEVER), _Filter.of(http.logging_headers, _SECRETS)


def _pattern(text: str) -> re.Pattern[str]:
    """A header pattern as a regex: ``*`` matches anything, every other character itself."""
    return re.compile("".join(".*" if char == "*" else re.escape(char) for char in text.lower()), re.DOTALL)
