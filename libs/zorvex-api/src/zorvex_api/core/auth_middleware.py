"""Authentication: the ``@auth.authenticate`` handler of zorvex.json ``auth.path``, run for each request."""

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import structlog
from fastapi import HTTPException
from langgraph_sdk import Auth
from starlette.requests import HTTPConnection

from zorvex_api.config import load_config
from zorvex_api.core import loopback
from zorvex_api.models.auth import User
from zorvex_api.utils.loader import load_object

logger = structlog.getLogger(__name__)

# What the handler can ask for, by parameter name, as ``Auth.authenticate`` documents it.
_REQUEST_PARTS: dict[str, Callable[[HTTPConnection], Any]] = {
    "request": lambda request: request,
    "path": lambda request: request.scope["path"],
    "method": lambda request: request.scope.get("method", "GET"),
    "path_params": lambda request: request.path_params,
    "query_params": lambda request: dict(request.query_params),
    "headers": lambda request: dict(request.scope["headers"]),
    "authorization": lambda request: request.headers.get("authorization"),
}

_USER_ATTRIBUTES = ("identity", "display_name", "permissions", "is_authenticated")


class AuthBackend:
    """Authenticates requests with the ``@auth.authenticate`` handler of ``auth.path``."""

    def __init__(self) -> None:
        path = load_config().auth.path
        self.auth_instance: Auth | None = _load_auth(path) if path else None
        self._handler = self.auth_instance._authenticate_handler if self.auth_instance else None
        self._parameters = _parameters(self._handler) if self._handler else ()
        if self.auth_instance is None:
            logger.warning(
                "No auth file configured — all requests share a single 'anonymous' identity. "
                "Data is NOT isolated between users in this mode. "
                "Configure 'auth.path' in zorvex.json before serving multiple users."
            )

    async def authenticate(self, request: HTTPConnection) -> User:
        """The request's user. Raises HTTPException when authentication fails."""
        if self.auth_instance is None:
            # Every caller shares this identity: no tenant isolation (warned once at startup).
            return loopback.ANONYMOUS
        if self._handler is None:
            raise HTTPException(401, "Authentication required")
        try:
            result = await self._handler(**{name: _REQUEST_PARTS[name](request) for name in self._parameters})
        except Auth.exceptions.HTTPException as e:
            raise HTTPException(e.status_code, e.detail, headers=e.headers) from e
        return _user(result)


def _load_auth(path: str) -> Auth:
    """The Auth instance ``auth.path`` names. A configured auth that cannot load stops the server."""
    auth = load_object(path, module_name="zorvex_auth")
    if not isinstance(auth, Auth):
        raise TypeError(f"auth.path {path!r} names a {type(auth).__name__}, not a langgraph_sdk Auth")
    logger.info("Loaded auth", path=path)
    return auth


def _parameters(handler: Callable[..., Any]) -> tuple[str, ...]:
    """The request parts the handler takes, by name; all of them for ``**kwargs``."""
    parameters = inspect.signature(handler).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        return tuple(_REQUEST_PARTS)
    unknown = [p.name for p in parameters if p.name not in _REQUEST_PARTS and p.default is p.empty]
    if unknown:
        raise TypeError(f"@auth.authenticate takes {unknown}; it can take any of {sorted(_REQUEST_PARTS)}")
    return tuple(p.name for p in parameters if p.name in _REQUEST_PARTS)


def _user(result: Any) -> User:
    """The user a handler returned: a user id, or a mapping or object with an ``identity``."""
    if isinstance(result, str):
        data: dict[str, Any] = {"identity": result}
    elif isinstance(result, Mapping):
        data = dict(result)
    else:
        data = {key: getattr(result, key) for key in _USER_ATTRIBUTES if hasattr(result, key)}
    if not data.get("identity"):
        raise TypeError("@auth.authenticate must return a user id, or a mapping or object with an identity")
    data["display_name"] = data.get("display_name") or data["identity"]
    return User.model_validate(data)


@functools.lru_cache(maxsize=1)
def get_auth_backend() -> AuthBackend:
    """The authentication backend, built once on first use.

    zorvex.json ``auth.path`` alone decides whether requests are authenticated.
    """
    return AuthBackend()


def get_auth_instance() -> Auth | None:
    """The loaded Auth instance, or None when no auth.path is configured."""
    return get_auth_backend().auth_instance
