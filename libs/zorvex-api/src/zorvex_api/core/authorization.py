"""Authorization: the ``@auth.on`` handlers of the auth file, run as the LangGraph SDK defines them.

Every protocol route names the event it authorizes as, once, with ``Authorize``::

    async def get_thread(..., authorize: Authorization = Authorize("threads", "read")):
        filters = await authorize({"thread_id": thread_id})

``await authorize(value)`` runs the most specific handler for the event. A
handler changes the operation by changing ``value`` in place (metadata to add,
the namespace of a store item, the kwargs of a run), and narrows what the
operation sees by returning a filter, which the route applies to the metadata of
the rows it reads. Store items have no metadata, so a store handler scopes by
changing ``value["namespace"]``.

A route that returns without calling its ``Authorization`` fails with a 500
instead of completing unauthorized, and ``check_authorization`` refuses to start
a server with a protocol route that declares no event.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from langgraph_sdk import Auth

from zorvex_api.core import loopback
from zorvex_api.core.auth_deps import require_auth
from zorvex_api.core.auth_middleware import get_auth_instance
from zorvex_api.models.auth import User

Resource = Literal["assistants", "threads", "crons", "store"]
Action = Literal["create", "read", "update", "delete", "search", "create_run", "put", "get", "list_namespaces"]
Filters = dict[str, Any]


class Authorization:
    """One user's attempt at one ``@auth.on`` event; ``await authorization(value)`` runs its handler."""

    __slots__ = ("action", "dispatched", "resource", "trusted", "user")

    def __init__(self, user: User, resource: Resource, action: Action, *, trusted: bool = False) -> None:
        self.user = user
        self.resource: Resource = resource
        self.action: Action = action
        # A loopback request's: allowed without running a handler, as on the Agent Server.
        self.trusted = trusted
        self.dispatched = False

    async def __call__(self, value: dict[str, Any]) -> Filters | None:
        """Run the event's handler on *value*; returns the filter it set, or None.

        A handler that returns False or fails an assertion answers 403, and its
        own ``Auth.exceptions.HTTPException`` keeps its status. With no auth file,
        or no handler for the event, the event is allowed.
        """
        self.dispatched = True
        auth = get_auth_instance()
        handler = _handler(auth, self.resource, self.action) if auth is not None and not self.trusted else None
        if handler is None:
            return None
        ctx = Auth.types.AuthContext(
            user=self.user, permissions=self.user.permissions, resource=self.resource, action=self.action
        )
        try:
            result = await handler(ctx=ctx, value=value)
        except Auth.exceptions.HTTPException as e:
            raise HTTPException(e.status_code, e.detail, headers=e.headers) from e
        except AssertionError as e:
            raise HTTPException(403, str(e) or "Forbidden") from e
        if result is None or result is True:
            return None
        if result is False:
            raise HTTPException(403, "Forbidden")
        if isinstance(result, dict):
            return result
        raise TypeError(
            f"The @auth.on handler for {self.resource}.{self.action} returned {type(result).__name__}; "
            "expected None, a bool or a filter dict"
        )


def _handler(auth: Auth, resource: str, action: str) -> Any | None:
    """The most specific handler: for the resource and action, the resource, the action, then the global one."""
    for key in ((resource, action), (resource, "*"), ("*", action)):
        if handlers := auth._handlers.get(key):
            return handlers[-1]
    return auth._global_handlers[-1] if auth._global_handlers else None


class _EventDependency:
    """The dependency behind ``Authorize``: the route's ``Authorization``, checked once the route returns."""

    def __init__(self, resource: Resource, action: Action) -> None:
        self.resource: Resource = resource
        self.action: Action = action

    async def __call__(self, request: Request, user: User = Depends(require_auth)) -> AsyncIterator[Authorization]:
        authorization = Authorization(user, self.resource, self.action, trusted=loopback.is_loopback(request.scope))
        yield authorization
        # Reached only when the route returned normally; a raised error never gets here.
        if not authorization.dispatched:
            raise RuntimeError(
                f"{request.method} {request.url.path} returned without authorizing {self.resource}.{self.action}"
            )


def Authorize(resource: Resource, action: Action) -> Any:
    """Declare the ``@auth.on`` event a route authorizes as; the parameter receives its ``Authorization``.

    Named like ``Depends``, which it wraps, since it reads as a declaration.
    """
    # Function scope: the check above runs before the response is sent.
    return Depends(_EventDependency(resource, action), scope="function")


def _api_routes(routes: Iterable[Any]) -> Iterator[APIRoute]:
    """Every ``APIRoute`` under ``routes``, descending into included routers."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
            continue
        # FastAPI wraps included routers; newer versions expose the wrapped
        # router as `original_router` rather than `routes`.
        nested = getattr(route, "original_router", None)
        if nested is not None:
            yield from _api_routes(nested.routes)
        elif hasattr(route, "routes"):
            yield from _api_routes(route.routes)


def _declared_events(dependant: Dependant) -> list[_EventDependency]:
    """The ``Authorize`` declarations in a route's dependency tree."""
    found = [dependant.call] if isinstance(dependant.call, _EventDependency) else []
    for sub in dependant.dependencies:
        found += _declared_events(sub)
    return found


def check_authorization(app: FastAPI, protocol_routers: Sequence[APIRouter]) -> None:
    """Raise unless every route serving a protocol path declares exactly one ``Authorize`` event.

    A custom route mounted at a protocol path serves in the protocol route's
    place, so it must declare one too.
    """
    protocol_paths = {
        (method, route.path)
        for router in protocol_routers
        for route in _api_routes(router.routes)
        for method in route.methods or ()
    }
    problems = [
        f"{method} {route.path}: declares {len(events)} Authorize events, not one"
        for route in _api_routes(app.routes)
        for method in sorted(route.methods or ())
        if (method, route.path) in protocol_paths and len(events := _declared_events(route.dependant)) != 1
    ]
    if problems:
        raise RuntimeError("Routes without one @auth.on event:\n  " + "\n  ".join(problems))
