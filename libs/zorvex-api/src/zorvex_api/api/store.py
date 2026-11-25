"""Store endpoints for Agent Protocol.

Each operation authorizes with the value the SDK's ``Auth.types`` names for it
(``StorePut``, ``StoreGet``, ...). A handler scopes an operation by changing
``value["namespace"]`` in place, and the operation runs on what the handler
left; store items have no metadata, so a filter a handler returns has nothing
to narrow. On top, every namespace is scoped to its user (or to a configured
``store.scopes`` attribute), so users never share items by accident.
"""

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from langgraph.store.base import Item

from zorvex_api.config import load_config
from zorvex_api.core.auth import auth_dependency
from zorvex_api.core.authorization import Authorization, Authorize
from zorvex_api.core.database import db_manager
from zorvex_api.models import (
    StoreDeleteRequest,
    StoreItem,
    StoreListNamespacesRequest,
    StoreListNamespacesResponse,
    StorePutRequest,
    StoreSearchRequest,
    StoreSearchResponse,
    User,
)
from zorvex_api.models.errors import BAD_REQUEST, NOT_FOUND
from zorvex_api.models.store import StoreSearchItem

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Store"], dependencies=auth_dependency)


@router.put("/store/items", status_code=204)
async def put_store_item(
    request: StorePutRequest,
    authorize: Authorization = Authorize("store", "put"),
) -> Response:
    """Create or update an item in the store.

    If an item with the same namespace and key already exists, its value is
    overwritten. Values must be JSON objects (dictionaries).
    """
    user = authorize.user
    value: dict[str, Any] = {
        "namespace": tuple(request.namespace),
        "key": request.key,
        "value": request.value,
        "index": request.index,
    }
    await authorize(value)
    await db_manager.get_store().aput(
        _scoped(value["namespace"], user),
        value["key"],
        value["value"],
        index=value["index"],
        **({} if request.ttl is None else {"ttl": request.ttl}),
    )
    return Response(status_code=204)


@router.get("/store/items", response_model=StoreItem, responses={**BAD_REQUEST, **NOT_FOUND})
async def get_store_item(
    key: str | None = Query(None, description="Key of the item to retrieve."),
    namespace: str = Query("", description="Namespace path, its labels joined with dots."),
    refresh_ttl: bool | None = Query(None, description="Refresh the item's TTL on read."),
    authorize: Authorization = Authorize("store", "get"),
) -> StoreItem:
    """Get an item from the store by key.

    Returns 400 without a key, 404 if no item exists at the given namespace and key.
    """
    user = authorize.user
    if not key:
        raise HTTPException(400, "Key is required")
    value: dict[str, Any] = {"namespace": tuple(label for label in namespace.split(".") if label), "key": key}
    await authorize(value)
    item = await db_manager.get_store().aget(
        _scoped(value["namespace"], user),
        value["key"],
        **({} if refresh_ttl is None else {"refresh_ttl": refresh_ttl}),
    )
    if item is None:
        raise HTTPException(404, "Item not found")
    return StoreItem(**_item_fields(item))


@router.delete("/store/items", status_code=204)
async def delete_store_item(
    request: StoreDeleteRequest,
    authorize: Authorization = Authorize("store", "delete"),
) -> Response:
    """Delete an item from the store."""
    user = authorize.user
    value: dict[str, Any] = {"namespace": tuple(request.namespace), "key": request.key}
    await authorize(value)
    await db_manager.get_store().adelete(_scoped(value["namespace"], user), value["key"])
    return Response(status_code=204)


@router.post("/store/items/search", response_model=StoreSearchResponse)
async def search_store_items(
    request: StoreSearchRequest,
    authorize: Authorization = Authorize("store", "search"),
) -> StoreSearchResponse:
    """Search items in the store.

    Filter items by namespace prefix, key-value metadata filters, or semantic
    query. Results are paginated via `limit` and `offset`.
    """
    user = authorize.user
    value: dict[str, Any] = {
        "namespace": tuple(request.namespace_prefix),
        "filter": request.filter,
        "limit": request.limit,
        "offset": request.offset,
        "query": request.query,
    }
    await authorize(value)
    items = await db_manager.get_store().asearch(
        _scoped(value["namespace"], user),
        query=value["query"],
        filter=value["filter"],
        limit=value["limit"],
        offset=value["offset"],
        **({} if request.refresh_ttl is None else {"refresh_ttl": request.refresh_ttl}),
    )
    return StoreSearchResponse(items=[StoreSearchItem(**_item_fields(item), score=item.score) for item in items])


@router.post("/store/namespaces", response_model=StoreListNamespacesResponse)
async def list_namespaces(
    request: StoreListNamespacesRequest,
    authorize: Authorization = Authorize("store", "list_namespaces"),
) -> StoreListNamespacesResponse:
    """List namespaces in the store.

    Returns the namespace paths that contain items. Filter by prefix, suffix,
    or maximum depth.
    """
    user = authorize.user
    value: dict[str, Any] = {
        "namespace": tuple(request.prefix) if request.prefix else None,
        "suffix": tuple(request.suffix) if request.suffix else None,
        "max_depth": request.max_depth,
        "limit": request.limit,
        "offset": request.offset,
    }
    await authorize(value)
    namespaces = await db_manager.get_store().alist_namespaces(
        prefix=_scoped(value["namespace"] or (), user),
        suffix=tuple(value["suffix"]) if value["suffix"] else None,
        max_depth=value["max_depth"],
        limit=value["limit"],
        offset=value["offset"],
    )
    return StoreListNamespacesResponse(namespaces=[list(namespace) for namespace in namespaces])


def _item_fields(item: Item) -> dict[str, Any]:
    """The API fields of a stored item."""
    return {
        "namespace": list(item.namespace),
        "key": item.key,
        "value": item.value,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


_USER_SCOPE_PREFIX = "users"


@cache
def _scope_attr_map() -> Mapping[str, list[str]]:
    """Map of namespace prefix -> list of User attributes, from zorvex.json store.scopes.

    Empty unless configured — configurable scopes are entirely opt-in. The
    reserved "users" prefix is dropped so it can never be remapped away from
    per-user isolation.
    """
    scopes = dict(load_config().store.scopes)
    if scopes.pop(_USER_SCOPE_PREFIX, None) is not None:
        logger.warning("store.scopes key %r is reserved and was ignored", _USER_SCOPE_PREFIX)
    return scopes


def _scoped(namespace: Sequence[str], user: User) -> tuple[str, ...]:
    """The namespace under the user's scope, ``users/<identity>`` unless it opts into a configured one.

    A leading label naming a ``store.scopes`` prefix isolates the data under
    ``[prefix, *<user attributes>]``, so every user sharing those attribute
    values (an org_id, say) shares it. A namespace already under its scope is
    left as it is.
    """
    scopes = _scope_attr_map()
    if namespace and (prefix := namespace[0]) in scopes:
        values: list[str] = []
        for attr in scopes[prefix]:
            value = getattr(user, attr, None)
            if value is None or value == "":
                raise HTTPException(403, f"User has no {attr!r} required for {prefix!r}-scoped store access")
            values.append(str(value))
        head = (prefix, *values)
    else:
        head = (_USER_SCOPE_PREFIX, user.identity)
    namespace = tuple(namespace)
    return namespace if namespace[: len(head)] == head else (*head, *namespace)
