"""Store models for Agent Protocol, shaped as the LangGraph SDK's store client sends and reads them."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from zorvex_api.models.search import SearchLimit, SearchOffset

# The store separates labels with dots, so a label cannot hold one.
NamespaceLabel = Annotated[str, Field(min_length=1, pattern=r"^[^.]+$")]


class StorePutRequest(BaseModel):
    """Request model for storing items"""

    namespace: list[NamespaceLabel] = Field(
        default_factory=list, description="Storage namespace; the root when omitted."
    )
    key: str = Field(description="Item key")
    value: dict[str, Any] = Field(description="Item value (a JSON object)")
    index: Literal[False] | list[str] | None = Field(
        default=None, description="Fields to index for semantic search; False disables indexing for this item."
    )
    ttl: float | None = Field(default=None, description="Time-to-live in minutes for this item.")


class StoreDeleteRequest(BaseModel):
    """Request body for deleting a store item."""

    namespace: list[NamespaceLabel] = Field(
        default_factory=list, description="Namespace path of the item to delete; the root when omitted."
    )
    key: str = Field(description="Key of the item to delete.")


class StoreSearchRequest(BaseModel):
    """Request model for searching store items"""

    namespace_prefix: list[NamespaceLabel] = Field(description="Namespace prefix to search")
    filter: dict[str, Any] | None = Field(default=None, description="Key-value pairs the items' values must match.")
    query: str | None = Field(default=None, description="Natural-language query for semantic search.")
    limit: SearchLimit = 10
    offset: SearchOffset = 0
    refresh_ttl: bool | None = Field(default=None, description="Refresh the matched items' TTL on read.")


class StoreListNamespacesRequest(BaseModel):
    """Request model for listing store namespaces"""

    prefix: list[NamespaceLabel] | None = Field(default=None, description="Filter by namespace prefix")
    suffix: list[NamespaceLabel] | None = Field(default=None, description="Filter by namespace suffix")
    max_depth: int | None = Field(default=None, le=100, ge=1, description="Maximum namespace depth to return")
    limit: SearchLimit = 100
    offset: SearchOffset = 0


class StoreItem(BaseModel):
    """A stored item, as the SDK's ``Item``."""

    namespace: list[str] = Field(description="The namespace path where this item is stored.")
    key: str = Field(description="The item's key within its namespace.")
    value: dict[str, Any] = Field(description="The stored value.")
    created_at: datetime = Field(description="When the item was first stored.")
    updated_at: datetime = Field(description="When the item was last written.")


class StoreSearchItem(StoreItem):
    """A search result, as the SDK's ``SearchItem``."""

    score: float | None = Field(default=None, description="Relevance to the query; null without a query.")


class StoreSearchResponse(BaseModel):
    """Response model for store search"""

    items: list[StoreSearchItem]


class StoreListNamespacesResponse(BaseModel):
    """Response model for listing store namespaces"""

    namespaces: list[list[str]]
