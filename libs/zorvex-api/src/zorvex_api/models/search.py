"""What every search request shares: its page, its order and the fields it returns."""

from typing import Annotated

from pydantic import BaseModel, Field

from zorvex_api.models.enums import SortOrder
from zorvex_api.settings import settings

# Capped by MAX_SEARCH_LIMIT. Requests default to 10 results, as the SDK asks for.
SearchLimit = Annotated[int, Field(ge=1, le=settings.app.MAX_SEARCH_LIMIT, description="Maximum number of results.")]
SearchOffset = Annotated[int, Field(ge=0, description="Number of results to skip.")]


class SearchPage[Selected: str](BaseModel):
    """A page of search results, narrowed to the *Selected* fields; each search adds the ``sort_by`` it takes."""

    limit: SearchLimit = 10
    offset: SearchOffset = 0
    sort_order: SortOrder = Field(default="desc", description="Sort direction.")
    select: list[Selected] | None = Field(default=None, description="Fields to return; all fields when omitted.")
