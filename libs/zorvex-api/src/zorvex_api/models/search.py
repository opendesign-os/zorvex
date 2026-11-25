"""Paging fields that every search request shares."""

from typing import Annotated

from pydantic import Field

from zorvex_api.settings import settings

# Capped by MAX_SEARCH_LIMIT. Requests default to 10 results, as the SDK asks for.
SearchLimit = Annotated[int, Field(ge=1, le=settings.app.MAX_SEARCH_LIMIT, description="Maximum number of results.")]
SearchOffset = Annotated[int, Field(ge=0, description="Number of results to skip.")]
