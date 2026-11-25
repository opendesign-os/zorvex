"""The pagination headers of search responses, as the Agent Server sends them."""


def pagination_headers(count: int, *, limit: int, offset: int) -> dict[str, str]:
    """``X-Pagination-Total`` and, when more follow, ``X-Pagination-Next``, for a page read one row past its limit.

    *count* is the number of rows the read returned, ``limit + 1`` when another page follows.
    """
    if count > limit:
        next_offset = offset + limit
        return {"X-Pagination-Total": str(next_offset + 1), "X-Pagination-Next": str(next_offset)}
    return {"X-Pagination-Total": str(offset + count)}
