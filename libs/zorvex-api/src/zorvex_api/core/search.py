"""Searching what the API lists (assistants, threads, crons): one query and one answer for all of them.

A page is read one row past its limit, to tell whether another page follows,
and answered with the pagination headers the Agent Server sends.
"""

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute


class Page(Protocol):
    """What a search request says about its page: a ``SearchPage`` with its ``sort_by``."""

    @property
    def limit(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def sort_by(self) -> str: ...
    @property
    def sort_order(self) -> str: ...


async def page(
    session: AsyncSession,
    key: InstrumentedAttribute[Any],
    conditions: Iterable[ColumnElement[bool]],
    request: Page,
) -> tuple[list[Any], dict[str, str]]:
    """The rows on *request*'s page, ordered by its sort and then by *key*, and the page's headers.

    The *key* tie-break keeps offset pages stable when the sort column repeats.
    """
    table = key.class_
    column = getattr(table, request.sort_by)
    stmt = (
        select(table)
        .where(*conditions)
        .order_by(column.asc() if request.sort_order == "asc" else column.desc(), key.asc())
        .offset(request.offset)
        .limit(request.limit + 1)
    )
    rows = list((await session.scalars(stmt)).all())
    end = request.offset + request.limit
    if len(rows) > request.limit:
        headers = {"X-Pagination-Total": str(end + 1), "X-Pagination-Next": str(end)}
    else:
        headers = {"X-Pagination-Total": str(request.offset + len(rows))}
    return rows[: request.limit], headers


async def count(session: AsyncSession, table: type, conditions: Iterable[ColumnElement[bool]]) -> int:
    """How many rows of *table* match *conditions*."""
    return await session.scalar(select(func.count()).select_from(table).where(*conditions)) or 0


def respond(items: Sequence[BaseModel], headers: dict[str, str], selected: Sequence[str] | None) -> JSONResponse:
    """The page's items as JSON, each narrowed to the *selected* fields when the request names some."""
    include = set(selected) if selected else None
    return JSONResponse([item.model_dump(mode="json", include=include) for item in items], headers=headers)
