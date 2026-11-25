"""Compile an ``@auth.on`` handler's filter into a JSONB predicate over a metadata column.

A handler narrows what an operation sees by returning a filter
(``Auth.types.FilterType``), matched against the metadata of the rows:

    {"owner": "u1"}                       -> metadata @> {"owner": "u1"}        (eq)
    {"owner": {"$eq": "u1"}}              -> same, explicit
    {"tags": {"$contains": "x"}}          -> metadata->'tags' @> ["x"]          (array contains)
    {"tags": {"$contains": ["x", "y"]}}   -> metadata->'tags' @> ["x", "y"]
    {"$or": [f1, f2, ...]}                -> OR of branches (>= 2, depth <= 2)
    {"$and": [f1, f2, ...]}               -> AND of branches (>= 2)
    multiple top-level keys               -> AND (logical conjunction)

A malformed filter is a bug in the auth file: it raises ValueError, a 500 with
the reason in the server log, rather than matching the wrong rows.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

_MAX_FILTER_DEPTH = 2


def build_metadata_filter(
    column: InstrumentedAttribute[Any],
    filters: dict[str, Any] | None,
) -> ColumnElement[bool] | None:
    """The predicate *filters* puts on the JSONB metadata ``column``, or None when there is none."""
    if not filters:
        return None
    clauses = _compile(column, filters, depth=0)
    return and_(*clauses) if len(clauses) > 1 else clauses[0]


def _compile(
    column: InstrumentedAttribute[Any],
    filters: dict[str, Any],
    *,
    depth: int,
) -> list[ColumnElement[bool]]:
    return [
        _compile_bool_op(column, key, value, depth=depth)
        if key in ("$or", "$and")
        else _compile_field(column, key, value)
        for key, value in filters.items()
    ]


def _compile_bool_op(
    column: InstrumentedAttribute[Any],
    op: str,
    branches: Any,
    *,
    depth: int,
) -> ColumnElement[bool]:
    if depth >= _MAX_FILTER_DEPTH:
        raise ValueError(f"An @auth.on filter nests deeper than {_MAX_FILTER_DEPTH} levels")
    if not isinstance(branches, list) or len(branches) < 2:
        raise ValueError(f"An @auth.on filter's '{op}' must be a list of at least 2 filters")
    # An empty branch would compile to and_() == TRUE, making OR collapse to TRUE
    # and silently bypass the constraint. Reject it.
    if any(not isinstance(branch, dict) or not branch for branch in branches):
        raise ValueError(f"An @auth.on filter's '{op}' holds an empty or non-dict branch")
    compiled = [and_(*_compile(column, branch, depth=depth + 1)) for branch in branches]
    return or_(*compiled) if op == "$or" else and_(*compiled)


def _compile_field(
    column: InstrumentedAttribute[Any],
    key: str,
    value: Any,
) -> ColumnElement[bool]:
    if not isinstance(value, dict):
        # Bare value -> equality via JSONB containment: metadata @> {key: value}.
        return column.op("@>")({key: value})
    if len(value) != 1:
        raise ValueError(f"An @auth.on filter on '{key}' must hold exactly one operator, $eq or $contains")
    operator, operand = next(iter(value.items()))
    if operator == "$eq":
        return column.op("@>")({key: operand})
    if operator == "$contains":
        items = operand if isinstance(operand, list) else [operand]
        # metadata->'key' @> '[items...]' : the array at `key` contains every item.
        return column[key].op("@>")(items)
    raise ValueError(f"An @auth.on filter uses operator '{operator}'; only $eq and $contains are supported")
