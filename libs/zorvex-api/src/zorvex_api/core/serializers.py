"""JSON-compatible conversion for graph values: messages, models, dataclasses, interrupts and the rest."""

import dataclasses
import inspect
from base64 import b64encode
from collections import deque
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID


def to_jsonable(obj: Any) -> Any:
    """``obj`` as JSON-compatible data. Also works as ``json.dumps(default=to_jsonable)``."""
    # A class (e.g. a Pydantic class passed to with_structured_output) is named, not dumped.
    if inspect.isclass(obj):
        return f"{obj.__module__}.{obj.__qualname__}"
    if hasattr(obj, "model_dump") and callable(obj.model_dump):  # Pydantic v2
        return to_jsonable(obj.model_dump())
    if hasattr(obj, "dict") and callable(obj.dict):  # LangChain objects, Pydantic v1
        return to_jsonable(obj.dict())
    if dataclasses.is_dataclass(obj):  # LangGraph Interrupt, Command, Send
        return {field.name: to_jsonable(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if hasattr(obj, "_asdict") and callable(obj._asdict):  # NamedTuples such as PregelTask
        return {key: to_jsonable(value) for key, value in obj._asdict().items()}
    if isinstance(obj, bytes):
        return b64encode(obj).decode("ascii")
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (UUID, Decimal, PurePath)):
        return str(obj)
    if isinstance(obj, Exception):
        return {"type": type(obj).__name__, "message": str(obj)}
    if isinstance(obj, (set, frozenset, deque, tuple, list)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return _mapping(obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _key(key: Any) -> str:
    converted = to_jsonable(key)
    if isinstance(converted, str):
        return converted
    if isinstance(converted, bool):
        return "true" if converted else "false"
    if converted is None:
        return "null"
    return str(converted)


def _mapping(mapping: dict[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    original_keys: dict[str, Any] = {}
    for key, value in mapping.items():
        converted = _key(key)
        if converted in result and original_keys[converted] != key:
            raise ValueError(f"Mapping keys {original_keys[converted]!r} and {key!r} serialize to the same key")
        original_keys[converted] = key
        result[converted] = to_jsonable(value)
    return result
