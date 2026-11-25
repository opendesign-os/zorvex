"""A2UI v0.9 over A2A, as the Agent Server carries it: negotiation, and the payloads checked against the spec.

A2UI is on for an assistant whose ``metadata.a2a.a2ui`` is true, or, unless
that says false, for every assistant when ``A2A_A2UI_ENABLED`` is set. A client
activates it by naming the extension in an ``A2A-Extensions`` (or
``X-A2A-Extensions``) header; it is negotiated when both hold. A2UI travels in
data parts whose ``metadata.mimeType`` is the A2UI MIME type: inbound they
become content blocks of the user's message, outbound the agent's A2UI blocks
and the results of the tools ``A2A_A2UI_TOOL_CALL_RESULTS`` names become such
parts. Payloads are checked against the A2UI v0.9 schemas (a2ui_schemas/).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, TypeGuard

import structlog
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.protocols import Validator
from starlette.datastructures import Headers

from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.9"
MIME_TYPE = "application/json+a2ui"
# The other spelling clients send; answered in the negotiated one.
_ALTERNATE_MIME_TYPE = "application/a2ui+json"
EXTENSION_HEADERS = ("a2a-extensions", "x-a2a-extensions")

_SCHEMAS = Path(__file__).with_name("a2ui_schemas")
Direction = Literal["client", "server"]


class A2UIValidationError(ValueError):
    """A2UI metadata or a payload that does not match A2UI v0.9."""


def is_mime_type(value: Any) -> TypeGuard[str]:
    return value in (MIME_TYPE, _ALTERNATE_MIME_TYPE)


def enabled_for(assistant_metadata: Mapping[str, Any] | None) -> bool:
    """Whether A2UI is on for an assistant: its ``metadata.a2a.a2ui``, else the deployment's setting."""
    a2a = (assistant_metadata or {}).get("a2a")
    override = a2a.get("a2ui") if isinstance(a2a, dict) else None
    return override if isinstance(override, bool) else settings.a2a.A2A_A2UI_ENABLED


def extensions_requested(headers: Headers) -> list[str]:
    """The extension URIs the request's extension headers name, each once, in order."""
    found: list[str] = []
    for header in EXTENSION_HEADERS:
        for value in headers.getlist(header):
            for item in value.split(","):
                if (uri := item.strip()) and uri not in found:
                    found.append(uri)
    return found


@dataclass(frozen=True)
class A2UIContext:
    """What one exchange settled about A2UI."""

    enabled: bool = False
    active: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: list[str] = field(default_factory=list)
    # In history, an invalid payload is left out instead of failing the request.
    drop_invalid: bool = False
    mime_type: str = MIME_TYPE

    @property
    def negotiated(self) -> bool:
        return self.enabled and self.active

    def part_for(self, data: Any, metadata: Mapping[str, Any], *, direction: Direction) -> dict[str, Any] | None:
        """*data* as an A2UI data part, or None when A2UI is not negotiated (or the payload invalid and dropped)."""
        if not self.negotiated:
            return None
        try:
            validate_payload(data, direction=direction)
        except A2UIValidationError:
            if not self.drop_invalid:
                raise
            logger.warning("Dropping invalid A2UI payload", direction=direction)
            return None
        return {"kind": "data", "data": data, "metadata": normalize_metadata(metadata, mime_type=self.mime_type)}

    def parts_per_message(
        self, data: Any, metadata: Mapping[str, Any], *, direction: Direction
    ) -> list[dict[str, Any]]:
        """One A2UI part per message of *data*, a message or a list of them."""
        part = self.part_for(data, metadata, direction=direction)
        if part is None:
            return []
        messages = part["data"] if isinstance(part["data"], list) else [part["data"]]
        return [{**part, "data": message} for message in messages]


DISABLED = A2UIContext()


def message_context(
    message: Mapping[str, Any],
    assistant_metadata: Mapping[str, Any] | None,
    headers: Headers,
    *,
    default_mime_type: str = MIME_TYPE,
) -> A2UIContext:
    """What an inbound message settles: A2UI on and active, its metadata, extensions and MIME type.

    Raises:
        A2UIValidationError: For message metadata that is not an object.
    """
    metadata = message.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise A2UIValidationError("Invalid A2A message metadata")
    extensions = extensions_requested(headers)
    mime_type = default_mime_type
    for part in message.get("parts") or []:
        part_metadata = part.get("metadata") if isinstance(part, dict) else None
        if isinstance(part_metadata, dict) and is_mime_type(part_metadata.get("mimeType")):
            mime_type = part_metadata["mimeType"]
            break
    return A2UIContext(
        enabled=enabled_for(assistant_metadata),
        active=EXTENSION_URI in extensions,
        metadata=metadata,
        extensions=extensions,
        mime_type=mime_type,
    )


def run_context(run_metadata: Mapping[str, Any] | None) -> A2UIContext:
    """The A2UI a task's run was started with, as its ``a2a`` metadata recorded it."""
    a2a = (run_metadata or {}).get("a2a")
    a2a = a2a if isinstance(a2a, dict) else {}
    mime_type = a2a.get("a2ui_mime_type")
    return A2UIContext(
        enabled=a2a.get("a2ui") is True,
        active=a2a.get("a2ui_active") is True,
        mime_type=mime_type if is_mime_type(mime_type) else MIME_TYPE,
    )


def run_metadata(
    assistant_metadata: Mapping[str, Any] | None,
    context: A2UIContext,
    *,
    task_id: str | None,
    task_start_id: str | None,
) -> dict[str, Any]:
    """The ``a2a`` metadata of a task's run: its task, and with A2UI on, the assistant's ``a2a`` settings and A2UI's."""
    a2a = (assistant_metadata or {}).get("a2a")
    inherited = (
        {key: value for key, value in a2a.items() if key not in ("task_id", "task_start_id")}
        if context.enabled and isinstance(a2a, dict)
        else {}
    )
    recorded: dict[str, Any] = {**inherited}
    if task_id:
        recorded["task_id"] = task_id
    if task_start_id:
        recorded["task_start_id"] = task_start_id
    if context.enabled:
        recorded |= {"a2ui": True, "a2ui_active": context.active}
    if context.negotiated:
        recorded["a2ui_mime_type"] = context.mime_type
    return {"a2a": recorded}


def normalize_metadata(metadata: Mapping[str, Any], *, mime_type: str = MIME_TYPE) -> dict[str, Any]:
    """Part metadata with an A2UI MIME type spelled as negotiated."""
    if is_mime_type(metadata.get("mimeType")):
        return {**metadata, "mimeType": mime_type}
    return dict(metadata)


def validate_client_metadata(metadata: Mapping[str, Any]) -> None:
    """Check the A2UI client capabilities and data model a message's metadata carries, if any."""
    for key, schema in (
        ("a2uiClientCapabilities", "client_capabilities"),
        ("a2uiClientDataModel", "client_data_model"),
    ):
        value = metadata.get(key)
        if value is not None and not _validator(schema).is_valid(value):
            raise A2UIValidationError("Invalid A2UI message metadata")


def validate_payload(data: Any, *, direction: Direction) -> None:
    """Check an A2UI payload, one message or a list of them, against the client's or the server's schema."""
    if isinstance(data, list):
        messages: Iterable[Any] = data
    elif isinstance(data, dict):
        messages = (data,)
    else:
        raise A2UIValidationError("A2UI DataPart data must be an array or object")
    validator = _validator("client_to_server" if direction == "client" else "server_to_client")
    if not all(validator.is_valid(message) for message in messages):
        raise A2UIValidationError(f"Invalid A2UI {direction} message")


@cache
def _validator(name: str) -> Validator:
    schema = _relax_catalog_refs(json.loads((_SCHEMAS / f"{name}.json").read_text(encoding="utf-8")))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _relax_catalog_refs(value: Any) -> Any:
    """The schema with references into the catalog read as any object, so custom catalogs stay opaque."""
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("catalog.json#"):
            return {"type": "object"}
        return {key: _relax_catalog_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relax_catalog_refs(item) for item in value]
    return value
