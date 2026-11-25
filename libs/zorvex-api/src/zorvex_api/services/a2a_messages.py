"""A2A messages and LangGraph messages, each into the other, in the Agent Server's wire format.

The Agent Server answers in JSON of its own: A2A 1.0's enum names
(``TASK_STATE_*``, ``ROLE_*``) with v0.3's ``kind`` discriminators and part
shapes (``text``; ``file`` holding ``bytes`` or ``uri``; ``data``).
``to_spec_format`` turns it into v0.3's for the v0.3 method names.

Inbound, a message's text parts become the graph's messages, or one message
of content blocks when the message carries files or A2UI; its object data
parts become the run input. Outbound, the thread's messages become history
attributed to tasks, the run's reply the "Assistant Response" artifact and its
interrupts the "Interrupt" artifact; tool results are data parts.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import structlog
from langchain_core.utils.uuid import uuid7

from zorvex_api.services import a2ui
from zorvex_api.services.a2ui import A2UIContext
from zorvex_api.settings import settings

logger = structlog.getLogger(__name__)

Role = Literal["ROLE_USER", "ROLE_AGENT"]
HistoryScope = Literal["context", "task"]

_ROLES_V03 = {"ROLE_USER": "user", "ROLE_AGENT": "agent"}
_STATES_V03 = {
    "TASK_STATE_SUBMITTED": "submitted",
    "TASK_STATE_WORKING": "working",
    "TASK_STATE_COMPLETED": "completed",
    "TASK_STATE_FAILED": "failed",
    "TASK_STATE_CANCELED": "canceled",
    "TASK_STATE_INPUT_REQUIRED": "input-required",
    "TASK_STATE_REJECTED": "rejected",
    "TASK_STATE_AUTH_REQUIRED": "auth-required",
}
_ROLES_FROM_V03 = {v03: v1 for v1, v03 in _ROLES_V03.items()}
STATES_FROM_V03 = {v03: v1 for v1, v03 in _STATES_V03.items()}

_USER_TYPES = frozenset({"human", "humanmessage", "humanmessagechunk"})
_AGENT_TYPES = frozenset({"ai", "aimessage", "aimessagechunk"})
_ROLE_DEFINED_TYPES = frozenset({"chat", "chatmessage", "chatmessagechunk"})
_USER_ROLES = frozenset({"human", "user"})
_AGENT_ROLES = frozenset({"agent", "assistant"})
_MEDIA_BLOCK_TYPES = frozenset({"audio", "file", "image", "video"})


class InvalidAgentResponseError(ValueError):
    """Agent output that no A2A part can carry: -32006."""


# --- inbound


def normalize_role(role: Any) -> Any:
    """A message role as A2A 1.0 names it; v0.3's ``user`` and ``agent`` are accepted."""
    return _ROLES_FROM_V03.get(role, role)


def message_input(
    parts: Sequence[Any],
    role: Any,
    message_id: str,
    *,
    a2ui_context: A2UIContext,
    task_id: str | None = None,
    task_start_id: str | None = None,
) -> dict[str, Any]:
    """The run input a message makes: ``messages`` from its text, files and A2UI, and its data parts' keys.

    Text alone makes one message per text part; with a file or an A2UI part,
    one message of content blocks in part order. Each carries, under
    ``additional_kwargs.a2a``, the task it belongs to (or the start of a new
    one) and the message's metadata and the extensions the client named.

    Raises:
        ValueError: For a part that is not text, a file or data, a data part that is not an
            object, a malformed file part, or a message carrying nothing.
        A2UIValidationError: For A2UI the assistant does not take, or that does not validate.
    """
    a2ui.validate_client_metadata(a2ui_context.metadata)
    has_blocks = any(isinstance(part, dict) and ("file" in part or is_a2ui_part(part)) for part in parts)
    langgraph_role = "human" if role == "ROLE_USER" else "assistant"
    messages: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    data: dict[str, Any] = {}

    for part in parts:
        if not isinstance(part, dict):
            raise ValueError(
                "Each message part must be an object. A2A agents support 'text', 'data', and 'file' parts only."
            )
        present = [key for key in ("text", "file", "data") if key in part]
        if len(present) > 1:
            raise ValueError(
                f"Each message part must contain exactly one of 'text', 'file', or 'data'. Got multiple: {present}."
            )
        if "text" in part:
            if has_blocks:
                blocks.append({"type": "text", "text": part["text"]})
            else:
                messages.append({"role": langgraph_role, "content": part["text"], "id": message_id})
        elif "file" in part:
            blocks.append(_file_block(part))
        elif "data" in part:
            value = part.get("data", {})
            if is_a2ui_part(part):
                if not a2ui_context.enabled:
                    raise a2ui.A2UIValidationError("A2UI is not enabled for this agent")
                if not a2ui_context.active:
                    raise a2ui.A2UIValidationError("The A2UI extension must be activated")
                a2ui.validate_payload(value, direction="client")
                metadata = a2ui.normalize_metadata(part["metadata"], mime_type=a2ui_context.mime_type)
                blocks.append({"type": "data", "data": value, "metadata": metadata})
                continue
            if not isinstance(value, dict):
                raise ValueError("DataPart must contain a JSON object in the 'data' field")
            data.update(value)
        else:
            raise ValueError("Unsupported part type. A2A agents support 'text', 'data', and 'file' parts only.")

    if has_blocks:
        messages.append({"role": langgraph_role, "content": blocks, "id": message_id})
    if task_id or task_start_id or a2ui_context.metadata or a2ui_context.extensions:
        stamp: dict[str, Any] = {}
        if task_id:
            stamp["task_id"] = task_id
        if task_start_id:
            stamp["task_start_id"] = task_start_id
        if a2ui_context.metadata:
            stamp["metadata"] = a2ui_context.metadata
        if a2ui_context.extensions:
            stamp["extensions"] = a2ui_context.extensions
        for message in messages:
            message["additional_kwargs"] = {"a2a": stamp}
    if not messages and not data:
        raise ValueError("Message must contain at least one valid text, data, or file part")

    run_input: dict[str, Any] = {}
    if messages:
        run_input["messages"] = messages
    run_input.update(data)
    return run_input


def text_of_parts(parts: Sequence[Any]) -> str | None:
    """The message's text parts joined by newlines, or None when it has none."""
    texts = [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    return "\n".join(texts) if texts else None


def resume_values(parts: Sequence[Any]) -> list[Any]:
    """The ``resume`` values of the message's object data parts."""
    return [
        part["data"].get("resume")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("data"), dict) and "resume" in part["data"]
    ]


def is_a2ui_part(part: Mapping[str, Any]) -> bool:
    """Whether a part is A2UI: a data part whose ``metadata.mimeType`` is the A2UI MIME type."""
    metadata = part.get("metadata")
    return "data" in part and isinstance(metadata, dict) and a2ui.is_mime_type(metadata.get("mimeType"))


def _file_block(part: Mapping[str, Any]) -> dict[str, Any]:
    """A file part as a LangChain content block; a URI is passed on, never fetched."""
    file = part.get("file")
    if not isinstance(file, dict):
        raise ValueError("FilePart must contain a 'file' object with 'bytes' or 'uri'.")
    data, uri = file.get("bytes"), file.get("uri")
    if data is None and uri is None:
        raise ValueError("FilePart 'file' must contain either 'bytes' or 'uri'.")
    if data is not None and uri is not None:
        raise ValueError("FilePart 'file' must contain only one of 'bytes' or 'uri'.")
    mime_type = file.get("mimeType")
    kind = mime_type.split("/", 1)[0].strip().lower() if isinstance(mime_type, str) else ""
    block: dict[str, Any] = {"type": kind if kind in ("image", "audio", "video") else "file"}
    if data is not None:
        block["base64"] = data
    else:
        block["url"] = uri
    if mime_type:
        block["mime_type"] = mime_type
    if file.get("name"):
        block["extras"] = {"filename": file["name"]}
    return block


# --- outbound


def role_for_message(message: Mapping[str, Any]) -> Role | None:
    """The A2A role of a public LangChain message; None for an internal one (system, tool, function...)."""
    message_type = _label(message.get("type"))
    message_role = _label(message.get("role"))
    by_type: Role | None = None
    by_role: Role | None = None
    if message_type:
        if message_type in _USER_TYPES:
            by_type = "ROLE_USER"
        elif message_type in _AGENT_TYPES:
            by_type = "ROLE_AGENT"
        elif message_type not in _ROLE_DEFINED_TYPES:
            return None
    if message_role:
        if message_role in _USER_ROLES:
            by_role = "ROLE_USER"
        elif message_role in _AGENT_ROLES:
            by_role = "ROLE_AGENT"
        else:
            return None
    if by_type is not None and by_role is not None and by_type != by_role:
        return None
    return by_type or by_role


def _label(value: Any) -> str:
    return value.lower().replace("_", "").replace("-", "") if isinstance(value, str) else ""


def content_parts(
    content: Any, *, a2ui_context: A2UIContext, direction: a2ui.Direction = "server"
) -> list[dict[str, Any]]:
    """Message content as parts, in order: runs of text merged, media as file parts, A2UI as data parts.

    Raises:
        InvalidAgentResponseError: For a media block no file part can carry.
        A2UIValidationError: For an A2UI block that does not validate (unless dropped).
    """
    if isinstance(content, str):
        return [{"kind": "text", "text": content}]
    if content is None:
        return []
    if not isinstance(content, list):
        return [{"kind": "text", "text": str(content)}]
    parts: list[dict[str, Any]] = []
    texts: list[str] = []

    def flush() -> None:
        if texts:
            parts.append({"kind": "text", "text": "".join(texts)})
            texts.clear()

    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif not isinstance(block, dict):
            continue
        elif block.get("type") == "text":
            if isinstance(block.get("text"), str):
                texts.append(block["text"])
        elif block.get("type") == "data":
            metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
            if a2ui.is_mime_type(metadata.get("mimeType")):
                part = a2ui_context.part_for(block.get("data"), metadata, direction=direction)
                if part is not None:
                    flush()
                    parts.append(part)
        elif block.get("type") in _MEDIA_BLOCK_TYPES:
            flush()
            parts.append(_file_part(block))
    flush()
    return parts


def parts_have_content(parts: Sequence[Mapping[str, Any]]) -> bool:
    """Whether parts carry anything public: data, a file, or non-empty text."""
    return any(
        part.get("kind") in ("data", "file") or (part.get("kind") == "text" and part.get("text")) for part in parts
    )


def content_text(content: Any) -> str:
    """Message content as plain text: its text blocks joined."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
    return "".join(texts)


def _file_part(block: Mapping[str, Any]) -> dict[str, Any]:
    """A LangChain media block as a file part: its base64 (normalized), data or URL, MIME type and name."""
    source = next((key for key in ("base64", "data", "url") if block.get(key) is not None), None)
    if source is None:
        raise InvalidAgentResponseError("Media content blocks must contain 'base64', 'data', or 'url'.")
    value = block[source]
    file: dict[str, Any] = {}
    if source == "url":
        if not isinstance(value, str) or not value:
            raise InvalidAgentResponseError("Media content block 'url' must be a non-empty string.")
        file["uri"] = value
    else:
        if source == "data" and isinstance(value, (bytes, bytearray)):
            value = base64.b64encode(value).decode("ascii")
        if not isinstance(value, str):
            raise InvalidAgentResponseError(f"Media content block '{source}' must be base64-encoded text or bytes.")
        compact = "".join(value.split())
        try:
            decoded = base64.b64decode(compact + "=" * (-len(compact) % 4), altchars=b"-_", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidAgentResponseError(f"Media content block '{source}' must contain valid base64 data.") from exc
        file["bytes"] = base64.b64encode(decoded).decode("ascii")
    mime_type = block.get("mime_type")
    if mime_type is not None:
        if not isinstance(mime_type, str):
            raise InvalidAgentResponseError("Media content block 'mime_type' must be a string.")
        if mime_type:
            file["mimeType"] = mime_type
    filename = block.get("filename")
    if filename is None and isinstance(block.get("extras"), dict):
        filename = block["extras"].get("filename")
    if filename is not None:
        if not isinstance(filename, str):
            raise InvalidAgentResponseError("Media content block filename must be a string.")
        if filename:
            file["name"] = filename
    return {"kind": "file", "file": file}


def current_turn(messages: Sequence[Any]) -> list[Any]:
    """The messages after the last user message."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and role_for_message(message) == "ROLE_USER":
            return list(messages[index + 1 :])
    return list(messages)


def response_parts(
    result: Mapping[str, Any], *, a2ui_context: A2UIContext, turn: Sequence[Any] | None = None
) -> list[dict[str, Any]]:
    """The run's reply: the last agent message of the turn that says something, then the turn's tool UI.

    A result without messages is its text (as Python prints it), as on the Agent Server.
    """
    messages = result.get("messages")
    if "__error__" in result or not isinstance(messages, list) or not messages:
        return [{"kind": "text", "text": str(result)}]
    turn = current_turn(messages) if turn is None else list(turn)
    tool_ui = [part for message in turn for part in tool_result_a2ui_parts(message, a2ui_context)]
    for message in reversed(turn):
        if isinstance(message, dict) and role_for_message(message) == "ROLE_AGENT" and "content" in message:
            reply = content_parts(message["content"], a2ui_context=a2ui_context)
            if parts_have_content(reply):
                return [*reply, *tool_ui]
    return tool_ui or [{"kind": "text", "text": ""}]


def interrupt_artifact(interrupts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The interrupts a run paused on, one data part each: id, value, and response schema when there is one."""
    parts = []
    for interrupt in interrupts:
        data = {"id": interrupt.get("id"), "value": interrupt.get("value")}
        if interrupt.get("response_schema") is not None:
            data["response_schema"] = interrupt["response_schema"]
        parts.append({"kind": "data", "data": data})
    return {
        "artifactId": str(uuid4()),
        "name": "Interrupt",
        "description": "Agent requires input to continue",
        "parts": parts,
    }


def response_artifact(parts: list[dict[str, Any]], assistant_id: str) -> dict[str, Any]:
    return {
        "artifactId": str(uuid4()),
        "name": "Assistant Response",
        "description": f"Response from assistant {assistant_id}",
        "parts": parts,
    }


def response_artifact_update(
    *, task_id: str, context_id: str, parts: list[dict[str, Any]], assistant_id: str
) -> dict[str, Any]:
    return {
        "taskId": task_id,
        "contextId": context_id,
        "kind": "artifact-update",
        "artifact": response_artifact(parts, assistant_id),
        "append": False,
        "lastChunk": True,
    }


def interrupt_artifact_update(
    *, task_id: str, context_id: str, interrupts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "taskId": task_id,
        "contextId": context_id,
        "kind": "artifact-update",
        "artifact": interrupt_artifact(interrupts),
    }


def tool_result(message: Mapping[str, Any]) -> dict[str, Any] | None:
    """A tool message as a tool result, when it answers a tool call and its tool's results are shown."""
    tool_call_id = _tool_call_id(message)
    allowed = settings.a2a.allowed_tool_call_results
    if tool_call_id is None or (allowed is not None and message.get("name") not in allowed):
        return None
    result: dict[str, Any] = {"toolCallId": tool_call_id}
    if message.get("content") not in (None, ""):
        result["content"] = message["content"]
    for key in ("name", "status"):
        if isinstance(message.get(key), str) and message[key]:
            result[key] = message[key]
    return result


def tool_results_part(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "data", "data": {"tool_results": results}}


def tool_result_a2ui_parts(message: Any, a2ui_context: A2UIContext) -> list[dict[str, Any]]:
    """A2UI messages a listed tool returned, one part each, when A2UI is negotiated."""
    if (
        not a2ui_context.negotiated
        or not isinstance(message, dict)
        or message.get("name") not in settings.a2a.a2ui_tool_call_results
        or message.get("status") == "error"
        or _tool_call_id(message) is None
    ):
        return []
    content = message.get("content")
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        logger.warning("Dropping non-JSON A2UI tool result", tool=message.get("name"))
        return []
    return replace(a2ui_context, drop_invalid=True).parts_per_message(
        data, {"mimeType": a2ui_context.mime_type}, direction="server"
    )


def _tool_call_id(message: Mapping[str, Any]) -> str | None:
    tool_call_id = message.get("tool_call_id")
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def working_update(items: Sequence[Any], *, task_id: str, context_id: str) -> dict[str, Any]:
    """A WORKING status update carrying streamed messages: the public ones' text, then any tool results."""
    texts: list[str] = []
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if role_for_message(item) is None:
            if (found := tool_result(item)) is not None:
                results.append(found)
            continue
        if text := content_text(item.get("content")):
            texts.append(text)
    parts: list[dict[str, Any]] = []
    if texts:
        parts.append({"kind": "text", "text": "".join(texts)})
    if results:
        parts.append(tool_results_part(results))
    message = {
        "kind": "message",
        "role": "ROLE_AGENT",
        "parts": parts or [{"kind": "text", "text": ""}],
        "messageId": str(uuid4()),
        "taskId": task_id,
        "contextId": context_id,
    }
    return {
        "taskId": task_id,
        "contextId": context_id,
        "kind": "status-update",
        "status": {"state": "TASK_STATE_WORKING", "message": message, "timestamp": now_iso()},
        "final": False,
    }


def agent_message(text: str, *, task_id: str, context_id: str | None = None) -> dict[str, Any]:
    """A one-text-part agent message, for a status."""
    message: dict[str, Any] = {
        "kind": "message",
        "role": "ROLE_AGENT",
        "parts": [{"kind": "text", "text": text}],
        "messageId": str(uuid4()),
        "taskId": task_id,
    }
    if context_id is not None:
        message["contextId"] = context_id
    return message


@dataclass(frozen=True)
class View:
    """How a request sees a task's history: up to *length* messages (None for all, 0 for none) in its
    *scope*, A2UI parts as *a2ui* renders them, and the tasks earlier turns started, by start id."""

    length: int | None
    scope: HistoryScope
    a2ui: A2UIContext = field(default_factory=A2UIContext)
    task_ids: Mapping[str, str] | None = None


def history(messages: Sequence[Any], task_id: str, context_id: str, view: View) -> list[dict[str, Any]]:
    """The thread's public messages as A2A messages, each attributed to its task.

    A message stamped with a task (or a task's start) begins that task's
    segment, which runs until the next stamp; one stamped with a task of
    another assistant leaves out its segment. In the context scope, messages
    before the first stamp belong to the requested task; in the task scope,
    only the requested task's messages are kept. Tool results are agent data
    parts; system prompts and tool output not answering a call are left out.
    """
    history_a2ui = replace(view.a2ui, drop_invalid=True)
    current: str | None = None if view.scope == "task" else task_id
    known = set(view.task_ids.values()) if view.task_ids is not None else None
    converted: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        kwargs = message.get("additional_kwargs")
        stamp = kwargs.get("a2a") if isinstance(kwargs, dict) else None
        stamp = stamp if isinstance(stamp, dict) else {}
        stamped_task, stamped_start = stamp.get("task_id"), stamp.get("task_start_id")
        if isinstance(stamped_task, str) and stamped_task:
            current = stamped_task if known is None or stamped_task == task_id or stamped_task in known else None
        elif isinstance(stamped_start, str) and stamped_start:
            if view.task_ids is not None:
                current = view.task_ids.get(stamped_start)
            elif view.scope == "task":
                current = None
        if current is None:
            continue
        role = role_for_message(message)
        if role is None:
            parts: list[dict[str, Any]] = []
            if (found := tool_result(message)) is not None:
                parts.append(tool_results_part([found]))
            parts.extend(tool_result_a2ui_parts(message, history_a2ui))
            if not parts:
                continue
            role = "ROLE_AGENT"
        else:
            parts = content_parts(
                message.get("content", ""),
                a2ui_context=history_a2ui,
                direction="client" if role == "ROLE_USER" else "server",
            )
            if role == "ROLE_AGENT" and not parts_have_content(parts):
                continue
            parts = parts or [{"kind": "text", "text": ""}]
        converted.append(
            {
                "kind": "message",
                "role": role,
                "parts": parts,
                "messageId": message.get("id") or str(uuid7()),
                "taskId": current,
                "contextId": context_id,
            }
        )
    if view.scope == "task":
        converted = [message for message in converted if message["taskId"] == task_id]
    if view.length is None:
        return converted
    return converted[-view.length :] if view.length else []


@dataclass(frozen=True)
class RunMessages:
    """A run's messages, and where the ones it added begin (None when that cannot be told)."""

    messages: list[Any]
    added_from: int | None

    @classmethod
    def between(cls, before: list[Any], messages: list[Any]) -> RunMessages:
        """The messages a run ended with, against those it started with."""
        if messages[: len(before)] == before:
            return cls(messages, len(before))
        before_ids = {message.get("id") for message in before if isinstance(message, dict)}
        if len(before_ids) != len(before) or None in before_ids:
            return cls(messages, None)
        last_kept = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, dict) and message.get("id") in before_ids
            ),
            default=-1,
        )
        return cls(messages, last_kept + 1)

    @property
    def added(self) -> list[Any] | None:
        return None if self.added_from is None else self.messages[self.added_from :]

    def stamped(self, task_id: str) -> list[Any]:
        """The messages, the first the run added stamped with *task_id* unless it carries a stamp already."""
        if self.added_from is None or self.added_from >= len(self.messages):
            return self.messages
        message = self.messages[self.added_from]
        if not isinstance(message, dict):
            return self.messages
        kwargs = (
            dict(message.get("additional_kwargs") or {}) if isinstance(message.get("additional_kwargs"), dict) else {}
        )
        stamp = dict(kwargs.get("a2a") or {}) if isinstance(kwargs.get("a2a"), dict) else {}
        if stamp.get("task_id") or stamp.get("task_start_id"):
            return self.messages
        messages = list(self.messages)
        messages[self.added_from] = {**message, "additional_kwargs": {**kwargs, "a2a": {**stamp, "task_id": task_id}}}
        return messages


def to_spec_format(data: Any) -> Any:
    """A response in v0.3's spelling: lower-case roles and states, and a task no longer wrapped in ``task``."""
    if isinstance(data, dict):
        converted: dict[str, Any] = {}
        for key, value in data.items():
            if key == "role" and isinstance(value, str):
                converted[key] = _ROLES_V03.get(value, value)
            elif key == "state" and isinstance(value, str):
                converted[key] = _STATES_V03.get(value, value)
            elif key == "result" and isinstance(value, dict) and list(value) == ["task"]:
                converted[key] = to_spec_format(value["task"])
            else:
                converted[key] = to_spec_format(value)
        return converted
    if isinstance(data, list):
        return [to_spec_format(item) for item in data]
    return data


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
