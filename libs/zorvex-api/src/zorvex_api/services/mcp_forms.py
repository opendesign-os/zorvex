"""Interrupt answers as MCP elicitation forms, as the Agent Server asks for them.

An MCP client renders a form from a flat object schema of primitive fields:
text, number, boolean, and single or multiple choice. A graph's response schema
is narrowed to that shape (local ``$ref`` resolved, a nullable field made
optional, keywords a form cannot show dropped), and a schema that is not an
object becomes a form of one field, ``response``. An answer is checked against
the form before it resumes the graph.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.protocols import Validator

Schema = dict[str, Any]

WRAPPED_FIELD = "response"
MAX_FIELDS = 100
MAX_FORM_BYTES = 1024 * 1024

_STRING_FORMATS = frozenset({"email", "uri", "date", "date-time"})
_DEFS_PREFIX = "#/$defs/"
_ROOT_KEYS = frozenset({"type", "properties", "required"})
_DROPPED_ROOT_KEYS = frozenset({"$schema", "$defs", "title", "description", "additionalProperties"})
_DROPPED_KEYS = frozenset({"$defs", "$schema", "examples", "$comment", "deprecated", "readOnly", "writeOnly"})
_ANNOTATIONS = frozenset({"title", "description", "default"})
# The keywords each kind of form field may carry.
_TEXT_KEYWORDS = _ANNOTATIONS | {"type", "minLength", "maxLength", "format"}
_CHOICE_KEYWORDS = _ANNOTATIONS | {"type", "enum", "enumNames"}
_TITLED_CHOICE_KEYWORDS = _ANNOTATIONS | {"type", "oneOf"}
_NUMBER_KEYWORDS = _ANNOTATIONS | {"type", "minimum", "maximum"}
_BOOLEAN_KEYWORDS = _ANNOTATIONS | {"type"}
_MULTI_SELECT_KEYWORDS = _ANNOTATIONS | {"type", "items", "minItems", "maxItems"}
_TITLED = frozenset({"const", "title"})
_NULL: Schema = {"type": "null"}


class UnsupportedSchema(ValueError):
    """A schema no MCP form can ask for."""


class FormTooLarge(UnsupportedSchema):
    """A form larger than a client is asked to render."""


@dataclass(frozen=True, slots=True)
class FormSchema:
    """An elicitation form's ``requestedSchema``, and how an answer to it reads."""

    schema: Schema
    validator: Validator = field(compare=False, repr=False)
    wrapped: bool = False

    @classmethod
    def of(cls, source: Mapping[str, Any]) -> FormSchema:
        """The form an object schema asks for.

        Raises:
            UnsupportedSchema: For a schema that is not one flat object of form fields.
        """
        if source.get("type") != "object":
            raise UnsupportedSchema("A form schema must describe an object")
        return cls._built(_form_root(dict(source), _Definitions.of(source)))

    @classmethod
    def wrapping(cls, source: Mapping[str, Any]) -> FormSchema:
        """The form any schema asks for: an object's own, or one required field, ``response``, holding the value."""
        if source.get("type") == "object":
            return cls.of(source)
        one_field: Schema = {
            "type": "object",
            "properties": _form_fields({WRAPPED_FIELD: dict(source)}, _Definitions.of(source)),
            "required": [WRAPPED_FIELD],
        }
        return cls._built(one_field, wrapped=True)

    @classmethod
    def _built(cls, schema: Schema, *, wrapped: bool = False) -> FormSchema:
        _check_size(schema)
        _check_fields(schema)
        return cls(schema, _validator({**schema, "additionalProperties": False}), wrapped)

    def parse(self, content: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """The answer when *content* fits the form, else None: fields left null are unset, repeated choices once."""
        if content is None:
            return None
        fields = self.schema.get("properties") or {}
        answer = {key: _each_once(value) for key, value in content.items() if value is not None or key not in fields}
        if not self.validator.is_valid(answer) or not _encodable(answer):
            return None
        return answer

    def value(self, content: Mapping[str, Any]) -> Any:
        """What an answer resumes the graph with: a wrapped schema's ``response``, else the answer itself."""
        return content[WRAPPED_FIELD] if self.wrapped else dict(content)


@dataclass(frozen=True, slots=True)
class _Definitions:
    """A schema's ``$defs``, which its fields may reference."""

    schemas: Mapping[str, Any]

    @classmethod
    def of(cls, source: Mapping[str, Any]) -> _Definitions:
        schemas = source.get("$defs")
        return cls(schemas if isinstance(schemas, dict) else {})

    def resolved(self, node: Schema) -> Schema:
        """*node* with its reference replaced by the definition (less its title), *node*'s own keywords kept."""
        all_of = node.get("allOf")
        if (
            isinstance(all_of, list)
            and len(all_of) == 1
            and isinstance(all_of[0], dict)
            and isinstance(all_of[0].get("$ref"), str)
        ):
            siblings = {key: value for key, value in node.items() if key != "allOf"}
            return self.resolved({**all_of[0], **siblings})
        if "$ref" not in node:
            return node
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith(_DEFS_PREFIX):
            raise UnsupportedSchema("Form schemas may only use local $defs")
        target = self.schemas.get(ref.removeprefix(_DEFS_PREFIX))
        if not isinstance(target, dict):
            raise UnsupportedSchema("Form schemas may only use known $defs")
        untitled = {key: value for key, value in target.items() if key != "title"}
        return {**untitled, **{key: value for key, value in node.items() if key != "$ref"}}


def _form_root(schema: Schema, definitions: _Definitions) -> Schema:
    properties = schema.get("properties")
    if (
        set(schema) - _ROOT_KEYS - _DROPPED_ROOT_KEYS
        or schema.get("additionalProperties", False)
        or not isinstance(properties, dict)
    ):
        raise UnsupportedSchema("A form schema must be one flat object")
    if not 0 < len(properties) <= MAX_FIELDS:
        raise UnsupportedSchema(f"A form needs 1 to {MAX_FIELDS} fields")
    kept = {key: value for key, value in schema.items() if key in _ROOT_KEYS}
    return {**kept, "properties": _form_fields(properties, definitions)}


def _form_fields(properties: Mapping[str, Any], definitions: _Definitions) -> Schema:
    fields = {name: _form_field(prop, definitions) for name, prop in properties.items()}
    _check_size(fields)
    return fields


def _form_field(prop: Any, definitions: _Definitions) -> Any:
    """A field as a form asks for it; one that is no form field is left for the check to refuse."""
    if not isinstance(prop, dict):
        return prop
    narrowed = _without_null(definitions.resolved(prop), definitions)
    kept = {key: value for key, value in narrowed.items() if key not in _DROPPED_KEYS and not key.startswith("x-")}
    if "default" in kept and kept["default"] is None:
        del kept["default"]
    const = kept.get("const")
    if isinstance(const, str):
        del kept["const"]
        kept |= {"type": "string", "enum": [const]}
    items = kept.get("items")
    if kept.get("type") == "array" and isinstance(items, dict):
        kept.pop("uniqueItems", None)
        kept["items"] = _multi_select_items(definitions.resolved(items))
    return kept


def _without_null(prop: Schema, definitions: _Definitions) -> Schema:
    """A nullable field as the optional field it stands for: its one other type, null dropped."""
    for union in ("anyOf", "oneOf"):
        options = prop.get(union)
        if isinstance(options, list) and _NULL in options:
            others = [option for option in options if option != _NULL]
            if len(others) != 1 or not isinstance(others[0], dict):
                raise UnsupportedSchema("A form field can't mix several types")
            outer = {key: value for key, value in prop.items() if key != union}
            return {**definitions.resolved(others[0]), **outer}
    kind = prop.get("type")
    if isinstance(kind, list) and "null" in kind:
        others = [option for option in kind if option != "null"]
        if len(others) != 1:
            raise UnsupportedSchema("A form field can't mix several types")
        return {**prop, "type": others[0]}
    return prop


def _multi_select_items(items: Schema) -> Schema:
    if "enum" in items:
        return {"type": "string", "enum": items["enum"]}
    if "anyOf" in items:
        return {"anyOf": items["anyOf"]}
    return items


def _check_size(value: Any) -> None:
    try:
        size = len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())
    except (TypeError, ValueError):
        size = MAX_FORM_BYTES + 1
    if size > MAX_FORM_BYTES:
        raise FormTooLarge(f"A form can't be over {MAX_FORM_BYTES} bytes")


def _check_fields(schema: Schema) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise UnsupportedSchema("A form schema must be one flat object")
    for name, prop in properties.items():
        if not isinstance(prop, dict) or not _is_form_field(prop):
            raise UnsupportedSchema(f"A form can't ask for field {name!r}")
        if "default" in prop and not _validator(prop).is_valid(prop["default"]):
            raise UnsupportedSchema(f"Field {name!r} has an invalid default")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(key, str) and key in properties for key in required):
        raise UnsupportedSchema("A form can only require its own fields")


def _is_form_field(prop: Schema) -> bool:
    keys = set(prop)
    kind = prop.get("type")
    if kind == "string" and "oneOf" in prop:
        return keys <= _TITLED_CHOICE_KEYWORDS and _is_titled_choices(prop["oneOf"])
    if kind == "string" and "enum" in prop:
        choices = prop["enum"]
        names = prop.get("enumNames", choices)
        return keys <= _CHOICE_KEYWORDS and _is_named_choices(choices, names)
    if kind == "string":
        string_format = prop.get("format")
        return keys <= _TEXT_KEYWORDS and ("format" not in prop or string_format in _STRING_FORMATS)
    if kind in ("number", "integer"):
        return keys <= _NUMBER_KEYWORDS
    if kind == "boolean":
        return keys <= _BOOLEAN_KEYWORDS
    if kind == "array" and "items" in prop:
        return keys <= _MULTI_SELECT_KEYWORDS and _is_multi_select(prop["items"])
    return False


def _is_named_choices(choices: Any, names: Any) -> bool:
    return (
        _is_string_enum(choices)
        and isinstance(names, list)
        and len(names) == len(choices)
        and all(isinstance(name, str) for name in names)
    )


def _is_multi_select(items: Any) -> bool:
    if not isinstance(items, dict):
        return False
    if set(items) == {"type", "enum"} and items["type"] == "string":
        return _is_string_enum(items["enum"])
    if set(items) == {"anyOf"}:
        return _is_titled_choices(items["anyOf"])
    return False


def _is_string_enum(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) for value in values)
        and len(set(values)) == len(values)
    )


def _is_titled_choices(choices: Any) -> bool:
    """Whether *choices* are ``{"const", "title"}`` pairs of distinct strings."""
    if not isinstance(choices, list):
        return False
    consts = [
        choice["const"]
        for choice in choices
        if isinstance(choice, dict) and set(choice) == _TITLED and isinstance(choice["title"], str)
    ]
    return len(consts) == len(choices) and _is_string_enum(consts)


def _validator(schema: Schema) -> Validator:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise UnsupportedSchema("Invalid form schema") from None
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _each_once(value: Any) -> Any:
    if isinstance(value, list) and all(isinstance(choice, str) for choice in value):
        return list(dict.fromkeys(value))
    return value


def _encodable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
