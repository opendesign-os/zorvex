"""What a paused run asks, as MCP elicitation requests, and what the answers resume it with.

Over MCP 2026-07-28 a tool call that needs human input answers
``input_required`` with the requests the client puts to its user, then is
called again with the answers. The interrupts a run paused on make one round of
questions, as the Agent Server asks them:

- an interrupt asking a question (a text value, or any value with a
  ``response_schema``) is one form, and its answer resumes it;
- an ``mcp_elicitation`` interrupt relays the elicitation requests its value
  holds, and is resumed with every response, accepted or not;
- a human-in-the-loop review (``action_requests`` with ``review_configs``) and
  a ``tool_approval`` ask one form per action: approve, edit, reject or
  respond, as the action allows.

An interrupt no form can ask for (or a URL that is not safe to send a user to)
is ``UnaskableInterrupt``; the call then ends with an error result.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal, Protocol
from urllib.parse import urlsplit

import mcp_types
import structlog
from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import SchemaError
from jsonschema.protocols import Validator
from mcp.shared.exceptions import MCPError
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError
from referencing.exceptions import Unresolvable

from zorvex_api.constants import LOOPBACK_HOSTS
from zorvex_api.services.mcp_forms import FormSchema, FormTooLarge, UnsupportedSchema

logger = structlog.getLogger(__name__)

Mode = Literal["form", "url"]
Refusal = Literal["decline", "cancel"]
Results = Mapping[str, mcp_types.ElicitResult]

UNASKABLE_INPUT_MESSAGE = "This tool needs input that can't be asked over MCP"
_ENDED: dict[str, str] = {"decline": "declined", "cancel": "cancelled"}
_TEXT_ANSWER: dict[str, Any] = {"type": "string"}
_ELICIT_PARAMS: TypeAdapter[mcp_types.ElicitRequestParams] = TypeAdapter(mcp_types.ElicitRequestParams)
_INTERRUPT_ID = re.compile(r"[0-9a-f]{32}")
_MAX_URL_LENGTH = 4096
_UNSAFE_URL = "An elicitation URL needs https (or http on this machine), a host, and no credentials"
_DIGEST_LENGTH = 16

# The decisions a review offers, as a form titles them.
_DECISION_TITLES = {
    "approve": "Approve",
    "edit": "Edit the arguments",
    "reject": "Reject",
    "respond": "Answer for the tool",
}
_MESSAGE_DECISIONS = frozenset({"reject", "respond"})
_REVIEW_MESSAGE: dict[str, Any] = {
    "type": "string",
    "title": "Message",
    "description": "Why you reject the call, or the answer to send in its place",
}
_REVIEW_ARGS: dict[str, Any] = {
    "type": "string",
    "title": "Arguments",
    "description": "The arguments to run the tool with, as a JSON object",
}


def digest(*parts: Any) -> str:
    """A short, stable digest of JSON values."""
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:_DIGEST_LENGTH]


class UnaskableInterrupt(ValueError):
    """An interrupt that no MCP elicitation can ask."""

    @property
    def message(self) -> str:
        return f"{UNASKABLE_INPUT_MESSAGE}: {self}"


class InvalidAnswer(MCPError):
    """An answer that does not fit what was asked: invalid params."""

    def __init__(self, message: str) -> None:
        super().__init__(mcp_types.INVALID_PARAMS, message)


@dataclass(frozen=True, slots=True)
class Resume:
    """The answers make a resume value."""

    value: Any

    @classmethod
    def single(cls, interrupt_id: str, value: Any) -> Resume:
        """One interrupt's value; keyed by its id when it would otherwise read as a map of interrupt ids."""
        if isinstance(value, dict) and all(_INTERRUPT_ID.fullmatch(key) for key in value):
            return cls({interrupt_id: value})
        return cls(value)


@dataclass(frozen=True, slots=True)
class EndCall:
    """The user declined or dismissed a request the graph cannot go on without."""

    refusal: Refusal

    @property
    def message(self) -> str:
        return f"The request for input was {_ENDED[self.refusal]}."


Reply = Resume | EndCall


@dataclass(frozen=True, slots=True)
class FormQuestion:
    """A form the user fills in."""

    key: str
    message: str
    form: FormSchema
    mode: ClassVar[Mode] = "form"

    @property
    def request(self) -> mcp_types.ElicitRequest:
        params = mcp_types.ElicitRequestFormParams(message=self.message, requested_schema=self.form.schema)
        return mcp_types.ElicitRequest(params=params)

    def answer(self, result: mcp_types.ElicitResult) -> Any:
        return self.form.value(self._content(result))

    def relayed(self, result: mcp_types.ElicitResult) -> dict[str, Any]:
        if result.action != "accept":
            return {"action": result.action}
        return {"action": "accept", "content": self._content(result)}

    def _content(self, result: mcp_types.ElicitResult) -> dict[str, Any]:
        content = self.form.parse(result.content)
        if content is None:
            raise InvalidAnswer("Form content does not match requestedSchema")
        return content


@dataclass(frozen=True, slots=True)
class UrlQuestion:
    """A page the user is sent to."""

    key: str
    message: str
    url: str
    mode: ClassVar[Mode] = "url"

    @property
    def request(self) -> mcp_types.ElicitRequest:
        return mcp_types.ElicitRequest(params=mcp_types.ElicitRequestURLParams(message=self.message, url=self.url))

    def relayed(self, result: mcp_types.ElicitResult) -> dict[str, Any]:
        return {"action": result.action}


Question = FormQuestion | UrlQuestion


class Prompt(Protocol):
    """What one interrupt asks, and the value its answers resume it with."""

    @property
    def questions(self) -> Sequence[Question]: ...

    def scoped(self, scope: str) -> Prompt: ...

    def refusal(self, results: Results) -> EndCall | None: ...

    def resume_value(self, results: Results) -> Any: ...


@dataclass(frozen=True, slots=True)
class AnswerPrompt:
    """An interrupt asking a question: a text value, or any value with a response schema."""

    question: FormQuestion

    @classmethod
    def of(cls, interrupt: Mapping[str, Any]) -> AnswerPrompt:
        value = interrupt.get("value")
        text = _text(value)
        schema = _answer_schema(text, interrupt.get("response_schema"))
        with _unaskable_schemas():
            form = FormSchema.wrapping(schema)
        message = text or _compact(value)
        return cls(FormQuestion(f"answer-{digest(value, schema)}", message, form))

    @property
    def questions(self) -> tuple[FormQuestion]:
        return (self.question,)

    def scoped(self, scope: str) -> AnswerPrompt:
        return self

    def refusal(self, results: Results) -> EndCall | None:
        action = results[self.question.key].action
        return EndCall(action) if action in ("decline", "cancel") else None

    def resume_value(self, results: Results) -> Any:
        return self.question.answer(results[self.question.key])


@dataclass(frozen=True, slots=True)
class RelayedRequests:
    """An ``mcp_elicitation`` interrupt: requests the graph wrote for the client, answered as they come."""

    requests: tuple[Question, ...]
    scope: str = ""

    @property
    def questions(self) -> tuple[Question, ...]:
        return tuple(replace(request, key=self.scope + request.key) for request in self.requests)

    def scoped(self, scope: str) -> RelayedRequests:
        return replace(self, scope=f"{scope}/")

    def refusal(self, results: Results) -> None:
        return None

    def resume_value(self, results: Results) -> Any:
        return {
            "responses": {request.key: request.relayed(results[self.scope + request.key]) for request in self.requests}
        }


@dataclass(frozen=True, slots=True)
class _Dialect:
    """How a review names its keys: snake_case (Python) or camelCase (JS)."""

    allowed_decisions: str
    edited_action: str

    def decisions_of(self, config: Any) -> Any:
        return config.get(self.allowed_decisions) if isinstance(config, dict) else None

    def edit(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"type": "edit", self.edited_action: {"name": name, "args": args}}


_SNAKE_CASE = _Dialect("allowed_decisions", "edited_action")
_CAMEL_CASE = _Dialect("allowedDecisions", "editedAction")


@dataclass(slots=True)
class _ReviewKeys:
    """Keys for the review forms of one round: the same action asked twice gets a count."""

    asked: Counter[str] = field(default_factory=Counter)

    def next(self, *action: Any) -> str:
        reviewed = digest(*action)
        self.asked[reviewed] += 1
        return f"review-{reviewed}.{self.asked[reviewed]}"


@dataclass(frozen=True, slots=True)
class _ReviewedAction:
    """One tool call put to the user, with the decisions it allows."""

    name: str
    decisions: tuple[str, ...]
    question: FormQuestion
    dialect: _Dialect

    @classmethod
    def of(cls, request: Any, decisions: Any, dialect: _Dialect, *, keys: _ReviewKeys) -> _ReviewedAction:
        if not (
            isinstance(request, dict)
            and isinstance(request.get("name"), str)
            and isinstance(request.get("args"), dict)
            and _are_decisions(decisions)
        ):
            raise UnaskableInterrupt("A review needs a tool name, its arguments and its decisions")
        name, args = request["name"], request["args"]
        allowed = tuple(dict.fromkeys(decisions))
        message = _text(request.get("description")) or f"Run {name}?"
        key = keys.next(name, args, message, allowed)
        return cls(name, allowed, FormQuestion(key, message, _review_form(args, allowed)), dialect)

    def refusal(self, results: Results) -> EndCall | None:
        action = results[self.question.key].action
        if action == "cancel":
            return EndCall("cancel")
        if action == "decline" and "reject" not in self.decisions:
            return EndCall("decline")
        return None

    def decision(self, results: Results) -> dict[str, Any]:
        result = results[self.question.key]
        if result.action == "decline":
            return {"type": "reject"}
        return _decision(self.name, self.question.answer(result), self.dialect)


@dataclass(frozen=True, slots=True)
class ReviewPrompt:
    """A human-in-the-loop review: one form per action, resumed with the list of decisions."""

    actions: tuple[_ReviewedAction, ...]

    @classmethod
    def of(
        cls, requests: Sequence[Any], configs: Sequence[Any], dialect: _Dialect, *, keys: _ReviewKeys
    ) -> ReviewPrompt:
        if len(requests) != len(configs):
            raise UnaskableInterrupt("A review needs one review config per action")
        return cls(
            tuple(
                _ReviewedAction.of(request, dialect.decisions_of(config), dialect, keys=keys)
                for request, config in zip(requests, configs, strict=True)
            )
        )

    @property
    def questions(self) -> tuple[FormQuestion, ...]:
        return tuple(action.question for action in self.actions)

    def scoped(self, scope: str) -> ReviewPrompt:
        return self

    def refusal(self, results: Results) -> EndCall | None:
        for action in self.actions:
            if (refusal := action.refusal(results)) is not None:
                return refusal
        return None

    def resume_value(self, results: Results) -> Any:
        return {"decisions": [action.decision(results) for action in self.actions]}


@dataclass(frozen=True, slots=True)
class ToolApprovalPrompt:
    """A ``tool_approval`` interrupt: one action, resumed with a decision its response schema admits."""

    action: _ReviewedAction
    schema: Any

    @classmethod
    def of(cls, approval: Any, schema: Any, *, keys: _ReviewKeys) -> ToolApprovalPrompt:
        action = _ReviewedAction.of(approval, _approval_decisions(schema), _SNAKE_CASE, keys=keys)
        return cls(action, schema)

    @property
    def questions(self) -> tuple[FormQuestion]:
        return (self.action.question,)

    def scoped(self, scope: str) -> ToolApprovalPrompt:
        return self

    def refusal(self, results: Results) -> EndCall | None:
        return self.action.refusal(results)

    def resume_value(self, results: Results) -> Any:
        decision = self.action.decision(results)
        validator = _decision_validator(self.schema)
        try:
            error = next(iter(validator.iter_errors(decision)), None) if validator is not None else None
        except Unresolvable as unresolvable:
            # The server never fetches a reference; a schema leaning on one checks nothing.
            logger.warning("A tool approval's response_schema has an unresolvable reference", error=str(unresolvable))
            error = None
        if error is not None:
            raise InvalidAnswer(f"The decision does not fit the tool: {_reason(error)}")
        return decision


@dataclass(frozen=True, slots=True)
class Round:
    """The questions the interrupts of one pause ask, by key, and the prompts they come from, by interrupt id."""

    prompts: Mapping[str, Prompt]
    questions: Mapping[str, Question]

    @classmethod
    def of(cls, interrupts: Sequence[Mapping[str, Any]]) -> Round:
        keys = _ReviewKeys()
        prompts = {_interrupt_id(interrupt): _prompt(interrupt, keys) for interrupt in interrupts}
        if len(prompts) > 1:
            scopes = _scopes(interrupts)
            prompts = {interrupt_id: prompt.scoped(scopes[interrupt_id]) for interrupt_id, prompt in prompts.items()}
        questions: dict[str, Question] = {}
        for prompt in prompts.values():
            for question in prompt.questions:
                if questions.setdefault(question.key, question) != question:
                    raise UnaskableInterrupt(f"Two different questions use the key {question.key!r}")
        return cls(prompts, questions)

    @property
    def digest(self) -> str:
        """What the round asks, to tell an answer to this round from one to another."""
        return digest(
            {
                key: question.request.model_dump(mode="json", exclude_none=True)
                for key, question in self.questions.items()
            }
        )

    @property
    def modes(self) -> frozenset[Mode]:
        return frozenset(question.mode for question in self.questions.values())

    @property
    def input_requests(self) -> dict[str, mcp_types.InputRequest]:
        return {key: question.request for key, question in self.questions.items()}

    def answered_by(self, answers: Mapping[str, mcp_types.InputResponse]) -> bool:
        return self.questions.keys() <= answers.keys()

    def reply(self, answers: Mapping[str, mcp_types.InputResponse]) -> Reply:
        """What the answers to every question make: a refusal that ends the call, or the value to resume with."""
        results = {key: _elicit_result(answers[key]) for key in self.questions}
        for prompt in self.prompts.values():
            if (refusal := prompt.refusal(results)) is not None:
                return refusal
        resumed = {interrupt_id: prompt.resume_value(results) for interrupt_id, prompt in self.prompts.items()}
        if len(resumed) == 1:
            ((interrupt_id, value),) = resumed.items()
            return Resume.single(interrupt_id, value)
        return Resume(resumed)


def _prompt(interrupt: Mapping[str, Any], keys: _ReviewKeys) -> Prompt:
    value = interrupt.get("value")
    if isinstance(value, dict) and value.get("type") == "mcp_elicitation":
        requests = value.get("requests")
        if isinstance(requests, list) and requests:
            return RelayedRequests(tuple(_listed_question(request) for request in requests))
        if isinstance(requests, dict) and requests:
            return RelayedRequests(tuple(_relayed_question(key, request) for key, request in requests.items()))
        raise UnaskableInterrupt("An mcp_elicitation interrupt needs a request")
    if isinstance(value, dict) and value.get("type") == "tool_approval":
        return ToolApprovalPrompt.of(value, interrupt.get("response_schema"), keys=keys)
    if isinstance(value, dict):
        for requests_key, configs_key, dialect in (
            ("action_requests", "review_configs", _SNAKE_CASE),
            ("actionRequests", "reviewConfigs", _CAMEL_CASE),
        ):
            requests, configs = value.get(requests_key), value.get(configs_key)
            if isinstance(requests, list) and requests and isinstance(configs, list):
                return ReviewPrompt.of(requests, configs, dialect, keys=keys)
    return AnswerPrompt.of(interrupt)


def _scopes(interrupts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Per interrupt, the scope that keeps its relayed keys apart from another's: its question and a count."""
    asked: Counter[str] = Counter()
    scopes: dict[str, str] = {}
    for interrupt in interrupts:
        question = digest(interrupt.get("value"))
        asked[question] += 1
        scopes[_interrupt_id(interrupt)] = f"{question}.{asked[question]}"
    return scopes


def _listed_question(request: Any) -> Question:
    return _relayed_question(request.get("key") if isinstance(request, dict) else None, request)


def _relayed_question(key: Any, request: Any) -> Question:
    params = _elicit_params(request)
    if isinstance(key, str) and key and params is not None and _text(params.message):
        if isinstance(params, mcp_types.ElicitRequestFormParams):
            with _unaskable_schemas():
                return FormQuestion(key, params.message, FormSchema.of(params.requested_schema))
        return UrlQuestion(key, params.message, _safe_url(params.url))
    raise UnaskableInterrupt("An mcp_elicitation request needs a key, a message, and a form schema or a URL")


def _elicit_params(request: Any) -> mcp_types.ElicitRequestParams | None:
    try:
        return _ELICIT_PARAMS.validate_python(request)
    except PydanticValidationError:
        return None


def _are_decisions(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and value in _DECISION_TITLES for value in values)
    )


def _approval_decisions(schema: Any) -> Any:
    """The decision types a ``tool_approval`` response schema admits: its discriminator's, or each branch's."""
    if not isinstance(schema, dict):
        return [None]
    discriminator = schema.get("discriminator")
    if isinstance(discriminator, dict) and isinstance(discriminator.get("mapping"), dict):
        return list(discriminator["mapping"])
    for union in ("oneOf", "anyOf"):
        if isinstance(schema.get(union), list):
            return [_decision_type(branch) for branch in schema[union]]
    return [_decision_type(schema)]


def _decision_type(branch: Any) -> Any:
    properties = branch.get("properties") if isinstance(branch, dict) else None
    kind = properties.get("type") if isinstance(properties, dict) else None
    const = kind.get("const") if isinstance(kind, dict) else None
    return const if isinstance(const, str) else None


def _decision_validator(schema: Any) -> Validator | None:
    """A validator for the decision, or None to take any: a schema that does not compile checks nothing."""
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        logger.warning(
            "A tool approval's response_schema does not compile, so this answer is not checked", error=error.message
        )
        return None
    return Draft202012Validator(schema)


def _reason(error: ValidationError) -> str:
    """Why a decision fails: within a union, the branch of its own type says best."""
    if error.validator in ("oneOf", "anyOf") and error.context:
        branches: dict[Any, list[ValidationError]] = {}
        for sub in error.context:
            branches.setdefault(sub.relative_schema_path[0], []).append(sub)
        for errors in branches.values():  # in branch order, as the union lists them
            if not any(list(sub.absolute_path) == ["type"] for sub in errors):
                return errors[0].message
    return error.message


def _review_form(args: dict[str, Any], decisions: tuple[str, ...]) -> FormSchema:
    """The form reviewing one action; too large with its arguments to edit, it leaves editing out."""
    with _unaskable_schemas():
        try:
            return FormSchema.of(_review_schema(args, decisions))
        except FormTooLarge:
            uneditable = tuple(decision for decision in decisions if decision != "edit")
            if not uneditable:
                raise
            return FormSchema.of(_review_schema(args, uneditable))


def _review_schema(args: dict[str, Any], decisions: tuple[str, ...]) -> dict[str, Any]:
    choices = [{"const": decision, "title": _DECISION_TITLES[decision]} for decision in decisions]
    fields: dict[str, Any] = {"decision": {"type": "string", "title": "Decision", "oneOf": choices}}
    if _MESSAGE_DECISIONS.intersection(decisions):
        fields["message"] = _REVIEW_MESSAGE
    if "edit" in decisions:
        fields["args"] = {**_REVIEW_ARGS, "default": _compact(args)}
    return {"type": "object", "properties": fields, "required": ["decision"]}


def _decision(name: str, answer: Any, dialect: _Dialect) -> dict[str, Any]:
    """A filled-in review form as the decision the graph resumes with."""
    chosen = answer.get("decision") if isinstance(answer, dict) else None
    if chosen == "approve":
        return {"type": "approve"}
    if chosen == "reject":
        reason = _text(answer.get("message"))
        return {"type": "reject", "message": reason} if reason else {"type": "reject"}
    if chosen == "respond" and _text(answer.get("message")):
        return {"type": "respond", "message": answer["message"]}
    if chosen == "edit" and isinstance(answer.get("args"), str):
        return dialect.edit(name, _json_object(answer["args"]))
    raise InvalidAnswer("A decision needs its message or its arguments")


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise InvalidAnswer("Edited arguments must be a JSON object")
    return value


def _safe_url(url: str) -> str:
    """*url*, when it is safe to send a user to: https (or http to this machine), a host, no credentials."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise UnaskableInterrupt(_UNSAFE_URL) from None
    secure = parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS)
    if (
        not secure
        or not parsed.hostname
        or parsed.username is not None
        or port == 0
        or len(url) > _MAX_URL_LENGTH
        or any(char.isspace() or not char.isprintable() for char in url)
    ):
        raise UnaskableInterrupt(_UNSAFE_URL)
    return url


@contextmanager
def _unaskable_schemas() -> Generator[None]:
    try:
        yield
    except UnsupportedSchema as error:
        raise UnaskableInterrupt(str(error)) from None


def _elicit_result(response: mcp_types.InputResponse) -> mcp_types.ElicitResult:
    if isinstance(response, mcp_types.ElicitResult):
        return response
    raise InvalidAnswer("Expected an elicitation result")


def _interrupt_id(interrupt: Mapping[str, Any]) -> str:
    interrupt_id = interrupt.get("id")
    if isinstance(interrupt_id, str) and interrupt_id:
        return interrupt_id
    raise UnaskableInterrupt("An interrupt without an id cannot be resumed")


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def _answer_schema(text: str | None, declared: Any) -> dict[str, Any]:
    if isinstance(declared, dict) and declared:
        return declared
    if text is not None:
        return _TEXT_ANSWER
    raise UnaskableInterrupt("An interrupt that is not a text question needs a response_schema")
