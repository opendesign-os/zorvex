"""Error response models for Agent Protocol"""

from http import HTTPStatus
from typing import Any

from pydantic import BaseModel, Field


class AgentProtocolError(BaseModel):
    """Standard Agent Protocol error response"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")


def _response(status: int, description: str) -> dict[int | str, dict[str, Any]]:
    return {status: {"model": AgentProtocolError, "description": description}}


# Reusable OpenAPI error response declarations for endpoint decorators
BAD_REQUEST = _response(400, "Invalid request")
UNAUTHORIZED = _response(401, "Authentication failed")
FORBIDDEN = _response(403, "An @auth.on handler denied the request")
NOT_FOUND = _response(404, "Resource not found")
CONFLICT = _response(409, "Resource state conflict")
VALIDATION_ERROR = _response(422, "The request does not match the endpoint's parameters")

# SSE streaming response for OpenAPI documentation
SSE_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Server-Sent Events stream",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    },
}

# The statuses whose ``error`` is not their reason phrase.
_ERROR_TYPES = {422: "validation_error", 500: "internal_error"}


def get_error_type(status_code: int) -> str:
    """The ``error`` of a status: its HTTP reason phrase in snake case, as ``not_found`` for 404."""
    if status_code in _ERROR_TYPES:
        return _ERROR_TYPES[status_code]
    try:
        return HTTPStatus(status_code).phrase.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        return "error"
