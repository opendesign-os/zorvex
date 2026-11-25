"""The authenticated user."""

from typing import Any, cast

from langgraph_sdk.auth.types import BaseUser
from pydantic import BaseModel, ConfigDict, Field


class User(BaseModel):
    """The authenticated user: an identity plus whatever else the auth handler returned.

    Extra fields are kept, and readable as attributes or keys (``user.team_id``,
    ``user["team_id"]``, ``"team_id" in user``) as the SDK's BaseUser protocol expects.
    """

    model_config = ConfigDict(extra="allow")

    identity: str
    is_authenticated: bool = True
    permissions: list[str] = []
    display_name: str = Field(default_factory=lambda data: data["identity"])
    # Common optional fields, typed for IDE hints.
    org_id: str | None = None
    email: str | None = None

    def __getitem__(self, key: str) -> Any:
        if key in type(self).model_fields:
            return getattr(self, key)
        extra = self.model_extra or {}
        if key in extra:
            return extra[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in type(self).model_fields or key in (self.model_extra or {}))


def as_base_user(user: User) -> BaseUser:
    """*user* as the SDK's ``BaseUser``, which its auth contexts and runtimes take.

    A User is one, but the protocol leaves ``__contains__`` and ``__iter__`` unannotated,
    which pyright reads as returning None, so it matches no user class there.
    """
    return cast(BaseUser, user)
