"""The authentication dependency for FastAPI routes."""

from fastapi import Depends, Request

from zorvex_api.core import loopback
from zorvex_api.core.auth_middleware import get_auth_backend
from zorvex_api.models.auth import User

# Several dependencies on one request can call require_auth; the backend must run once
# because a credential may be single-use.
_AUTH_RESULT_SCOPE_KEY = "zorvex.auth_result"


async def require_auth(request: Request) -> User:
    """The request's authenticated user, resolved once per request.

    A loopback request (core/loopback.py) is not authenticated: it acts as the user its graph acts for.
    """
    user = request.scope.get(_AUTH_RESULT_SCOPE_KEY)
    if user is None:
        if loopback.is_loopback(request.scope):
            user = loopback.acting_user.get() or loopback.ANONYMOUS
        else:
            user = await get_auth_backend().authenticate(request)
        request.scope[_AUTH_RESULT_SCOPE_KEY] = user
    return user


# For applying to entire routers
auth_dependency = [Depends(require_auth)]
