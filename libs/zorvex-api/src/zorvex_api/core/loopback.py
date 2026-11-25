"""Graphs reaching their own server in process: the LangGraph SDK's loopback client, as the Agent Server serves it.

``langgraph_sdk.get_client()`` with no url talks to the server it runs in over
an in-process ASGI transport rooted at ``/noauth``. As on the Agent Server,
such a request is trusted: it skips authentication and the ``@auth.on``
handlers. It acts as the user the graph is built or run for (outside one, as
the anonymous user), so it reaches that user's threads and runs.

The SDK finds the server it runs in at ``langgraph_api.server.app``, and leaves
a client made before there is one for the server to wire up. ``serve`` does
both: it wires the clients made so far and publishes the app under that module
name alone, without a ``langgraph_api`` package, so nothing else mistakes this
server for that one.
"""

import sys
from contextvars import ContextVar
from types import ModuleType
from typing import Any

from langgraph_sdk.client import configure_loopback_transports
from starlette.types import Scope

from zorvex_api.models.auth import User

ROOT_PATH = "/noauth"
ANONYMOUS = User(identity="anonymous", display_name="Anonymous User", is_authenticated=False)
_SERVER_MODULE = "langgraph_api.server"

# The user the graph being built or run acts for.
acting_user: ContextVar[User | None] = ContextVar("acting_user", default=None)


def is_loopback(scope: Scope) -> bool:
    """Whether a request came over the loopback transport; a client cannot set its root path."""
    root_path = scope.get("root_path") or ""
    return root_path == ROOT_PATH or root_path.startswith(ROOT_PATH + "/")


def serve(app: Any) -> None:
    """Make *app* the server loopback clients reach: those made already, and those made from now on."""
    configure_loopback_transports(app)
    server = ModuleType(_SERVER_MODULE)
    vars(server)["app"] = app
    sys.modules.setdefault(_SERVER_MODULE, server)
