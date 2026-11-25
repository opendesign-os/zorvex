"""The v2 thread stream over a WebSocket, as the Agent Server serves it.

The client sends commands as JSON text; the server answers each with its
response envelope and pushes the events of its subscriptions, each a protocol
event envelope. ``subscription.subscribe`` takes what the SSE body takes
(channels, namespaces, depth, since); ``subscription.reconnect`` takes the same
with a required ``since``; ``subscription.unsubscribe`` ends one. Any other
command runs as on ``POST /threads/{thread_id}/commands``. Access failures close
the socket with 4000 plus the HTTP status; a thread that does not exist yet
waits for the ``run.start`` that creates it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import structlog
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from zorvex_api.core import loopback
from zorvex_api.core.auth import require_auth
from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models import User
from zorvex_api.services.event_streaming.commands import INVALID_COMMAND, handle_command
from zorvex_api.services.event_streaming.protocol import Response, build_error, build_success
from zorvex_api.services.event_streaming.session import RunLister, ThreadEventSession, validate_channels
from zorvex_api.services.ownership import owned_thread

logger = structlog.getLogger(__name__)

_SUBSCRIBE = "subscription.subscribe"
_RECONNECT = "subscription.reconnect"
_UNSUBSCRIBE = "subscription.unsubscribe"


class _Socket:
    def __init__(
        self, websocket: WebSocket, thread_id: str, authorize: Authorization, lister: Callable[[User], RunLister]
    ) -> None:
        self.websocket = websocket
        self.thread_id = thread_id
        self.authorize = authorize
        self.user = authorize.user
        self.lister = lister
        self.subscriptions: dict[str, asyncio.Task[None]] = {}
        self.lock = asyncio.Lock()

    async def send(self, message: Mapping[str, Any]) -> None:
        async with self.lock:
            await self.websocket.send_text(json.dumps(message, default=str))

    async def pump(self, subscription_id: str, session: ThreadEventSession) -> None:
        try:
            async for envelope in session.stream():
                await self.send(envelope)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Thread event socket failed", thread_id=self.thread_id, subscription=subscription_id)
            failure = {"type": "error", "id": None, "error": "unknown_error", "message": "Thread event stream failed."}
            with contextlib.suppress(Exception):
                await self.send(failure)
                await self.websocket.close(code=1011, reason="thread stream ended")

    def subscribe(self, command_id: int, params: dict[str, Any], method: str) -> Response:
        channels, invalid = validate_channels(params.get("channels"))
        if invalid:
            return build_error(command_id, "invalid_argument", f"Unsupported channels: {', '.join(invalid)}")
        since = params.get("since")
        if method == _RECONNECT and not (isinstance(since, int) and not isinstance(since, bool)):
            return build_error(command_id, "invalid_argument", "subscription.reconnect requires an integer since.")
        depth = params.get("depth")
        namespaces = params.get("namespaces")
        session = ThreadEventSession(
            channels=channels,
            list_run_ids=self.lister(self.user),
            since=since if isinstance(since, int) and not isinstance(since, bool) and since >= 0 else None,
            namespaces=[list(ns) for ns in namespaces if isinstance(ns, list)]
            if isinstance(namespaces, list)
            else None,
            depth=depth if isinstance(depth, int) and not isinstance(depth, bool) and depth >= 0 else None,
        )
        subscription_id = params.get("subscription_id") if method == _RECONNECT else None
        if not isinstance(subscription_id, str) or not subscription_id:
            subscription_id = str(uuid.uuid4())
        if old := self.subscriptions.pop(subscription_id, None):
            old.cancel()
        self.subscriptions[subscription_id] = asyncio.create_task(self.pump(subscription_id, session))
        return build_success(command_id, {"subscription_id": subscription_id, "replayed_events": 0})

    def unsubscribe(self, command_id: int, params: dict[str, Any]) -> Response:
        task = self.subscriptions.pop(str(params.get("subscription_id")), None)
        if task is None:
            return build_error(command_id, "no_such_subscription", "No such subscription.")
        task.cancel()
        return build_success(command_id, {})

    async def command(self, body: Any) -> tuple[Response, str | None]:
        """One command's response envelope, and its method when it was one."""
        command_id = body.get("id") if isinstance(body, dict) else None
        method = body.get("method") if isinstance(body, dict) else None
        if not isinstance(command_id, int) or isinstance(command_id, bool) or not isinstance(method, str):
            return build_error(None, "invalid_argument", INVALID_COMMAND), None
        params = body.get("params") if isinstance(body.get("params"), dict) else {}
        if method in (_SUBSCRIBE, _RECONNECT):
            return self.subscribe(command_id, params, method), method
        if method == _UNSUBSCRIBE:
            return self.unsubscribe(command_id, params), method
        async with get_session_maker()() as session:
            response = await handle_command(
                {**body, "params": params}, session=session, thread_id=self.thread_id, authorize=self.authorize
            )
        return response, method

    def close(self) -> None:
        for task in self.subscriptions.values():
            task.cancel()
        self.subscriptions.clear()


async def serve(websocket: WebSocket, thread_id: str, lister: Callable[[User], RunLister]) -> None:
    """Serve one client's socket on *thread_id* until it goes."""
    await websocket.accept()
    try:
        user = await require_auth(websocket)
        # A graph's own socket is trusted, as its loopback requests are.
        authorize = Authorization(user, "threads", "read", trusted=loopback.is_loopback(websocket.scope))
        filters = await authorize({"thread_id": thread_id})
    except HTTPException as exc:
        await websocket.close(code=4000 + exc.status_code)
        return
    missing = False
    async with get_session_maker()() as session:
        try:
            await owned_thread(session, thread_id, user, filters=filters)
        except HTTPException as exc:
            if exc.status_code != 404:
                await websocket.close(code=4000 + exc.status_code)
                return
            missing = True

    socket = _Socket(websocket, thread_id, authorize, lister)
    try:
        while True:
            text = await websocket.receive_text()
            try:
                body = json.loads(text)
            except ValueError:
                await socket.send(
                    {
                        "type": "error",
                        "id": None,
                        "error": "invalid_argument",
                        "message": "Protocol commands must be valid JSON.",
                    }
                )
                continue
            if missing and not (isinstance(body, dict) and body.get("method") == "run.start"):
                await websocket.close(code=4404)
                return
            response, method = await socket.command(body)
            if method == "run.start" and response.get("type") == "success":
                missing = False
            await socket.send(response)
    except WebSocketDisconnect:
        pass
    finally:
        socket.close()
