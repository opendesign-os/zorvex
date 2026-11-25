"""Route merging utilities for combining custom apps with Zorvex core routes"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
    websocket_request_validation_exception_handler,
)

logger = structlog.get_logger(__name__)

_FASTAPI_DEFAULT_HANDLERS = (
    http_exception_handler,
    request_validation_exception_handler,
    websocket_request_validation_exception_handler,
)


def merge_lifespans(user_app: FastAPI, core_lifespan: Callable) -> FastAPI:
    """Merge user lifespan with Zorvex's core lifespan.

    Both lifespans will run, with core lifespan wrapping user lifespan.
    This ensures Zorvex's initialization (database, services) happens before
    user initialization, and cleanup happens in reverse order.

    Args:
        user_app: User's FastAPI/Starlette application
        core_lifespan: Zorvex's core lifespan context manager

    Returns:
        Modified user_app with merged lifespan
    """
    user_lifespan = user_app.router.lifespan_context

    # Check for deprecated on_startup/on_shutdown handlers
    if user_app.router.on_startup or user_app.router.on_shutdown:
        raise ValueError(
            f"Cannot merge lifespans with on_startup or on_shutdown handlers. "
            f"Please use lifespan context manager instead. "
            f"Found: on_startup={user_app.router.on_startup}, "
            f"on_shutdown={user_app.router.on_shutdown}"
        )

    @asynccontextmanager
    async def combined_lifespan(app):
        async with core_lifespan(app), user_lifespan(app):
            yield

    user_app.router.lifespan_context = combined_lifespan
    return user_app


def merge_exception_handlers(
    user_app: FastAPI, core_exception_handlers: dict[type[Exception], Callable[..., Any]]
) -> FastAPI:
    """Add the core exception handlers to the custom app, where its author set none.

    Every FastAPI app starts with FastAPI's default handlers; those give way,
    so protocol routes answer in the protocol's error format.
    """
    for exc_type, handler in core_exception_handlers.items():
        if user_app.exception_handlers.get(exc_type) in (None, *_FASTAPI_DEFAULT_HANDLERS):
            user_app.exception_handlers[exc_type] = handler
        else:
            logger.debug(f"User app overrides exception handler for {exc_type}")

    return user_app
