"""Run priority: the langgraph_sdk ``Priority`` zorvex.json's ``priority.path`` names.

Its ``@priority.assign`` handler is asked once per thread, when the thread's
first run is created, and its answer is kept on the thread. While a handler is
configured, every run on the thread inherits that priority: -5 starts first,
5 last, and the worker queue orders by it (services/job_queue.py). The answer's
fairness group is kept too, though nothing schedules by it yet, as the SDK
documents. With no handler, runs have priority 0.
"""

from dataclasses import dataclass
from functools import cache
from typing import Any

import structlog
from fastapi import HTTPException
from langgraph_sdk import Auth, Priority, PriorityContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zorvex_api.config import load_config
from zorvex_api.utils.loader import load_object

logger = structlog.getLogger(__name__)


@dataclass(frozen=True)
class ThreadPriority:
    """What a thread keeps from the handler's answer."""

    priority: int = 0
    fairness_group: str = "default"


class _Answer(BaseModel):
    """The handler's answer, checked strictly; an absent or null field means its default."""

    model_config = ConfigDict(extra="forbid", strict=True)

    priority: int | None = Field(default=None, ge=-5, le=5)
    fairness_group: str | None = Field(default=None, max_length=256)


@cache
def callback() -> Any | None:
    """The configured ``@priority.assign`` handler, or None when zorvex.json has no ``priority``.

    Called at startup, so a path that does not name a usable handler stops the server there.
    """
    config = load_config().priority
    if config is None:
        return None
    target = load_object(config.path, module_name="zorvex_priority")
    if not isinstance(target, Priority):
        raise TypeError(f"priority.path {config.path!r} is a {type(target).__name__}; it must name a Priority")
    if target._assign_handler is None:
        raise ValueError(f"priority.path {config.path!r} names a Priority with no @priority.assign handler")
    logger.info("Using the priority handler", path=config.path)
    return target._assign_handler


async def assign(context: PriorityContext) -> ThreadPriority:
    """Ask the handler about the thread *context* describes.

    An ``Auth.exceptions.HTTPException`` it raises keeps its status; anything
    else it raises, or an answer that fails the checks, is a 500. Nothing is
    kept in those cases, so the thread's next run asks again.
    """
    handler = callback()
    if handler is None:
        raise RuntimeError("zorvex.json configures no priority handler")
    try:
        answer = await handler(context)
    except Auth.exceptions.HTTPException as e:
        raise HTTPException(e.status_code, e.detail, headers=e.headers) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("priority.assign raised", thread_id=context.thread_id, graph_id=context.graph_id)
        raise HTTPException(500, f"priority.assign raised {type(e).__name__}: {e}") from e
    if isinstance(answer, dict) and "fairness_weight" in answer:
        # A key some SDK releases typed; there is no weight to keep, so it goes.
        logger.warning("Dropping the fairness_weight priority.assign returned", thread_id=context.thread_id)
        answer = {key: value for key, value in answer.items() if key != "fairness_weight"}
    try:
        checked = _Answer.model_validate(answer or {})
    except ValidationError as e:
        logger.error("priority.assign returned settings out of range", thread_id=context.thread_id, error=str(e))
        raise HTTPException(500, f"priority.assign returned settings out of range: {e}") from e
    return ThreadPriority(
        priority=checked.priority if checked.priority is not None else 0,
        fairness_group=checked.fairness_group or "default",
    )
