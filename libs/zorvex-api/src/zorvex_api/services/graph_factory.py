"""Graph factories: callables zorvex.json names in place of a graph.

As in the Agent Server, a factory is called on every access to its graph (a
run, a state read or update, an introspection request) with a
``RunnableConfig`` and/or a ``ServerRuntime``, whichever its parameters ask
for, so it can build the graph for that user and config. The contract is
``langgraph_sdk.runtime.ServerRuntime``'s.
"""

import asyncio
import dataclasses
import inspect
import typing
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, get_args, get_origin

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph_sdk.auth.types import BaseUser
from langgraph_sdk.runtime import AccessContext, ServerRuntime, _ExecutionRuntime, _ReadRuntime
from pydantic import BaseModel

# The classes ServerRuntime is the union of.
_RUNTIME_CLASSES = (_ExecutionRuntime, _ReadRuntime)
# Names a factory may import only under TYPE_CHECKING and still annotate with.
_LOCALNS = {"ServerRuntime": ServerRuntime, "RunnableConfig": RunnableConfig}


@dataclasses.dataclass(frozen=True, slots=True)
class GraphFactory:
    """A graph factory and the parameters it takes the config and the runtime by."""

    fn: Callable[..., Any]
    config_param: str | None = None
    runtime_param: str | None = None
    # The T of a ServerRuntime[T] annotation, which a run's context is coerced to.
    context_type: Any = None

    @classmethod
    def of(cls, fn: Callable[..., Any]) -> "GraphFactory":
        """Classify *fn* by its parameters: none, a config, a ``ServerRuntime``, or one of each, in either order.

        The runtime parameter is the one annotated ``ServerRuntime`` (or
        ``ServerRuntime[T]``); any other parameter takes the config.

        Raises:
            ValueError: If *fn* takes anything else.
        """
        params = list(inspect.signature(fn).parameters.values())
        try:
            hints = typing.get_type_hints(fn, localns=_LOCALNS)
        except Exception:  # annotations that do not resolve are matched as written
            hints = {}
        runtime = [p.name for p in params if _is_runtime(hints.get(p.name, p.annotation))]
        config = [p.name for p in params if p.name not in runtime]
        if len(runtime) > 1 or len(config) > 1:
            raise ValueError(
                f"Graph factory {fn!r} can take a RunnableConfig and a ServerRuntime, "
                f"got parameters {[p.name for p in params]}"
            )
        runtime_param = runtime[0] if runtime else None
        return cls(
            fn,
            config_param=config[0] if config else None,
            runtime_param=runtime_param,
            context_type=_context_type(hints.get(runtime_param)) if runtime_param else None,
        )

    def __call__(self, config: RunnableConfig, runtime: ServerRuntime) -> Any:
        """Call the factory with whichever of *config* and *runtime* it takes."""
        kwargs: dict[str, Any] = {}
        if self.config_param:
            kwargs[self.config_param] = config
        if self.runtime_param:
            kwargs[self.runtime_param] = runtime
        return self.fn(**kwargs)

    def runtime(
        self, access_context: AccessContext, *, store: BaseStore, user: BaseUser | None, context: Any = None
    ) -> ServerRuntime:
        """The runtime for *access_context*; only a run's carries the run's *context*, as the factory's T."""
        if access_context != "threads.create_run":
            return _ReadRuntime(access_context=access_context, user=user, store=store)
        return _ExecutionRuntime(access_context=access_context, user=user, store=store, context=self._coerce(context))

    def _coerce(self, context: Any) -> Any:
        """A dict context as T when T is a pydantic model or a dataclass, as LangGraph coerces ``context_schema``.

        The run's graph only gets the keys its context schema declares, so a
        dataclass is built from its fields' keys alone. Anything else is copied,
        so the factory cannot change the context the run gets.
        """
        if not isinstance(context, dict):
            return context
        context_type = self.context_type
        if isinstance(context_type, type) and issubclass(context_type, BaseModel):
            return context_type.model_validate(context)
        if isinstance(context_type, type) and dataclasses.is_dataclass(context_type):
            fields = {f.name for f in dataclasses.fields(context_type) if f.init}
            return context_type(**{k: v for k, v in context.items() if k in fields})
        return dict(context)


def _is_runtime(annotation: Any) -> bool:
    """Whether *annotation* is ``ServerRuntime``, ``ServerRuntime[T]``, a runtime class, or one wrapped in a union or ``Annotated``."""
    if annotation is ServerRuntime:
        return True
    if isinstance(annotation, type):
        return issubclass(annotation, _RUNTIME_CLASSES)
    origin = get_origin(annotation)
    if origin is ServerRuntime:
        return True
    return origin is not None and any(_is_runtime(arg) for arg in get_args(annotation))


def _context_type(annotation: Any) -> Any:
    """The T of a ``ServerRuntime[T]`` annotation (also inside a union or ``Annotated``), or None."""
    origin = get_origin(annotation)
    if origin is ServerRuntime:
        return next(iter(get_args(annotation)), None)
    if origin is None:
        return None
    return next((t for t in map(_context_type, get_args(annotation)) if t is not None), None)


@asynccontextmanager
async def generate_graph(value: Any) -> AsyncGenerator[Any]:
    """What a factory returned, entered when it is a context manager and awaited when it is a coroutine."""
    if hasattr(value, "__aenter__") and hasattr(value, "__aexit__"):
        async with value as graph:
            yield graph
    elif hasattr(value, "__enter__") and hasattr(value, "__exit__"):
        with value as graph:
            yield graph
    elif asyncio.iscoroutine(value):
        yield await value
    else:
        yield value
