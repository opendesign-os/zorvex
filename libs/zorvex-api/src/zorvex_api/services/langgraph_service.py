"""The graphs zorvex.json declares, and the configs graphs are built and run with.

Graphs load once, at startup. A compiled graph (or a StateGraph, compiled then)
is handed out as a copy with the Postgres checkpointer and store attached; a
factory is called on every access, as the Agent Server calls it, with the
config the caller passes and a ServerRuntime for the access.
"""

import sys
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid5

import structlog
from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from langgraph_sdk.runtime import AccessContext
from sqlalchemy import func, select

from zorvex_api.config import GraphEntry, config_dir, load_config
from zorvex_api.constants import ASSISTANT_NAMESPACE_UUID
from zorvex_api.core import loopback
from zorvex_api.core.database import db_manager
from zorvex_api.core.orm import Assistant as AssistantORM
from zorvex_api.core.orm import AssistantVersion as AssistantVersionORM
from zorvex_api.core.orm import get_session_maker
from zorvex_api.models.auth import User
from zorvex_api.models.run_job import RunJob
from zorvex_api.services.graph_factory import GraphFactory, generate_graph
from zorvex_api.utils.headers import request_headers, with_headers
from zorvex_api.utils.loader import load_object

logger = structlog.get_logger(__name__)

# The name LangGraph gives a compiled graph nobody named; the server names it after its graph id.
_UNNAMED_GRAPH = "LangGraph"


def _module_name_for(graph_id: str) -> str:
    """The ``sys.modules`` key a graph file imports under, kept from shadowing real packages."""
    safe_id = graph_id.replace(".", "_").replace("/", "_").replace("-", "_")
    return f"zorvex_graphs.{safe_id}"


def _load_graph(graph_id: str, path: str) -> Pregel | GraphFactory:
    """The compiled graph or the factory zorvex.json names for *graph_id*.

    Raises:
        TypeError: If *path* names neither a graph nor a callable.
    """
    graph = load_object(path, module_name=_module_name_for(graph_id))
    if isinstance(graph, Pregel | StateGraph):
        return _compiled(graph, graph_id)
    if callable(graph):
        return GraphFactory.of(graph)
    raise TypeError(f"Graph '{graph_id}' ({path}) is a {type(graph).__name__}, not a graph or a graph factory")


def _compiled(graph: Any, graph_id: str) -> Pregel:
    """*graph* compiled, or a TypeError naming what was found instead."""
    if isinstance(graph, StateGraph):
        return graph.compile()
    if isinstance(graph, Pregel):
        return graph
    raise TypeError(f"Graph '{graph_id}' is a {type(graph).__name__}, not a compiled graph or a StateGraph")


class LangGraphService:
    """The graphs zorvex.json declares, by graph id."""

    def __init__(self) -> None:
        self._entries: dict[str, GraphEntry] = {}
        self._graphs: dict[str, Pregel | GraphFactory] = {}

    async def initialize(self) -> None:
        """Load the graphs, then give each its system assistant."""
        config = load_config()
        # Dependency paths go on sys.path before the graph modules import.
        _add_dependency_paths(config.dependencies)
        entries = {
            graph_id: entry if isinstance(entry, GraphEntry) else GraphEntry(path=entry)
            for graph_id, entry in config.graphs.items()
        }
        modules: dict[str, str] = {}
        for graph_id in entries:
            if (other := modules.setdefault(_module_name_for(graph_id), graph_id)) != graph_id:
                raise ValueError(f"Graph ids '{other}' and '{graph_id}' import as the same module")
        self._graphs = {graph_id: _load_graph(graph_id, entry.path) for graph_id, entry in entries.items()}
        self._entries = entries
        await self._ensure_system_assistants()

    def list_graphs(self) -> dict[str, str]:
        """The graph ids zorvex.json declares, with the path each loads from."""
        return {graph_id: entry.path for graph_id, entry in self._entries.items()}

    @asynccontextmanager
    async def get_graph(
        self,
        graph_id: str,
        config: Mapping[str, Any],
        *,
        access_context: AccessContext,
        user: User | None,
        context: Any = None,
    ) -> AsyncGenerator[Pregel]:
        """The graph, with the Postgres checkpointer and store attached.

        A factory is called with *config*, carrying *user* in its configurable,
        and a ServerRuntime for *access_context*; a run's runtime also carries
        the run's *context*. Every access must build the run's topology, so
        *config* is what the graph runs with: the run's config for a run, the
        thread's (see :func:`thread_graph_config`) for its state, and the
        assistant's for introspection.

        Raises:
            HTTPException: 404 when zorvex.json declares no graph *graph_id*.
        """
        value = self._graphs.get(graph_id)
        if value is None:
            raise HTTPException(404, f"Graph '{graph_id}' not found")
        store = db_manager.get_store()
        # A loopback client the graph makes acts for this user.
        acting = loopback.acting_user.set(user)
        try:
            if isinstance(value, GraphFactory):
                runtime = value.runtime(access_context, store=store, user=user, context=context)
                # A run keeps the headers it was created with; any other access sees the request's.
                if access_context != "threads.create_run":
                    config = with_headers(dict(config), request_headers.get())
                value = value(ensure_config(with_user(config, user)), runtime)
            async with generate_graph(value) as built:
                graph = _compiled(built, graph_id)
                update: dict[str, Any] = {"checkpointer": db_manager.get_checkpointer(), "store": store}
                if graph.name == _UNNAMED_GRAPH:
                    update["name"] = graph_id
                yield graph.copy(update=update)
        finally:
            loopback.acting_user.reset(acting)

    async def _ensure_system_assistants(self) -> None:
        """Give each graph a system assistant whose id derives from the graph id, so clients can pass the graph id.

        A restart keeps the assistant, and versions it when its name (the
        graph's, else the graph id) or zorvex.json's description changed.
        """
        async with get_session_maker()() as session:
            for graph_id, entry in self._entries.items():
                assistant_id = str(uuid5(ASSISTANT_NAMESPACE_UUID, graph_id))
                graph = self._graphs[graph_id]
                name = graph.name if isinstance(graph, Pregel) and graph.name != _UNNAMED_GRAPH else graph_id
                description = entry.description
                assistant = await session.scalar(
                    select(AssistantORM).where(AssistantORM.assistant_id == assistant_id).with_for_update()
                )
                if assistant is None:
                    session.add(
                        AssistantORM(
                            assistant_id=assistant_id,
                            name=name,
                            description=description,
                            graph_id=graph_id,
                            config={},
                            user_id="system",
                            metadata_dict={"created_by": "system"},
                        )
                    )
                    session.add(
                        AssistantVersionORM(
                            assistant_id=assistant_id,
                            version=1,
                            name=name,
                            description=description,
                            graph_id=graph_id,
                            config={},
                            context={},
                            metadata_dict={"created_by": "system"},
                        )
                    )
                    continue
                if assistant.user_id != "system" or (assistant.name, assistant.description) == (name, description):
                    continue
                latest = await session.scalar(
                    select(func.max(AssistantVersionORM.version)).where(
                        AssistantVersionORM.assistant_id == assistant_id
                    )
                )
                version = max(assistant.version, latest or 0) + 1
                now = datetime.now(UTC)
                session.add(
                    AssistantVersionORM(
                        assistant_id=assistant_id,
                        version=version,
                        name=name,
                        description=description,
                        graph_id=assistant.graph_id,
                        config=assistant.config,
                        context=assistant.context,
                        metadata_dict=assistant.metadata_dict,
                        created_at=now,
                    )
                )
                assistant.name, assistant.description = name, description
                assistant.version, assistant.updated_at = version, now
            await session.commit()


def _add_dependency_paths(dependencies: list[str]) -> None:
    """Put zorvex.json's dependency directories on sys.path, the first listed first.

    Relative paths resolve against the config file's directory.
    """
    for dependency in reversed(dependencies):
        path = (config_dir() / dependency).resolve()
        if not path.exists():
            logger.warning("Dependency path does not exist", path=str(path))
        elif str(path) not in sys.path:
            sys.path.insert(0, str(path))
            logger.info("Added dependency path to sys.path", path=str(path))


_langgraph_service: LangGraphService | None = None


def get_langgraph_service() -> LangGraphService:
    """The process's graph service."""
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphService()
    return _langgraph_service


def with_user(config: Mapping[str, Any], user: User | None) -> RunnableConfig:
    """*config* with *user* in its configurable, under the keys the Agent Server puts it.

    ``langgraph_auth_user`` is the user object, ``langgraph_auth_user_id`` its
    identity and ``langgraph_auth_permissions`` its permissions; ``user_id``
    defaults to the identity.
    """
    configurable = dict(config.get("configurable") or {})
    if user is not None:
        configurable.setdefault("user_id", user.identity)
        configurable["langgraph_auth_user"] = user
        configurable["langgraph_auth_user_id"] = user.identity
        configurable["langgraph_auth_permissions"] = list(user.permissions)
    return {**config, "configurable": configurable}


def create_thread_config(thread_id: str, user: User | None) -> RunnableConfig:
    """The config a thread's state is read and written with."""
    return with_user({"configurable": {"thread_id": thread_id}}, user)


def thread_graph_config(thread_config: Mapping[str, Any] | None, config: Mapping[str, Any]) -> RunnableConfig:
    """The config a thread's graph is built with to read or write its state.

    The thread's: the config its last run's graph was built from, without
    that run's own configurable (see ``run_preparation``), with *config*'s
    configurable on top, as the Agent Server builds it.
    """
    base = thread_config or {}
    return {
        **base,
        **config,
        "configurable": {**(base.get("configurable") or {}), **(config.get("configurable") or {})},
    }


def create_run_config(job: RunJob) -> RunnableConfig:
    """The config a run executes with: its kwargs' config, and the server's keys.

    The run's ids are the server's to set: a ``configurable.thread_id`` from
    the request would point the run at another thread's checkpoints (the
    checkpointer keys on thread_id alone), so they overwrite the request's.
    """
    identity = job.identity
    cfg: dict[str, Any] = deepcopy(job.kwargs.config)
    cfg.setdefault("configurable", {}).update(
        run_id=identity.run_id,
        thread_id=identity.thread_id,
        graph_id=identity.graph_id,
        assistant_id=identity.assistant_id,
    )
    user = job.user
    # The root run's id, so the run's callbacks and astream_events report it.
    cfg.setdefault("run_id", identity.run_id)
    return with_user(cfg, user)
