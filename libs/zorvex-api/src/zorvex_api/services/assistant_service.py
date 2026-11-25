"""Assistants: creating, reading, searching, versioning and deleting them.

An assistant's row is its current version, and ``assistant_versions`` keeps
every version. Its context is its config's ``configurable``. Each method runs
the route's ``Authorization`` before it reads or writes, and applies the
handler's filter to the assistants it touches.
"""

from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException
from langchain_core.runnables.utils import create_model
from langgraph.pregel import Pregel
from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core import search
from zorvex_api.core.auth import require_auth
from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import Assistant as AssistantORM
from zorvex_api.core.orm import AssistantVersion as AssistantVersionORM
from zorvex_api.core.orm import SessionDep
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.models import Assistant, AssistantCountRequest, AssistantCreate, AssistantSearchRequest, AssistantUpdate
from zorvex_api.models.assistants import AssistantVersionsRequest
from zorvex_api.models.auth import User
from zorvex_api.models.enums import OnConflictBehavior
from zorvex_api.services import run_cleanup
from zorvex_api.services.langgraph_service import LangGraphService, get_langgraph_service
from zorvex_api.services.ownership import owned_assistant, readable_assistants

# The columns an assistant's row shares with each of its versions.
_VERSIONED = ("name", "description", "graph_id", "config", "metadata_dict")
_BOTH_CONTEXTS = "Cannot specify both configurable and context. Prefer setting context alone."


def _state_jsonschema(graph: Any) -> dict[str, Any]:
    """The graph's state, one field per channel; a channel whose type has no schema is Any."""
    fields: dict[str, Any] = {}
    for key in graph.stream_channels_list:
        update_type = graph.channels[key].UpdateType
        try:
            create_model(key, __root__=(update_type, None)).model_json_schema()
            fields[key] = (update_type, None)
        except Exception:
            fields[key] = (Any, None)
    return create_model(graph.get_name("State"), **fields).model_json_schema()


def _graph_schemas(graph: Any) -> dict[str, Any]:
    """The graph's schemas; one the server cannot derive is None. Its config schema is its context's."""

    def derived(schema: Any) -> Any:
        try:
            return schema()
        except Exception:
            return None

    try:
        context = graph.get_context_jsonschema()
    except Exception:
        context = config = None
    else:
        config = context or {}
    return {
        "input_schema": derived(graph.get_input_jsonschema),
        "output_schema": derived(graph.get_output_jsonschema),
        "state_schema": derived(lambda: _state_jsonschema(graph)),
        "config_schema": config,
        "context_schema": context,
    }


def _escape_like(value: str) -> str:
    """*value* with LIKE's wildcards and escape character escaped, backslash first."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _injected_metadata(base: dict[str, Any] | None, value: dict[str, Any]) -> dict[str, Any] | None:
    """The request's metadata with what a create or update handler added to ``value["metadata"]`` in place."""
    injected = value.get("metadata")
    if isinstance(injected, dict) and injected:
        return {**(base or {}), **injected}
    return base


def _with_context(config: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
    """*config* with *context* as its ``configurable``, which context replaces; 400 when both are set."""
    if not context:
        return config
    if config.get("configurable"):
        raise HTTPException(400, _BOTH_CONTEXTS)
    return {**config, "configurable": context}


def _version_view(version: AssistantVersionORM) -> Assistant:
    """A version as the API shows it: an assistant as it was, updated when it was made."""
    return Assistant.model_validate(
        {
            "assistant_id": version.assistant_id,
            "version": version.version,
            "created_at": version.created_at,
            "updated_at": version.created_at,
            "metadata": version.metadata_dict,
            **{column: getattr(version, column) for column in _VERSIONED if column != "metadata_dict"},
        }
    )


class AssistantService:
    def __init__(self, session: AsyncSession, user: User, langgraph_service: LangGraphService):
        self.session = session
        self.user = user
        self.langgraph_service = langgraph_service

    async def create_assistant(self, authorize: Authorization, request: AssistantCreate) -> Assistant:
        value = request.model_dump()
        await authorize(value)
        self.langgraph_service.require_graph(request.graph_id)
        assistant_id = request.assistant_id or str(uuid4())
        current = {
            "name": request.name or f"Assistant for {request.graph_id}",
            "description": request.description,
            "graph_id": request.graph_id,
            "config": _with_context(dict(request.config or {}), request.context),
            "metadata_dict": _injected_metadata(request.metadata, value) or {},
        }
        # The primary key arbitrates: two creates of one id cannot both insert.
        created = await self.session.scalar(
            pg_insert(AssistantORM)
            .values(assistant_id=assistant_id, user_id=self.user.identity, version=1, **current)
            .on_conflict_do_nothing(index_elements=[AssistantORM.assistant_id])
            .returning(AssistantORM)
        )
        if created is None:
            return await self._resolve_create_conflict(assistant_id, request.if_exists)
        self.session.add(AssistantVersionORM(assistant_id=assistant_id, version=1, **current))
        await self.session.commit()
        return Assistant.model_validate(created)

    async def _resolve_create_conflict(self, assistant_id: str, if_exists: OnConflictBehavior) -> Assistant:
        """Apply ``if_exists`` to the caller's assistant that won the insert; 409 for anyone else's."""
        existing = await self.session.scalar(
            select(AssistantORM).where(
                AssistantORM.user_id == self.user.identity, AssistantORM.assistant_id == assistant_id
            )
        )
        if existing is not None and if_exists == "do_nothing":
            return Assistant.model_validate(existing)
        raise HTTPException(409, f"Assistant '{assistant_id}' already exists")

    def _search_conditions(
        self, request: AssistantCountRequest, filters: dict[str, Any] | None
    ) -> list[ColumnElement[bool]]:
        """The caller's assistants and the system's, narrowed by the request's filters and the handler's."""
        conditions = [readable_assistants(self.user, filters)]
        if request.name:
            conditions.append(AssistantORM.name.ilike(f"%{_escape_like(request.name)}%", escape="\\"))
        if request.graph_id:
            self.langgraph_service.require_graph(request.graph_id)
            conditions.append(AssistantORM.graph_id == request.graph_id)
        if request.metadata:
            conditions.append(AssistantORM.metadata_dict.op("@>")(request.metadata))
        return conditions

    async def search_assistants(
        self, authorize: Authorization, request: AssistantSearchRequest
    ) -> tuple[list[Assistant], dict[str, str]]:
        """A page of the matching assistants, and its pagination headers."""
        filters = await authorize(request.model_dump())
        conditions = self._search_conditions(request, filters)
        rows, headers = await search.page(self.session, AssistantORM.assistant_id, conditions, request)
        return [Assistant.model_validate(row) for row in rows], headers

    async def count_assistants(self, authorize: Authorization, request: AssistantCountRequest) -> int:
        filters = await authorize(request.model_dump())
        return await search.count(self.session, AssistantORM, self._search_conditions(request, filters))

    async def _readable(self, authorize: Authorization, assistant_id: str) -> AssistantORM:
        """The assistant, after authorizing reading it, within the handler's filter; 404 otherwise."""
        filters = await authorize({"assistant_id": assistant_id})
        return await owned_assistant(self.session, assistant_id, self.user, filters=filters)

    async def get_assistant(self, authorize: Authorization, assistant_id: str) -> Assistant:
        return Assistant.model_validate(await self._readable(authorize, assistant_id))

    async def update_assistant(
        self, authorize: Authorization, assistant_id: str, request: AssistantUpdate
    ) -> Assistant:
        """Make a new version of the assistant from the fields supplied; the others keep their value.

        Supplied ``metadata`` merges into the stored metadata, as the LangGraph SDK expects.
        """
        supplied = request.model_dump(exclude_unset=True)
        # A handler injects by mutating value["metadata"] in place, so the dict fields are always dicts.
        value = {
            **request.model_dump(),
            "config": request.config or {},
            "context": request.context or {},
            "metadata": request.metadata or {},
            "assistant_id": assistant_id,
        }
        filters = await authorize(value)
        assistant = await owned_assistant(self.session, assistant_id, self.user, filters=filters, writable=True)

        config = (supplied["config"] or {}) if "config" in supplied else assistant.config
        if "context" in supplied:
            context = supplied["context"] or {}
            if "config" in supplied and context and config.get("configurable"):
                raise HTTPException(400, _BOTH_CONTEXTS)
            config = {**config, "configurable": context}
        current = {
            "name": request.name or assistant.name,
            # .get takes a supplied null, which clears the description.
            "description": supplied.get("description", assistant.description),
            "graph_id": request.graph_id or assistant.graph_id,
            "config": config,
            "metadata_dict": {**assistant.metadata_dict, **(_injected_metadata(request.metadata, value) or {})},
        }
        latest = await self.session.scalar(
            select(func.max(AssistantVersionORM.version)).where(AssistantVersionORM.assistant_id == assistant_id)
        )
        version = (latest or 0) + 1
        self.session.add(AssistantVersionORM(assistant_id=assistant_id, version=version, **current))
        return await self._make_current(assistant_id, version, current)

    async def set_assistant_latest(self, authorize: Authorization, assistant_id: str, version: int) -> Assistant:
        """Make one of the assistant's versions its current one."""
        filters = await authorize({"assistant_id": assistant_id, "version": version})
        await owned_assistant(self.session, assistant_id, self.user, filters=filters, writable=True)
        chosen = await self.session.get(AssistantVersionORM, (assistant_id, version))
        if chosen is None:
            raise HTTPException(404, f"Version '{version}' for Assistant '{assistant_id}' not found")
        return await self._make_current(
            assistant_id, version, {column: getattr(chosen, column) for column in _VERSIONED}
        )

    async def _make_current(self, assistant_id: str, version: int, current: dict[str, Any]) -> Assistant:
        """Make *current* the assistant's row, as its *version*, and commit; the row as it now is."""
        updated = await self.session.scalar(
            update(AssistantORM)
            .where(AssistantORM.assistant_id == assistant_id)
            .values(**current, version=version, updated_at=datetime.now(UTC))
            .returning(AssistantORM)
        )
        await self.session.commit()
        return Assistant.model_validate(updated)

    async def delete_assistant(
        self, authorize: Authorization, assistant_id: str, *, delete_threads: bool = False
    ) -> dict[str, Any]:
        """Delete the assistant and its versions, and with *delete_threads* the threads bound to it."""
        filters = await authorize({"assistant_id": assistant_id})
        assistant = await owned_assistant(self.session, assistant_id, self.user, filters=filters, writable=True)
        result: dict[str, Any] = {"status": "deleted"}
        if delete_threads:
            thread_ids = await self.session.scalars(
                select(ThreadORM.thread_id).where(
                    ThreadORM.user_id == self.user.identity,
                    ThreadORM.metadata_dict.op("@>")({"assistant_id": assistant_id}),
                )
            )
            result["deleted_threads"] = await run_cleanup.delete_threads(
                self.session, thread_ids.all(), user_id=self.user.identity
            )
        await self.session.delete(assistant)
        await self.session.commit()
        return result

    async def list_assistant_versions(
        self, authorize: Authorization, assistant_id: str, request: AssistantVersionsRequest
    ) -> list[Assistant]:
        """A page of the assistant's versions, newest first."""
        filters = await authorize({"assistant_id": assistant_id, "metadata": request.metadata})
        await owned_assistant(self.session, assistant_id, self.user, filters=filters)
        stmt = select(AssistantVersionORM).where(AssistantVersionORM.assistant_id == assistant_id)
        if request.metadata:
            stmt = stmt.where(AssistantVersionORM.metadata_dict.op("@>")(request.metadata))
        versions = await self.session.scalars(
            stmt.order_by(AssistantVersionORM.version.desc()).offset(request.offset).limit(request.limit)
        )
        return [_version_view(version) for version in versions]

    async def get_assistant_schemas(self, authorize: Authorization, assistant_id: str) -> dict[str, Any]:
        """The input, output, state, config and context schemas of the assistant's graph."""
        assistant = await self._readable(authorize, assistant_id)
        async with self._graph(assistant) as graph:
            return {"graph_id": assistant.graph_id, **_graph_schemas(graph)}

    async def get_assistant_graph(
        self, authorize: Authorization, assistant_id: str, xray: bool | int
    ) -> dict[str, Any]:
        """The assistant's graph as nodes and edges, for drawing; *xray* expands subgraphs."""
        if isinstance(xray, int) and not isinstance(xray, bool) and xray <= 0:
            raise HTTPException(422, detail="Invalid xray value")
        assistant = await self._readable(authorize, assistant_id)
        async with self._graph(assistant) as graph:
            try:
                drawable = await graph.aget_graph(xray=xray)
            except NotImplementedError as e:
                raise HTTPException(422, detail="The graph does not support visualization") from e
        json_graph = drawable.to_json()
        for node in json_graph.get("nodes", []):
            if isinstance(data := node.get("data"), dict):
                data.pop("id", None)
        return json_graph

    async def get_assistant_subgraphs(
        self, authorize: Authorization, assistant_id: str, namespace: str | None, recurse: bool
    ) -> dict[str, Any]:
        """The schemas of the assistant's subgraphs, by namespace; only *namespace*'s when given."""
        assistant = await self._readable(authorize, assistant_id)
        async with self._graph(assistant) as graph:
            try:
                return {
                    ns: _graph_schemas(subgraph)
                    async for ns, subgraph in graph.aget_subgraphs(namespace=namespace, recurse=recurse)
                }
            except NotImplementedError as e:
                raise HTTPException(422, detail="The graph does not support subgraphs") from e

    def _graph(self, assistant: AssistantORM) -> AbstractAsyncContextManager[Pregel]:
        """The assistant's graph, built as a factory builds it for introspection: with the assistant's config."""
        return self.langgraph_service.get_graph(
            assistant.graph_id, assistant.config, access_context="assistants.read", user=self.user
        )


def get_assistant_service(
    session: SessionDep,
    user: User = Depends(require_auth),
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
) -> AssistantService:
    return AssistantService(session, user, langgraph_service)
