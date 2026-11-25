"""Service layer for assistant business logic

This service encapsulates all business logic for assistant management, following
a layered architecture pattern. The code was extracted from api/assistants.py
to separate concerns and improve maintainability.

Responsibilities:
- Business logic and validation
- Database operations via SQLAlchemy ORM
- Graph schema extraction and manipulation
- Coordination between different components

This is the first service layer implementation in Zorvex. The pattern will be
applied to other APIs (runs, threads, crons) as part of ongoing refactoring.
"""

from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException
from langchain_core.runnables.utils import create_model
from langgraph.pregel import Pregel
from pydantic import TypeAdapter
from sqlalchemy import ColumnElement, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.auth_deps import require_auth
from zorvex_api.core.auth_filters import build_metadata_filter
from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import Assistant as AssistantORM
from zorvex_api.core.orm import AssistantVersion as AssistantVersionORM
from zorvex_api.core.orm import SessionDep
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.models import (
    Assistant,
    AssistantCountRequest,
    AssistantCreate,
    AssistantSearchRequest,
    AssistantUpdate,
)
from zorvex_api.models.auth import User
from zorvex_api.models.enums import OnConflictBehavior
from zorvex_api.services import run_cleanup
from zorvex_api.services.langgraph_service import LangGraphService, get_langgraph_service


def _state_jsonschema(graph) -> dict | None:
    """Extract state schema from graph channels"""
    fields: dict = {}
    for k in graph.stream_channels_list:
        v = graph.channels[k]
        try:
            create_model(k, __root__=(v.UpdateType, None)).model_json_schema()
            fields[k] = (v.UpdateType, None)
        except Exception:
            fields[k] = (Any, None)
    return create_model(graph.get_name("State"), **fields).model_json_schema()


# Configurable keys the server sets, left out of an assistant's config schema as the Agent Server leaves them.
_SERVER_CONFIGURABLE = ("__pregel_checkpointer", "__pregel_store", "checkpoint_id", "checkpoint_ns", "thread_id")


def _get_configurable_jsonschema(graph) -> dict:
    """Get the JSON schema for the configurable part of the graph"""
    config_schema = graph.config_schema()
    model_fields = getattr(config_schema, "model_fields", None) or getattr(config_schema, "__fields__", None)

    if model_fields is not None and "configurable" in model_fields:
        configurable = TypeAdapter(model_fields["configurable"].annotation)
        json_schema = configurable.json_schema()
        if json_schema:
            for key in _SERVER_CONFIGURABLE:
                json_schema["properties"].pop(key, None)
        if hasattr(graph, "config_type") and graph.config_type is not None and hasattr(graph.config_type, "__name__"):
            json_schema["title"] = graph.config_type.__name__
        return json_schema
    return {}


def _extract_graph_schemas(graph) -> dict:
    """Extract schemas from a compiled LangGraph graph object"""
    try:
        input_schema = graph.get_input_jsonschema()
    except Exception:
        input_schema = None

    try:
        output_schema = graph.get_output_jsonschema()
    except Exception:
        output_schema = None

    try:
        state_schema = _state_jsonschema(graph)
    except Exception:
        state_schema = None

    try:
        config_schema = _get_configurable_jsonschema(graph)
    except Exception:
        config_schema = None

    try:
        context_schema = graph.get_context_jsonschema()
    except Exception:
        context_schema = None

    return {
        "input_schema": input_schema,
        "output_schema": output_schema,
        "state_schema": state_schema,
        "config_schema": config_schema,
        "context_schema": context_schema,
    }


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``, ``\\``) in user input.
    Backslash is replaced first so subsequent escapes are not double-escaped.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _injected_metadata(
    base: dict[str, Any] | None,
    value: dict[str, Any],
) -> dict[str, Any] | None:
    """Fold metadata a create/update handler injected into ``value`` onto the request.

    Handlers inject by mutating ``value["metadata"]`` in place (e.g.
    ``value["metadata"]["created_by"] = ctx.user.identity``). The handler's
    *return* is a query filter with no insert meaning, so it is not read here.
    """
    value_meta = value.get("metadata")
    if isinstance(value_meta, dict) and value_meta:
        return {**(base or {}), **value_meta}
    return base


class AssistantService:
    """Service for managing assistants.

    Each method takes the route's ``Authorization`` and runs it before it reads
    or writes, applying the handler's filter to the assistants it touches.
    """

    def __init__(self, session: AsyncSession, user: User, langgraph_service: LangGraphService):
        self.session = session
        self.user = user
        self.langgraph_service = langgraph_service

    def _require_graph(self, graph_id: str) -> None:
        """404 unless the deployment registers *graph_id*."""
        graphs = self.langgraph_service.list_graphs()
        if graph_id not in graphs:
            raise HTTPException(404, f"Graph '{graph_id}' not found. Expected one of: {sorted(graphs)}")

    async def create_assistant(self, authorize: Authorization, request: AssistantCreate) -> Assistant:
        """Create a new assistant"""
        value = request.model_dump()
        await authorize(value)
        request.metadata = _injected_metadata(request.metadata, value)

        graph_id = request.graph_id
        self._require_graph(graph_id)

        config = dict(request.config or {})
        context = request.context or {}

        if config.get("configurable") and context:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both configurable and context. Prefer setting context alone. Context was introduced in LangGraph 0.6.0 and is the long term planned replacement for configurable.",
            )

        # Keep config and context up to date with one another
        if config.get("configurable"):
            context = config["configurable"]
        elif context:
            config["configurable"] = context

        # Generate assistant_id if not provided
        assistant_id = request.assistant_id or str(uuid4())

        # Generate name if not provided
        name = request.name or f"Assistant for {graph_id}"

        metadata = request.metadata or {}

        # Insert first and let the primary key arbitrate: a SELECT-then-INSERT lets
        # two creates collide, and the loser's UniqueViolationError 500s past if_exists.
        insert_stmt = (
            pg_insert(AssistantORM)
            .values(
                assistant_id=assistant_id,
                name=name,
                description=request.description,
                config=config,
                context=context,
                graph_id=graph_id,
                user_id=self.user.identity,
                metadata_dict=metadata,
                version=1,
            )
            .on_conflict_do_nothing(index_elements=[AssistantORM.assistant_id])
            .returning(AssistantORM)
        )
        created = (await self.session.scalars(insert_stmt)).first()

        if created is None:
            return await self._resolve_create_conflict(assistant_id, request.if_exists)

        # Version 1 commits with the assistant it describes: an assistant whose only
        # version went missing 404s out of list_assistant_versions while it is live.
        assistant_version_orm = AssistantVersionORM(
            assistant_id=assistant_id,
            version=1,
            graph_id=graph_id,
            config=config,
            context=context,
            created_at=datetime.now(UTC),
            name=name,
            description=request.description,
            metadata_dict=metadata,
        )
        self.session.add(assistant_version_orm)
        await self.session.commit()

        return Assistant.model_validate(created)

    async def _resolve_create_conflict(self, assistant_id: str, if_exists: OnConflictBehavior) -> Assistant:
        """Apply ``if_exists`` to the row that won the insert, or 409."""
        # Scoped to the caller while assistant_pkey is global, so an incumbent owned
        # by someone else is deliberately not found and conflicts instead of being adopted.
        existing = await self.session.scalar(
            select(AssistantORM).where(
                AssistantORM.user_id == self.user.identity, AssistantORM.assistant_id == assistant_id
            )
        )

        if existing is not None and if_exists == "do_nothing":
            return Assistant.model_validate(existing)

        raise HTTPException(409, f"Assistant '{assistant_id}' already exists")

    async def _search_conditions(
        self, authorize: Authorization, request: AssistantCountRequest
    ) -> list[ColumnElement[bool]]:
        """WHERE clauses of an assistant search or count, after authorizing it.

        The caller's assistants and the system ones, narrowed by the request's
        filters and the handler's.
        """
        filters = await authorize(request.model_dump())
        conditions: list[ColumnElement[bool]] = [
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system")
        ]
        if request.name:
            conditions.append(AssistantORM.name.ilike(f"%{_escape_like(request.name)}%", escape="\\"))
        if request.graph_id:
            self._require_graph(request.graph_id)
            conditions.append(AssistantORM.graph_id == request.graph_id)
        if request.metadata:
            conditions.append(AssistantORM.metadata_dict.op("@>")(request.metadata))
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            conditions.append(auth_filter)
        return conditions

    async def search_assistants(self, authorize: Authorization, request: AssistantSearchRequest) -> list[Assistant]:
        """Search assistants with filters"""
        column = getattr(AssistantORM, request.sort_by)
        stmt = (
            select(AssistantORM)
            .where(*await self._search_conditions(authorize, request))
            # Tie-break on assistant_id keeps offset pagination stable when the sort column has duplicates.
            .order_by(column.asc() if request.sort_order == "asc" else column.desc(), AssistantORM.assistant_id.asc())
            .offset(request.offset)
            .limit(request.limit)
        )
        result = await self.session.scalars(stmt)
        return [Assistant.model_validate(row) for row in result.all()]

    async def count_assistants(self, authorize: Authorization, request: AssistantCountRequest) -> int:
        """Count assistants with filters"""
        stmt = select(func.count()).select_from(AssistantORM).where(*await self._search_conditions(authorize, request))
        return await self.session.scalar(stmt) or 0

    async def _read_owned_assistant(self, authorize: Authorization, assistant_id: str) -> AssistantORM:
        """Authorize, then load the row with the handler's filter applied; 404 if
        the row is absent or the filter excludes it.

        Shared by every read-derived endpoint so a handler's metadata filter is
        enforced uniformly — not only on GET /assistants/{id}.
        """
        filters = await authorize({"assistant_id": assistant_id})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system"),
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)

        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")
        return assistant

    async def get_assistant(self, authorize: Authorization, assistant_id: str) -> Assistant:
        """Get assistant by ID"""
        return Assistant.model_validate(await self._read_owned_assistant(authorize, assistant_id))

    async def update_assistant(
        self, authorize: Authorization, assistant_id: str, request: AssistantUpdate
    ) -> Assistant:
        """Partially update an assistant.

        Fields the caller omitted keep their stored value; supplied ``metadata``
        is merged into the stored metadata, matching the LangGraph SDK contract.
        Omission is read from ``exclude_unset`` rather than from ``None``, so a
        caller can still clear a field by sending it empty (``{"config": {}}``).
        """
        supplied = request.model_dump(exclude_unset=True)
        # Handlers inject by mutating value["metadata"] in place, so the dispatch
        # payload keeps the dict shape the auth API documents even when unset.
        value = {
            **request.model_dump(),
            "config": request.config or {},
            "context": request.context or {},
            "metadata": request.metadata or {},
            "assistant_id": assistant_id,
        }
        filters = await authorize(value)
        request.metadata = _injected_metadata(request.metadata, value)

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            AssistantORM.user_id == self.user.identity,
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        # NOT NULL columns: an empty value cannot be stored, and falling back to
        # the stored one silently is the bug this endpoint is fixing.
        for field in ("name", "graph_id"):
            if field in supplied and not supplied[field]:
                raise HTTPException(
                    status_code=422,
                    detail=f"{field} must be a non-empty string when supplied; omit it to leave it unchanged",
                )

        config = (supplied["config"] or {}) if "config" in supplied else (assistant.config or {})
        context = (supplied["context"] or {}) if "context" in supplied else (assistant.context or {})

        if "config" in supplied and "context" in supplied and config.get("configurable") and context:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both configurable and context. Use only one.",
            )

        metadata = {**(assistant.metadata_dict or {}), **(request.metadata or {})}

        # context mirrors config["configurable"], so whichever one the caller
        # supplied wins and the other is derived from it.
        if "context" in supplied:
            config = {**config, "configurable": context}
        elif "config" in supplied:
            context = config.get("configurable") or {}
        elif config.get("configurable"):
            context = config["configurable"]
        elif context:
            config = {**config, "configurable": context}

        now = datetime.now(UTC)
        version_stmt = select(func.max(AssistantVersionORM.version)).where(
            AssistantVersionORM.assistant_id == assistant_id
        )
        max_version = await self.session.scalar(version_stmt)
        new_version = (max_version or 1) + 1 if max_version is not None else 1

        new_version_details = {
            "assistant_id": assistant_id,
            "version": new_version,
            # .get returns a supplied null rather than the stored value, which
            # is what lets an explicit null clear the nullable description.
            "graph_id": supplied.get("graph_id", assistant.graph_id),
            "config": config,
            "context": context,
            "created_at": now,
            "name": supplied.get("name", assistant.name),
            "description": supplied.get("description", assistant.description),
            "metadata_dict": metadata,
        }

        assistant_version_orm = AssistantVersionORM(**new_version_details)
        self.session.add(assistant_version_orm)
        await self.session.commit()

        assistant_update = (
            update(AssistantORM)
            .where(
                AssistantORM.assistant_id == assistant_id,
                AssistantORM.user_id == self.user.identity,
            )
            .values(
                name=new_version_details["name"],
                description=new_version_details["description"],
                graph_id=new_version_details["graph_id"],
                config=new_version_details["config"],
                context=new_version_details["context"],
                metadata_dict=new_version_details["metadata_dict"],
                version=new_version,
                updated_at=now,
            )
        )
        await self.session.execute(assistant_update)
        await self.session.commit()
        updated_assistant = await self.session.scalar(stmt)
        return Assistant.model_validate(updated_assistant)

    async def delete_assistant(
        self, authorize: Authorization, assistant_id: str, *, delete_threads: bool = False
    ) -> dict:
        """Delete assistant by ID, optionally with the threads bound to it."""
        filters = await authorize({"assistant_id": assistant_id})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            AssistantORM.user_id == self.user.identity,
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)

        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        deleted_threads = 0
        if delete_threads:
            thread_ids = await self.session.scalars(
                select(ThreadORM.thread_id).where(
                    ThreadORM.user_id == self.user.identity,
                    ThreadORM.metadata_dict.op("@>")({"assistant_id": assistant_id}),
                )
            )
            deleted_threads = await run_cleanup.delete_threads(
                self.session, thread_ids.all(), user_id=self.user.identity
            )

        await self.session.delete(assistant)
        await self.session.commit()

        if not delete_threads:
            return {"status": "deleted"}
        return {"status": "deleted", "deleted_threads": deleted_threads}

    async def set_assistant_latest(self, authorize: Authorization, assistant_id: str, version: int) -> Assistant:
        """Set the given version as the latest version of an assistant"""
        filters = await authorize({"assistant_id": assistant_id, "version": version})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            AssistantORM.user_id == self.user.identity,
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        version_stmt = select(AssistantVersionORM).where(
            AssistantVersionORM.assistant_id == assistant_id,
            AssistantVersionORM.version == version,
        )
        assistant_version = await self.session.scalar(version_stmt)
        if not assistant_version:
            raise HTTPException(404, f"Version '{version}' for Assistant '{assistant_id}' not found")

        assistant_update = (
            update(AssistantORM)
            .where(
                AssistantORM.assistant_id == assistant_id,
                AssistantORM.user_id == self.user.identity,
            )
            .values(
                name=assistant_version.name,
                description=assistant_version.description,
                config=assistant_version.config,
                context=assistant_version.context,
                graph_id=assistant_version.graph_id,
                metadata_dict=assistant_version.metadata_dict,
                version=version,
                updated_at=datetime.now(UTC),
            )
        )
        await self.session.execute(assistant_update)
        await self.session.commit()
        updated_assistant = await self.session.scalar(stmt)
        return Assistant.model_validate(updated_assistant)

    async def list_assistant_versions(self, authorize: Authorization, assistant_id: str) -> list[Assistant]:
        """List all versions of an assistant"""
        filters = await authorize({"assistant_id": assistant_id, "metadata": None})

        stmt = select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            or_(AssistantORM.user_id == self.user.identity, AssistantORM.user_id == "system"),
        )
        auth_filter = build_metadata_filter(AssistantORM.metadata_dict, filters)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
        assistant = await self.session.scalar(stmt)
        if not assistant:
            raise HTTPException(404, f"Assistant '{assistant_id}' not found")

        stmt = (
            select(AssistantVersionORM)
            .where(AssistantVersionORM.assistant_id == assistant_id)
            .order_by(AssistantVersionORM.version.desc())
        )
        result = await self.session.scalars(stmt)
        versions = result.all()

        if not versions:
            raise HTTPException(404, f"No versions found for Assistant '{assistant_id}'")

        # Convert to Pydantic models
        return [
            Assistant(
                assistant_id=assistant_id,
                name=v.name,
                description=v.description,
                config=v.config,
                context=v.context,
                graph_id=v.graph_id,
                version=v.version,
                created_at=v.created_at,
                updated_at=v.created_at,
                metadata=v.metadata_dict or {},
            )
            for v in versions
        ]

    async def get_assistant_schemas(self, authorize: Authorization, assistant_id: str) -> dict[str, Any]:
        """Get input, output, state, config and context schemas for an assistant"""
        assistant = await self._read_owned_assistant(authorize, assistant_id)
        async with self._graph(assistant) as graph:
            return {"graph_id": assistant.graph_id, **_extract_graph_schemas(graph)}

    async def get_assistant_graph(
        self, authorize: Authorization, assistant_id: str, xray: bool | int
    ) -> dict[str, Any]:
        """Get the graph structure for visualization"""
        if isinstance(xray, int) and not isinstance(xray, bool) and xray <= 0:
            raise HTTPException(422, detail="Invalid xray value")
        assistant = await self._read_owned_assistant(authorize, assistant_id)
        async with self._graph(assistant) as graph:
            try:
                drawable_graph = await graph.aget_graph(xray=xray)
            except NotImplementedError as e:
                raise HTTPException(422, detail="The graph does not support visualization") from e
        json_graph = drawable_graph.to_json()
        for node in json_graph.get("nodes", []):
            if (data := node.get("data")) and isinstance(data, dict):
                data.pop("id", None)
        return json_graph

    async def get_assistant_subgraphs(
        self,
        authorize: Authorization,
        assistant_id: str,
        namespace: str | None,
        recurse: bool,
    ) -> dict[str, Any]:
        """Get subgraphs of an assistant"""
        assistant = await self._read_owned_assistant(authorize, assistant_id)
        async with self._graph(assistant) as graph:
            try:
                return {
                    ns: _extract_graph_schemas(subgraph)
                    async for ns, subgraph in graph.aget_subgraphs(namespace=namespace, recurse=recurse)
                }
            except NotImplementedError as e:
                raise HTTPException(422, detail="The graph does not support subgraphs") from e

    def _graph(self, assistant: AssistantORM) -> AbstractAsyncContextManager[Pregel]:
        """The assistant's graph, built as a factory builds it for introspection: with the assistant's config."""
        return self.langgraph_service.get_graph(
            assistant.graph_id, assistant.config or {}, access_context="assistants.read", user=self.user
        )


def get_assistant_service(
    session: SessionDep,
    user: User = Depends(require_auth),
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
) -> AssistantService:
    """Dependency injection for AssistantService"""
    return AssistantService(session, user, langgraph_service)
