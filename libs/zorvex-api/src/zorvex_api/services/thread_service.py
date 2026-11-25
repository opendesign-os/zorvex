"""Creating, reading, updating and deleting threads, for the thread routes and the protocol servers.

Each operation authorizes as its ``@auth.on.threads`` event, and a thread that
is another user's, or that the event's filter excludes, reads as missing (404).
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from langgraph.types import StateUpdate
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from zorvex_api.core.authorization import Authorization
from zorvex_api.core.orm import Thread as ThreadORM
from zorvex_api.core.orm import ThreadTTL as ThreadTTLORM
from zorvex_api.models import ThreadCreate, ThreadTTLSpec, ThreadUpdate, User
from zorvex_api.services.langgraph_service import create_thread_config, get_langgraph_service
from zorvex_api.services.ownership import owned_thread, visible_threads
from zorvex_api.services.run_cleanup import delete_threads
from zorvex_api.services.thread_state_service import snapshot_values
from zorvex_api.services.thread_ttl import get_thread_ttl_config
from zorvex_api.utils.jsonb import jsonb_patch, jsonb_shallow_merge


class ThreadService:
    """One user's threads, over one database session."""

    def __init__(self, session: AsyncSession, user: User) -> None:
        self.session = session
        self.user = user

    async def create(self, authorize: Authorization, request: ThreadCreate) -> ThreadORM:
        """Create the thread, or with ``if_exists="do_nothing"`` find the caller's thread of that id.

        Raises:
            HTTPException: 409 when the id is taken, 422 for a ttl the server cannot apply.
        """
        thread_id = request.thread_id or str(uuid4())
        value: dict[str, Any] = {
            "thread_id": thread_id,
            "metadata": dict(request.metadata or {}),
            "if_exists": request.if_exists,
            "ttl": request.ttl.model_dump() if request.ttl else None,
        }
        filters = await authorize(value)
        # The thread gets the metadata as the handler left it.
        metadata = value["metadata"] or {}
        # Resolved before the insert, so an invalid ttl answers 422 without database work.
        ttl_row = ttl_for(thread_id, request.ttl)

        # Insert first and let the primary key arbitrate: a SELECT-then-INSERT lets
        # two requests for the same id both miss, and the loser's unique violation
        # would answer 500 instead of honouring if_exists.
        insert_stmt = (
            pg_insert(ThreadORM)
            .values(thread_id=thread_id, status="idle", metadata_dict=metadata, user_id=self.user.identity)
            .on_conflict_do_nothing(index_elements=["thread_id"])
            .returning(ThreadORM)
        )
        created = (await self.session.scalars(insert_stmt)).first()
        # One transaction with the thread; a conflict leaves the incumbent's ttl alone.
        if created is not None and ttl_row is not None:
            self.session.add(ttl_row)
        await self.session.commit()

        if created is not None:
            if request.supersteps:
                created.values, created.interrupts = await _apply_supersteps(
                    thread_id, request.supersteps, metadata, self.user
                )
                await self.session.commit()
            return created

        # The id is taken. This read is the caller's only while thread ids are
        # global, so another user's thread is not adopted: it is a conflict.
        existing = await self.session.scalar(
            select(ThreadORM).where(
                ThreadORM.thread_id == thread_id,
                ThreadORM.user_id == self.user.identity,
                *visible_threads(filters),
            )
        )
        if existing is not None and request.if_exists == "do_nothing":
            return existing
        raise HTTPException(409, f"Thread '{thread_id}' already exists")

    async def get(self, authorize: Authorization, thread_id: str) -> ThreadORM:
        """The thread; 404 when the caller cannot see it."""
        filters = await authorize({"thread_id": thread_id})
        return await owned_thread(self.session, thread_id, self.user, filters=filters)

    async def update(self, authorize: Authorization, thread_id: str, request: ThreadUpdate) -> ThreadORM:
        """Merge metadata into the thread (shallowly), and replace its ttl, whose clock restarts."""
        value: dict[str, Any] = {"thread_id": thread_id, "metadata": dict(request.metadata or {})}
        filters = await authorize(value)
        # The metadata to merge, as the handler left it.
        metadata = value["metadata"]

        # The 404 probe and ownership check. The merge below reads no column into
        # Python, so it cannot lose a concurrent writer's keys.
        thread = await owned_thread(self.session, thread_id, self.user, filters=filters)

        values: dict[str, Any] = {"updated_at": datetime.now(UTC)}
        if metadata:
            values["metadata_dict"] = jsonb_shallow_merge(
                ThreadORM.metadata_dict, jsonb_patch(metadata, "metadata_patch")
            )
        await self.session.execute(
            update(ThreadORM)
            .where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == self.user.identity)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if request.ttl is not None and (ttl := ttl_for(thread_id, request.ttl)) is not None:
            replacement = {"strategy": ttl.strategy, "ttl_minutes": ttl.ttl_minutes, "expires_at": ttl.expires_at}
            await self.session.execute(
                pg_insert(ThreadTTLORM)
                .values(thread_id=thread_id, **replacement)
                .on_conflict_do_update(index_elements=[ThreadTTLORM.thread_id], set_=replacement)
            )
        await self.session.commit()
        await self.session.refresh(thread)
        return thread

    async def delete(self, authorize: Authorization, thread_id: str) -> None:
        """Delete the thread with its runs (cancelling the active ones) and its checkpoints."""
        filters = await authorize({"thread_id": thread_id})
        thread = await owned_thread(self.session, thread_id, self.user, filters=filters)
        await delete_threads(self.session, [thread.thread_id], user_id=self.user.identity)
        await self.session.commit()


def ttl_for(thread_id: str, requested: ThreadTTLSpec | None) -> ThreadTTLORM | None:
    """The thread's ttl row, or None when no ttl applies.

    The server's ``checkpointer.ttl`` gives the defaults, which a requested ttl
    overrides field by field; without server defaults a request must name its ttl.
    """
    config = get_thread_ttl_config()
    if config is None and requested is None:
        return None

    requested_ttl = requested.ttl if requested else None
    requested_strategy = requested.strategy if requested else None
    if config is not None:
        ttl_minutes = requested_ttl if requested_ttl is not None else config.default_ttl
        strategy = requested_strategy or config.strategy
    else:
        if requested_ttl is None:
            raise HTTPException(422, "ttl.ttl is required when no server-side TTL default is configured")
        ttl_minutes = requested_ttl
        strategy = requested_strategy or "delete"

    return ThreadTTLORM(
        thread_id=thread_id,
        strategy=strategy,
        ttl_minutes=ttl_minutes,
        expires_at=datetime.now(UTC) + timedelta(minutes=ttl_minutes),
    )


async def _apply_supersteps(
    thread_id: str,
    supersteps: list[dict[str, Any]],
    metadata: dict[str, Any],
    user: User,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Seed a new thread with checkpoint history; returns the values and interrupts it ends at."""
    graph_id = metadata.get("graph_id")
    if not graph_id:
        raise HTTPException(422, "supersteps requires 'graph_id' in metadata")

    config = create_thread_config(thread_id, user)
    batches = [
        [StateUpdate(values=change.get("values"), as_node=change.get("as_node")) for change in step["updates"]]
        for step in supersteps
    ]
    # A new thread has no config yet: no run has built its graph.
    async with get_langgraph_service().get_graph(graph_id, config, access_context="threads.update", user=user) as agent:
        await agent.abulk_update_state(config, batches)
        return snapshot_values(await agent.aget_state(config))
