"""Work on LangGraph's checkpoint tables that its Postgres saver leaves unimplemented: copy, delete, measure.

``acopy_thread`` and ``adelete_for_runs`` are stubs on the Postgres saver, and
it reports no sizes, so these run SQL on its tables. Column lists come from the
catalog rather than being hardcoded: LangGraph adds columns across versions.
"""

from typing import Any

from psycopg import sql

from zorvex_api.core.database import db_manager

_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


async def copy(source_thread_id: str, target_thread_id: str) -> None:
    """Copy every checkpoint, blob and pending write of one thread to another, all or nothing."""
    async with db_manager.get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        for table in _TABLES:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_schema = current_schema() AND table_name = %s ORDER BY ordinal_position",
                (table,),
            )
            columns = [row["column_name"] for row in await cur.fetchall()]
            targets = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
            projection = sql.SQL(", ").join(
                sql.Placeholder() if c == "thread_id" else sql.Identifier(c) for c in columns
            )
            await cur.execute(
                sql.SQL("INSERT INTO {table} ({targets}) SELECT {projection} FROM {table} WHERE thread_id = %s").format(
                    table=sql.Identifier(table), targets=targets, projection=projection
                ),
                (target_thread_id, source_thread_id),
            )


async def delete_run(thread_id: str, run_id: str) -> None:
    """Delete the checkpoints *run_id* wrote on *thread_id* (subgraphs' too), their writes, and orphaned blobs.

    A checkpoint names its run in its metadata, a write its checkpoint, and a
    blob stays while some checkpoint's channel versions use it.
    """
    async with db_manager.get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            "WITH gone AS ("
            " DELETE FROM checkpoints WHERE thread_id = %s AND metadata ->> 'run_id' = %s"
            " RETURNING checkpoint_ns, checkpoint_id)"
            " DELETE FROM checkpoint_writes w USING gone g"
            " WHERE w.thread_id = %s AND w.checkpoint_ns = g.checkpoint_ns AND w.checkpoint_id = g.checkpoint_id",
            (thread_id, run_id, thread_id),
        )
        await cur.execute(
            "DELETE FROM checkpoint_blobs b WHERE b.thread_id = %s AND NOT EXISTS ("
            " SELECT 1 FROM checkpoints c"
            " WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns"
            " AND c.checkpoint -> 'channel_versions' ->> b.channel = b.version)",
            (thread_id,),
        )


async def storage(thread_id: str) -> dict[str, Any]:
    """The bytes a thread's history takes, as the Agent Server reports them: checkpoints, channel values, writes.

    Checkpoints count their stored document and metadata (values small enough
    to live inline included), channel values one blob per channel version, and
    pending writes their serialized values; every namespace counts.
    """
    async with db_manager.get_pool().connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT count(*) AS count,"
            " coalesce(sum(pg_column_size(checkpoint) + pg_column_size(metadata)), 0) AS bytes"
            " FROM checkpoints WHERE thread_id = %s",
            (thread_id,),
        )
        saved = await cur.fetchone() or {"count": 0, "bytes": 0}
        await cur.execute(
            "SELECT channel, count(*) AS versions, coalesce(sum(octet_length(blob)), 0) AS bytes"
            " FROM checkpoint_blobs WHERE thread_id = %s GROUP BY channel ORDER BY channel",
            (thread_id,),
        )
        channels = {
            row["channel"]: {"versions": row["versions"], "bytes": row["bytes"]} for row in await cur.fetchall()
        }
        await cur.execute(
            "SELECT count(*) AS count, coalesce(sum(octet_length(blob)), 0) AS bytes"
            " FROM checkpoint_writes WHERE thread_id = %s",
            (thread_id,),
        )
        writes = await cur.fetchone() or {"count": 0, "bytes": 0}
    values = {
        "count": sum(c["versions"] for c in channels.values()),
        "bytes": sum(c["bytes"] for c in channels.values()),
        "by_channel": channels,
    }
    return {
        "thread_id": thread_id,
        "total_bytes": saved["bytes"] + values["bytes"] + writes["bytes"],
        "checkpoints": {"count": saved["count"], "bytes": saved["bytes"]},
        "channel_values": values,
        "pending_writes": {"count": writes["count"], "bytes": writes["bytes"]},
    }
