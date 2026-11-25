"""Baseline schema

Revision ID: 0001
Revises:
Create Date: 2026-11-25

Zorvex's own tables, as core/orm.py defines them; LangGraph creates and migrates
its checkpoint and store tables itself. A database from an earlier schema has no
upgrade path: recreate it.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant",
        sa.Column("assistant_id", sa.Text(), server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("graph_id", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("assistant_id"),
    )
    op.create_index(
        "idx_assistant_metadata_gin",
        "assistant",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    )
    op.create_index("idx_assistant_user", "assistant", ["user_id"], unique=False)
    op.create_table(
        "thread",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'idle'"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("values", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("interrupts", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("state_updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("run_priority", sa.Integer(), nullable=True),
        sa.Column("fairness_group", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_index(
        "idx_thread_metadata_gin",
        "thread",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    )
    op.create_index("idx_thread_user", "thread", ["user_id"], unique=False)
    op.create_table(
        "assistant_versions",
        sa.Column("assistant_id", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("graph_id", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["assistant_id"], ["assistant.assistant_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("assistant_id", "version"),
    )
    op.create_table(
        "cron",
        sa.Column("cron_id", sa.Text(), server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("assistant_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("schedule", sa.Text(), nullable=False),
        sa.Column("timezone", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("on_run_completed", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("end_time", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("next_run_date", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("claimed_until", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assistant_id"], ["assistant.assistant_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("cron_id"),
    )
    op.create_index("idx_cron_assistant_id", "cron", ["assistant_id"], unique=False)
    op.create_index("idx_cron_next_run", "cron", ["enabled", "next_run_date"], unique=False)
    op.create_index("idx_cron_thread_id", "cron", ["thread_id"], unique=False)
    op.create_index("idx_cron_user", "cron", ["user_id"], unique=False)
    op.create_table(
        "run",
        sa.Column("run_id", sa.Text(), server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("assistant_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("priority", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("execution_params", postgresql.JSONB(), nullable=False),
        sa.Column("claimed_by", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("retries", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cancel_action", sa.Text(), nullable=True),
        sa.Column("cancel_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["assistant_id"], ["assistant.assistant_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("idx_run_assistant_id", "run", ["assistant_id"], unique=False)
    op.create_index("idx_run_status_lease", "run", ["status", "lease_expires_at"], unique=False)
    op.create_index("idx_run_thread_created", "run", ["thread_id", "created_at"], unique=False)
    op.create_table(
        "thread_ttl",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("strategy", sa.Text(), server_default=sa.text("'delete'"), nullable=False),
        sa.Column("ttl_minutes", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_index("idx_thread_ttl_expires_at", "thread_ttl", ["expires_at"], unique=False)


def downgrade() -> None:
    for table in ("thread_ttl", "run", "cron", "assistant_versions", "thread", "assistant"):
        op.drop_table(table)
