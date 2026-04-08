"""add queue task id to operator runs

Revision ID: 20260405_0008
Revises: 20260404_0007
Create Date: 2026-04-05 11:35:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260405_0008"
down_revision = "20260404_0007"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(column.get("name") == column_name for column in inspector.get_columns(table_name))


def _has_index(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    table_name = "operator_runs"
    if not _has_table(table_name):
        return
    if not _has_column(table_name, "queue_task_id"):
        op.add_column(table_name, sa.Column("queue_task_id", sa.String(length=128), nullable=True))
    if not _has_index(table_name, "ix_operator_runs_queue_task_id"):
        op.create_index("ix_operator_runs_queue_task_id", table_name, ["queue_task_id"], unique=False)


def downgrade() -> None:
    table_name = "operator_runs"
    if not _has_table(table_name):
        return
    if _has_index(table_name, "ix_operator_runs_queue_task_id"):
        op.drop_index("ix_operator_runs_queue_task_id", table_name=table_name)
    if _has_column(table_name, "queue_task_id"):
        op.drop_column(table_name, "queue_task_id")
