"""add operator run status_reason diagnostics

Revision ID: 20260420_0010
Revises: 20260418_0009
Create Date: 2026-04-20 11:10:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260420_0010"
down_revision = "20260418_0009"
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
    return any(index.get("name") == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    table_name = "operator_runs"
    if not _has_table(table_name):
        return

    if not _has_column(table_name, "status_reason"):
        op.add_column(table_name, sa.Column("status_reason", sa.String(length=64), nullable=True))

    if not _has_index(table_name, "ix_operator_runs_status_reason"):
        op.create_index("ix_operator_runs_status_reason", table_name, ["status_reason"], unique=False)


def downgrade() -> None:
    # Keep downgrade non-blocking: preserve repaired schema state.
    return

