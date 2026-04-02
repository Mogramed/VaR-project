"""extend reconciliation acknowledgements into incident workflow

Revision ID: 20260401_0006
Revises: 20260401_0005
Create Date: 2026-04-01 22:15:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260401_0006"
down_revision = "20260401_0005"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _has_index(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    table_name = "reconciliation_acknowledgements"
    if not _has_table(table_name):
        return

    if not _has_column(table_name, "incident_status"):
        op.add_column(table_name, sa.Column("incident_status", sa.String(length=32), nullable=True))
    if not _has_column(table_name, "resolution_note"):
        op.add_column(table_name, sa.Column("resolution_note", sa.String(length=512), nullable=True))
    if not _has_column(table_name, "resolved_at"):
        op.add_column(table_name, sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True))

    if not _has_index(table_name, "ix_reconciliation_acknowledgements_incident_status"):
        op.create_index(
            "ix_reconciliation_acknowledgements_incident_status",
            table_name,
            ["incident_status"],
        )
    if not _has_index(table_name, "ix_reconciliation_acknowledgements_resolved_at"):
        op.create_index(
            "ix_reconciliation_acknowledgements_resolved_at",
            table_name,
            ["resolved_at"],
        )


def downgrade() -> None:
    table_name = "reconciliation_acknowledgements"
    if not _has_table(table_name):
        return

    for index_name in (
        "ix_reconciliation_acknowledgements_resolved_at",
        "ix_reconciliation_acknowledgements_incident_status",
    ):
        if _has_index(table_name, index_name):
            op.drop_index(index_name, table_name=table_name)

    for column_name in ("resolved_at", "resolution_note", "incident_status"):
        if _has_column(table_name, column_name):
            op.drop_column(table_name, column_name)
