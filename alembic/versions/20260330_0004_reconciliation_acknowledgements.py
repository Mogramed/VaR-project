"""persist reconciliation acknowledgements

Revision ID: 20260330_0004
Revises: 20260329_0003
Create Date: 2026-03-30 10:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260330_0004"
down_revision = "20260329_0003"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if _has_table("reconciliation_acknowledgements"):
        return

    op.create_table(
        "reconciliation_acknowledgements",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("portfolio_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.String(length=128), nullable=True),
        sa.Column("operator_note", sa.String(length=512), nullable=True),
        sa.Column("mismatch_status", sa.String(length=32), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("portfolio_id", "symbol", name="uq_reconciliation_ack_portfolio_symbol"),
    )
    op.create_index(
        "ix_reconciliation_acknowledgements_portfolio_id",
        "reconciliation_acknowledgements",
        ["portfolio_id"],
    )
    op.create_index(
        "ix_reconciliation_acknowledgements_symbol",
        "reconciliation_acknowledgements",
        ["symbol"],
    )
    op.create_index(
        "ix_reconciliation_acknowledgements_mismatch_status",
        "reconciliation_acknowledgements",
        ["mismatch_status"],
    )
    op.create_index(
        "ix_reconciliation_acknowledgements_acknowledged_at",
        "reconciliation_acknowledgements",
        ["acknowledged_at"],
    )


def downgrade() -> None:
    if not _has_table("reconciliation_acknowledgements"):
        return

    for index_name in (
        "ix_reconciliation_acknowledgements_acknowledged_at",
        "ix_reconciliation_acknowledgements_mismatch_status",
        "ix_reconciliation_acknowledgements_symbol",
        "ix_reconciliation_acknowledgements_portfolio_id",
    ):
        op.drop_index(index_name, table_name="reconciliation_acknowledgements")
    op.drop_table("reconciliation_acknowledgements")
