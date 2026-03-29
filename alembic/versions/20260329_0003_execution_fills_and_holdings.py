"""execution fills and paper trade removal

Revision ID: 20260329_0003
Revises: 20260329_0002
Create Date: 2026-03-29 18:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260329_0003"
down_revision = "20260329_0002"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return False
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    if _has_table("execution_results"):
        for column in (
            sa.Column("requested_volume_lots", sa.Float(), nullable=True),
            sa.Column("approved_volume_lots", sa.Float(), nullable=True),
            sa.Column("submitted_volume_lots", sa.Float(), nullable=True),
            sa.Column("filled_volume_lots", sa.Float(), nullable=True),
            sa.Column("remaining_volume_lots", sa.Float(), nullable=True),
            sa.Column("fill_ratio", sa.Float(), nullable=True),
            sa.Column("broker_status", sa.String(length=32), nullable=True),
            sa.Column("position_id", sa.Integer(), nullable=True),
            sa.Column("slippage_points", sa.Float(), nullable=True),
            sa.Column("reconciliation_status", sa.String(length=32), nullable=True),
        ):
            if not _has_column("execution_results", column.name):
                op.add_column("execution_results", column)

    if not _has_table("execution_fills"):
        op.create_table(
            "execution_fills",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("execution_result_id", sa.Integer(), nullable=True),
            sa.Column("portfolio_id", sa.Integer(), nullable=True),
            sa.Column("symbol", sa.String(length=32), nullable=False),
            sa.Column("order_ticket", sa.Integer(), nullable=True),
            sa.Column("deal_ticket", sa.Integer(), nullable=True),
            sa.Column("position_id", sa.Integer(), nullable=True),
            sa.Column("side", sa.String(length=16), nullable=True),
            sa.Column("entry", sa.String(length=32), nullable=True),
            sa.Column("volume_lots", sa.Float(), nullable=False),
            sa.Column("price", sa.Float(), nullable=True),
            sa.Column("profit", sa.Float(), nullable=True),
            sa.Column("commission", sa.Float(), nullable=True),
            sa.Column("swap", sa.Float(), nullable=True),
            sa.Column("fee", sa.Float(), nullable=True),
            sa.Column("reason", sa.String(length=64), nullable=True),
            sa.Column("comment", sa.String(length=255), nullable=True),
            sa.Column("is_manual", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("slippage_points", sa.Float(), nullable=True),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("time_utc", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_execution_fills_execution_result_id", "execution_fills", ["execution_result_id"])
        op.create_index("ix_execution_fills_portfolio_id", "execution_fills", ["portfolio_id"])
        op.create_index("ix_execution_fills_symbol", "execution_fills", ["symbol"])
        op.create_index("ix_execution_fills_order_ticket", "execution_fills", ["order_ticket"])
        op.create_index("ix_execution_fills_deal_ticket", "execution_fills", ["deal_ticket"])
        op.create_index("ix_execution_fills_position_id", "execution_fills", ["position_id"])
        op.create_index("ix_execution_fills_is_manual", "execution_fills", ["is_manual"])
        op.create_index("ix_execution_fills_time_utc", "execution_fills", ["time_utc"])
        op.create_index("ix_execution_fills_created_at", "execution_fills", ["created_at"])

    if _has_table("paper_trades"):
        op.drop_table("paper_trades")


def downgrade() -> None:
    if _has_table("paper_trades"):
        pass
    else:
        op.create_table(
            "paper_trades",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("portfolio_id", sa.Integer(), nullable=True),
            sa.Column("decision_id", sa.Integer(), nullable=True),
            sa.Column("symbol", sa.String(length=32), nullable=False),
            sa.Column("requested_delta_position_eur", sa.Float(), nullable=False),
            sa.Column("executed_delta_position_eur", sa.Float(), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if _has_table("execution_fills"):
        for index_name in (
            "ix_execution_fills_created_at",
            "ix_execution_fills_time_utc",
            "ix_execution_fills_is_manual",
            "ix_execution_fills_position_id",
            "ix_execution_fills_deal_ticket",
            "ix_execution_fills_order_ticket",
            "ix_execution_fills_symbol",
            "ix_execution_fills_portfolio_id",
            "ix_execution_fills_execution_result_id",
        ):
            op.drop_index(index_name, table_name="execution_fills")
        op.drop_table("execution_fills")

    if _has_table("execution_results"):
        for column_name in (
            "reconciliation_status",
            "slippage_points",
            "position_id",
            "broker_status",
            "fill_ratio",
            "remaining_volume_lots",
            "filled_volume_lots",
            "submitted_volume_lots",
            "approved_volume_lots",
            "requested_volume_lots",
        ):
            if _has_column("execution_results", column_name):
                op.drop_column("execution_results", column_name)
