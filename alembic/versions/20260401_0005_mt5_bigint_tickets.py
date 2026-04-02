"""widen mt5 broker ticket columns to bigint

Revision ID: 20260401_0005
Revises: 20260330_0004
Create Date: 2026-04-01 15:25:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260401_0005"
down_revision = "20260330_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table_name, columns in (
        (
            "execution_results",
            (
                ("position_id", True),
                ("mt5_order_ticket", True),
                ("mt5_deal_ticket", True),
            ),
        ),
        (
            "execution_fills",
            (
                ("order_ticket", True),
                ("deal_ticket", True),
                ("position_id", True),
            ),
        ),
        (
            "mt5_order_history",
            (
                ("ticket", False),
                ("position_id", True),
            ),
        ),
        (
            "mt5_deal_history",
            (
                ("ticket", False),
                ("order_ticket", True),
                ("position_id", True),
            ),
        ),
    ):
        with op.batch_alter_table(table_name) as batch_op:
            for column_name, nullable in columns:
                batch_op.alter_column(
                    column_name,
                    existing_type=sa.Integer(),
                    type_=sa.BigInteger(),
                    existing_nullable=nullable,
                )


def downgrade() -> None:
    for table_name, columns in (
        (
            "mt5_deal_history",
            (
                ("position_id", True),
                ("order_ticket", True),
                ("ticket", False),
            ),
        ),
        (
            "mt5_order_history",
            (
                ("position_id", True),
                ("ticket", False),
            ),
        ),
        (
            "execution_fills",
            (
                ("position_id", True),
                ("deal_ticket", True),
                ("order_ticket", True),
            ),
        ),
        (
            "execution_results",
            (
                ("mt5_deal_ticket", True),
                ("mt5_order_ticket", True),
                ("position_id", True),
            ),
        ),
    ):
        with op.batch_alter_table(table_name) as batch_op:
            for column_name, nullable in columns:
                batch_op.alter_column(
                    column_name,
                    existing_type=sa.BigInteger(),
                    type_=sa.Integer(),
                    existing_nullable=nullable,
                )
