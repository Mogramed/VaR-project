"""mt5 canonical cache schema

Revision ID: 20260329_0002
Revises: 20260329_0001
Create Date: 2026-03-29 12:00:00
"""

from __future__ import annotations

from alembic import op

from var_project.storage.models import (
    InstrumentRecord,
    MT5DealHistoryRecord,
    MT5OrderHistoryRecord,
    MarketBarRecord,
    MarketDataSyncRecord,
)


revision = "20260329_0002"
down_revision = "20260329_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    for table in (
        InstrumentRecord.__table__,
        MarketDataSyncRecord.__table__,
        MarketBarRecord.__table__,
        MT5OrderHistoryRecord.__table__,
        MT5DealHistoryRecord.__table__,
    ):
        table.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table in (
        MT5DealHistoryRecord.__table__,
        MT5OrderHistoryRecord.__table__,
        MarketBarRecord.__table__,
        MarketDataSyncRecord.__table__,
        InstrumentRecord.__table__,
    ):
        table.drop(bind=bind, checkfirst=True)
