from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from var_project.storage.serialization import utcnow


class Base(DeclarativeBase):
    pass


class PortfolioRecord(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    base_currency: Mapped[str] = mapped_column(String(16))
    symbols_json: Mapped[list[Any]] = mapped_column(JSON)
    positions_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_type: Mapped[str] = mapped_column(String(80), index=True)
    format: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class BacktestRunRecord(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    artifact_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(16), nullable=True)
    days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    alpha: Mapped[float] = mapped_column(Float)
    window: Mapped[int] = mapped_column(Integer)
    n_rows: Mapped[int] = mapped_column(Integer)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ValidationRunRecord(Base):
    __tablename__ = "validation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    source_artifact_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    alpha: Mapped[float] = mapped_column(Float)
    expected_rate: Mapped[float] = mapped_column(Float)
    best_model: Mapped[str | None] = mapped_column(String(32), nullable=True)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class SnapshotRecord(Base):
    __tablename__ = "risk_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    artifact_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(16), nullable=True)
    days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    live_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    live_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    breach_hist: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class AlertRecord(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    snapshot_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    validation_run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    code: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(String(512))
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class DecisionRecord(Base):
    __tablename__ = "risk_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    requested_delta_position_eur: Mapped[float] = mapped_column(Float)
    approved_delta_position_eur: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    model_used: Mapped[str] = mapped_column(String(32), index=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class CapitalSnapshotRecord(Base):
    __tablename__ = "capital_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    reference_model: Mapped[str] = mapped_column(String(32), index=True)
    total_budget_eur: Mapped[float] = mapped_column(Float)
    consumed_eur: Mapped[float] = mapped_column(Float)
    reserved_eur: Mapped[float] = mapped_column(Float)
    remaining_eur: Mapped[float] = mapped_column(Float)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ExecutionRecord(Base):
    __tablename__ = "execution_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    requested_delta_position_eur: Mapped[float] = mapped_column(Float)
    approved_delta_position_eur: Mapped[float] = mapped_column(Float)
    executed_delta_position_eur: Mapped[float] = mapped_column(Float)
    requested_volume_lots: Mapped[float | None] = mapped_column(Float, nullable=True)
    approved_volume_lots: Mapped[float | None] = mapped_column(Float, nullable=True)
    submitted_volume_lots: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_volume_lots: Mapped[float | None] = mapped_column(Float, nullable=True)
    remaining_volume_lots: Mapped[float | None] = mapped_column(Float, nullable=True)
    fill_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    broker_status: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    slippage_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    reconciliation_status: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    mt5_order_ticket: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    mt5_deal_ticket: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ExecutionFillRecord(Base):
    __tablename__ = "execution_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_result_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    order_ticket: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    deal_ticket: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    entry: Mapped[str | None] = mapped_column(String(32), nullable=True)
    volume_lots: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    commission: Mapped[float | None] = mapped_column(Float, nullable=True)
    swap: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comment: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_manual: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    slippage_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ReconciliationAcknowledgementRecord(Base):
    __tablename__ = "reconciliation_acknowledgements"
    __table_args__ = (
        UniqueConstraint("portfolio_id", "symbol", name="uq_reconciliation_ack_portfolio_symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    operator_note: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mismatch_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    acknowledged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class AuditRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    action_type: Mapped[str] = mapped_column(String(64), index=True)
    object_type: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    object_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class InstrumentRecord(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    asset_class: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    contract_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    quote_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    profit_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    margin_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    tick_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    tick_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_step: Mapped[float | None] = mapped_column(Float, nullable=True)
    trading_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="mt5", index=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class MarketDataSyncRecord(Base):
    __tablename__ = "market_data_sync_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    portfolio_slug: Mapped[str | None] = mapped_column(String(160), index=True, nullable=True)
    mode: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class MarketBarRecord(Base):
    __tablename__ = "market_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "time_utc", name="uq_market_bars_symbol_timeframe_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sync_run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    tick_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    real_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="mt5", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class MT5OrderHistoryRecord(Base):
    __tablename__ = "mt5_order_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sync_run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    ticket: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    order_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    state: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    volume_initial: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_manual: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    time_setup_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    time_done_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), default="mt5", index=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class MT5DealHistoryRecord(Base):
    __tablename__ = "mt5_deal_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sync_run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    portfolio_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    ticket: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    order_ticket: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    entry: Mapped[str | None] = mapped_column(String(32), nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    commission: Mapped[float | None] = mapped_column(Float, nullable=True)
    swap: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comment: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_manual: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), default="mt5", index=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
