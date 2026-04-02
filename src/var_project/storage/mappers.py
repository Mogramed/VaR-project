from __future__ import annotations

from typing import Any

from var_project.storage.models import (
    AlertRecord,
    ArtifactRecord,
    AuditRecord,
    BacktestRunRecord,
    CapitalSnapshotRecord,
    DecisionRecord,
    ExecutionFillRecord,
    ExecutionRecord,
    InstrumentRecord,
    MT5DealHistoryRecord,
    MT5OrderHistoryRecord,
    MarketBarRecord,
    MarketDataSyncRecord,
    PortfolioRecord,
    ReconciliationAcknowledgementRecord,
    SnapshotRecord,
    ValidationRunRecord,
)
from var_project.storage.serialization import to_iso


def portfolio_to_dict(record: PortfolioRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "slug": record.slug,
        "name": record.name,
        "base_currency": record.base_currency,
        "symbols": list(record.symbols_json or []),
        "positions": dict(record.positions_json or {}),
        "created_at": to_iso(record.created_at),
        "updated_at": to_iso(record.updated_at),
    }


def artifact_to_dict(record: ArtifactRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "artifact_type": record.artifact_type,
        "format": record.format,
        "path": record.path,
        "size_bytes": record.size_bytes,
        "details": record.details_json or {},
        "created_at": to_iso(record.created_at),
        "updated_at": to_iso(record.updated_at),
    }


def backtest_to_dict(record: BacktestRunRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "portfolio_id": record.portfolio_id,
        "artifact_id": record.artifact_id,
        "timeframe": record.timeframe,
        "days": record.days,
        "alpha": record.alpha,
        "window": record.window,
        "n_rows": record.n_rows,
        "summary": record.summary_json or {},
        "created_at": to_iso(record.created_at),
    }


def validation_to_dict(record: ValidationRunRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "portfolio_id": record.portfolio_id,
        "source_artifact_id": record.source_artifact_id,
        "alpha": record.alpha,
        "expected_rate": record.expected_rate,
        "best_model": record.best_model,
        "summary": record.summary_json or {},
        "created_at": to_iso(record.created_at),
    }


def snapshot_to_dict(record: SnapshotRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "portfolio_id": record.portfolio_id,
        "artifact_id": record.artifact_id,
        "source": record.source,
        "alpha": record.alpha,
        "timeframe": record.timeframe,
        "days": record.days,
        "window": record.window,
        "live_pnl": record.live_pnl,
        "live_loss": record.live_loss,
        "breach_hist": record.breach_hist,
        "payload": record.payload_json or {},
        "created_at": to_iso(record.created_at),
    }


def alert_to_dict(record: AlertRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "portfolio_id": record.portfolio_id,
        "snapshot_id": record.snapshot_id,
        "validation_run_id": record.validation_run_id,
        "source": record.source,
        "severity": record.severity,
        "code": record.code,
        "message": record.message,
        "context": record.context_json or {},
        "created_at": to_iso(record.created_at),
    }


def decision_to_dict(record: DecisionRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("symbol", record.symbol)
    payload.setdefault("requested_exposure_change", record.requested_delta_position_eur)
    payload.setdefault("approved_exposure_change", record.approved_delta_position_eur)
    payload.setdefault("decision", record.decision)
    payload.setdefault("model_used", record.model_used)
    payload.setdefault("created_at", to_iso(record.created_at))
    return payload


def capital_snapshot_to_dict(record: CapitalSnapshotRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("source", record.source)
    payload.setdefault("reference_model", record.reference_model)
    payload.setdefault("total_capital_budget_eur", record.total_budget_eur)
    payload.setdefault("total_capital_consumed_eur", record.consumed_eur)
    payload.setdefault("total_capital_reserved_eur", record.reserved_eur)
    payload.setdefault("total_capital_remaining_eur", record.remaining_eur)
    payload.setdefault("created_at", to_iso(record.created_at))
    return payload


def execution_to_dict(record: ExecutionRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("decision_id", record.decision_id)
    payload.setdefault("symbol", record.symbol)
    payload.setdefault("requested_exposure_change", record.requested_delta_position_eur)
    payload.setdefault("approved_exposure_change", record.approved_delta_position_eur)
    payload.setdefault("executed_exposure_change", record.executed_delta_position_eur)
    payload.setdefault("requested_volume_lots", record.requested_volume_lots)
    payload.setdefault("approved_volume_lots", record.approved_volume_lots)
    payload.setdefault("submitted_volume_lots", record.submitted_volume_lots)
    payload.setdefault("filled_volume_lots", record.filled_volume_lots)
    payload.setdefault("remaining_volume_lots", record.remaining_volume_lots)
    payload.setdefault("fill_ratio", record.fill_ratio)
    payload.setdefault("broker_status", record.broker_status)
    payload.setdefault("position_id", record.position_id)
    payload.setdefault("slippage_points", record.slippage_points)
    payload.setdefault("reconciliation_status", record.reconciliation_status)
    payload.setdefault("mt5_order_ticket", record.mt5_order_ticket)
    payload.setdefault("mt5_deal_ticket", record.mt5_deal_ticket)
    payload.setdefault("status", record.status)
    payload.setdefault("created_at", to_iso(record.created_at))
    return payload


def execution_fill_to_dict(record: ExecutionFillRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("execution_result_id", record.execution_result_id)
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("symbol", record.symbol)
    payload.setdefault("order_ticket", record.order_ticket)
    payload.setdefault("deal_ticket", record.deal_ticket)
    payload.setdefault("position_id", record.position_id)
    payload.setdefault("side", record.side)
    payload.setdefault("entry", record.entry)
    payload.setdefault("volume_lots", record.volume_lots)
    payload.setdefault("price", record.price)
    payload.setdefault("profit", record.profit)
    payload.setdefault("commission", record.commission)
    payload.setdefault("swap", record.swap)
    payload.setdefault("fee", record.fee)
    payload.setdefault("reason", record.reason)
    payload.setdefault("comment", record.comment)
    payload.setdefault("is_manual", bool(record.is_manual))
    payload.setdefault("slippage_points", record.slippage_points)
    payload.setdefault("time_utc", to_iso(record.time_utc))
    payload.setdefault("created_at", to_iso(record.created_at))
    return payload


def reconciliation_acknowledgement_to_dict(record: ReconciliationAcknowledgementRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("symbol", record.symbol)
    payload.setdefault("reason", record.reason)
    payload.setdefault("operator_note", record.operator_note)
    payload.setdefault("mismatch_status", record.mismatch_status)
    payload.setdefault("incident_status", record.incident_status)
    payload.setdefault("resolution_note", record.resolution_note)
    payload.setdefault("acknowledged_at", to_iso(record.acknowledged_at))
    payload.setdefault("resolved_at", to_iso(record.resolved_at))
    payload.setdefault("updated_at", to_iso(record.updated_at))
    return payload


def audit_to_dict(record: AuditRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("actor", record.actor)
    payload.setdefault("action_type", record.action_type)
    payload.setdefault("object_type", record.object_type)
    payload.setdefault("object_id", record.object_id)
    payload.setdefault("created_at", to_iso(record.created_at))
    return payload


def instrument_to_dict(record: InstrumentRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("symbol", record.symbol)
    payload.setdefault("asset_class", record.asset_class)
    payload.setdefault("contract_size", record.contract_size)
    payload.setdefault("base_currency", record.base_currency)
    payload.setdefault("quote_currency", record.quote_currency)
    payload.setdefault("profit_currency", record.profit_currency)
    payload.setdefault("margin_currency", record.margin_currency)
    payload.setdefault("tick_size", record.tick_size)
    payload.setdefault("tick_value", record.tick_value)
    payload.setdefault("volume_min", record.volume_min)
    payload.setdefault("volume_max", record.volume_max)
    payload.setdefault("volume_step", record.volume_step)
    payload.setdefault("trading_mode", record.trading_mode)
    payload.setdefault("source", record.source)
    payload.setdefault("synced_at", to_iso(record.synced_at))
    payload.setdefault("created_at", to_iso(record.created_at))
    payload.setdefault("updated_at", to_iso(record.updated_at))
    return payload


def market_bar_to_dict(record: MarketBarRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "sync_run_id": record.sync_run_id,
        "symbol": record.symbol,
        "timeframe": record.timeframe,
        "time_utc": to_iso(record.time_utc),
        "open": record.open,
        "high": record.high,
        "low": record.low,
        "close": record.close,
        "tick_volume": record.tick_volume,
        "spread": record.spread,
        "real_volume": record.real_volume,
        "source": record.source,
        "created_at": to_iso(record.created_at),
    }


def market_data_sync_to_dict(record: MarketDataSyncRecord) -> dict[str, Any]:
    return {
        "id": int(record.id),
        "portfolio_id": record.portfolio_id,
        "portfolio_slug": record.portfolio_slug,
        "mode": record.mode,
        "status": record.status,
        "details": record.details_json or {},
        "synced_at": to_iso(record.synced_at),
    }


def mt5_order_history_to_dict(record: MT5OrderHistoryRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("sync_run_id", record.sync_run_id)
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("ticket", record.ticket)
    payload.setdefault("position_id", record.position_id)
    payload.setdefault("symbol", record.symbol)
    payload.setdefault("side", record.side)
    payload.setdefault("order_type", record.order_type)
    payload.setdefault("state", record.state)
    payload.setdefault("volume_initial", record.volume_initial)
    payload.setdefault("volume_current", record.volume_current)
    payload.setdefault("price_open", record.price_open)
    payload.setdefault("price_current", record.price_current)
    payload.setdefault("comment", record.comment)
    payload.setdefault("is_manual", bool(record.is_manual))
    payload.setdefault("time_setup_utc", to_iso(record.time_setup_utc))
    payload.setdefault("time_done_utc", to_iso(record.time_done_utc))
    payload.setdefault("source", record.source)
    payload.setdefault("synced_at", to_iso(record.synced_at))
    payload.setdefault("updated_at", to_iso(record.updated_at))
    return payload


def mt5_deal_history_to_dict(record: MT5DealHistoryRecord) -> dict[str, Any]:
    payload = dict(record.payload_json or {})
    payload.setdefault("id", int(record.id))
    payload.setdefault("sync_run_id", record.sync_run_id)
    payload.setdefault("portfolio_id", record.portfolio_id)
    payload.setdefault("ticket", record.ticket)
    payload.setdefault("order_ticket", record.order_ticket)
    payload.setdefault("position_id", record.position_id)
    payload.setdefault("symbol", record.symbol)
    payload.setdefault("side", record.side)
    payload.setdefault("entry", record.entry)
    payload.setdefault("volume", record.volume)
    payload.setdefault("price", record.price)
    payload.setdefault("profit", record.profit)
    payload.setdefault("commission", record.commission)
    payload.setdefault("swap", record.swap)
    payload.setdefault("fee", record.fee)
    payload.setdefault("reason", record.reason)
    payload.setdefault("comment", record.comment)
    payload.setdefault("is_manual", bool(record.is_manual))
    payload.setdefault("time_utc", to_iso(record.time_utc))
    payload.setdefault("source", record.source)
    payload.setdefault("synced_at", to_iso(record.synced_at))
    payload.setdefault("updated_at", to_iso(record.updated_at))
    return payload
