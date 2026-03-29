from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from sqlalchemy import select

from var_project.alerts.engine import AlertEvent
from var_project.storage.mappers import (
    alert_to_dict,
    artifact_to_dict,
    audit_to_dict,
    backtest_to_dict,
    capital_snapshot_to_dict,
    decision_to_dict,
    execution_fill_to_dict,
    execution_to_dict,
    instrument_to_dict,
    market_bar_to_dict,
    market_data_sync_to_dict,
    mt5_deal_history_to_dict,
    mt5_order_history_to_dict,
    portfolio_to_dict,
    snapshot_to_dict,
    validation_to_dict,
)
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
    SnapshotRecord,
    ValidationRunRecord,
)
from var_project.storage.serialization import coerce_datetime, jsonable, slugify_label, utcnow
from var_project.storage.settings import StorageSettings
from var_project.validation.model_validation import ValidationSummary


class StorageWriteRepository:
    def __init__(self, session_factory: Any, settings: StorageSettings):
        self.session_factory = session_factory
        self.settings = settings

    def upsert_portfolio(
        self,
        *,
        name: str,
        base_currency: str,
        symbols: list[str],
        positions: Mapping[str, Any],
        slug: str | None = None,
    ) -> int:
        portfolio_slug = slug or slugify_label(name)
        with self.session_factory() as session:
            record = session.scalar(select(PortfolioRecord).where(PortfolioRecord.slug == portfolio_slug))
            if record is None:
                record = PortfolioRecord(
                    slug=portfolio_slug,
                    name=str(name),
                    base_currency=str(base_currency),
                    symbols_json=jsonable(list(symbols)),
                    positions_json=jsonable(dict(positions)),
                )
                session.add(record)
            else:
                record.name = str(name)
                record.base_currency = str(base_currency)
                record.symbols_json = jsonable(list(symbols))
                record.positions_json = jsonable(dict(positions))
                record.updated_at = utcnow()
            session.commit()
            session.refresh(record)
            return int(record.id)

    def register_artifact(
        self,
        *,
        path: Path,
        artifact_type: str,
        format: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        artifact_format = str(format or resolved.suffix.lstrip(".") or self.settings.analytics_format)
        size_bytes = resolved.stat().st_size if resolved.exists() else None

        with self.session_factory() as session:
            record = session.scalar(select(ArtifactRecord).where(ArtifactRecord.path == str(resolved)))
            payload = jsonable(dict(details or {}))
            if record is None:
                record = ArtifactRecord(
                    artifact_type=str(artifact_type),
                    format=artifact_format,
                    path=str(resolved),
                    size_bytes=size_bytes,
                    details_json=payload,
                )
                session.add(record)
            else:
                record.artifact_type = str(artifact_type)
                record.format = artifact_format
                record.size_bytes = size_bytes
                if details is None:
                    record.details_json = record.details_json
                else:
                    merged_details = dict(record.details_json or {})
                    merged_details.update(dict(details))
                    record.details_json = jsonable(merged_details)
                record.updated_at = utcnow()
            session.commit()
            session.refresh(record)
            return int(record.id)

    def write_dataframe_artifact(
        self,
        frame: pd.DataFrame,
        *,
        path: Path,
        artifact_type: str,
        index: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            frame.to_parquet(path, index=index)
            artifact_format = "parquet"
        else:
            frame.to_csv(path, index=index)
            artifact_format = "csv"
        return self.register_artifact(
            path=path,
            artifact_type=artifact_type,
            format=artifact_format,
            details=details,
        )

    def write_json_artifact(
        self,
        payload: Mapping[str, Any] | ValidationSummary,
        *,
        path: Path,
        artifact_type: str,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
        return self.register_artifact(path=path, artifact_type=artifact_type, format="json", details=details)

    def record_backtest_run(
        self,
        *,
        portfolio_id: int | None,
        artifact_id: int | None,
        timeframe: str | None,
        days: int | None,
        alpha: float,
        window: int,
        n_rows: int,
        summary: Mapping[str, Any] | None = None,
    ) -> int:
        with self.session_factory() as session:
            record = BacktestRunRecord(
                portfolio_id=portfolio_id,
                artifact_id=artifact_id,
                timeframe=str(timeframe) if timeframe is not None else None,
                days=int(days) if days is not None else None,
                alpha=float(alpha),
                window=int(window),
                n_rows=int(n_rows),
                summary_json=jsonable(dict(summary or {})),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def record_validation_run(
        self,
        summary: ValidationSummary,
        *,
        portfolio_id: int | None,
        source_artifact_id: int | None = None,
    ) -> int:
        payload = summary.to_dict()
        with self.session_factory() as session:
            record = ValidationRunRecord(
                portfolio_id=portfolio_id,
                source_artifact_id=source_artifact_id,
                alpha=float(summary.alpha),
                expected_rate=float(summary.expected_rate),
                best_model=summary.best_model,
                summary_json=jsonable(payload),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def record_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        portfolio_id: int | None,
        artifact_id: int | None = None,
        source: str = "live",
    ) -> int:
        payload = jsonable(dict(snapshot))
        live_pnl = payload.get("live_pnl")
        if live_pnl is None:
            live_pnl = payload.get("live_pnl_proxy")
        live_loss = payload.get("live_loss")
        if live_loss is None:
            live_loss = payload.get("live_loss_proxy")
        created_at = coerce_datetime(payload.get("time_utc")) or utcnow()

        with self.session_factory() as session:
            record = SnapshotRecord(
                portfolio_id=portfolio_id,
                artifact_id=artifact_id,
                source=str(source),
                alpha=float(payload["alpha"]) if payload.get("alpha") is not None else None,
                timeframe=str(payload["timeframe"]) if payload.get("timeframe") is not None else None,
                days=int(payload["days"]) if payload.get("days") is not None else None,
                window=int(payload["window"]) if payload.get("window") is not None else None,
                live_pnl=float(live_pnl) if live_pnl is not None else None,
                live_loss=float(live_loss) if live_loss is not None else None,
                breach_hist=bool(payload["breach_hist"]) if payload.get("breach_hist") is not None else None,
                payload_json=payload,
                created_at=created_at,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def record_alerts(
        self,
        alerts: Iterable[AlertEvent | Mapping[str, Any]],
        *,
        portfolio_id: int | None = None,
        snapshot_id: int | None = None,
        validation_run_id: int | None = None,
    ) -> list[int]:
        inserted: list[int] = []
        with self.session_factory() as session:
            for alert in alerts:
                payload = jsonable(alert)
                record = AlertRecord(
                    portfolio_id=portfolio_id,
                    snapshot_id=snapshot_id,
                    validation_run_id=validation_run_id,
                    source=str(payload.get("source") or "unknown"),
                    severity=str(payload.get("severity") or "INFO"),
                    code=str(payload.get("code") or "UNKNOWN"),
                    message=str(payload.get("message") or ""),
                    context_json=jsonable(payload.get("context") or {}),
                )
                session.add(record)
                session.flush()
                inserted.append(int(record.id))
            session.commit()
        return inserted

    def record_decision(
        self,
        decision: Mapping[str, Any],
        *,
        portfolio_id: int | None = None,
    ) -> int:
        payload = jsonable(dict(decision))
        created_at = coerce_datetime(payload.get("created_at") or payload.get("time_utc")) or utcnow()
        with self.session_factory() as session:
            record = DecisionRecord(
                portfolio_id=portfolio_id,
                symbol=str(payload.get("symbol") or ""),
                requested_delta_position_eur=float(payload.get("requested_delta_position_eur", 0.0)),
                approved_delta_position_eur=float(payload.get("approved_delta_position_eur", 0.0)),
                decision=str(payload.get("decision") or "REJECT"),
                model_used=str(payload.get("model_used") or "hist"),
                payload_json=payload,
                created_at=created_at,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def record_capital_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        portfolio_id: int | None = None,
        source: str = "historical",
    ) -> int:
        payload = jsonable(dict(snapshot))
        created_at = coerce_datetime(payload.get("snapshot_timestamp") or payload.get("created_at")) or utcnow()
        with self.session_factory() as session:
            record = CapitalSnapshotRecord(
                portfolio_id=portfolio_id,
                source=str(source),
                reference_model=str(payload.get("reference_model") or "hist"),
                total_budget_eur=float(payload.get("total_capital_budget_eur", 0.0)),
                consumed_eur=float(payload.get("total_capital_consumed_eur", 0.0)),
                reserved_eur=float(payload.get("total_capital_reserved_eur", 0.0)),
                remaining_eur=float(payload.get("total_capital_remaining_eur", 0.0)),
                payload_json=payload,
                created_at=created_at,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def record_execution_result(
        self,
        payload: Mapping[str, Any],
        *,
        portfolio_id: int | None = None,
        decision_id: int | None = None,
    ) -> int:
        execution = jsonable(dict(payload))
        created_at = coerce_datetime(execution.get("created_at") or execution.get("time_utc")) or utcnow()
        mt5_result = dict(execution.get("mt5_result") or {})
        with self.session_factory() as session:
            record = ExecutionRecord(
                portfolio_id=portfolio_id,
                decision_id=decision_id,
                symbol=str(execution.get("symbol") or ""),
                requested_delta_position_eur=float(execution.get("requested_delta_position_eur", 0.0)),
                approved_delta_position_eur=float(execution.get("approved_delta_position_eur", 0.0)),
                executed_delta_position_eur=float(execution.get("executed_delta_position_eur", 0.0)),
                requested_volume_lots=None if execution.get("requested_volume_lots") is None else float(execution.get("requested_volume_lots")),
                approved_volume_lots=None if execution.get("approved_volume_lots") is None else float(execution.get("approved_volume_lots")),
                submitted_volume_lots=None if execution.get("submitted_volume_lots") is None else float(execution.get("submitted_volume_lots")),
                filled_volume_lots=None if execution.get("filled_volume_lots") is None else float(execution.get("filled_volume_lots")),
                remaining_volume_lots=None if execution.get("remaining_volume_lots") is None else float(execution.get("remaining_volume_lots")),
                fill_ratio=None if execution.get("fill_ratio") is None else float(execution.get("fill_ratio")),
                broker_status=None if execution.get("broker_status") is None else str(execution.get("broker_status")),
                position_id=None if execution.get("position_id") is None else int(execution.get("position_id")),
                slippage_points=None if execution.get("slippage_points") is None else float(execution.get("slippage_points")),
                reconciliation_status=None if execution.get("reconciliation_status") is None else str(execution.get("reconciliation_status")),
                status=str(execution.get("status") or "UNKNOWN"),
                mt5_order_ticket=None if mt5_result.get("order") is None else int(mt5_result.get("order")),
                mt5_deal_ticket=None if mt5_result.get("deal") is None else int(mt5_result.get("deal")),
                payload_json=execution,
                created_at=created_at,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            fills = list(execution.get("fills") or [])
            for item in fills:
                fill = jsonable(dict(item))
                session.add(
                    ExecutionFillRecord(
                        execution_result_id=int(record.id),
                        portfolio_id=portfolio_id,
                        symbol=str(fill.get("symbol") or execution.get("symbol") or ""),
                        order_ticket=None if fill.get("order_ticket") is None else int(fill.get("order_ticket")),
                        deal_ticket=None if fill.get("deal_ticket") is None else int(fill.get("deal_ticket")),
                        position_id=None if fill.get("position_id") is None else int(fill.get("position_id")),
                        side=None if fill.get("side") is None else str(fill.get("side")),
                        entry=None if fill.get("entry") is None else str(fill.get("entry")),
                        volume_lots=float(fill.get("volume_lots", 0.0)),
                        price=None if fill.get("price") is None else float(fill.get("price")),
                        profit=None if fill.get("profit") is None else float(fill.get("profit")),
                        commission=None if fill.get("commission") is None else float(fill.get("commission")),
                        swap=None if fill.get("swap") is None else float(fill.get("swap")),
                        fee=None if fill.get("fee") is None else float(fill.get("fee")),
                        reason=None if fill.get("reason") is None else str(fill.get("reason")),
                        comment=None if fill.get("comment") is None else str(fill.get("comment")),
                        is_manual=bool(fill.get("is_manual")),
                        slippage_points=None if fill.get("slippage_points") is None else float(fill.get("slippage_points")),
                        payload_json=fill,
                        time_utc=coerce_datetime(fill.get("time_utc")),
                        created_at=coerce_datetime(fill.get("created_at") or fill.get("time_utc")) or created_at,
                    )
                )
            session.commit()
            return int(record.id)

    def record_audit_event(
        self,
        *,
        actor: str,
        action_type: str,
        object_type: str | None = None,
        object_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
        portfolio_id: int | None = None,
    ) -> int:
        with self.session_factory() as session:
            record = AuditRecord(
                portfolio_id=portfolio_id,
                actor=str(actor),
                action_type=str(action_type),
                object_type=None if object_type is None else str(object_type),
                object_id=object_id,
                payload_json=jsonable(dict(payload or {})),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def upsert_instrument(
        self,
        payload: Mapping[str, Any],
        *,
        source: str = "mt5",
    ) -> int:
        instrument = jsonable(dict(payload))
        symbol = str(instrument.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Instrument payload must include a symbol.")

        with self.session_factory() as session:
            record = session.scalar(select(InstrumentRecord).where(InstrumentRecord.symbol == symbol))
            if record is None:
                record = InstrumentRecord(symbol=symbol, source=str(source))
                session.add(record)
            record.asset_class = None if instrument.get("asset_class") is None else str(instrument.get("asset_class"))
            record.contract_size = None if instrument.get("contract_size") is None else float(instrument.get("contract_size"))
            record.base_currency = None if instrument.get("base_currency") is None else str(instrument.get("base_currency"))
            record.quote_currency = None if instrument.get("quote_currency") is None else str(instrument.get("quote_currency"))
            record.profit_currency = None if instrument.get("profit_currency") is None else str(instrument.get("profit_currency"))
            record.margin_currency = None if instrument.get("margin_currency") is None else str(instrument.get("margin_currency"))
            record.tick_size = None if instrument.get("tick_size") is None else float(instrument.get("tick_size"))
            record.tick_value = None if instrument.get("tick_value") is None else float(instrument.get("tick_value"))
            record.volume_min = None if instrument.get("volume_min") is None else float(instrument.get("volume_min"))
            record.volume_max = None if instrument.get("volume_max") is None else float(instrument.get("volume_max"))
            record.volume_step = None if instrument.get("volume_step") is None else float(instrument.get("volume_step"))
            record.trading_mode = None if instrument.get("trading_mode") is None else str(instrument.get("trading_mode"))
            record.source = str(source)
            record.payload_json = instrument
            record.synced_at = utcnow()
            record.updated_at = utcnow()
            session.commit()
            session.refresh(record)
            return int(record.id)

    def record_market_data_sync(
        self,
        *,
        portfolio_id: int | None,
        portfolio_slug: str | None,
        mode: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        with self.session_factory() as session:
            record = MarketDataSyncRecord(
                portfolio_id=portfolio_id,
                portfolio_slug=None if portfolio_slug is None else str(portfolio_slug),
                mode=str(mode),
                status=str(status),
                details_json=jsonable(dict(details or {})),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def sync_market_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        bars: Iterable[Mapping[str, Any]],
        sync_run_id: int | None = None,
        source: str = "mt5",
    ) -> int:
        inserted = 0
        normalized_symbol = str(symbol).upper()
        normalized_timeframe = str(timeframe).upper()
        with self.session_factory() as session:
            for bar in bars:
                payload = jsonable(dict(bar))
                time_utc = coerce_datetime(payload.get("time") or payload.get("time_utc"))
                if time_utc is None:
                    continue
                record = session.scalar(
                    select(MarketBarRecord).where(
                        MarketBarRecord.symbol == normalized_symbol,
                        MarketBarRecord.timeframe == normalized_timeframe,
                        MarketBarRecord.time_utc == time_utc,
                    )
                )
                if record is None:
                    record = MarketBarRecord(
                        symbol=normalized_symbol,
                        timeframe=normalized_timeframe,
                        time_utc=time_utc,
                    )
                    session.add(record)
                record.sync_run_id = sync_run_id
                record.open = float(payload.get("open", 0.0))
                record.high = float(payload.get("high", 0.0))
                record.low = float(payload.get("low", 0.0))
                record.close = float(payload.get("close", 0.0))
                record.tick_volume = None if payload.get("tick_volume") is None else float(payload.get("tick_volume"))
                record.spread = None if payload.get("spread") is None else float(payload.get("spread"))
                record.real_volume = None if payload.get("real_volume") is None else float(payload.get("real_volume"))
                record.source = str(source)
                inserted += 1
            session.commit()
        return inserted

    def sync_mt5_order_history(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        sync_run_id: int | None = None,
        portfolio_id: int | None = None,
        source: str = "mt5",
    ) -> int:
        upserted = 0
        with self.session_factory() as session:
            for item in records:
                payload = jsonable(dict(item))
                ticket = payload.get("ticket")
                if ticket is None:
                    continue
                record = session.scalar(select(MT5OrderHistoryRecord).where(MT5OrderHistoryRecord.ticket == int(ticket)))
                if record is None:
                    record = MT5OrderHistoryRecord(ticket=int(ticket))
                    session.add(record)
                record.sync_run_id = sync_run_id
                record.portfolio_id = portfolio_id
                record.position_id = None if payload.get("position_id") is None else int(payload.get("position_id"))
                record.symbol = str(payload.get("symbol") or "").upper()
                record.side = None if payload.get("side") is None else str(payload.get("side"))
                record.order_type = None if payload.get("order_type") is None else str(payload.get("order_type"))
                record.state = None if payload.get("state") is None else str(payload.get("state"))
                record.volume_initial = None if payload.get("volume_initial") is None else float(payload.get("volume_initial"))
                record.volume_current = None if payload.get("volume_current") is None else float(payload.get("volume_current"))
                record.price_open = None if payload.get("price_open") is None else float(payload.get("price_open"))
                record.price_current = None if payload.get("price_current") is None else float(payload.get("price_current"))
                record.comment = None if payload.get("comment") is None else str(payload.get("comment"))
                record.is_manual = bool(payload.get("is_manual", False))
                record.time_setup_utc = coerce_datetime(payload.get("time_setup_utc") or payload.get("time_setup"))
                record.time_done_utc = coerce_datetime(payload.get("time_done_utc") or payload.get("time_done"))
                record.source = str(source)
                record.payload_json = payload
                record.synced_at = utcnow()
                record.updated_at = utcnow()
                upserted += 1
            session.commit()
        return upserted

    def sync_mt5_deal_history(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        sync_run_id: int | None = None,
        portfolio_id: int | None = None,
        source: str = "mt5",
    ) -> int:
        upserted = 0
        with self.session_factory() as session:
            for item in records:
                payload = jsonable(dict(item))
                ticket = payload.get("ticket")
                if ticket is None:
                    continue
                record = session.scalar(select(MT5DealHistoryRecord).where(MT5DealHistoryRecord.ticket == int(ticket)))
                if record is None:
                    record = MT5DealHistoryRecord(ticket=int(ticket))
                    session.add(record)
                record.sync_run_id = sync_run_id
                record.portfolio_id = portfolio_id
                record.order_ticket = None if payload.get("order_ticket") is None else int(payload.get("order_ticket"))
                record.position_id = None if payload.get("position_id") is None else int(payload.get("position_id"))
                record.symbol = str(payload.get("symbol") or "").upper()
                record.side = None if payload.get("side") is None else str(payload.get("side"))
                record.entry = None if payload.get("entry") is None else str(payload.get("entry"))
                record.volume = None if payload.get("volume") is None else float(payload.get("volume"))
                record.price = None if payload.get("price") is None else float(payload.get("price"))
                record.profit = None if payload.get("profit") is None else float(payload.get("profit"))
                record.commission = None if payload.get("commission") is None else float(payload.get("commission"))
                record.swap = None if payload.get("swap") is None else float(payload.get("swap"))
                record.fee = None if payload.get("fee") is None else float(payload.get("fee"))
                record.reason = None if payload.get("reason") is None else str(payload.get("reason"))
                record.comment = None if payload.get("comment") is None else str(payload.get("comment"))
                record.is_manual = bool(payload.get("is_manual", False))
                record.time_utc = coerce_datetime(payload.get("time_utc") or payload.get("time"))
                record.source = str(source)
                record.payload_json = payload
                record.synced_at = utcnow()
                record.updated_at = utcnow()
                upserted += 1
            session.commit()
        return upserted


class StorageReadRepository:
    def __init__(self, session_factory: Any):
        self.session_factory = session_factory

    def latest_snapshot(self, *, source: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        with self.session_factory() as session:
            stmt = select(SnapshotRecord)
            if source:
                stmt = stmt.where(SnapshotRecord.source == source)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return None
                stmt = stmt.where(SnapshotRecord.portfolio_id == portfolio_id)
            record = session.scalars(stmt.order_by(SnapshotRecord.created_at.desc(), SnapshotRecord.id.desc())).first()
            return None if record is None else snapshot_to_dict(record)

    def latest_backtest_run(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        with self.session_factory() as session:
            stmt = select(BacktestRunRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return None
                stmt = stmt.where(BacktestRunRecord.portfolio_id == portfolio_id)
            record = session.scalars(stmt.order_by(BacktestRunRecord.created_at.desc(), BacktestRunRecord.id.desc())).first()
            return None if record is None else backtest_to_dict(record)

    def latest_validation_run(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        with self.session_factory() as session:
            stmt = select(ValidationRunRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return None
                stmt = stmt.where(ValidationRunRecord.portfolio_id == portfolio_id)
            record = session.scalars(stmt.order_by(ValidationRunRecord.created_at.desc(), ValidationRunRecord.id.desc())).first()
            return None if record is None else validation_to_dict(record)

    def recent_alerts(self, *, limit: int = 25) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(AlertRecord).order_by(AlertRecord.created_at.desc(), AlertRecord.id.desc())
            records = session.scalars(stmt.limit(int(limit))).all()
            return [alert_to_dict(record) for record in records]

    def latest_artifact(self, artifact_type: str, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        records = self.recent_artifacts(artifact_type, limit=50, portfolio_slug=portfolio_slug)
        return records[0] if records else None

    def recent_artifacts(
        self,
        artifact_type: str,
        *,
        limit: int = 25,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = (
                select(ArtifactRecord)
                .where(ArtifactRecord.artifact_type == artifact_type)
                .order_by(ArtifactRecord.updated_at.desc(), ArtifactRecord.id.desc())
            )
            records = [artifact_to_dict(record) for record in session.scalars(stmt.limit(max(int(limit), 1) * 4)).all()]
            if portfolio_slug is None:
                return records[: int(limit)]
            filtered = [
                record
                for record in records
                if str((record.get("details") or {}).get("portfolio_slug") or "") == str(portfolio_slug)
            ]
            return filtered[: int(limit)]

    def artifact_by_id(self, artifact_id: int) -> dict[str, Any] | None:
        with self.session_factory() as session:
            record = session.get(ArtifactRecord, int(artifact_id))
            return None if record is None else artifact_to_dict(record)

    def latest_capital_snapshot(self, *, source: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        with self.session_factory() as session:
            stmt = select(CapitalSnapshotRecord)
            if source:
                stmt = stmt.where(CapitalSnapshotRecord.source == source)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return None
                stmt = stmt.where(CapitalSnapshotRecord.portfolio_id == portfolio_id)
            record = session.scalars(stmt.order_by(CapitalSnapshotRecord.created_at.desc(), CapitalSnapshotRecord.id.desc())).first()
            return None if record is None else capital_snapshot_to_dict(record)

    def capital_history(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(CapitalSnapshotRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return []
                stmt = stmt.where(CapitalSnapshotRecord.portfolio_id == portfolio_id)
            records = session.scalars(stmt.order_by(CapitalSnapshotRecord.created_at.desc(), CapitalSnapshotRecord.id.desc()).limit(int(limit))).all()
            return [capital_snapshot_to_dict(record) for record in records]

    def recent_decisions(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(DecisionRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return []
                stmt = stmt.where(DecisionRecord.portfolio_id == portfolio_id)
            records = session.scalars(stmt.order_by(DecisionRecord.created_at.desc(), DecisionRecord.id.desc()).limit(int(limit))).all()
            return [decision_to_dict(record) for record in records]

    def recent_execution_results(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(ExecutionRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return []
                stmt = stmt.where(ExecutionRecord.portfolio_id == portfolio_id)
            records = session.scalars(stmt.order_by(ExecutionRecord.created_at.desc(), ExecutionRecord.id.desc()).limit(int(limit))).all()
            return [execution_to_dict(record) for record in records]

    def recent_execution_fills(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(ExecutionFillRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return []
                stmt = stmt.where(ExecutionFillRecord.portfolio_id == portfolio_id)
            records = session.scalars(
                stmt.order_by(ExecutionFillRecord.time_utc.desc(), ExecutionFillRecord.id.desc()).limit(int(limit))
            ).all()
            return [execution_fill_to_dict(record) for record in records]

    def recent_audit_events(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(AuditRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return []
                stmt = stmt.where(AuditRecord.portfolio_id == portfolio_id)
            records = session.scalars(stmt.order_by(AuditRecord.created_at.desc(), AuditRecord.id.desc()).limit(int(limit))).all()
            return [audit_to_dict(record) for record in records]

    def list_portfolios(self) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            records = session.scalars(select(PortfolioRecord).order_by(PortfolioRecord.slug.asc())).all()
            return [portfolio_to_dict(record) for record in records]

    def list_instruments(self, *, symbols: Iterable[str] | None = None) -> list[dict[str, Any]]:
        allowed = {str(symbol).upper() for symbol in symbols or []}
        with self.session_factory() as session:
            stmt = select(InstrumentRecord).order_by(InstrumentRecord.symbol.asc())
            if allowed:
                stmt = stmt.where(InstrumentRecord.symbol.in_(sorted(allowed)))
            records = session.scalars(stmt).all()
            return [instrument_to_dict(record) for record in records]

    def latest_market_data_sync(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        with self.session_factory() as session:
            stmt = select(MarketDataSyncRecord)
            if portfolio_slug:
                stmt = stmt.where(MarketDataSyncRecord.portfolio_slug == str(portfolio_slug))
            record = session.scalars(stmt.order_by(MarketDataSyncRecord.synced_at.desc(), MarketDataSyncRecord.id.desc())).first()
            return None if record is None else market_data_sync_to_dict(record)

    def market_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        since: Any | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(MarketBarRecord).where(
                MarketBarRecord.symbol == str(symbol).upper(),
                MarketBarRecord.timeframe == str(timeframe).upper(),
            )
            if since is not None:
                since_dt = coerce_datetime(since)
                if since_dt is not None:
                    stmt = stmt.where(MarketBarRecord.time_utc >= since_dt)
            stmt = stmt.order_by(MarketBarRecord.time_utc.asc(), MarketBarRecord.id.asc())
            if limit is not None:
                stmt = stmt.limit(int(limit))
            records = session.scalars(stmt).all()
            return [market_bar_to_dict(record) for record in records]

    def latest_market_bar_times(
        self,
        *,
        symbols: Iterable[str],
        timeframe: str,
    ) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        normalized_timeframe = str(timeframe).upper()
        with self.session_factory() as session:
            for symbol in symbols:
                record = session.scalars(
                    select(MarketBarRecord)
                    .where(
                        MarketBarRecord.symbol == str(symbol).upper(),
                        MarketBarRecord.timeframe == normalized_timeframe,
                    )
                    .order_by(MarketBarRecord.time_utc.desc(), MarketBarRecord.id.desc())
                    .limit(1)
                ).first()
                result[str(symbol).upper()] = None if record is None else market_bar_to_dict(record).get("time_utc")
        return result

    def recent_mt5_order_history(
        self,
        *,
        limit: int = 50,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(MT5OrderHistoryRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return []
                stmt = stmt.where(MT5OrderHistoryRecord.portfolio_id == portfolio_id)
            records = session.scalars(
                stmt.order_by(MT5OrderHistoryRecord.time_setup_utc.desc(), MT5OrderHistoryRecord.ticket.desc()).limit(int(limit))
            ).all()
            return [mt5_order_history_to_dict(record) for record in records]

    def recent_mt5_deal_history(
        self,
        *,
        limit: int = 50,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            stmt = select(MT5DealHistoryRecord)
            if portfolio_slug:
                portfolio_id = self._portfolio_id(session, portfolio_slug)
                if portfolio_id is None:
                    return []
                stmt = stmt.where(MT5DealHistoryRecord.portfolio_id == portfolio_id)
            records = session.scalars(
                stmt.order_by(MT5DealHistoryRecord.time_utc.desc(), MT5DealHistoryRecord.ticket.desc()).limit(int(limit))
            ).all()
            return [mt5_deal_history_to_dict(record) for record in records]

    @staticmethod
    def _portfolio_id(session: Any, portfolio_slug: str) -> int | None:
        record = session.scalar(select(PortfolioRecord).where(PortfolioRecord.slug == portfolio_slug))
        return None if record is None else int(record.id)
