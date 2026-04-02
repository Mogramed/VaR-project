from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from var_project.alerts.engine import AlertEvent
from var_project.storage.models import Base
from var_project.storage.repositories import StorageReadRepository, StorageWriteRepository
from var_project.storage.settings import StorageSettings
from var_project.validation.model_validation import ValidationSummary


class AppStorage:
    def __init__(self, settings: StorageSettings):
        self.settings = settings
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        self.engine = create_engine(settings.database_url, future=True, connect_args=connect_args)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.reads = StorageReadRepository(self.session_factory)
        self.writes = StorageWriteRepository(self.session_factory, settings)

    @classmethod
    def from_root(cls, root: Path, raw_config: Mapping[str, Any] | None = None) -> "AppStorage":
        return cls(StorageSettings.from_root(root, raw_config))

    def initialize(self, *, create_schema: bool = False) -> None:
        if self.settings.database_path is not None:
            self.settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.analytics_dir.mkdir(parents=True, exist_ok=True)
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
        self.settings.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.settings.tick_archive_dir.mkdir(parents=True, exist_ok=True)
        if create_schema:
            Base.metadata.create_all(self.engine)

    def ping(self) -> tuple[bool, str | None]:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True, None
        except Exception as exc:  # pragma: no cover - driver-specific failures
            return False, str(exc)

    def schema_ready(self) -> bool:
        try:
            inspector = inspect(self.engine)
            return inspector.has_table("portfolios") and inspector.has_table("artifacts")
        except Exception:
            return False

    def upsert_portfolio(
        self,
        name: str,
        base_currency: str,
        symbols: list[str],
        positions: Mapping[str, Any],
        slug: str | None = None,
    ) -> int:
        return self.writes.upsert_portfolio(
            name=name,
            base_currency=base_currency,
            symbols=symbols,
            positions=positions,
            slug=slug,
        )

    def register_artifact(
        self,
        path: Path,
        artifact_type: str,
        format: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        return self.writes.register_artifact(
            path=path,
            artifact_type=artifact_type,
            format=format,
            details=details,
        )

    def write_dataframe_artifact(
        self,
        frame: pd.DataFrame,
        path: Path,
        artifact_type: str,
        *,
        index: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        return self.writes.write_dataframe_artifact(
            frame,
            path=path,
            artifact_type=artifact_type,
            index=index,
            details=details,
        )

    def write_json_artifact(
        self,
        payload: Mapping[str, Any] | ValidationSummary,
        path: Path,
        artifact_type: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        return self.writes.write_json_artifact(
            payload,
            path=path,
            artifact_type=artifact_type,
            details=details,
        )

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
        return self.writes.record_backtest_run(
            portfolio_id=portfolio_id,
            artifact_id=artifact_id,
            timeframe=timeframe,
            days=days,
            alpha=alpha,
            window=window,
            n_rows=n_rows,
            summary=summary,
        )

    def record_validation_run(
        self,
        summary: ValidationSummary,
        *,
        portfolio_id: int | None,
        source_artifact_id: int | None = None,
    ) -> int:
        return self.writes.record_validation_run(
            summary,
            portfolio_id=portfolio_id,
            source_artifact_id=source_artifact_id,
        )

    def record_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        portfolio_id: int | None,
        artifact_id: int | None = None,
        source: str = "live",
    ) -> int:
        return self.writes.record_snapshot(
            snapshot,
            portfolio_id=portfolio_id,
            artifact_id=artifact_id,
            source=source,
        )

    def record_alerts(
        self,
        alerts: Iterable[AlertEvent | Mapping[str, Any]],
        *,
        portfolio_id: int | None = None,
        snapshot_id: int | None = None,
        validation_run_id: int | None = None,
    ) -> list[int]:
        return self.writes.record_alerts(
            alerts,
            portfolio_id=portfolio_id,
            snapshot_id=snapshot_id,
            validation_run_id=validation_run_id,
        )

    def record_decision(
        self,
        decision: Mapping[str, Any],
        *,
        portfolio_id: int | None = None,
    ) -> int:
        return self.writes.record_decision(decision, portfolio_id=portfolio_id)

    def record_capital_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        portfolio_id: int | None = None,
        source: str = "historical",
    ) -> int:
        return self.writes.record_capital_snapshot(snapshot, portfolio_id=portfolio_id, source=source)

    def record_execution_result(
        self,
        payload: Mapping[str, Any],
        *,
        portfolio_id: int | None = None,
        decision_id: int | None = None,
    ) -> int:
        return self.writes.record_execution_result(payload, portfolio_id=portfolio_id, decision_id=decision_id)

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
        return self.writes.record_audit_event(
            actor=actor,
            action_type=action_type,
            object_type=object_type,
            object_id=object_id,
            payload=payload,
            portfolio_id=portfolio_id,
        )

    def upsert_reconciliation_acknowledgement(
        self,
        *,
        portfolio_id: int | None,
        symbol: str,
        reason: str = "",
        operator_note: str = "",
        mismatch_status: str | None = None,
        incident_status: str | None = None,
        resolution_note: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        return self.writes.upsert_reconciliation_acknowledgement(
            portfolio_id=portfolio_id,
            symbol=symbol,
            reason=reason,
            operator_note=operator_note,
            mismatch_status=mismatch_status,
            incident_status=incident_status,
            resolution_note=resolution_note,
            payload=payload,
        )

    def latest_snapshot(self, *, source: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_snapshot(source=source, portfolio_slug=portfolio_slug)

    def recent_snapshots(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.recent_snapshots(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def latest_backtest_run(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_backtest_run(portfolio_slug=portfolio_slug)

    def latest_validation_run(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_validation_run(portfolio_slug=portfolio_slug)

    def recent_alerts(self, *, limit: int = 25) -> list[dict[str, Any]]:
        return self.reads.recent_alerts(limit=limit)

    def latest_artifact(self, artifact_type: str, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_artifact(artifact_type, portfolio_slug=portfolio_slug)

    def recent_artifacts(
        self,
        artifact_type: str,
        *,
        limit: int = 25,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.recent_artifacts(artifact_type, limit=limit, portfolio_slug=portfolio_slug)

    def artifact_by_id(self, artifact_id: int) -> dict[str, Any] | None:
        return self.reads.artifact_by_id(artifact_id)

    def latest_capital_snapshot(self, *, source: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_capital_snapshot(source=source, portfolio_slug=portfolio_slug)

    def capital_history(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def recent_decisions(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_decisions(limit=limit, portfolio_slug=portfolio_slug)

    def recent_execution_results(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_execution_results(limit=limit, portfolio_slug=portfolio_slug)

    def recent_execution_fills(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_execution_fills(limit=limit, portfolio_slug=portfolio_slug)

    def recent_audit_events(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_audit_events(limit=limit, portfolio_slug=portfolio_slug)

    def reconciliation_acknowledgements(
        self,
        *,
        portfolio_slug: str | None = None,
        symbol: str | None = None,
        incident_status: str | None = None,
        include_resolved: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.reconciliation_acknowledgements(
            portfolio_slug=portfolio_slug,
            symbol=symbol,
            incident_status=incident_status,
            include_resolved=include_resolved,
            limit=limit,
        )

    def list_portfolios(self) -> list[dict[str, Any]]:
        return self.reads.list_portfolios()

    def upsert_instrument(self, payload: Mapping[str, Any], *, source: str = "mt5") -> int:
        return self.writes.upsert_instrument(payload, source=source)

    def record_market_data_sync(
        self,
        *,
        portfolio_id: int | None,
        portfolio_slug: str | None,
        mode: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        return self.writes.record_market_data_sync(
            portfolio_id=portfolio_id,
            portfolio_slug=portfolio_slug,
            mode=mode,
            status=status,
            details=details,
        )

    def sync_market_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        bars: Iterable[Mapping[str, Any]],
        sync_run_id: int | None = None,
        source: str = "mt5",
    ) -> int:
        return self.writes.sync_market_bars(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            sync_run_id=sync_run_id,
            source=source,
        )

    def sync_mt5_order_history(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        sync_run_id: int | None = None,
        portfolio_id: int | None = None,
        source: str = "mt5",
    ) -> int:
        return self.writes.sync_mt5_order_history(
            records,
            sync_run_id=sync_run_id,
            portfolio_id=portfolio_id,
            source=source,
        )

    def sync_mt5_deal_history(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        sync_run_id: int | None = None,
        portfolio_id: int | None = None,
        source: str = "mt5",
    ) -> int:
        return self.writes.sync_mt5_deal_history(
            records,
            sync_run_id=sync_run_id,
            portfolio_id=portfolio_id,
            source=source,
        )

    def list_instruments(self, *, symbols: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return self.reads.list_instruments(symbols=symbols)

    def latest_market_data_sync(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_market_data_sync(portfolio_slug=portfolio_slug)

    def market_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        since: Any | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.market_bars(symbol=symbol, timeframe=timeframe, since=since, limit=limit)

    def latest_market_bar_times(self, *, symbols: Iterable[str], timeframe: str) -> dict[str, str | None]:
        return self.reads.latest_market_bar_times(symbols=symbols, timeframe=timeframe)

    def recent_mt5_order_history(
        self,
        *,
        limit: int = 50,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.recent_mt5_order_history(limit=limit, portfolio_slug=portfolio_slug)

    def recent_mt5_deal_history(
        self,
        *,
        limit: int = 50,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.recent_mt5_deal_history(limit=limit, portfolio_slug=portfolio_slug)
