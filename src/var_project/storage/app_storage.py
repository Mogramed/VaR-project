from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from var_project.alerts.engine import AlertEvent
from var_project.storage.migrations import expected_head_revision
from var_project.storage.models import Base
from var_project.storage.repositories import StorageReadRepository, StorageWriteRepository
from var_project.storage.schema_checks import validate_storage_schema
from var_project.storage.settings import StorageSettings
from var_project.validation.model_validation import ValidationSummary


class AppStorage:
    def __init__(self, settings: StorageSettings, *, root: Path | None = None):
        self.settings = settings
        self.root = (root or Path.cwd()).resolve()
        is_sqlite = settings.database_url.startswith("sqlite")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        engine_kwargs: dict[str, Any] = {
            "future": True,
            "connect_args": connect_args,
            "pool_pre_ping": True,
        }
        if not is_sqlite:
            engine_kwargs["pool_recycle"] = int(os.getenv("VAR_PROJECT_DB_POOL_RECYCLE_SECONDS", "300"))
            engine_kwargs["pool_size"] = int(os.getenv("VAR_PROJECT_DB_POOL_SIZE", "8"))
            engine_kwargs["max_overflow"] = int(os.getenv("VAR_PROJECT_DB_MAX_OVERFLOW", "8"))
            engine_kwargs["pool_timeout"] = int(os.getenv("VAR_PROJECT_DB_POOL_TIMEOUT_SECONDS", "15"))
        self.engine = create_engine(settings.database_url, **engine_kwargs)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.reads = StorageReadRepository(self.session_factory)
        self.writes = StorageWriteRepository(self.session_factory, settings)

    @classmethod
    def from_root(cls, root: Path, raw_config: Mapping[str, Any] | None = None) -> "AppStorage":
        return cls(StorageSettings.from_root(root, raw_config), root=root)

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

    def _current_alembic_revision(self) -> str | None:
        try:
            inspector = inspect(self.engine)
            if not inspector.has_table("alembic_version"):
                return None
            with self.engine.connect() as connection:
                rows = connection.execute(text("SELECT version_num FROM alembic_version ORDER BY version_num")).scalars().all()
        except Exception:
            return None
        revisions = [str(item) for item in rows if item not in {None, "", "null"}]
        if not revisions:
            return None
        if len(revisions) == 1:
            return revisions[0]
        return ",".join(revisions)

    def schema_status(self, *, strict_revision: bool = True) -> dict[str, Any]:
        issues: list[str] = []
        expected_revision: str | None = None
        current_revision: str | None = None
        try:
            issues.extend(validate_storage_schema(self.engine))
            current_revision = self._current_alembic_revision()
            if strict_revision:
                expected_revision_error = False
                try:
                    expected_revision = expected_head_revision(self.root)
                except Exception as exc:
                    expected_revision_error = True
                    issues.append(f"Cannot determine expected Alembic head revision: {exc}")
                if not expected_revision_error and expected_revision in {None, "", "null"}:
                    issues.append("Cannot determine expected Alembic head revision.")
                if current_revision is None:
                    issues.append("Missing Alembic revision marker in table `alembic_version`.")
                elif expected_revision and current_revision != expected_revision:
                    issues.append(
                        "Alembic revision mismatch: "
                        f"current '{current_revision}' != expected '{expected_revision}'."
                    )
        except Exception as exc:
            issues.append(f"Schema inspection failed: {exc}")

        deduped_issues = list(dict.fromkeys(str(item) for item in issues if str(item).strip()))
        ready = len(deduped_issues) == 0
        hint = "Run `var-project db upgrade` (or `alembic upgrade head`) then retry."
        detail = (
            "Database schema is ready."
            if ready
            else f"Database schema is invalid. {deduped_issues[0]} {hint}"
        )
        return {
            "ready": ready,
            "strict_revision": bool(strict_revision),
            "issues": deduped_issues,
            "detail": detail,
            "hint": hint,
            "current_revision": current_revision,
            "expected_revision": expected_revision,
            "target": self.settings.database_url,
        }

    def schema_ready(self, *, strict_revision: bool = True) -> bool:
        try:
            return bool(self.schema_status(strict_revision=strict_revision).get("ready"))
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

    def update_execution_reconciliation(
        self,
        execution_id: int,
        *,
        reconciliation_status: str | None = None,
        filled_volume_lots: float | None = None,
        remaining_volume_lots: float | None = None,
        fill_ratio: float | None = None,
        broker_status: str | None = None,
        position_id: int | None = None,
        mt5_order_ticket: int | None = None,
        mt5_deal_ticket: int | None = None,
    ) -> dict[str, Any] | None:
        return self.writes.update_execution_reconciliation(
            execution_id,
            reconciliation_status=reconciliation_status,
            filled_volume_lots=filled_volume_lots,
            remaining_volume_lots=remaining_volume_lots,
            fill_ratio=fill_ratio,
            broker_status=broker_status,
            position_id=position_id,
            mt5_order_ticket=mt5_order_ticket,
            mt5_deal_ticket=mt5_deal_ticket,
        )

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

    def create_operator_run(
        self,
        *,
        portfolio_id: int | None,
        portfolio_slug: str | None,
        action: str,
        request_id: str,
        status: str,
        stage: str,
        request_payload: Mapping[str, Any] | None = None,
        artifact_refs: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        hint: str | None = None,
        queue_task_id: str | None = None,
        reused_run_id: int | None = None,
        started_at: Any | None = None,
        finished_at: Any | None = None,
    ) -> int:
        return self.writes.create_operator_run(
            portfolio_id=portfolio_id,
            portfolio_slug=portfolio_slug,
            action=action,
            request_id=request_id,
            status=status,
            stage=stage,
            request_payload=request_payload,
            artifact_refs=artifact_refs,
            result=result,
            error_code=error_code,
            error_message=error_message,
            hint=hint,
            queue_task_id=queue_task_id,
            reused_run_id=reused_run_id,
            started_at=started_at,
            finished_at=finished_at,
        )

    def update_operator_run(
        self,
        run_id: int,
        *,
        status: str | None = None,
        stage: str | None = None,
        artifact_refs: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        hint: str | None = None,
        queue_task_id: str | None = None,
        reused_run_id: int | None = None,
        started_at: Any | None = None,
        finished_at: Any | None = None,
    ) -> dict[str, Any] | None:
        return self.writes.update_operator_run(
            run_id,
            status=status,
            stage=stage,
            artifact_refs=artifact_refs,
            result=result,
            error_code=error_code,
            error_message=error_message,
            hint=hint,
            queue_task_id=queue_task_id,
            reused_run_id=reused_run_id,
            started_at=started_at,
            finished_at=finished_at,
        )

    def claim_operator_run(
        self,
        run_id: int,
        *,
        stage: str = "starting",
        started_at: Any | None = None,
    ) -> dict[str, Any] | None:
        return self.writes.claim_operator_run(
            run_id,
            stage=stage,
            started_at=started_at,
        )

    def operator_run_by_id(self, run_id: int) -> dict[str, Any] | None:
        return self.reads.operator_run_by_id(run_id)

    def list_operator_runs(
        self,
        *,
        portfolio_slug: str | None = None,
        action: str | None = None,
        statuses: Iterable[str] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        return self.reads.list_operator_runs(
            portfolio_slug=portfolio_slug,
            action=action,
            statuses=statuses,
            limit=limit,
        )

    def latest_active_operator_run(
        self,
        *,
        portfolio_slug: str,
        action: str,
        request_payload: Mapping[str, Any] | None = None,
        statuses: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        return self.reads.latest_active_operator_run(
            portfolio_slug=portfolio_slug,
            action=action,
            request_payload=request_payload,
            statuses=statuses,
        )

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

    def update_market_data_sync(
        self,
        sync_run_id: int,
        *,
        status: str | None = None,
        details: Mapping[str, Any] | None = None,
        synced_at: Any | None = None,
    ) -> dict[str, Any] | None:
        return self.writes.update_market_data_sync(
            sync_run_id,
            status=status,
            details=details,
            synced_at=synced_at,
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
