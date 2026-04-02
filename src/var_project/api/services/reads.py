from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from var_project.api.services.runtime import DeskServiceRuntime
from var_project.desk.overview import build_desk_snapshot
from var_project.jobs import build_worker_status
from var_project.validation.model_validation import ValidationSummary, build_champion_challenger_summary


class DeskReadService:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

    def health(self) -> dict[str, Any]:
        latest_artifacts = {
            artifact_type: (
                self.runtime.storage.latest_artifact(artifact_type)["path"]
                if self.runtime.storage_ready and self.runtime.storage.latest_artifact(artifact_type) is not None
                else None
            )
            for artifact_type in ("backtest_compare", "validation_summary", "daily_report", "live_snapshot")
        }
        return {
            "status": "ok",
            "repo_root": str(self.runtime.root.resolve()),
            "database_url": self.runtime.storage.settings.database_url,
            "portfolio_slug": self.runtime.portfolio["slug"],
            "portfolio_count": len(self.runtime.portfolios),
            "desk_slug": self.runtime.desk.slug,
            "portfolio_mode": self.runtime.portfolio.get("mode"),
            "latest_artifacts": latest_artifacts,
            "defaults": {
                "timeframes": self.runtime.data_defaults["timeframes"],
                "history_days_list": self.runtime.data_defaults["history_days_list"],
                "market_history_days": self.runtime.data_defaults["market_history_days"],
                "min_coverage": self.runtime.data_defaults["min_coverage"],
                "alpha": self.runtime.risk_defaults["alpha"],
                "alphas": self.runtime.risk_defaults["alphas"],
                "horizons": self.runtime.risk_defaults["horizons"],
                "window": self.runtime.risk_defaults["window"],
                "estimation_window_days": self.runtime.risk_defaults["estimation_window_days"],
                "minimum_valid_days": self.runtime.risk_defaults["minimum_valid_days"],
                "garch": self.runtime.risk_defaults["garch"],
            },
            "dependencies": {
                "database": self.runtime.database_dependency(),
                "mt5": self.runtime.mt5_dependency(),
                "mt5_live": self.runtime.mt5_live_dependency(self.runtime.portfolio["slug"]),
                "market_data": self.runtime.market_data.market_data_status(portfolio_slug=self.runtime.portfolio["slug"]),
            },
        }

    def list_portfolios(self) -> list[dict[str, Any]]:
        if self.runtime.storage_ready:
            return self.runtime.storage.list_portfolios()
        return [
            {
                "id": index + 1,
                "slug": portfolio["slug"],
                "name": portfolio["name"],
                "base_currency": portfolio["base_currency"],
                "symbols": list(portfolio["symbols"]),
                "positions": dict(portfolio["positions"]),
                "created_at": None,
                "updated_at": None,
            }
            for index, portfolio in enumerate(self.runtime.portfolios)
        ]

    def list_desks(self) -> list[dict[str, Any]]:
        return [self.runtime.desk.to_dict()]

    def latest_snapshot(self, *, source: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None
        return self.runtime.storage.latest_snapshot(source=source, portfolio_slug=portfolio_slug)

    def latest_backtest(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None
        return self.runtime.storage.latest_backtest_run(portfolio_slug=portfolio_slug)

    def latest_validation(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None
        return self.runtime.storage.latest_validation_run(portfolio_slug=portfolio_slug)

    def recent_alerts(self, *, limit: int = 25) -> list[dict[str, Any]]:
        if not self.runtime.storage_ready:
            return []
        return self.runtime.storage.recent_alerts(limit=limit)

    def recent_decisions(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        if not self.runtime.storage_ready:
            return []
        return self.runtime.storage.recent_decisions(limit=limit, portfolio_slug=portfolio_slug)

    def latest_capital(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        resolved_slug = self.runtime._resolve_portfolio_context(portfolio_slug)["slug"]
        capital = (
            self.runtime.storage.latest_capital_snapshot(source="historical", portfolio_slug=resolved_slug)
            if self.runtime.storage_ready
            else None
        )
        if capital is not None:
            return capital
        bundle = self.runtime._compute_portfolio_state(portfolio_slug=resolved_slug)
        return bundle["capital"]

    def capital_history(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.runtime.storage_ready:
            return []
        return self.runtime.storage.capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def recent_execution_results(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        if not self.runtime.storage_ready:
            return []
        return self.runtime.storage.recent_execution_results(limit=limit, portfolio_slug=portfolio_slug)

    def recent_execution_fills(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        if not self.runtime.storage_ready:
            return []
        return self.runtime.storage.recent_execution_fills(limit=limit, portfolio_slug=portfolio_slug)

    def recent_audit_events(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        if not self.runtime.storage_ready:
            return []
        return self.runtime.storage.recent_audit_events(limit=limit, portfolio_slug=portfolio_slug)

    def report_decision_history(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.recent_decisions(limit=limit, portfolio_slug=portfolio_slug)

    def report_capital_history(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def portfolio_capital(self, portfolio_slug: str) -> dict[str, Any]:
        return self.latest_capital(portfolio_slug=portfolio_slug)

    def desk_overview(self, *, desk_slug: str | None = None) -> dict[str, Any]:
        if desk_slug is not None and desk_slug != self.runtime.desk.slug:
            raise ValueError(f"Unknown desk '{desk_slug}'.")
        snapshots = [self.latest_capital(portfolio_slug=portfolio["slug"]) for portfolio in self.runtime.portfolios]
        alert_counts = self.runtime._alert_counts_by_portfolio()
        portfolio_map = {portfolio["slug"]: portfolio for portfolio in self.runtime.portfolios}
        return build_desk_snapshot(self.runtime.desk.to_dict(), snapshots, portfolio_map, alerts_by_portfolio=alert_counts).to_dict()

    def latest_artifact(self, artifact_type: str) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None
        return self.runtime.storage.latest_artifact(artifact_type)

    def jobs_status(self) -> dict[str, Any]:
        return build_worker_status(self.runtime.root, storage=self.runtime.storage)

    def latest_model_comparison(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        validation = self.latest_validation(portfolio_slug=portfolio_slug)
        if validation is None:
            return None

        summary = ValidationSummary.from_dict(dict(validation.get("summary") or validation))
        snapshot = self.latest_snapshot(source="historical", portfolio_slug=portfolio_slug)
        if snapshot is None:
            snapshot = self.latest_snapshot(source="live", portfolio_slug=portfolio_slug)
        snapshot_payload = {} if snapshot is None else dict(snapshot.get("payload") or snapshot)

        comparison = build_champion_challenger_summary(
            summary,
            current_var=dict(snapshot_payload.get("var") or {}),
            current_es=dict(snapshot_payload.get("es") or {}),
        ).to_dict()
        comparison["snapshot_source"] = None if snapshot is None else str(snapshot.get("source") or snapshot_payload.get("source") or "")
        comparison["snapshot_timestamp"] = None if snapshot is None else str(snapshot.get("created_at") or snapshot_payload.get("time_utc") or "")
        return comparison

    def latest_risk_attribution(
        self,
        *,
        source: str = "historical",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        snapshot = self.latest_snapshot(source=source, portfolio_slug=portfolio_slug)
        if snapshot is None:
            return None

        payload = dict(snapshot.get("payload") or snapshot)
        attribution = dict(payload.get("attribution") or {})
        if not attribution:
            return None

        attribution["snapshot_source"] = str(snapshot.get("source") or payload.get("source") or source)
        attribution["snapshot_timestamp"] = str(snapshot.get("created_at") or payload.get("time_utc") or "")
        return attribution

    def latest_risk_budget(
        self,
        *,
        source: str = "historical",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        snapshot = self.latest_snapshot(source=source, portfolio_slug=portfolio_slug)
        if snapshot is None:
            return None

        payload = dict(snapshot.get("payload") or snapshot)
        budget = dict(payload.get("risk_budget") or {})
        if not budget:
            return None

        budget["snapshot_source"] = str(snapshot.get("source") or payload.get("source") or source)
        budget["snapshot_timestamp"] = str(snapshot.get("created_at") or payload.get("time_utc") or "")
        return budget

    def latest_backtest_frame(self, *, limit: int = 400, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None

        artifact = None
        backtest = self.runtime.storage.latest_backtest_run(portfolio_slug=portfolio_slug)
        artifact_id = None if backtest is None else backtest.get("artifact_id")
        if artifact_id:
            artifact = self.runtime.storage.artifact_by_id(int(artifact_id))
        if artifact is None:
            artifact = self.runtime.storage.latest_artifact("backtest_compare", portfolio_slug=portfolio_slug)
        if artifact is None:
            return None

        compare_csv = Path(artifact["path"])
        if not compare_csv.exists():
            return None

        frame = pd.read_csv(compare_csv)
        if limit > 0 and len(frame) > limit:
            frame = frame.tail(limit).reset_index(drop=True)
        frame = frame.where(pd.notna(frame), None)
        return {
            "compare_csv": str(compare_csv.resolve()),
            "portfolio_slug": str((artifact.get("details") or {}).get("portfolio_slug") or portfolio_slug or ""),
            "rows": frame.to_dict(orient="records"),
        }

    def latest_report_content(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None

        artifact = self.runtime.storage.latest_artifact("daily_report", portfolio_slug=portfolio_slug)
        if artifact is None:
            return None

        report_path = Path(artifact["path"])
        if not report_path.exists():
            return None

        chart_paths: list[str] = []
        for chart_path in (
            report_path.with_name(f"{report_path.stem}_exceptions.png"),
            report_path.with_name(f"{report_path.stem}_pnl_var.png"),
        ):
            if chart_path.exists():
                chart_paths.append(str(chart_path.resolve()))

        return {
            "report_markdown": str(report_path.resolve()),
            "portfolio_slug": str((artifact.get("details") or {}).get("portfolio_slug") or portfolio_slug or ""),
            "content": report_path.read_text(encoding="utf-8"),
            "chart_paths": chart_paths,
        }
