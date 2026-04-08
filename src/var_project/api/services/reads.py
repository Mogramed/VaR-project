from __future__ import annotations

from datetime import datetime, timezone
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

    def _preferred_snapshot_sources(
        self,
        *,
        portfolio_slug: str | None = None,
        source: str | None = None,
    ) -> list[str]:
        normalized_source = str(source or "auto").strip().lower()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if self.runtime.is_live_portfolio(portfolio):
            live_sources = list(self.runtime.strict_live_sources())
            if normalized_source not in {"", "auto"} and normalized_source in live_sources:
                return [normalized_source]
            # Live portfolios must stay broker-backed, never historical fallback.
            return live_sources
        if normalized_source not in {"", "auto"}:
            return [normalized_source]
        return ["historical", "mt5_live_bridge", "mt5_live"]

    @staticmethod
    def _parse_backtest_timeline_column(values: pd.Series, *, column: str) -> pd.Series:
        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().any():
            max_abs = float(numeric_values.abs().max())
            if max_abs < 1e11:
                unit = "s"
            elif max_abs < 1e14:
                unit = "ms"
            elif max_abs < 1e17:
                unit = "us"
            else:
                unit = "ns"
            parsed = pd.to_datetime(numeric_values, unit=unit, utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(values, utc=True, errors="coerce")
        if not parsed.notna().any():
            raise RuntimeError(
                f"Backtest artifact has an invalid '{column}' timeline column. Re-run backtest after refreshing data."
            )
        invalid_epoch = parsed.notna() & (parsed.dt.year < 2000)
        if bool(invalid_epoch.any()):
            raise RuntimeError(
                "Backtest timeline is invalid (epoch-style timestamps detected). Re-run the backtest after refreshing data."
            )
        return parsed

    @staticmethod
    def _normalize_backtest_timeline(frame: pd.DataFrame) -> pd.DataFrame:
        timestamp_columns = [
            column
            for column in ("date", "time", "time_utc", "timestamp")
            if column in frame.columns
        ]
        if not timestamp_columns:
            raise RuntimeError(
                "Backtest artifact is missing a timestamp column. Re-run the backtest with normalized market data."
            )
        for column in timestamp_columns:
            parsed = DeskReadService._parse_backtest_timeline_column(frame[column], column=column)
            frame[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return frame

    def health(self) -> dict[str, Any]:
        latest_artifacts = {
            artifact_type: (
                self.runtime.storage.latest_artifact(artifact_type)["path"]
                if self.runtime.storage_ready and self.runtime.storage.latest_artifact(artifact_type) is not None
                else None
            )
            for artifact_type in ("backtest_compare", "validation_summary", "daily_report", "live_snapshot")
        }
        mt5_configured = bool(
            self.runtime._has_custom_mt5_factory
            or self.runtime.mt5_config.agent_base_url
            or self.runtime.mt5_config.path
            or self.runtime.mt5_config.login
            or self.runtime.mt5_config.server
        )
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
                "database": {
                    "mode": "database",
                    "configured": True,
                    "reachable": bool(self.runtime.storage_ready),
                    "schema_ready": bool(self.runtime.storage_ready),
                    "detail": (
                        "Database schema is ready."
                        if self.runtime.storage_ready
                        else "Database schema is not ready. Run `var-project db upgrade`."
                    ),
                    "target": self.runtime.storage.settings.database_url,
                },
                "mt5": {
                    "mode": "configuration",
                    "configured": mt5_configured,
                    "reachable": None,
                    "schema_ready": None,
                    "detail": (
                        "MT5 configuration is present."
                        if mt5_configured
                        else "MT5 configuration is not set in this API process."
                    ),
                    "target": self.runtime.mt5_config.agent_base_url or self.runtime.mt5_config.path,
                },
            },
        }

    def health_dependencies(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "portfolio_slug": self.runtime.portfolio["slug"],
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
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        normalized_source = str(source or "").strip().lower()
        candidate_sources = self._preferred_snapshot_sources(
            portfolio_slug=portfolio["slug"],
            source=source,
        )
        if self.runtime.is_live_portfolio(portfolio) and normalized_source in {"mt5_live", "mt5_live_bridge"}:
            candidate_sources = [
                normalized_source,
                *[item for item in self.runtime.strict_live_sources() if item != normalized_source],
            ]
        for candidate_source in candidate_sources:
            snapshot = self.runtime.storage.latest_snapshot(
                source=candidate_source,
                portfolio_slug=portfolio["slug"],
            )
            if snapshot is not None:
                payload = dict(snapshot)
                actual_source = str(payload.get("source") or "").strip().lower()
                if normalized_source == "mt5_live" and actual_source == "mt5_live_bridge":
                    payload["source"] = "mt5_live"
                return payload
        return None

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

    def latest_capital(
        self,
        *,
        portfolio_slug: str | None = None,
        source: str | None = "auto",
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        for candidate_source in self._preferred_snapshot_sources(
            portfolio_slug=portfolio["slug"],
            source=source,
        ):
            capital = (
                self.runtime.storage.latest_capital_snapshot(
                    source=candidate_source,
                    portfolio_slug=portfolio["slug"],
                )
                if self.runtime.storage_ready
                else None
            )
            if capital is not None:
                return capital
        if self.runtime.strict_live_required(portfolio):
            raise self.runtime.strict_live_unavailable_error(portfolio=portfolio)
        bundle = self.runtime._compute_portfolio_state(portfolio_slug=portfolio["slug"])
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
        snapshot = None
        for source in self._preferred_snapshot_sources(portfolio_slug=portfolio_slug, source="auto"):
            snapshot = self.latest_snapshot(source=source, portfolio_slug=portfolio_slug)
            if snapshot is not None:
                break
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
        source: str = "auto",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        for candidate_source in self._preferred_snapshot_sources(
            portfolio_slug=portfolio_slug,
            source=source,
        ):
            snapshot = self.latest_snapshot(source=candidate_source, portfolio_slug=portfolio_slug)
            if snapshot is None:
                continue
            payload = dict(snapshot.get("payload") or snapshot)
            attribution = dict(payload.get("attribution") or {})
            if not attribution:
                continue
            attribution["snapshot_source"] = str(
                snapshot.get("source") or payload.get("source") or candidate_source
            )
            attribution["snapshot_timestamp"] = str(
                snapshot.get("created_at") or payload.get("time_utc") or ""
            )
            return attribution
        return None

    def latest_risk_budget(
        self,
        *,
        source: str = "auto",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        for candidate_source in self._preferred_snapshot_sources(
            portfolio_slug=portfolio_slug,
            source=source,
        ):
            snapshot = self.latest_snapshot(source=candidate_source, portfolio_slug=portfolio_slug)
            if snapshot is None:
                continue
            payload = dict(snapshot.get("payload") or snapshot)
            budget = dict(payload.get("risk_budget") or {})
            if not budget:
                continue
            budget["snapshot_source"] = str(
                snapshot.get("source") or payload.get("source") or candidate_source
            )
            budget["snapshot_timestamp"] = str(
                snapshot.get("created_at") or payload.get("time_utc") or ""
            )
            return budget
        return None

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
        frame = self._normalize_backtest_timeline(frame)
        frame = frame.where(pd.notna(frame), None)
        return {
            "compare_csv": str(compare_csv.resolve()),
            "portfolio_slug": str((artifact.get("details") or {}).get("portfolio_slug") or portfolio_slug or ""),
            "rows": frame.to_dict(orient="records"),
        }

    def latest_report_content(
        self,
        *,
        portfolio_slug: str | None = None,
        report_id: int | None = None,
    ) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None

        artifact = None
        if report_id is not None:
            candidate = self.runtime.storage.artifact_by_id(int(report_id))
            if candidate is not None and str(candidate.get("artifact_type") or "") == "daily_report":
                details = dict(candidate.get("details") or {})
                if portfolio_slug is None or str(details.get("portfolio_slug") or "") == str(portfolio_slug):
                    artifact = candidate
        else:
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
            "report_id": int(artifact["id"]),
            "report_markdown": str(report_path.resolve()),
            "portfolio_slug": str((artifact.get("details") or {}).get("portfolio_slug") or portfolio_slug or ""),
            "content": report_path.read_text(encoding="utf-8"),
            "chart_paths": chart_paths,
        }
