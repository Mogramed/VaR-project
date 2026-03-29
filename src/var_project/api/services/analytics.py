from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from var_project.alerts.engine import alerts_from_capital_snapshot, alerts_from_risk_budget
from var_project.api.services.runtime import DeskServiceRuntime
from var_project.engine.risk_engine import RiskEngine
from var_project.reporting.render import render_daily_markdown
from var_project.validation.workflows import persist_validation_summary


class DeskAnalyticsService:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

    def run_snapshot(
        self,
        *,
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        alpha: float | None = None,
        window: int | None = None,
        n_sims: int | None = None,
        dist: str | None = None,
        df_t: int | None = None,
        seed: int | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            self.runtime.market_data.sync_market_data(
                portfolio_slug=portfolio["slug"],
                days=int(days or self.runtime._default_days()),
                timeframes=[str(timeframe or self.runtime._default_timeframe())],
            )
        selected_timeframe = timeframe or self.runtime._default_timeframe()
        selected_days = int(days or self.runtime._default_days())
        selected_min_coverage = float(min_coverage or self.runtime.data_defaults["min_coverage"])
        selected_window = int(window or self.runtime.risk_defaults["window"])
        config = self.runtime._build_risk_model_config(alpha, n_sims, dist, df_t, seed)
        bundle = self.runtime._compute_portfolio_state(
            portfolio_slug=portfolio["slug"],
            timeframe=selected_timeframe,
            days=selected_days,
            min_coverage=selected_min_coverage,
            config=config,
            window=selected_window,
        )
        sample = bundle["sample"]
        snapshot = bundle["snapshot"]
        attribution = bundle["attribution"]
        risk_budget = bundle["risk_budget"]
        capital_snapshot = bundle["capital"]

        now = datetime.now(timezone.utc)
        payload = {
            "time_utc": now.isoformat(),
            "source": "historical",
            "alpha": config.alpha,
            "timeframe": selected_timeframe,
            "days": selected_days,
            "window": selected_window,
            "holdings": list(bundle["holdings"]),
            "exposure_by_symbol": dict(bundle["exposure_by_symbol"]),
            "positions_eur": dict(bundle["exposure_by_symbol"]),
            "var": snapshot.vars_dict(),
            "es": snapshot.es_dict(),
            "attribution": attribution.to_dict(),
            "risk_budget": risk_budget.to_dict(),
            "capital_usage": capital_snapshot,
            "sample_size": snapshot.sample_size,
            "latest_observation": sample.index[-1].isoformat(),
        }
        out_path = self.runtime.storage.settings.snapshots_dir / f"historical_snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"
        artifact_id = self.runtime.storage.write_json_artifact(
            payload,
            out_path,
            artifact_type="historical_snapshot",
            details={
                "portfolio": portfolio["name"],
                "portfolio_slug": portfolio["slug"],
                "timeframe": selected_timeframe,
                "days": selected_days,
                "alpha": config.alpha,
                "window": selected_window,
            },
        )
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        snapshot_id = self.runtime.storage.record_snapshot(
            payload,
            portfolio_id=portfolio_id,
            artifact_id=artifact_id,
            source="historical",
        )
        capital_snapshot["artifact_id"] = artifact_id
        capital_snapshot["snapshot_id"] = snapshot_id
        capital_snapshot["snapshot_timestamp"] = now.isoformat()
        self.runtime.storage.record_capital_snapshot(capital_snapshot, portfolio_id=portfolio_id, source="historical")
        budget_alerts = alerts_from_risk_budget(risk_budget.to_dict())
        capital_alerts = alerts_from_capital_snapshot(capital_snapshot)
        if budget_alerts:
            self.runtime.storage.record_alerts(budget_alerts, portfolio_id=portfolio_id, snapshot_id=snapshot_id)
        if capital_alerts:
            self.runtime.storage.record_alerts(capital_alerts, portfolio_id=portfolio_id, snapshot_id=snapshot_id)
        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="snapshot.run",
            object_type="risk_snapshot",
            object_id=snapshot_id,
            payload={"artifact_id": artifact_id, "portfolio_slug": portfolio["slug"], "timeframe": selected_timeframe, "days": selected_days},
            portfolio_id=portfolio_id,
        )
        return {
            "snapshot_id": snapshot_id,
            "artifact_id": artifact_id,
            "artifact_path": str(out_path.resolve()),
            "portfolio_slug": portfolio["slug"],
            "source": "historical",
            "snapshot": payload,
        }

    def run_backtest(
        self,
        *,
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        alpha: float | None = None,
        window: int | None = None,
        n_sims: int | None = None,
        dist: str | None = None,
        df_t: int | None = None,
        seed: int | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            self.runtime.market_data.sync_market_data(
                portfolio_slug=portfolio["slug"],
                days=int(days or self.runtime._default_days()),
                timeframes=[str(timeframe or self.runtime._default_timeframe())],
            )
        selected_timeframe = timeframe or self.runtime._default_timeframe()
        selected_days = int(days or self.runtime._default_days())
        selected_min_coverage = float(min_coverage or self.runtime.data_defaults["min_coverage"])
        selected_window = int(window or self.runtime.risk_defaults["window"])
        config = self.runtime._build_risk_model_config(alpha, n_sims, dist, df_t, seed)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])

        bundle = self.runtime._compute_portfolio_state(
            portfolio_slug=portfolio["slug"],
            timeframe=selected_timeframe,
            days=selected_days,
            min_coverage=selected_min_coverage,
            config=config,
            window=selected_window,
        )
        engine = RiskEngine(bundle["holdings"], base_currency=str(portfolio["base_currency"]))
        daily_rets = bundle["daily_returns"][bundle["portfolio_symbols"]]
        exposure_by_symbol = dict(bundle["exposure_by_symbol"])
        gross_notional = float(sum(abs(value) for value in exposure_by_symbol.values()))
        symbols = list(bundle["portfolio_symbols"])

        backtest = engine.backtest(
            returns_wide=daily_rets,
            window=selected_window,
            config=config,
            metadata={
                "portfolio": portfolio["name"],
                "base_currency": portfolio["base_currency"],
                "symbols": ",".join(symbols),
                "exposure_by_symbol_json": self.runtime._positions_json(exposure_by_symbol),
                "positions_eur_json": self.runtime._positions_json(exposure_by_symbol),
                "holdings_json": json.dumps(list(bundle["holdings"])),
                "gross_notional": gross_notional,
                "timeframe": selected_timeframe,
                "days": selected_days,
            },
        )

        out_path = self.runtime.storage.settings.analytics_dir / (
            f"compare_{selected_timeframe}_{selected_days}d_alpha{int(config.alpha * 100)}"
            f"_pf{portfolio['slug']}_mc{config.mc.dist}_garch_{config.garch.dist}_ewma_fhs.csv"
        )
        compare_artifact_id = self.runtime.storage.write_dataframe_artifact(
            backtest,
            out_path,
            artifact_type="backtest_compare",
            index=False,
            details={
                "portfolio": portfolio["name"],
                "portfolio_slug": portfolio["slug"],
                "base_currency": portfolio["base_currency"],
                "symbols": symbols,
                "timeframe": selected_timeframe,
                "days": selected_days,
                "alpha": config.alpha,
                "window": selected_window,
            },
        )
        exception_counts = {
            "hist": int(backtest["exc_hist"].sum()),
            "param": int(backtest["exc_param"].sum()),
            "mc": int(backtest["exc_mc"].sum()),
            "ewma": int(backtest["exc_ewma"].sum()),
            "garch": int(backtest["exc_garch"].sum()) if "exc_garch" in backtest.columns else 0,
            "fhs": int(backtest["exc_fhs"].sum()),
        }
        backtest_run_id = self.runtime.storage.record_backtest_run(
            portfolio_id=portfolio_id,
            artifact_id=compare_artifact_id,
            timeframe=selected_timeframe,
            days=selected_days,
            alpha=config.alpha,
            window=selected_window,
            n_rows=len(backtest),
            summary={
                "portfolio": portfolio["name"],
                "gross_notional": gross_notional,
                "symbols": symbols,
                "exception_counts": exception_counts,
            },
        )

        validation_result = persist_validation_summary(
            storage=self.runtime.storage,
            compare_csv=out_path,
            alpha=config.alpha,
            source_artifact_id=compare_artifact_id,
            portfolio_id=portfolio_id,
            portfolio_name=portfolio["name"],
            portfolio_slug=portfolio["slug"],
            base_currency=portfolio["base_currency"],
            symbols=symbols,
            positions=exposure_by_symbol,
        )
        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="backtest.run",
            object_type="backtest_run",
            object_id=backtest_run_id,
            payload={
                "portfolio_slug": portfolio["slug"],
                "validation_run_id": validation_result["validation_run_id"],
                "best_model": validation_result["best_model"],
            },
            portfolio_id=portfolio_id,
        )

        return {
            "backtest_run_id": backtest_run_id,
            "validation_run_id": validation_result["validation_run_id"],
            "compare_artifact_id": compare_artifact_id,
            "validation_artifact_id": validation_result["validation_artifact_id"],
            "compare_csv": str(out_path.resolve()),
            "validation_json": validation_result["validation_json"],
            "best_model": validation_result["best_model"],
            "alert_count": validation_result["alert_count"],
            "exception_counts": exception_counts,
        }

    def run_validation(
        self,
        *,
        compare_path: str | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        compare_csv = self.runtime._resolve_compare_path(compare_path)
        if compare_csv is None or not compare_csv.exists():
            raise FileNotFoundError("No compare_*.csv found for validation.")

        selected_alpha = float(alpha or self.runtime.risk_defaults["alpha"])
        result = persist_validation_summary(
            storage=self.runtime.storage,
            compare_csv=compare_csv,
            alpha=selected_alpha,
        )
        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="validation.run",
            object_type="validation_run",
            object_id=result["validation_run_id"],
            payload={
                "portfolio_slug": result["portfolio_slug"],
                "compare_csv": str(compare_csv.resolve()),
                "best_model": result["best_model"],
            },
            portfolio_id=result["portfolio_id"],
        )
        return result

    def run_report(self, *, compare_path: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        compare_csv = self.runtime._resolve_compare_path(compare_path, portfolio_slug=portfolio_slug)
        if compare_csv is None or not compare_csv.exists():
            raise FileNotFoundError("No compare CSV available to generate report.")

        out_dir = self.runtime.storage.settings.reports_dir
        md_path = render_daily_markdown(
            compare_csv=compare_csv,
            out_dir=out_dir,
            snapshot_dir=self.runtime.storage.settings.snapshots_dir,
            risk_limits_yaml=self.runtime.root / "config" / "risk_limits.yaml",
        )
        self.runtime._append_governance_sections(md_path)
        compare_artifact = self.runtime.storage.register_artifact(compare_csv, artifact_type="backtest_compare")
        self.runtime.storage.register_artifact(
            md_path,
            artifact_type="daily_report",
            details={
                "portfolio_slug": portfolio_slug or self.runtime.portfolio["slug"],
                "compare_csv": str(compare_csv.resolve()),
                "source_artifact_id": compare_artifact,
            },
        )

        chart_paths: list[str] = []
        for chart_path in (
            out_dir / f"{compare_csv.stem}_exceptions.png",
            out_dir / f"{compare_csv.stem}_pnl_var.png",
        ):
            if chart_path.exists():
                chart_paths.append(str(chart_path.resolve()))
                self.runtime.storage.register_artifact(
                    chart_path,
                    artifact_type="report_chart",
                    details={"report_markdown": str(md_path.resolve())},
                )

        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="report.run",
            object_type="daily_report",
            payload={"report_markdown": str(md_path.resolve()), "compare_csv": str(compare_csv.resolve())},
            portfolio_id=None if portfolio_slug is None else self.runtime._resolve_portfolio_id(portfolio_slug),
        )
        return {
            "report_markdown": str(md_path.resolve()),
            "chart_paths": chart_paths,
        }
