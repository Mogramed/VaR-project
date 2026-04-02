from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from var_project.alerts.engine import alerts_from_capital_snapshot, alerts_from_risk_budget
from var_project.api.services.mt5 import DeskMt5Service
from var_project.api.services.runtime import DeskServiceRuntime
from var_project.engine.risk_engine import RiskEngine
from var_project.reporting.render import render_daily_markdown
from var_project.validation.workflows import persist_validation_summary


class DeskAnalyticsService:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

    def _resolve_report_snapshot(self, *, portfolio_slug: str) -> tuple[dict[str, Any] | None, str]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        preferred_sources: list[str] = []
        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            try:
                live_state = DeskMt5Service(self.runtime).live_state(portfolio_slug=portfolio["slug"])
            except Exception:
                live_state = None
            if live_state is not None and (live_state.get("risk_summary") or {}).get("source") == "mt5_live_bridge":
                preferred_sources.append("mt5_live_bridge")
        preferred_sources.extend(["historical", "live"])

        seen: set[str] = set()
        for source in preferred_sources:
            if source in seen:
                continue
            seen.add(source)
            snapshot = self.runtime.storage.latest_snapshot(source=source, portfolio_slug=portfolio["slug"])
            if snapshot is not None:
                return snapshot, source
        return None, "historical"

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
            "var": snapshot.vars_dict(),
            "es": snapshot.es_dict(),
            "risk_surface": dict(bundle["risk_surface"]),
            "headline_risk": list(bundle["headline_risk"]),
            "stress_surface": dict(bundle["stress_surface"]),
            "data_quality": dict(bundle["data_quality"]),
            "model_diagnostics": dict(bundle["risk_surface"].get("model_diagnostics") or {}),
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
        gross_exposure = float(sum(abs(value) for value in exposure_by_symbol.values()))
        symbols = list(bundle["portfolio_symbols"])

        backtest = engine.backtest(
            returns_wide=daily_rets,
            window=selected_window,
            config=config,
            alphas=[float(item) for item in self.runtime.risk_defaults["alphas"]],
            horizons=[int(item) for item in self.runtime.risk_defaults["horizons"]],
            metadata={
                "portfolio": portfolio["name"],
                "base_currency": portfolio["base_currency"],
                "symbols": ",".join(symbols),
                "exposure_by_symbol_json": self.runtime._positions_json(exposure_by_symbol),
                "holdings_json": json.dumps(list(bundle["holdings"])),
                "gross_exposure": gross_exposure,
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
            "hist": int(backtest["exc_hist"].sum()) if "exc_hist" in backtest.columns else 0,
            "param": int(backtest["exc_param"].sum()) if "exc_param" in backtest.columns else 0,
            "mc": int(backtest["exc_mc"].sum()) if "exc_mc" in backtest.columns else 0,
            "ewma": int(backtest["exc_ewma"].sum()) if "exc_ewma" in backtest.columns else 0,
            "garch": int(backtest["exc_garch"].sum()) if "exc_garch" in backtest.columns else 0,
            "fhs": int(backtest["exc_fhs"].sum()) if "exc_fhs" in backtest.columns else 0,
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
                "gross_exposure": gross_exposure,
                "symbols": symbols,
                "exception_counts": exception_counts,
            },
        )

        validation_result = persist_validation_summary(
            storage=self.runtime.storage,
            compare_csv=out_path,
            alpha=config.alpha,
            alphas=[float(item) for item in self.runtime.risk_defaults["alphas"]],
            horizons=[int(item) for item in self.runtime.risk_defaults["horizons"]],
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
            alphas=[float(item) for item in self.runtime.risk_defaults["alphas"]],
            horizons=[int(item) for item in self.runtime.risk_defaults["horizons"]],
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
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        compare_csv = self.runtime._resolve_compare_path(compare_path, portfolio_slug=portfolio["slug"])
        if compare_csv is None or not compare_csv.exists():
            raise FileNotFoundError("No compare CSV available to generate report.")

        out_dir = self.runtime.storage.settings.reports_dir
        report_snapshot, report_snapshot_source = self._resolve_report_snapshot(portfolio_slug=portfolio["slug"])
        snapshot_id = None if report_snapshot is None else report_snapshot.get("id")
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        md_path = render_daily_markdown(
            compare_csv=compare_csv,
            out_dir=out_dir,
            snapshot=report_snapshot,
            risk_limits_yaml=self.runtime.root / "config" / "risk_limits.yaml",
            report_label=portfolio["name"],
        )
        self.runtime._append_governance_sections(
            md_path,
            portfolio_slug=portfolio["slug"],
            capital_source=report_snapshot_source,
        )
        compare_artifact = self.runtime.storage.register_artifact(compare_csv, artifact_type="backtest_compare")
        self.runtime.storage.register_artifact(
            md_path,
            artifact_type="daily_report",
            details={
                "portfolio_slug": portfolio["slug"],
                "compare_csv": str(compare_csv.resolve()),
                "source_artifact_id": compare_artifact,
                "snapshot_source": report_snapshot_source,
                "snapshot_id": snapshot_id,
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
                    details={
                        "report_markdown": str(md_path.resolve()),
                        "portfolio_slug": portfolio["slug"],
                        "snapshot_source": report_snapshot_source,
                    },
                )

        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="report.run",
            object_type="daily_report",
            payload={
                "portfolio_slug": portfolio["slug"],
                "report_markdown": str(md_path.resolve()),
                "compare_csv": str(compare_csv.resolve()),
                "snapshot_source": report_snapshot_source,
                "snapshot_id": snapshot_id,
            },
            portfolio_id=portfolio_id,
        )
        return {
            "report_markdown": str(md_path.resolve()),
            "chart_paths": chart_paths,
        }

    def run_stress_test(
        self,
        *,
        portfolio_slug: str | None = None,
        scenarios: list[dict[str, Any]] | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        from var_project.risk.stress import StressScenario

        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        config = self.runtime._build_risk_model_config(alpha, None, None, None, None)
        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            self.runtime.market_data.sync_market_data(
                portfolio_slug=portfolio["slug"],
                days=int(self.runtime._default_days()),
                timeframes=[str(self.runtime._default_timeframe())],
            )
            live_state = DeskMt5Service(self.runtime).live_state(portfolio_slug=portfolio["slug"])
            live_holdings = list(live_state.get("holdings") or [])
            if live_holdings:
                bundle = self.runtime._compute_portfolio_state_for_holdings(
                    portfolio=portfolio,
                    holdings=live_holdings,
                    timeframe=self.runtime._default_timeframe(),
                    days=self.runtime._default_days(),
                    min_coverage=float(self.runtime.data_defaults["min_coverage"]),
                    config=config,
                    window=int(self.runtime.risk_defaults["window"]),
                    snapshot_source="mt5_live_bridge",
                    snapshot_timestamp=str(live_state.get("generated_at") or datetime.now(timezone.utc).isoformat()),
                )
            else:
                bundle = self.runtime._compute_portfolio_state(
                    portfolio_slug=portfolio["slug"],
                    config=config,
                )
        else:
            bundle = self.runtime._compute_portfolio_state(
                portfolio_slug=portfolio["slug"],
                config=config,
            )

        if not scenarios:
            scenarios = []

        stress_scenarios = [
            StressScenario(
                name=s["name"],
                vol_multiplier=float(s.get("vol_multiplier", 1.0)),
                shock_pnl=float(s.get("shock_pnl", 0.0)),
            )
            for s in scenarios
        ]
        stress_surface = bundle["stress_surface"]
        if stress_scenarios:
            stress_surface = RiskEngine(bundle["holdings"], base_currency=str(portfolio["base_currency"])).build_stress_surface(
                bundle["sample"][bundle["portfolio_symbols"]],
                config,
                alphas=[float(item) for item in self.runtime.risk_defaults["alphas"]],
                horizons=[int(item) for item in self.runtime.risk_defaults["horizons"]],
                estimation_window_days=int(self.runtime.risk_defaults["estimation_window_days"]),
                minimum_valid_days=int(self.runtime.risk_defaults["minimum_valid_days"]),
                scenarios=stress_scenarios,
            )

        target_alpha = float(config.alpha)
        surface_payload = dict(stress_surface.get("risk_surface") or {})
        reference_model = str(surface_payload.get("reference_model") or "hist").lower()
        baseline_headline = next(
            (
                item
                for item in list(surface_payload.get("points") or [])
                if not item.get("is_stressed")
                and str(item.get("model") or "").lower() == reference_model
                and int(item.get("horizon_days") or 0) == 1
                and abs(float(item.get("alpha") or 0.0) - target_alpha) <= 1e-9
            ),
            None,
        )
        if baseline_headline is None:
            baseline_headline = next(
                (
                    item
                    for item in list(stress_surface.get("headline_risk") or [])
                    if not item.get("is_stressed")
                    and int(item.get("horizon_days") or 0) == 1
                    and abs(float(item.get("alpha") or 0.0) - target_alpha) <= 1e-9
                ),
                None,
            )
        if baseline_headline is None:
            baseline_headline = next(
                (
                    item
                    for item in list(stress_surface.get("headline_risk") or [])
                    if not item.get("is_stressed") and item.get("key") == "live_1d_95"
                ),
                None,
            )
        scenario_rows = []
        for scenario in list(stress_surface.get("scenarios") or []):
            primary = dict(scenario.get("primary_metric") or {})
            scenario_rows.append(
                {
                    "name": scenario.get("name"),
                    "vol_multiplier": scenario.get("vol_multiplier"),
                    "shock_pnl": scenario.get("shock_pnl"),
                    "var": primary.get("var"),
                    "es": primary.get("es"),
                    "headline_risk": list(scenario.get("headline_risk") or []),
                    "risk_surface": dict(scenario.get("risk_surface") or {}),
                    "attribution": dict(scenario.get("attribution") or {}),
                    "primary_metric": primary,
                }
            )

        return {
            "portfolio_slug": portfolio["slug"],
            "alpha": config.alpha,
            "baseline_var": None if baseline_headline is None else baseline_headline.get("var"),
            "baseline_es": None if baseline_headline is None else baseline_headline.get("es"),
            "risk_surface": dict(stress_surface.get("risk_surface") or {}),
            "headline_risk": list(stress_surface.get("headline_risk") or []),
            "attribution": dict(stress_surface.get("attribution") or {}),
            "scenarios": scenario_rows,
            "historical_extremes": list(stress_surface.get("historical_extremes") or []),
        }
