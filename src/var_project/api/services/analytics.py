from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from var_project.alerts.engine import alerts_from_capital_snapshot, alerts_from_risk_budget
from var_project.api.services.mt5 import DeskMt5Service
from var_project.api.services.runtime import DeskServiceRuntime
from var_project.core.config_validation import validate_backtest_history_compatibility
from var_project.engine.risk_engine import RiskEngine
from var_project.reporting.render import render_daily_markdown
from var_project.validation.model_validation import recommended_backtest_history_days
from var_project.validation.workflows import persist_validation_summary


class DeskAnalyticsService:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

    def _default_backtest_window(self) -> int:
        raw_risk_cfg = dict(self.runtime.raw_config.get("risk") or {})
        configured_validation_window = raw_risk_cfg.get("validation_window_days")
        configured_live_window = self.runtime.risk_defaults.get("window")
        try:
            validation_window = (
                0
                if configured_validation_window in {None, "", "null"}
                else int(configured_validation_window)
            )
        except (TypeError, ValueError):
            validation_window = 0
        try:
            live_window = int(configured_live_window or 0)
        except (TypeError, ValueError):
            live_window = 0
        if validation_window > 0:
            return max(validation_window, live_window, 250)
        return max(live_window, 1)

    def _fallback_backtest_window_for_limited_history(
        self,
        *,
        selected_window: int,
        selected_days: int,
        explicit_window: int | None,
    ) -> int | None:
        if explicit_window is not None:
            return None
        raw_risk_cfg = dict(self.runtime.raw_config.get("risk") or {})
        configured_validation_window = raw_risk_cfg.get("validation_window_days")
        try:
            validation_window = (
                0
                if configured_validation_window in {None, "", "null"}
                else int(configured_validation_window)
            )
        except (TypeError, ValueError):
            validation_window = 0
        if validation_window <= 0 or validation_window >= 250:
            return None
        if int(selected_window) < 250:
            return None

        try:
            live_window = int(self.runtime.risk_defaults.get("window") or 0)
        except (TypeError, ValueError):
            live_window = 0
        horizons = [int(item) for item in list(self.runtime.risk_defaults.get("horizons") or []) if int(item) > 0]
        max_horizon = max(horizons, default=1)
        max_compatible_window = max(int(selected_days) - int(max_horizon), 1)
        fallback_window = min(max(live_window, validation_window, 1), int(max_compatible_window))
        if fallback_window >= int(selected_window):
            return None
        return int(fallback_window)

    @staticmethod
    def _extract_selected_model_from_report_markdown(report_path: Path) -> str | None:
        try:
            content = report_path.read_text(encoding="utf-8")
        except OSError:
            return None
        patterns = (
            r"^- Champion model:\s+\*\*(?P<model>[a-z0-9_]+)\*\*",
            r"^- Best model by score:\s+\*\*(?P<model>[a-z0-9_]+)\*\*",
        )
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
            if match is None:
                continue
            model = str(match.group("model")).strip().lower()
            if model:
                return model
        return None

    def _default_no_exposure_epsilon(self) -> float:
        raw_default = self.runtime.risk_defaults.get("no_exposure_epsilon_eur")
        try:
            epsilon = 1.0 if raw_default in {None, "", "null"} else float(raw_default)
        except (TypeError, ValueError):
            epsilon = 1.0
        if epsilon < 0.0:
            epsilon = 1.0
        return float(epsilon)

    def _no_exposure_epsilon_by_symbol(self, symbols: list[str]) -> dict[str, float]:
        configured = {
            str(symbol).upper(): value
            for symbol, value in dict(self.runtime.risk_defaults.get("no_exposure_epsilon_by_symbol") or {}).items()
            if symbol not in {None, ""}
        }
        default_epsilon = self._default_no_exposure_epsilon()
        epsilon_by_symbol: dict[str, float] = {}
        for symbol in symbols:
            normalized = str(symbol).upper()
            raw = configured.get(normalized)
            try:
                epsilon = default_epsilon if raw is None else float(raw)
            except (TypeError, ValueError):
                epsilon = default_epsilon
            if epsilon < 0.0:
                epsilon = default_epsilon
            epsilon_by_symbol[normalized] = float(epsilon)
        return epsilon_by_symbol

    @staticmethod
    def _parse_timeline_column(values: pd.Series, *, column: str) -> pd.Series:
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
                f"Backtest timeline is invalid: unable to parse column '{column}'. "
                "Re-run sync then backtest to regenerate normalized market data."
            )
        invalid_epoch = parsed.notna() & (parsed.dt.year < 2000)
        if bool(invalid_epoch.any()):
            raise RuntimeError(
                f"Backtest timeline is invalid: epoch-style timestamps detected in '{column}'. "
                "Refresh market data and rerun backtest."
            )
        return parsed

    def _normalize_backtest_timeline(self, frame: pd.DataFrame) -> pd.DataFrame:
        timeline_columns = [
            column
            for column in ("date", "time", "time_utc", "timestamp")
            if column in frame.columns
        ]
        if not timeline_columns:
            raise RuntimeError(
                "Backtest result is missing timeline columns. Expected one of date/time/time_utc/timestamp."
            )
        normalized = frame.copy()
        for column in timeline_columns:
            parsed = self._parse_timeline_column(normalized[column], column=column)
            normalized[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return normalized

    @staticmethod
    def _attach_alpha_backtest_var(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Add an adaptive `var_alpha` overlay from existing VaR models.

        The blend uses inverse trailing absolute error (vs realized loss) so
        better-performing models get higher weights while remaining robust when
        history is short or partially missing.
        """
        if frame.empty or "pnl" not in frame.columns:
            return frame

        candidate_models: list[str] = []
        for column in frame.columns:
            normalized = str(column).strip().lower()
            if not normalized.startswith("var_"):
                continue
            if "_a" in normalized and "_h" in normalized:
                continue
            model = normalized[len("var_") :]
            if not model or model == "alpha":
                continue
            candidate_models.append(model)

        deduped_models: list[str] = []
        seen_models: set[str] = set()
        for model in candidate_models:
            if model in seen_models:
                continue
            seen_models.add(model)
            deduped_models.append(model)
        if not deduped_models:
            return frame

        var_frame = pd.DataFrame(
            {
                model: pd.to_numeric(frame[f"var_{model}"], errors="coerce")
                for model in deduped_models
                if f"var_{model}" in frame.columns
            },
            index=frame.index,
        )
        if var_frame.empty or var_frame.notna().sum().sum() <= 0:
            return frame

        losses = -pd.to_numeric(frame["pnl"], errors="coerce")
        errors = var_frame.sub(losses, axis=0).abs()

        rolling_error = errors.rolling(window=60, min_periods=12).mean().shift(1)
        expanding_error = errors.expanding(min_periods=2).mean().shift(1)
        historical_error = rolling_error.fillna(expanding_error)

        quality = (1.0 / historical_error.clip(lower=1e-6)).fillna(0.0)
        available = var_frame.notna()
        weights = quality.where(available, 0.0)

        fallback_weights = available.astype(float)
        fallback_weights = fallback_weights.div(fallback_weights.sum(axis=1), axis=0).fillna(0.0)
        empty_rows = weights.sum(axis=1) <= 0.0
        if bool(empty_rows.any()):
            weights.loc[empty_rows, :] = fallback_weights.loc[empty_rows, :]

        weights = weights.div(weights.sum(axis=1).where(lambda series: series > 0.0), axis=0).fillna(0.0)
        alpha_var = (weights * var_frame).sum(axis=1)
        alpha_var = alpha_var.where(available.any(axis=1), float("nan"))
        alpha_var = alpha_var.abs()

        if "var_hist" in frame.columns:
            fallback_var = pd.to_numeric(frame["var_hist"], errors="coerce").abs()
        else:
            fallback_var = var_frame.mean(axis=1).abs()
        alpha_var = alpha_var.fillna(fallback_var)

        enriched = frame.copy()
        enriched["var_alpha"] = pd.to_numeric(alpha_var, errors="coerce")
        return enriched

    def _tracked_history_days(self) -> int:
        days_list = [
            int(value)
            for value in list(self.runtime.data_defaults.get("history_days_list") or [])
            if str(value).strip()
        ]
        configured_history_days = max(days_list, default=0)
        try:
            market_history_days = int(self.runtime.data_defaults.get("market_history_days") or 0)
        except (TypeError, ValueError):
            market_history_days = 0
        return int(max(configured_history_days, market_history_days, 0))

    def _resolve_backtest_days(
        self,
        *,
        requested_days: int | None,
        window: int,
        alphas: list[float],
        horizons: list[int],
        enforce_minimum_depth: bool = False,
    ) -> tuple[int, int, int]:
        normalized_requested_days = int(requested_days or self.runtime._default_days())
        tracked_history_days = self._tracked_history_days()
        recommended_days = recommended_backtest_history_days(
            alphas=alphas,
            horizons=horizons,
            window=window,
        )

        effective_days = normalized_requested_days
        if tracked_history_days > 0:
            effective_days = min(effective_days, tracked_history_days) if requested_days is not None else effective_days
            if requested_days is None:
                effective_days = max(effective_days, min(recommended_days, tracked_history_days))
        if enforce_minimum_depth and requested_days is not None:
            if tracked_history_days > 0:
                minimum_depth_days = min(int(recommended_days), int(tracked_history_days))
            else:
                minimum_depth_days = int(recommended_days)
            effective_days = max(int(effective_days), int(minimum_depth_days))
            if tracked_history_days > 0:
                effective_days = min(int(effective_days), int(tracked_history_days))
        effective_days = max(int(effective_days), 1)
        return normalized_requested_days, effective_days, int(recommended_days)

    def _resolve_report_snapshot(
        self,
        *,
        portfolio_slug: str,
        account_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        strict_live = self.runtime.strict_live_required(portfolio)
        preferred_sources: list[str] = []
        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            try:
                live_state = DeskMt5Service(self.runtime).live_state(portfolio_slug=portfolio["slug"], account_id=account_id)
            except Exception:
                live_state = None
            if live_state is not None and (live_state.get("risk_summary") or {}).get("source") == "mt5_live_bridge":
                preferred_sources.append("mt5_live_bridge")
        preferred_sources.extend(["mt5_live_bridge", "mt5_live"])
        if not strict_live:
            preferred_sources.extend(["historical", "live"])

        seen: set[str] = set()
        for source in preferred_sources:
            if source in seen:
                continue
            seen.add(source)
            snapshot = self.runtime.storage.latest_snapshot(source=source, portfolio_slug=portfolio["slug"])
            if snapshot is not None:
                return snapshot, source
        return None, "mt5_live_bridge" if strict_live else "historical"

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
        account_id: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
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
            allow_auto_sync=False,
        )
        sample = bundle["sample"]
        snapshot = bundle["snapshot"]
        attribution = bundle["attribution"]
        risk_budget = bundle["risk_budget"]
        capital_snapshot = bundle["capital"]
        snapshot_source = str(
            capital_snapshot.get("snapshot_source")
            or capital_snapshot.get("source")
            or "historical"
        )
        artifact_source = re.sub(r"[^a-z0-9_]+", "_", snapshot_source.lower()).strip("_") or "historical"
        artifact_type = "live_snapshot" if snapshot_source.startswith("mt5_live") else "historical_snapshot"

        now = datetime.now(timezone.utc)
        payload = {
            "time_utc": now.isoformat(),
            "source": snapshot_source,
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
        out_path = self.runtime.storage.settings.snapshots_dir / (
            f"{artifact_source}_snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"
        )
        artifact_id = self.runtime.storage.write_json_artifact(
            payload,
            out_path,
            artifact_type=artifact_type,
            details={
                "portfolio": portfolio["name"],
                "portfolio_slug": portfolio["slug"],
                "timeframe": selected_timeframe,
                "days": selected_days,
                "alpha": config.alpha,
                "window": selected_window,
                "source": snapshot_source,
            },
        )
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        snapshot_id = self.runtime.storage.record_snapshot(
            payload,
            portfolio_id=portfolio_id,
            artifact_id=artifact_id,
            source=snapshot_source,
        )
        capital_snapshot["artifact_id"] = artifact_id
        capital_snapshot["snapshot_id"] = snapshot_id
        capital_snapshot["snapshot_timestamp"] = now.isoformat()
        self.runtime.storage.record_capital_snapshot(
            capital_snapshot,
            portfolio_id=portfolio_id,
            source=snapshot_source,
        )
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
            payload={
                "artifact_id": artifact_id,
                "portfolio_slug": portfolio["slug"],
                "timeframe": selected_timeframe,
                "days": selected_days,
                "source": snapshot_source,
            },
            portfolio_id=portfolio_id,
        )
        return {
            "snapshot_id": snapshot_id,
            "artifact_id": artifact_id,
            "artifact_path": str(out_path.resolve()),
            "portfolio_slug": portfolio["slug"],
            "source": snapshot_source,
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
        account_id: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        selected_timeframe = timeframe or self.runtime._default_timeframe()
        selected_min_coverage = float(min_coverage or self.runtime.data_defaults["min_coverage"])
        selected_window = int(window or self._default_backtest_window())
        selected_alphas = [float(item) for item in self.runtime.risk_defaults["alphas"]]
        selected_horizons = [int(item) for item in self.runtime.risk_defaults["horizons"]]
        requested_days, selected_days, recommended_days = self._resolve_backtest_days(
            requested_days=days,
            window=selected_window,
            alphas=selected_alphas,
            horizons=selected_horizons,
            enforce_minimum_depth=self.runtime.is_live_portfolio(portfolio),
        )
        try:
            validate_backtest_history_compatibility(
                self.runtime.data_defaults,
                self.runtime.risk_defaults,
                days=selected_days,
                window=selected_window,
                horizons=selected_horizons,
                context="backtest",
            )
        except ValueError:
            fallback_window = self._fallback_backtest_window_for_limited_history(
                selected_window=selected_window,
                selected_days=selected_days,
                explicit_window=window,
            )
            if fallback_window is None:
                raise
            selected_window = int(fallback_window)
            requested_days, selected_days, recommended_days = self._resolve_backtest_days(
                requested_days=days,
                window=selected_window,
                alphas=selected_alphas,
                horizons=selected_horizons,
                enforce_minimum_depth=self.runtime.is_live_portfolio(portfolio),
            )
            validate_backtest_history_compatibility(
                self.runtime.data_defaults,
                self.runtime.risk_defaults,
                days=selected_days,
                window=selected_window,
                horizons=selected_horizons,
                context="backtest",
            )
        config = self.runtime._build_risk_model_config(alpha, n_sims, dist, df_t, seed)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])

        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            self.runtime.market_data.sync_market_data_if_stale(
                portfolio_slug=portfolio["slug"],
                account_id=account_id,
                max_age_seconds=300.0,
                days=selected_days,
                timeframes=[selected_timeframe],
            )

        bundle = self.runtime._compute_portfolio_state(
            portfolio_slug=portfolio["slug"],
            timeframe=selected_timeframe,
            days=selected_days,
            min_coverage=selected_min_coverage,
            config=config,
            window=selected_window,
            allow_auto_sync=False,
        )
        engine = RiskEngine(
            bundle["holdings"],
            base_currency=str(portfolio["base_currency"]),
            no_exposure_epsilon_by_symbol=self._no_exposure_epsilon_by_symbol(list(bundle["portfolio_symbols"])),
            default_no_exposure_epsilon=self._default_no_exposure_epsilon(),
        )
        return_columns = ["date", *bundle["portfolio_symbols"]]
        daily_rets = bundle["daily_returns"][return_columns].copy()
        exposure_by_symbol = dict(bundle["exposure_by_symbol"])
        gross_exposure = float(sum(abs(value) for value in exposure_by_symbol.values()))
        backtest_source = str(
            dict(bundle.get("capital") or {}).get("snapshot_source")
            or "historical"
        )
        flat_book = gross_exposure <= 1e-9
        symbols = list(bundle["portfolio_symbols"])

        backtest = engine.backtest(
            returns_wide=daily_rets,
            window=selected_window,
            config=config,
            alphas=selected_alphas,
            horizons=selected_horizons,
            metadata={
                "portfolio": portfolio["name"],
                "base_currency": portfolio["base_currency"],
                "symbols": ",".join(symbols),
                "exposure_by_symbol_json": self.runtime._positions_json(exposure_by_symbol),
                "holdings_json": json.dumps(list(bundle["holdings"])),
                "gross_exposure": gross_exposure,
                "timeframe": selected_timeframe,
                "days": selected_days,
                "requested_days": requested_days,
                "recommended_validation_days": recommended_days,
            },
        )
        backtest = self._attach_alpha_backtest_var(backtest)
        backtest = self._normalize_backtest_timeline(backtest)

        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        out_path = self.runtime.storage.settings.analytics_dir / (
            f"compare_{selected_timeframe}_{selected_days}d_alpha{int(config.alpha * 100)}"
            f"_pf{portfolio['slug']}_mc{config.mc.dist}_garch_{config.garch.dist}_ewma_fhs_{run_stamp}.csv"
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
                "requested_days": requested_days,
                "recommended_validation_days": recommended_days,
                "alpha": config.alpha,
                "window": selected_window,
                "source": backtest_source,
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
                "source": backtest_source,
                "flat_book": bool(flat_book),
                "requested_days": requested_days,
                "effective_days": selected_days,
                "recommended_validation_days": recommended_days,
            },
        )

        validation_result = persist_validation_summary(
            storage=self.runtime.storage,
            compare_csv=out_path,
            alpha=config.alpha,
            alphas=selected_alphas,
            horizons=selected_horizons,
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
            "source": backtest_source,
            "flat_book": bool(flat_book),
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

    def run_report(
        self,
        *,
        compare_path: str | None = None,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        compare_csv = self.runtime._resolve_compare_path(compare_path, portfolio_slug=portfolio["slug"])
        if compare_csv is None or not compare_csv.exists():
            raise FileNotFoundError("No compare CSV available to generate report.")

        out_dir = self.runtime.storage.settings.reports_dir
        report_snapshot, report_snapshot_source = self._resolve_report_snapshot(
            portfolio_slug=portfolio["slug"],
            account_id=resolved_account_id,
        )
        if self.runtime.strict_live_required(portfolio) and report_snapshot is None:
            raise self.runtime.strict_live_unavailable_error(
                portfolio=portfolio,
                reason="No MT5 live snapshot is available for report rendering.",
            )
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
        selected_model = self._extract_selected_model_from_report_markdown(md_path)
        compare_artifact = self.runtime.storage.register_artifact(compare_csv, artifact_type="backtest_compare")
        self.runtime.storage.register_artifact(
            md_path,
            artifact_type="daily_report",
            details={
                "portfolio_slug": portfolio["slug"],
                "account_id": resolved_account_id,
                "compare_csv": str(compare_csv.resolve()),
                "source_artifact_id": compare_artifact,
                "snapshot_source": report_snapshot_source,
                "snapshot_id": snapshot_id,
                "selected_model": selected_model,
                "report_contract_version": "report.v1",
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
                        "account_id": resolved_account_id,
                        "snapshot_source": report_snapshot_source,
                    },
                )

        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="report.run",
            object_type="daily_report",
            payload={
                "portfolio_slug": portfolio["slug"],
                "account_id": resolved_account_id,
                "report_markdown": str(md_path.resolve()),
                "compare_csv": str(compare_csv.resolve()),
                "snapshot_source": report_snapshot_source,
                "snapshot_id": snapshot_id,
                "selected_model": selected_model,
                "report_contract_version": "report.v1",
            },
            portfolio_id=portfolio_id,
        )
        return {
            "report_markdown": str(md_path.resolve()),
            "chart_paths": chart_paths,
            "account_id": resolved_account_id,
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
            # Stress requests are synchronous from the UI. Avoid forcing frequent heavy syncs
            # that can exceed gateway timeouts while still refreshing stale market history.
            stress_sync_max_age_seconds = max(
                300.0,
                float(self.runtime.mt5_config.live_history_poll_seconds or 0.0) * 8.0,
            )
            self.runtime.market_data.sync_market_data_if_stale(
                portfolio_slug=portfolio["slug"],
                max_age_seconds=stress_sync_max_age_seconds,
                days=int(self.runtime._default_days()),
                timeframes=[str(self.runtime._default_timeframe())],
            )
            live_holdings = self.runtime.market_data.live_holdings(portfolio_slug=portfolio["slug"])
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
                    snapshot_timestamp=datetime.now(timezone.utc).isoformat(),
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
            stress_surface = RiskEngine(
                bundle["holdings"],
                base_currency=str(portfolio["base_currency"]),
                no_exposure_epsilon_by_symbol=self._no_exposure_epsilon_by_symbol(list(bundle["portfolio_symbols"])),
                default_no_exposure_epsilon=self._default_no_exposure_epsilon(),
            ).build_stress_surface(
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
