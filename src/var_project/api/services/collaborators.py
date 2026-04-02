from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import httpx

from var_project.alerts.engine import (
    alerts_from_capital_snapshot,
    alerts_from_execution_result,
    alerts_from_risk_budget,
    alerts_from_risk_decision,
)
from var_project.core.exceptions import MT5ConnectionError
from var_project.engine.risk_engine import GarchConfig, MonteCarloConfig, RiskEngine, RiskModelConfig
from var_project.execution.mt5_live import (
    ORDER_CHECK_OK,
    ExecutionGuardDecision,
    MT5LiveGateway,
    MT5TerminalStatus,
)
from var_project.execution.mt5_remote import RemoteMT5Connector
from var_project.market_data.daily_returns import load_daily_simple_returns_from_processed
from var_project.portfolio.holdings import aggregate_exposure_by_symbol, holding_symbols, normalize_holdings
from var_project.reporting.render import render_daily_markdown
from var_project.risk.budgeting import build_risk_budget_snapshot
from var_project.risk.capital import build_capital_usage_snapshot
from var_project.risk.decisioning import TradeProposal, evaluate_trade_proposal
from var_project.storage.serialization import coerce_datetime

if TYPE_CHECKING:
    from var_project.api.services.runtime import DeskServiceRuntime


class PortfolioRiskCalculator:
    def __init__(self, runtime: "DeskServiceRuntime"):
        self.runtime = runtime

    def build_risk_model_config(
        self,
        alpha: float | None,
        n_sims: int | None,
        dist: str | None,
        df_t: int | None,
        seed: int | None,
    ) -> RiskModelConfig:
        mc_defaults = dict(self.runtime.risk_defaults["mc"])
        garch_defaults = dict(self.runtime.risk_defaults["garch"])
        return RiskModelConfig(
            alpha=float(alpha if alpha is not None else self.runtime.risk_defaults["alpha"]),
            ewma_lambda=float(self.runtime.risk_defaults["ewma_lambda"]),
            fhs_lambda=float(self.runtime.risk_defaults["fhs_lambda"]),
            mc=MonteCarloConfig(
                n_sims=int(n_sims if n_sims is not None else mc_defaults["n_sims"]),
                dist=str(dist if dist is not None else mc_defaults["dist"]),
                df_t=int(df_t if df_t is not None else mc_defaults["df_t"]),
                seed=seed if seed is not None else mc_defaults["seed"],
            ),
            garch=GarchConfig(
                enabled=bool(garch_defaults["enabled"]),
                p=int(garch_defaults["p"]),
                q=int(garch_defaults["q"]),
                dist=str(garch_defaults["dist"]),
                mean=str(garch_defaults["mean"]),
            ),
        )

    def compute_portfolio_state(
        self,
        *,
        portfolio_slug: str | None = None,
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        config: RiskModelConfig | None = None,
        window: int | None = None,
        allow_auto_sync: bool = True,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            live_holdings = self.runtime.market_data.live_holdings(portfolio_slug=portfolio["slug"])
            if live_holdings:
                return self.compute_portfolio_state_for_holdings(
                    portfolio=portfolio,
                    holdings=live_holdings,
                    timeframe=timeframe,
                    days=days,
                    min_coverage=min_coverage,
                    config=config,
                    window=window,
                    allow_auto_sync=allow_auto_sync,
                    snapshot_source="mt5_live",
                    snapshot_timestamp=datetime.now(timezone.utc).isoformat(),
                )
        return self.compute_portfolio_state_for_holdings(
            portfolio=portfolio,
            holdings=portfolio.get("configured_holdings") or [],
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            config=config,
            window=window,
            allow_auto_sync=allow_auto_sync,
            snapshot_source="historical",
        )

    def compute_portfolio_state_for_exposure(
        self,
        *,
        portfolio: Mapping[str, Any],
        exposure_by_symbol: Mapping[str, Any],
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        config: RiskModelConfig | None = None,
        window: int | None = None,
        allow_auto_sync: bool = True,
        snapshot_source: str = "historical",
        snapshot_timestamp: str | None = None,
    ) -> dict[str, Any]:
        return self.compute_portfolio_state_for_holdings(
            portfolio=portfolio,
            holdings=exposure_by_symbol,
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            config=config,
            window=window,
            allow_auto_sync=allow_auto_sync,
            snapshot_source=snapshot_source,
            snapshot_timestamp=snapshot_timestamp,
        )

    def compute_portfolio_state_for_positions(
        self,
        *,
        portfolio: Mapping[str, Any],
        positions_eur: Mapping[str, Any],
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        config: RiskModelConfig | None = None,
        window: int | None = None,
        allow_auto_sync: bool = True,
        snapshot_source: str = "historical",
        snapshot_timestamp: str | None = None,
    ) -> dict[str, Any]:
        return self.compute_portfolio_state_for_exposure(
            portfolio=portfolio,
            exposure_by_symbol=positions_eur,
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            config=config,
            window=window,
            allow_auto_sync=allow_auto_sync,
            snapshot_source=snapshot_source,
            snapshot_timestamp=snapshot_timestamp,
        )

    def compute_portfolio_state_for_holdings(
        self,
        *,
        portfolio: Mapping[str, Any],
        holdings: Mapping[str, Any] | list[Mapping[str, Any]],
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        config: RiskModelConfig | None = None,
        window: int | None = None,
        allow_auto_sync: bool = True,
        snapshot_source: str = "historical",
        snapshot_timestamp: str | None = None,
    ) -> dict[str, Any]:
        selected_timeframe = timeframe or self.default_timeframe()
        selected_days = int(days or self.default_days())
        selected_min_coverage = float(min_coverage or self.runtime.data_defaults["min_coverage"])
        selected_window = int(window or self.runtime.risk_defaults["window"])
        estimation_window_days = int(self.runtime.risk_defaults["estimation_window_days"])
        minimum_valid_days = int(self.runtime.risk_defaults["minimum_valid_days"])
        selected_alphas = [float(item) for item in self.runtime.risk_defaults["alphas"]]
        selected_horizons = [int(item) for item in self.runtime.risk_defaults["horizons"]]
        risk_config = config or self.build_risk_model_config(None, None, None, None, None)

        normalized_holdings = normalize_holdings(
            holdings,
            symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
            base_currency=str(portfolio["base_currency"]),
        )
        exposure_by_symbol = aggregate_exposure_by_symbol(normalized_holdings, base_currency=str(portfolio["base_currency"]))
        portfolio_symbols = holding_symbols(
            normalized_holdings,
            symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
            base_currency=str(portfolio["base_currency"]),
        )
        if not portfolio_symbols:
            raise ValueError("No holdings are available for the selected portfolio.")

        daily_returns = self.load_daily_returns_for_portfolio(
            portfolio,
            selected_timeframe,
            selected_days,
            selected_min_coverage,
            allow_auto_sync=allow_auto_sync,
            symbols=portfolio_symbols,
        )
        engine = RiskEngine(normalized_holdings, base_currency=str(portfolio["base_currency"]))
        portfolio_frame = engine.build_portfolio_frame(daily_returns)
        sample = portfolio_frame.iloc[-min(len(portfolio_frame), estimation_window_days) :].copy()
        if sample.empty:
            raise ValueError("No aligned returns available for the selected portfolio.")
        sample_symbols = engine.portfolio_symbols(sample)

        snapshot = engine.evaluate_models(pnl=sample["pnl"], returns_wide=sample[sample_symbols], config=risk_config)
        reference_model = self.runtime._decision_reference_model(portfolio["slug"])
        risk_surface = engine.build_risk_surface(
            sample[sample_symbols],
            risk_config,
            alphas=selected_alphas,
            horizons=selected_horizons,
            estimation_window_days=estimation_window_days,
            minimum_valid_days=minimum_valid_days,
            reference_model=reference_model,
        )
        stress_surface = engine.build_stress_surface(
            sample[sample_symbols],
            risk_config,
            alphas=selected_alphas,
            horizons=selected_horizons,
            estimation_window_days=estimation_window_days,
            minimum_valid_days=minimum_valid_days,
        )
        headline_risk = [dict(item) for item in risk_surface.to_dict().get("headline", [])]
        stressed_headline = [
            dict(item)
            for item in list(stress_surface.get("headline_risk") or [])
            if item.get("is_stressed")
        ]
        seen_headline_keys = {str(item.get("key")) for item in headline_risk}
        for item in stressed_headline:
            key = str(item.get("key"))
            if key in seen_headline_keys:
                continue
            headline_risk.append(item)
            seen_headline_keys.add(key)
        attribution = engine.attribute_from_returns(sample[sample_symbols], config=risk_config, base_snapshot=snapshot)
        risk_budget = build_risk_budget_snapshot(
            attribution,
            self.runtime.limits_config,
            exposure_by_symbol=exposure_by_symbol,
            preferred_model=self.runtime._preferred_model(portfolio["slug"]),
        )
        capital = build_capital_usage_snapshot(
            risk_budget.to_dict(),
            self.runtime.limits_config,
            portfolio_slug=portfolio["slug"],
            base_currency=portfolio["base_currency"],
            reference_model=reference_model,
            snapshot_source=snapshot_source,
            snapshot_timestamp=snapshot_timestamp or sample.index[-1].isoformat(),
        ).to_dict()
        return {
            "portfolio": dict(portfolio),
            "holdings": [holding.to_dict() for holding in normalized_holdings],
            "exposure_by_symbol": exposure_by_symbol,
            "portfolio_symbols": sample_symbols,
            "timeframe": selected_timeframe,
            "days": selected_days,
            "min_coverage": selected_min_coverage,
            "window": selected_window,
            "config": risk_config,
            "daily_returns": daily_returns,
            "portfolio_frame": portfolio_frame,
            "sample": sample,
            "snapshot": snapshot,
            "risk_surface": risk_surface.to_dict(),
            "headline_risk": headline_risk,
            "stress_surface": stress_surface,
            "data_quality": risk_surface.data_quality.to_dict(),
            "attribution": attribution,
            "risk_budget": risk_budget,
            "capital": capital,
        }

    def compute_live_portfolio_state(
        self,
        *,
        portfolio: Mapping[str, Any],
        live,
        allow_auto_sync: bool = True,
    ) -> dict[str, Any]:
        live_holdings = [item.to_dict() for item in live.holdings(symbols=None)]
        return self.compute_portfolio_state_for_holdings(
            portfolio=portfolio,
            holdings=live_holdings,
            allow_auto_sync=allow_auto_sync,
            snapshot_source="mt5_live_preview",
            snapshot_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def load_daily_returns(self, timeframe: str, days: int, min_coverage: float):
        return self.load_daily_returns_for_portfolio(self.runtime.portfolio, timeframe, days, min_coverage)

    def load_daily_returns_for_portfolio(
        self,
        portfolio: Mapping[str, Any],
        timeframe: str,
        days: int,
        min_coverage: float,
        *,
        allow_auto_sync: bool = True,
        symbols: list[str] | None = None,
    ):
        if self.runtime.market_data.should_use_mt5_market_data(portfolio):
            return self.runtime.market_data.load_daily_returns_for_portfolio(
                portfolio,
                timeframe=timeframe,
                days=days,
                min_coverage=min_coverage,
                ensure_sync=False,
                allow_auto_sync=allow_auto_sync,
                symbols=symbols,
            )
        return load_daily_simple_returns_from_processed(
            root=self.runtime.root,
            symbols=list(symbols or portfolio["symbols"]),
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
        )

    def default_timeframe(self) -> str:
        timeframes = self.runtime.data_defaults["timeframes"]
        return str(timeframes[0]) if timeframes else "H1"

    def default_days(self) -> int:
        days_list = self.runtime.data_defaults["history_days_list"]
        return int(days_list[-1]) if days_list else 365

    def exposures_json(self, exposure_by_symbol: Mapping[str, Any] | None = None) -> str:
        selected = dict(
            self.runtime.portfolio["configured_exposure"] if exposure_by_symbol is None else exposure_by_symbol
        )
        return "{" + ", ".join(f'"{symbol}": {float(value):.6f}' for symbol, value in selected.items()) + "}"


class DecisionPolicyEngine:
    def __init__(self, runtime: "DeskServiceRuntime"):
        self.runtime = runtime

    def evaluate_trade_decision_from_bundle(
        self,
        *,
        bundle: Mapping[str, Any],
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None,
        persist: bool,
        audit_action: str = "decision.evaluate",
    ) -> dict[str, Any]:
        portfolio = dict(bundle["portfolio"])
        exposure_by_symbol = dict(bundle["exposure_by_symbol"])
        sample = bundle["sample"]
        sample_symbols = list(bundle.get("portfolio_symbols") or portfolio["symbols"])
        selected_change = exposure_change if exposure_change is not None else delta_position_eur
        result = evaluate_trade_proposal(
            sample[sample_symbols],
            exposure_by_symbol=exposure_by_symbol,
            proposal=TradeProposal(symbol=str(symbol).upper(), exposure_change=selected_change, note=note),
            config=bundle["config"],
            limits_cfg=self.runtime.limits_config,
            reference_model=self.decision_reference_model(portfolio["slug"]),
        )
        now = datetime.now(timezone.utc)
        payload = {
            "time_utc": now.isoformat(),
            "decision_mode": self.decision_settings()["decision_mode"],
            "portfolio_slug": portfolio["slug"],
            "timeframe": bundle["timeframe"],
            "days": bundle["days"],
            "window": bundle["window"],
            **result.to_dict(),
        }
        payload["pre_trade"]["headline_risk"] = list(bundle.get("headline_risk") or [])
        payload["pre_trade"]["data_quality"] = dict(bundle.get("data_quality") or {})

        approved_change = float(payload.get("approved_exposure_change", 0.0))
        if abs(approved_change) > 1e-9 and str(symbol).upper() in exposure_by_symbol:
            post_exposure = dict(exposure_by_symbol)
            post_exposure[str(symbol).upper()] = float(post_exposure.get(str(symbol).upper(), 0.0) + approved_change)
            post_bundle = self.runtime._compute_portfolio_state_for_exposure(
                portfolio=portfolio,
                exposure_by_symbol=post_exposure,
                timeframe=bundle["timeframe"],
                days=bundle["days"],
                min_coverage=bundle["min_coverage"],
                config=bundle["config"],
                window=bundle["window"],
                snapshot_source="decision_preview",
                snapshot_timestamp=now.isoformat(),
            )
            payload["post_trade"]["headline_risk"] = list(post_bundle.get("headline_risk") or [])
            payload["post_trade"]["data_quality"] = dict(post_bundle.get("data_quality") or {})
        else:
            payload["post_trade"]["headline_risk"] = list(bundle.get("headline_risk") or [])
            payload["post_trade"]["data_quality"] = dict(bundle.get("data_quality") or {})
        if not persist:
            return payload

        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        decision_id = self.runtime.storage.record_decision(payload, portfolio_id=portfolio_id)
        payload["id"] = decision_id
        payload["created_at"] = now.isoformat()
        decision_alerts = alerts_from_risk_decision(payload)
        if decision_alerts:
            self.runtime.storage.record_alerts(decision_alerts, portfolio_id=portfolio_id)
        self.runtime.storage.record_audit_event(
            actor="api",
            action_type=audit_action,
            object_type="risk_decision",
            object_id=decision_id,
            payload=payload,
            portfolio_id=portfolio_id,
        )
        return payload

    def post_capital_after_trade(
        self,
        *,
        bundle: Mapping[str, Any],
        symbol: str,
        approved_exposure_change: float | None = None,
        approved_delta_position_eur: float | None = None,
        snapshot_source: str,
    ) -> dict[str, Any]:
        portfolio = dict(bundle["portfolio"])
        post_exposure = dict(bundle["exposure_by_symbol"])
        symbol_key = str(symbol).upper()
        if symbol_key not in post_exposure:
            return dict(bundle["capital"])
        selected_change = (
            approved_exposure_change if approved_exposure_change is not None else approved_delta_position_eur
        )
        post_exposure[symbol_key] = float(post_exposure.get(symbol_key, 0.0) + float(selected_change or 0.0))
        post_bundle = self.runtime._compute_portfolio_state_for_exposure(
            portfolio=portfolio,
            exposure_by_symbol=post_exposure,
            timeframe=bundle["timeframe"],
            days=bundle["days"],
            min_coverage=bundle["min_coverage"],
            config=bundle["config"],
            window=bundle["window"],
            snapshot_source=snapshot_source,
            snapshot_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return dict(post_bundle["capital"])

    def decision_settings(self) -> dict[str, Any]:
        decision_cfg = dict(self.runtime.limits_config.get("risk_decision") or {})
        budget_cfg = dict(self.runtime.limits_config.get("risk_budget") or {})
        return {
            "decision_mode": str(decision_cfg.get("decision_mode", "advisory")),
            "reference_model": str(decision_cfg.get("reference_model", "best_validation")),
            "warn_threshold": float(decision_cfg.get("warn_threshold", budget_cfg.get("utilisation_warn", 0.85))),
            "breach_threshold": float(decision_cfg.get("breach_threshold", budget_cfg.get("utilisation_breach", 1.0))),
            "min_fill_ratio": float(decision_cfg.get("min_fill_ratio", 0.25)),
            "allow_risk_reducing_override": bool(decision_cfg.get("allow_risk_reducing_override", True)),
        }

    def preferred_model(self, portfolio_slug: str | None = None) -> str | None:
        if not self.runtime.storage_ready:
            return None
        latest_validation = self.runtime.storage.latest_validation_run(portfolio_slug=portfolio_slug)
        if latest_validation is None:
            return None
        best_model = str(latest_validation.get("best_model") or "").strip().lower()
        return best_model or None

    def decision_reference_model(self, portfolio_slug: str | None = None) -> str:
        reference = str(self.decision_settings()["reference_model"]).strip().lower()
        if reference in {"best_validation", "best", "best_model"}:
            best_model = self.preferred_model(portfolio_slug)
            if best_model:
                return best_model
            budget_pref = str((self.runtime.limits_config.get("risk_budget") or {}).get("preferred_model", "")).strip().lower()
            if budget_pref and budget_pref not in {"best_validation", "best", "best_model", "auto"}:
                return budget_pref
            return "hist"
        return reference or "hist"


class MT5ExecutionOrchestrator:
    def __init__(self, runtime: "DeskServiceRuntime"):
        self.runtime = runtime

    def build_mt5_connector(self):
        if self.runtime.mt5_config.agent_base_url:
            return RemoteMT5Connector(self.runtime.mt5_config)
        return self.runtime.mt5_connector_factory(self.runtime.mt5_config)

    @contextmanager
    def mt5_gateway(self):
        connector = self.build_mt5_connector()
        connector.init()
        try:
            yield MT5LiveGateway(
                connector,
                config=self.runtime.mt5_config,
                base_currency=self.runtime.portfolio["base_currency"],
            )
        finally:
            connector.shutdown()

    def run_order_check_with_fill_fallback(
        self,
        *,
        live: MT5LiveGateway,
        order_request: Mapping[str, Any],
        fill_candidates: Any,
    ) -> dict[str, Any]:
        mutable_request = dict(order_request)
        candidates = [int(candidate) for candidate in list(fill_candidates or [])]
        initial = live.connector.order_check(mutable_request)
        if int(initial.get("retcode", ORDER_CHECK_OK)) != 10030:
            return initial

        for candidate in candidates:
            mutable_request["type_filling"] = candidate
            payload = live.connector.order_check(mutable_request)
            if int(payload.get("retcode", ORDER_CHECK_OK)) == ORDER_CHECK_OK:
                order_request["type_filling"] = candidate
                return payload
        return initial

    def _attach_position_target_if_reducing(
        self,
        *,
        live: MT5LiveGateway,
        symbol: str,
        order_request: Mapping[str, Any],
        request_side: str | None,
        requested_volume_lots: float,
    ) -> dict[str, Any]:
        normalized_side = str(request_side or "").upper()
        if normalized_side not in {"BUY", "SELL"}:
            return dict(order_request)
        opposite_side = "SELL" if normalized_side == "BUY" else "BUY"
        target_volume = float(requested_volume_lots or order_request.get("volume") or 0.0)
        if target_volume <= 1e-9:
            return dict(order_request)

        candidates = []
        for position in live.positions(symbols=[symbol]):
            if position.ticket is None:
                continue
            if str(position.side or "").upper() != opposite_side:
                continue
            live_volume = float(position.volume_lots or 0.0)
            if live_volume + 1e-9 < target_volume:
                continue
            candidates.append(position)

        if not candidates:
            return dict(order_request)

        candidates.sort(
            key=lambda item: (
                abs(float(item.volume_lots or 0.0) - target_volume),
                str(item.time_utc or ""),
                int(item.ticket or 0),
            )
        )
        targeted_request = dict(order_request)
        targeted_request["position"] = int(candidates[0].ticket)
        return targeted_request

    def build_execution_guard(
        self,
        *,
        live: MT5LiveGateway,
        terminal_status: MT5TerminalStatus,
        account: Mapping[str, Any],
        symbol: str,
        note: str | None,
        decision: Mapping[str, Any],
    ) -> tuple[ExecutionGuardDecision, dict[str, Any], dict[str, Any]]:
        requested_exposure_change = float(decision.get("requested_exposure_change", 0.0))
        approved_exposure_change = float(decision.get("approved_exposure_change", 0.0))
        model_used = str(decision.get("model_used") or "hist")
        risk_decision = str(decision.get("decision") or "REJECT")
        reasons: list[str] = list(decision.get("reasons") or [])
        suggested_exposure_change = decision.get("suggested_exposure_change")

        if risk_decision == "REJECT" or abs(approved_exposure_change) <= 1e-9 or not terminal_status.ready:
            if not terminal_status.ready:
                reasons.append(terminal_status.message)
            guard = ExecutionGuardDecision(
                decision="REJECT",
                risk_decision=risk_decision,
                requested_exposure_change=requested_exposure_change,
                approved_exposure_change=approved_exposure_change,
                executable_exposure_change=0.0,
                suggested_exposure_change=None if suggested_exposure_change is None else float(suggested_exposure_change),
                model_used=model_used,
                side=None,
                volume_lots=0.0,
                price=None,
                execution_enabled=bool(self.runtime.mt5_config.execution_enabled),
                submit_allowed=False,
                margin_ok=False,
                margin_required=None,
                free_margin_after=None,
                order_check_retcode=None,
                order_check_comment=None,
                reasons=reasons,
            )
            return guard, {}, {}

        order_request, meta = live.build_market_order(
            symbol=str(symbol).upper(),
            exposure_change=approved_exposure_change,
            note=note,
        )
        order_request = self._attach_position_target_if_reducing(
            live=live,
            symbol=str(symbol).upper(),
            order_request=order_request,
            request_side=None if meta.get("side") is None else str(meta.get("side")),
            requested_volume_lots=float(meta.get("volume_lots", 0.0)),
        )
        order_check = self.run_order_check_with_fill_fallback(
            live=live,
            order_request=order_request,
            fill_candidates=meta.get("fill_candidates") or [],
        )
        retcode = None if order_check.get("retcode") is None else int(order_check.get("retcode"))
        margin_ok = retcode == ORDER_CHECK_OK
        if not margin_ok:
            reasons.append(str(order_check.get("comment") or "Broker order_check rejected the request."))

        guard = ExecutionGuardDecision(
            decision=risk_decision if margin_ok else "REJECT",
            risk_decision=risk_decision,
            requested_exposure_change=requested_exposure_change,
            approved_exposure_change=approved_exposure_change,
            executable_exposure_change=float(meta.get("executable_exposure_change", 0.0)),
            suggested_exposure_change=None if suggested_exposure_change is None else float(suggested_exposure_change),
            model_used=model_used,
            side=None if meta.get("side") is None else str(meta.get("side")),
            volume_lots=float(meta.get("volume_lots", 0.0)),
            price=None if meta.get("price") is None else float(meta.get("price")),
            execution_enabled=bool(self.runtime.mt5_config.execution_enabled),
            submit_allowed=margin_ok and risk_decision != "REJECT" and terminal_status.ready,
            margin_ok=margin_ok,
            margin_required=None if order_check.get("margin") is None else float(order_check.get("margin")),
            free_margin_after=None if order_check.get("margin_free") is None else float(order_check.get("margin_free")),
            order_check_retcode=retcode,
            order_check_comment=None if order_check.get("comment") is None else str(order_check.get("comment")),
            reasons=reasons,
        )
        return guard, order_request, order_check

    def mt5_dependency(self) -> dict[str, Any]:
        if self.runtime.mt5_config.agent_base_url:
            target = str(self.runtime.mt5_config.agent_base_url)
            try:
                response = httpx.get(f"{target}/health", timeout=5.0)
                response.raise_for_status()
                payload = response.json()
                return {
                    "mode": "remote_agent",
                    "configured": True,
                    "reachable": True,
                    "schema_ready": None,
                    "detail": payload.get("agent_mode"),
                    "target": target,
                }
            except Exception as exc:  # pragma: no cover
                return {
                    "mode": "remote_agent",
                    "configured": True,
                    "reachable": False,
                    "schema_ready": None,
                    "detail": str(exc),
                    "target": target,
                }

        return {
            "mode": "direct_terminal",
            "configured": bool(self.runtime.mt5_config.path or self.runtime.mt5_config.login or self.runtime.mt5_config.server),
            "reachable": None,
            "schema_ready": None,
            "detail": "MT5 direct connector mode.",
            "target": self.runtime.mt5_config.path,
        }

    def mt5_live_dependency(self, portfolio_slug: str | None = None) -> dict[str, Any]:
        target = self.runtime.mt5_config.agent_base_url
        if not target:
            return {
                "mode": "direct_terminal",
                "configured": bool(self.runtime.mt5_config.path or self.runtime.mt5_config.login or self.runtime.mt5_config.server),
                "reachable": None,
                "schema_ready": None,
                "detail": "Live bridge is only available through the remote MT5 agent.",
                "target": self.runtime.mt5_config.path,
            }
        connector = self.build_mt5_connector()
        try:
            connector.init()
            if hasattr(connector, "live_state"):
                state = dict(connector.live_state())
                return {
                    "mode": "remote_agent_live_bridge",
                    "configured": True,
                    "reachable": bool(state.get("connected", False)),
                    "schema_ready": None,
                    "detail": str(state.get("status") or "unknown"),
                    "target": target,
                    "sequence": int(state.get("sequence") or 0),
                    "generated_at": state.get("generated_at"),
                    "stale": bool(state.get("stale", False)),
                    "degraded": bool(state.get("degraded", False)),
                }
            return {
                "mode": "remote_agent_live_bridge",
                "configured": True,
                "reachable": False,
                "schema_ready": None,
                "detail": "Remote connector does not expose a live bridge contract.",
                "target": target,
            }
        except Exception as exc:  # pragma: no cover
            return {
                "mode": "remote_agent_live_bridge",
                "configured": True,
                "reachable": False,
                "schema_ready": None,
                "detail": str(exc),
                "target": target,
            }
        finally:
            try:
                connector.shutdown()
            except Exception:
                pass


class GovernanceRecorder:
    def __init__(self, runtime: "DeskServiceRuntime"):
        self.runtime = runtime

    def _live_report_refresh_interval_seconds(self) -> float:
        return max(
            30.0,
            float(self.runtime.mt5_config.live_history_poll_seconds),
            float(self.runtime.mt5_config.live_poll_seconds) * 10.0,
        )

    def refresh_live_report_if_needed(
        self,
        *,
        portfolio: Mapping[str, Any],
        portfolio_id: int,
        snapshot_payload: Mapping[str, Any],
        snapshot_id: int,
        source: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if source != "mt5_live_bridge" or not self.runtime.storage_ready:
            return

        compare_csv = self.resolve_compare_path(None, portfolio_slug=portfolio["slug"])
        if compare_csv is None or not compare_csv.exists():
            return

        latest_report = self.runtime.storage.latest_artifact("daily_report", portfolio_slug=portfolio["slug"])
        latest_details = {} if latest_report is None else dict(latest_report.get("details") or {})
        normalized_metadata = dict(metadata or {})
        live_persistence_key = str(normalized_metadata.get("live_persistence_key") or "")

        if live_persistence_key and str(latest_details.get("live_persistence_key") or "") == live_persistence_key:
            return
        if latest_details.get("snapshot_id") is not None and int(latest_details["snapshot_id"]) == int(snapshot_id):
            return

        latest_updated_at = coerce_datetime(
            None if latest_report is None else latest_report.get("updated_at") or latest_report.get("created_at")
        )
        if latest_updated_at is not None:
            age_seconds = max((datetime.now(timezone.utc) - latest_updated_at).total_seconds(), 0.0)
            if age_seconds < self._live_report_refresh_interval_seconds():
                return

        rendered_snapshot = {
            "id": snapshot_id,
            "source": source,
            "created_at": snapshot_payload.get("time_utc"),
            "payload": dict(snapshot_payload),
        }
        report_path = render_daily_markdown(
            compare_csv=compare_csv,
            out_dir=self.runtime.storage.settings.reports_dir,
            snapshot=rendered_snapshot,
            risk_limits_yaml=self.runtime.root / "config" / "risk_limits.yaml",
            report_label=str(portfolio["name"]),
        )
        self.append_governance_sections(
            report_path,
            portfolio_slug=portfolio["slug"],
            capital_source=source,
        )

        compare_artifact_id = self.runtime.storage.register_artifact(
            compare_csv,
            artifact_type="backtest_compare",
        )
        report_artifact_id = self.runtime.storage.register_artifact(
            report_path,
            artifact_type="daily_report",
            details={
                "portfolio_slug": portfolio["slug"],
                "compare_csv": str(compare_csv.resolve()),
                "source_artifact_id": compare_artifact_id,
                "snapshot_source": source,
                "snapshot_id": snapshot_id,
                "auto_generated": True,
                "live_persistence_key": live_persistence_key,
            },
        )

        chart_paths: list[str] = []
        for chart_path in (
            report_path.with_name(f"{compare_csv.stem}_exceptions.png"),
            report_path.with_name(f"{compare_csv.stem}_pnl_var.png"),
        ):
            if not chart_path.exists():
                continue
            resolved_path = str(chart_path.resolve())
            chart_paths.append(resolved_path)
            self.runtime.storage.register_artifact(
                chart_path,
                artifact_type="report_chart",
                details={
                    "report_markdown": str(report_path.resolve()),
                    "portfolio_slug": portfolio["slug"],
                    "snapshot_source": source,
                    "snapshot_id": snapshot_id,
                    "auto_generated": True,
                },
            )

        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="report.auto_refresh",
            object_type="daily_report",
            object_id=report_artifact_id,
            payload={
                "portfolio_slug": portfolio["slug"],
                "report_markdown": str(report_path.resolve()),
                "compare_csv": str(compare_csv.resolve()),
                "snapshot_source": source,
                "snapshot_id": snapshot_id,
                "live_persistence_key": live_persistence_key,
                "chart_paths": chart_paths,
            },
            portfolio_id=portfolio_id,
        )

    def database_dependency(self) -> dict[str, Any]:
        reachable, detail = self.runtime.storage.ping()
        return {
            "mode": "database",
            "configured": True,
            "reachable": bool(reachable),
            "schema_ready": bool(self.runtime.storage_ready),
            "detail": detail,
            "target": self.runtime.storage.settings.database_url,
        }

    def persist_live_bundle(
        self,
        *,
        bundle: Mapping[str, Any],
        portfolio_id: int,
        source: str,
        metadata: Mapping[str, Any] | None = None,
        persist_alerts: bool = True,
        persist_audit: bool = True,
    ) -> None:
        portfolio = dict(bundle["portfolio"])
        sample = bundle["sample"]
        snapshot_payload = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "alpha": bundle["config"].alpha,
            "timeframe": bundle["timeframe"],
            "days": bundle["days"],
            "window": bundle["window"],
            "holdings": list(bundle["holdings"]),
            "exposure_by_symbol": dict(bundle["exposure_by_symbol"]),
            "var": bundle["snapshot"].vars_dict(),
            "es": bundle["snapshot"].es_dict(),
            "attribution": bundle["attribution"].to_dict(),
            "risk_budget": bundle["risk_budget"].to_dict(),
            "capital_usage": dict(bundle["capital"]),
            "sample_size": bundle["snapshot"].sample_size,
            "latest_observation": sample.index[-1].isoformat(),
            "risk_surface": dict(bundle.get("risk_surface") or {}),
            "headline_risk": list(bundle.get("headline_risk") or []),
            "stress_surface": dict(bundle.get("stress_surface") or {}),
            "data_quality": dict(bundle.get("data_quality") or {}),
            "risk_nowcast": dict(bundle.get("risk_nowcast") or {}),
            "microstructure": dict(bundle.get("microstructure") or {}),
            "tick_quality": dict(bundle.get("tick_quality") or {}),
            "pnl_explain": dict(bundle.get("pnl_explain") or {}),
            "metadata": dict(metadata or {}),
        }
        snapshot_id = self.runtime.storage.record_snapshot(snapshot_payload, portfolio_id=portfolio_id, source=source)
        self.runtime.storage.record_capital_snapshot(dict(bundle["capital"]), portfolio_id=portfolio_id, source=source)
        self.refresh_live_report_if_needed(
            portfolio=portfolio,
            portfolio_id=portfolio_id,
            snapshot_payload=snapshot_payload,
            snapshot_id=snapshot_id,
            source=source,
            metadata=metadata,
        )
        if persist_alerts:
            budget_alerts = alerts_from_risk_budget(bundle["risk_budget"].to_dict())
            capital_alerts = alerts_from_capital_snapshot(bundle["capital"])
            if budget_alerts:
                self.runtime.storage.record_alerts(budget_alerts, portfolio_id=portfolio_id, snapshot_id=snapshot_id)
            if capital_alerts:
                self.runtime.storage.record_alerts(capital_alerts, portfolio_id=portfolio_id, snapshot_id=snapshot_id)
        if persist_audit:
            self.runtime.storage.record_audit_event(
                actor="api",
                action_type="mt5.reconcile",
                object_type="risk_snapshot",
                object_id=snapshot_id,
                payload={"portfolio_slug": portfolio["slug"], "source": source, "metadata": dict(metadata or {})},
                portfolio_id=portfolio_id,
            )

    def alert_counts_by_portfolio(self) -> dict[str, int]:
        if not self.runtime.storage_ready:
            return {}
        portfolio_map = {portfolio["id"]: portfolio["slug"] for portfolio in self.runtime.storage.list_portfolios()}
        counts: dict[str, int] = {}
        for alert in self.runtime.storage.recent_alerts(limit=500):
            portfolio_id = alert.get("portfolio_id")
            slug = portfolio_map.get(portfolio_id)
            if slug is None:
                continue
            counts[slug] = int(counts.get(slug, 0) + 1)
        return counts

    def resolve_compare_path(self, compare_path: str | None, *, portfolio_slug: str | None = None) -> Path | None:
        if compare_path:
            candidate = Path(compare_path)
            return candidate.resolve() if candidate.is_absolute() else (self.runtime.root / candidate).resolve()
        if not self.runtime.storage_ready:
            return None
        if portfolio_slug:
            latest_backtest = self.runtime.storage.latest_backtest_run(portfolio_slug=portfolio_slug)
            artifact_id = None if latest_backtest is None else latest_backtest.get("artifact_id")
            if artifact_id:
                artifact = self.runtime.storage.artifact_by_id(int(artifact_id))
                if artifact is not None:
                    return Path(artifact["path"]).resolve()
        latest = self.runtime.storage.latest_artifact("backtest_compare", portfolio_slug=portfolio_slug)
        if latest is None:
            return None
        return Path(latest["path"]).resolve()

    def append_governance_sections(
        self,
        report_path: Path,
        *,
        portfolio_slug: str | None = None,
        capital_source: str | None = None,
    ) -> None:
        content = report_path.read_text(encoding="utf-8")
        decisions = (
            self.runtime.storage.recent_decisions(limit=5, portfolio_slug=portfolio_slug)
            if self.runtime.storage_ready
            else []
        )
        capital_history = (
            self.runtime.storage.capital_history(
                limit=5,
                source=capital_source,
                portfolio_slug=portfolio_slug,
            )
            if self.runtime.storage_ready
            else []
        )
        if self.runtime.storage_ready and capital_source is not None and not capital_history:
            capital_history = self.runtime.storage.capital_history(limit=5, portfolio_slug=portfolio_slug)
        audits = (
            self.runtime.storage.recent_audit_events(limit=8, portfolio_slug=portfolio_slug)
            if self.runtime.storage_ready
            else []
        )

        lines = [content.rstrip(), "", "## Decision History", ""]
        if not decisions:
            lines.append("- No persisted trade decisions yet.")
        else:
            for item in decisions:
                lines.append(
                    f"- {item.get('created_at') or item.get('time_utc')}: {item.get('symbol')} -> {item.get('decision')} "
                    f"(requested {item.get('requested_exposure_change')}, "
                    f"approved {item.get('approved_exposure_change')}, "
                    f"model {item.get('model_used')})"
                )

        lines.extend(["", "## Capital History", ""])
        if not capital_history:
            lines.append("- No capital snapshots persisted yet.")
        else:
            for item in capital_history:
                lines.append(
                    f"- {item.get('created_at') or item.get('snapshot_timestamp')}: "
                    f"consumed={item.get('total_capital_consumed_eur')} remaining={item.get('total_capital_remaining_eur')} "
                    f"status={item.get('status')} "
                    f"source={item.get('snapshot_source') or item.get('source') or 'n/a'}"
                )

        lines.extend(["", "## Audit Trail", ""])
        if not audits:
            lines.append("- No audit events persisted yet.")
        else:
            for item in audits:
                lines.append(
                    f"- {item.get('created_at')}: {item.get('actor')} {item.get('action_type')} "
                    f"{item.get('object_type') or ''}#{item.get('object_id') or ''}".strip()
                )

        report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
