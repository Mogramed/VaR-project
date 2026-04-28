from __future__ import annotations

from typing import Any

from var_project.api.services.runtime import DeskServiceRuntime


class DeskTradingService:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

    def evaluate_trade_decision(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        trade_action: str | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        normalized_exposure_change = None
        if exposure_change is not None or delta_position_eur is not None:
            normalized_exposure_change = (
                float(exposure_change) if exposure_change is not None else float(delta_position_eur or 0.0)
            )
        selected_timeframe = self.runtime._default_timeframe()
        selected_days = int(self.runtime._default_days())
        selected_window = int(self.runtime.risk_defaults["window"])
        config = self.runtime._build_risk_model_config(None, None, None, None, None)
        bundle = self.runtime._compute_portfolio_state(
            portfolio_slug=portfolio["slug"],
            timeframe=selected_timeframe,
            days=selected_days,
            min_coverage=float(self.runtime.data_defaults["min_coverage"]),
            config=config,
            window=selected_window,
        )
        return self.runtime._evaluate_trade_decision_from_bundle(
            bundle=bundle,
            symbol=symbol,
            exposure_change=normalized_exposure_change,
            trade_action=trade_action,
            note=note,
            account_id=account_id,
            persist=True,
            audit_action="decision.evaluate",
        )

    def decision_alpha_replay(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 200,
        lookback_days: int | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        return self.runtime.decisions.replay_decision_alpha(
            portfolio_slug=portfolio["slug"],
            limit=limit,
            lookback_days=lookback_days,
        )

    def decision_alpha_forecast(
        self,
        *,
        symbol: str,
        portfolio_slug: str | None = None,
        horizon_days: int = 5,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        return self.runtime.decisions.forecast_decision_alpha(
            symbol=symbol,
            horizon_days=horizon_days,
            portfolio_slug=portfolio["slug"],
        )

    def decision_alpha_backtest_trajectory(
        self,
        *,
        symbol: str,
        portfolio_slug: str | None = None,
        lookback_days: int = 90,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        return self.runtime.decisions.backtest_trajectory_decision_alpha(
            symbol=symbol,
            lookback_days=lookback_days,
            portfolio_slug=portfolio["slug"],
        )

    def decision_alpha_portfolio_forecast(
        self,
        *,
        portfolio_slug: str | None = None,
        horizon_days: int = 150,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        return self.runtime.decisions.portfolio_forecast_decision_alpha(
            portfolio_slug=portfolio["slug"],
            horizon_days=horizon_days,
            symbols=symbols,
        )
