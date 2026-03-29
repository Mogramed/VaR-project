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
        delta_position_eur: float,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
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
            delta_position_eur=delta_position_eur,
            note=note,
            persist=True,
            audit_action="decision.evaluate",
        )
