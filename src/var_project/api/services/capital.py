from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from var_project.alerts.engine import alerts_from_capital_snapshot
from var_project.api.services.runtime import DeskServiceRuntime
from var_project.risk.capital import build_capital_usage_snapshot


class DeskCapitalService:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

    def rebalance_capital(
        self,
        *,
        portfolio_slug: str | None = None,
        total_budget_eur: float | None = None,
        reserve_ratio: float | None = None,
        reference_model: str | None = None,
        symbol_weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        bundle = self.runtime._compute_portfolio_state(portfolio_slug=portfolio["slug"])
        overrides: dict[str, Any] = {}
        if total_budget_eur is not None:
            overrides["total_budget_eur"] = float(total_budget_eur)
        if reserve_ratio is not None:
            overrides["reserve_ratio"] = float(reserve_ratio)
        if reference_model is not None:
            overrides["reference_model"] = str(reference_model)
        if symbol_weights:
            overrides["symbol_weights"] = {str(symbol): float(value) for symbol, value in symbol_weights.items()}

        capital = build_capital_usage_snapshot(
            bundle["risk_budget"].to_dict(),
            self.runtime.limits_config,
            portfolio_slug=portfolio["slug"],
            base_currency=portfolio["base_currency"],
            reference_model=reference_model,
            snapshot_source="rebalance",
            snapshot_timestamp=datetime.now(timezone.utc).isoformat(),
            overrides=overrides,
        ).to_dict()
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        capital_id = self.runtime.storage.record_capital_snapshot(capital, portfolio_id=portfolio_id, source="rebalance")
        capital["id"] = capital_id
        capital_alerts = alerts_from_capital_snapshot(capital)
        if capital_alerts:
            self.runtime.storage.record_alerts(capital_alerts, portfolio_id=portfolio_id)
        self.runtime.storage.record_audit_event(
            actor="api",
            action_type="capital.rebalance",
            object_type="capital_snapshot",
            object_id=capital_id,
            payload={"portfolio_slug": portfolio["slug"], "overrides": overrides},
            portfolio_id=portfolio_id,
        )
        return capital
