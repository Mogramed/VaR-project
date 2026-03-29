from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from var_project.api.services.collaborators import (
    DecisionPolicyEngine,
    GovernanceRecorder,
    MT5ExecutionOrchestrator,
    PortfolioRiskCalculator,
)
from var_project.api.services.market_data import DeskMarketDataService
from var_project.connectors.mt5_connector import MT5Connector
from var_project.core.settings import (
    get_data_defaults,
    get_desk_context,
    get_mt5_config,
    get_portfolio_context,
    get_portfolio_contexts,
    get_risk_defaults,
    load_risk_limits,
    load_settings,
)
from var_project.desk.overview import DeskDefinition
from var_project.engine.risk_engine import RiskModelConfig
from var_project.execution.mt5_live import ExecutionGuardDecision, MT5LiveGateway, MT5TerminalStatus
from var_project.storage import AppStorage


class DeskServiceRuntime:
    def __init__(
        self,
        root: Path,
        mt5_connector_factory: type[MT5Connector] | None = None,
        *,
        bootstrap_storage: bool = False,
    ):
        self.root = root
        self.bootstrap_storage = bool(bootstrap_storage)
        self.raw_config = load_settings(root)
        self.limits_config = load_risk_limits(root)
        self.data_defaults = get_data_defaults(self.raw_config)
        self.risk_defaults = get_risk_defaults(self.raw_config)
        self.mt5_config = get_mt5_config(self.raw_config)
        self.mt5_connector_factory = mt5_connector_factory or MT5Connector
        self._has_custom_mt5_factory = mt5_connector_factory is not None
        self.portfolios = get_portfolio_contexts(self.raw_config)
        self.portfolio = get_portfolio_context(self.raw_config)
        self.portfolio_by_slug = {portfolio["slug"]: portfolio for portfolio in self.portfolios}
        self.desk = DeskDefinition(**get_desk_context(self.raw_config))
        self.storage = AppStorage.from_root(root, self.raw_config)
        self.storage.initialize(create_schema=self.bootstrap_storage)
        self.storage_ready = self.storage.schema_ready()
        self.portfolio_ids: dict[str, int] = {}
        if self.storage_ready:
            for portfolio in self.portfolios:
                self.portfolio_ids[portfolio["slug"]] = self.storage.upsert_portfolio(
                    name=portfolio["name"],
                    base_currency=portfolio["base_currency"],
                    symbols=portfolio["symbols"],
                    positions=portfolio["positions"],
                    slug=portfolio["slug"],
                )
        self.portfolio_id = self.portfolio_ids.get(self.portfolio["slug"])

        self.risk = PortfolioRiskCalculator(self)
        self.decisions = DecisionPolicyEngine(self)
        self.execution = MT5ExecutionOrchestrator(self)
        self.governance = GovernanceRecorder(self)
        self.market_data = DeskMarketDataService(self)

    def require_storage_ready(self) -> None:
        if not self.storage_ready:
            raise RuntimeError("Database schema is not initialized. Run `var-project db upgrade` first.")

    def _build_risk_model_config(
        self,
        alpha: float | None,
        n_sims: int | None,
        dist: str | None,
        df_t: int | None,
        seed: int | None,
    ) -> RiskModelConfig:
        return self.risk.build_risk_model_config(alpha, n_sims, dist, df_t, seed)

    def _compute_portfolio_state(
        self,
        *,
        portfolio_slug: str | None = None,
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        config: RiskModelConfig | None = None,
        window: int | None = None,
    ) -> dict[str, Any]:
        return self.risk.compute_portfolio_state(
            portfolio_slug=portfolio_slug,
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            config=config,
            window=window,
        )

    def _compute_portfolio_state_for_positions(
        self,
        *,
        portfolio: Mapping[str, Any],
        positions_eur: Mapping[str, Any],
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        config: RiskModelConfig | None = None,
        window: int | None = None,
        snapshot_source: str = "historical",
        snapshot_timestamp: str | None = None,
    ) -> dict[str, Any]:
        return self.risk.compute_portfolio_state_for_positions(
            portfolio=portfolio,
            positions_eur=positions_eur,
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            config=config,
            window=window,
            snapshot_source=snapshot_source,
            snapshot_timestamp=snapshot_timestamp,
        )

    def _compute_portfolio_state_for_holdings(
        self,
        *,
        portfolio: Mapping[str, Any],
        holdings: Mapping[str, Any] | list[Mapping[str, Any]],
        timeframe: str | None = None,
        days: int | None = None,
        min_coverage: float | None = None,
        config: RiskModelConfig | None = None,
        window: int | None = None,
        snapshot_source: str = "historical",
        snapshot_timestamp: str | None = None,
    ) -> dict[str, Any]:
        return self.risk.compute_portfolio_state_for_holdings(
            portfolio=portfolio,
            holdings=holdings,
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            config=config,
            window=window,
            snapshot_source=snapshot_source,
            snapshot_timestamp=snapshot_timestamp,
        )

    def _compute_live_portfolio_state(self, *, portfolio: Mapping[str, Any], live: MT5LiveGateway) -> dict[str, Any]:
        return self.risk.compute_live_portfolio_state(portfolio=portfolio, live=live)

    def _evaluate_trade_decision_from_bundle(
        self,
        *,
        bundle: Mapping[str, Any],
        symbol: str,
        delta_position_eur: float,
        note: str | None,
        persist: bool,
        audit_action: str = "decision.evaluate",
    ) -> dict[str, Any]:
        return self.decisions.evaluate_trade_decision_from_bundle(
            bundle=bundle,
            symbol=symbol,
            delta_position_eur=delta_position_eur,
            note=note,
            persist=persist,
            audit_action=audit_action,
        )

    def _post_capital_after_trade(
        self,
        *,
        bundle: Mapping[str, Any],
        symbol: str,
        approved_delta_position_eur: float,
        snapshot_source: str,
    ) -> dict[str, Any]:
        return self.decisions.post_capital_after_trade(
            bundle=bundle,
            symbol=symbol,
            approved_delta_position_eur=approved_delta_position_eur,
            snapshot_source=snapshot_source,
        )

    def _build_execution_guard(
        self,
        *,
        live: MT5LiveGateway,
        terminal_status: MT5TerminalStatus,
        account: Mapping[str, Any],
        symbol: str,
        note: str | None,
        decision: Mapping[str, Any],
    ) -> tuple[ExecutionGuardDecision, dict[str, Any], dict[str, Any]]:
        return self.execution.build_execution_guard(
            live=live,
            terminal_status=terminal_status,
            account=account,
            symbol=symbol,
            note=note,
            decision=decision,
        )

    def _run_order_check_with_fill_fallback(
        self,
        *,
        live: MT5LiveGateway,
        order_request: Mapping[str, Any],
        fill_candidates: Any,
    ) -> dict[str, Any]:
        return self.execution.run_order_check_with_fill_fallback(
            live=live,
            order_request=order_request,
            fill_candidates=fill_candidates,
        )

    def _persist_live_bundle(self, *, bundle: Mapping[str, Any], portfolio_id: int, source: str) -> None:
        self.governance.persist_live_bundle(bundle=bundle, portfolio_id=portfolio_id, source=source)

    def _mt5_gateway(self):
        return self.execution.mt5_gateway()

    def _build_mt5_connector(self):
        return self.execution.build_mt5_connector()

    def _load_daily_returns(self, timeframe: str, days: int, min_coverage: float):
        return self.risk.load_daily_returns(timeframe, days, min_coverage)

    def _load_daily_returns_for_portfolio(
        self,
        portfolio: Mapping[str, Any],
        timeframe: str,
        days: int,
        min_coverage: float,
    ):
        return self.risk.load_daily_returns_for_portfolio(portfolio, timeframe, days, min_coverage)

    def _should_use_mt5_market_data(self, portfolio: Mapping[str, Any]) -> bool:
        return self.market_data.should_use_mt5_market_data(portfolio)

    def _default_timeframe(self) -> str:
        return self.risk.default_timeframe()

    def _default_days(self) -> int:
        return self.risk.default_days()

    def database_dependency(self) -> dict[str, Any]:
        return self.governance.database_dependency()

    def mt5_dependency(self) -> dict[str, Any]:
        return self.execution.mt5_dependency()

    def _positions_json(self, positions: Mapping[str, Any] | None = None) -> str:
        return self.risk.exposures_json(positions)

    def _decision_settings(self) -> dict[str, Any]:
        return self.decisions.decision_settings()

    def _decision_reference_model(self, portfolio_slug: str | None = None) -> str:
        return self.decisions.decision_reference_model(portfolio_slug)

    def _preferred_model(self, portfolio_slug: str | None = None) -> str | None:
        return self.decisions.preferred_model(portfolio_slug)

    def _resolve_portfolio_context(self, portfolio_slug: str | None = None) -> dict[str, Any]:
        if portfolio_slug is None:
            return dict(self.portfolio)
        portfolio = self.portfolio_by_slug.get(str(portfolio_slug))
        if portfolio is None:
            raise ValueError(f"Unknown portfolio '{portfolio_slug}'.")
        return dict(portfolio)

    def _resolve_portfolio_id(self, portfolio_slug: str | None = None) -> int:
        self.require_storage_ready()
        portfolio = self._resolve_portfolio_context(portfolio_slug)
        return int(self.portfolio_ids[portfolio["slug"]])

    def _alert_counts_by_portfolio(self) -> dict[str, int]:
        return self.governance.alert_counts_by_portfolio()

    def _resolve_compare_path(self, compare_path: str | None, *, portfolio_slug: str | None = None) -> Path | None:
        return self.governance.resolve_compare_path(compare_path, portfolio_slug=portfolio_slug)

    def _append_governance_sections(self, report_path: Path) -> None:
        self.governance.append_governance_sections(report_path)
