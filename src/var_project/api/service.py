from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from var_project.api.services import (
    DeskAnalyticsService,
    DeskCapitalService,
    DeskMt5Service,
    DeskReadService,
    DeskServiceRuntime,
    DeskTradingService,
)
from var_project.connectors.mt5_connector import MT5Connector


class DeskApiService:
    def __init__(
        self,
        root: Path,
        mt5_connector_factory: type[MT5Connector] | None = None,
        *,
        bootstrap_storage: bool = False,
    ):
        self.runtime = DeskServiceRuntime(
            root,
            mt5_connector_factory=mt5_connector_factory,
            bootstrap_storage=bootstrap_storage,
        )
        self.reads = DeskReadService(self.runtime)
        self.analytics = DeskAnalyticsService(self.runtime)
        self.capital = DeskCapitalService(self.runtime)
        self.trading = DeskTradingService(self.runtime)
        self.mt5 = DeskMt5Service(self.runtime)
        self.market = self.runtime.market_data

        self.root = self.runtime.root
        self.raw_config = self.runtime.raw_config
        self.limits_config = self.runtime.limits_config
        self.data_defaults = self.runtime.data_defaults
        self.risk_defaults = self.runtime.risk_defaults
        self.mt5_config = self.runtime.mt5_config
        self.storage = self.runtime.storage
        self.portfolios = self.runtime.portfolios
        self.portfolio = self.runtime.portfolio
        self.portfolio_by_slug = self.runtime.portfolio_by_slug
        self.desk = self.runtime.desk
        self.portfolio_ids = self.runtime.portfolio_ids
        self.portfolio_id = self.runtime.portfolio_id

    def health(self) -> dict[str, Any]:
        return self.reads.health()

    def jobs_status(self) -> dict[str, Any]:
        return self.reads.jobs_status()

    def list_portfolios(self) -> list[dict[str, Any]]:
        return self.reads.list_portfolios()

    def list_desks(self) -> list[dict[str, Any]]:
        return self.reads.list_desks()

    def latest_snapshot(self, *, source: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_snapshot(source=source, portfolio_slug=portfolio_slug)

    def latest_backtest(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_backtest(portfolio_slug=portfolio_slug)

    def latest_validation(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_validation(portfolio_slug=portfolio_slug)

    def recent_alerts(self, *, limit: int = 25) -> list[dict[str, Any]]:
        return self.reads.recent_alerts(limit=limit)

    def recent_decisions(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_decisions(limit=limit, portfolio_slug=portfolio_slug)

    def latest_capital(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        return self.reads.latest_capital(portfolio_slug=portfolio_slug)

    def capital_history(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.capital_history(limit=limit, portfolio_slug=portfolio_slug)

    def recent_execution_results(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_execution_results(limit=limit, portfolio_slug=portfolio_slug)

    def recent_execution_fills(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_execution_fills(limit=limit, portfolio_slug=portfolio_slug)

    def recent_audit_events(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_audit_events(limit=limit, portfolio_slug=portfolio_slug)

    def report_decision_history(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.report_decision_history(limit=limit, portfolio_slug=portfolio_slug)

    def report_capital_history(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.report_capital_history(limit=limit, portfolio_slug=portfolio_slug)

    def portfolio_capital(self, portfolio_slug: str) -> dict[str, Any]:
        return self.reads.portfolio_capital(portfolio_slug)

    def desk_overview(self, *, desk_slug: str | None = None) -> dict[str, Any]:
        return self.reads.desk_overview(desk_slug=desk_slug)

    def latest_artifact(self, artifact_type: str) -> dict[str, Any] | None:
        return self.reads.latest_artifact(artifact_type)

    def latest_model_comparison(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_model_comparison(portfolio_slug=portfolio_slug)

    def latest_risk_attribution(
        self,
        *,
        source: str = "historical",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        return self.reads.latest_risk_attribution(source=source, portfolio_slug=portfolio_slug)

    def latest_risk_budget(
        self,
        *,
        source: str = "historical",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        return self.reads.latest_risk_budget(source=source, portfolio_slug=portfolio_slug)

    def mt5_status(self) -> dict[str, Any]:
        return self.mt5.mt5_status()

    def mt5_account(self) -> dict[str, Any]:
        return self.mt5.mt5_account()

    def mt5_positions(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.mt5.mt5_positions(portfolio_slug=portfolio_slug)

    def mt5_orders(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.mt5.mt5_orders(portfolio_slug=portfolio_slug)

    def mt5_live_state(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        return self.mt5.live_state(portfolio_slug=portfolio_slug)

    def mt5_live_events(
        self,
        *,
        portfolio_slug: str | None = None,
        after: int = 0,
        limit: int = 100,
        wait_seconds: float = 15.0,
    ) -> list[dict[str, Any]]:
        return self.mt5.live_events(
            portfolio_slug=portfolio_slug,
            after=after,
            limit=limit,
            wait_seconds=wait_seconds,
        )

    def mt5_history_orders(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.market.mt5_order_history(portfolio_slug=portfolio_slug, limit=limit)

    def mt5_history_deals(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.market.mt5_deal_history(portfolio_slug=portfolio_slug, limit=limit)

    def list_instruments(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.market.list_instruments(portfolio_slug=portfolio_slug)

    def live_holdings(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.market.live_holdings(portfolio_slug=portfolio_slug)

    def live_exposure(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        return self.market.live_exposure(portfolio_slug=portfolio_slug)

    def market_data_status(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        return self.market.market_data_status(portfolio_slug=portfolio_slug)

    def sync_market_data(
        self,
        *,
        portfolio_slug: str | None = None,
        days: int | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.market.sync_market_data(portfolio_slug=portfolio_slug, days=days, timeframes=timeframes)

    def reconciliation_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        return self.market.reconciliation_summary(portfolio_slug=portfolio_slug)

    def preview_execution(
        self,
        *,
        symbol: str,
        delta_position_eur: float,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        return self.mt5.preview_execution(
            symbol=symbol,
            delta_position_eur=delta_position_eur,
            note=note,
            portfolio_slug=portfolio_slug,
        )

    def submit_execution(
        self,
        *,
        symbol: str,
        delta_position_eur: float,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        return self.mt5.submit_execution(
            symbol=symbol,
            delta_position_eur=delta_position_eur,
            note=note,
            portfolio_slug=portfolio_slug,
        )

    def evaluate_trade_decision(
        self,
        *,
        symbol: str,
        delta_position_eur: float,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        return self.trading.evaluate_trade_decision(
            symbol=symbol,
            delta_position_eur=delta_position_eur,
            note=note,
            portfolio_slug=portfolio_slug,
        )

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
        return self.analytics.run_snapshot(
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            alpha=alpha,
            window=window,
            n_sims=n_sims,
            dist=dist,
            df_t=df_t,
            seed=seed,
            portfolio_slug=portfolio_slug,
        )

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
        return self.analytics.run_backtest(
            timeframe=timeframe,
            days=days,
            min_coverage=min_coverage,
            alpha=alpha,
            window=window,
            n_sims=n_sims,
            dist=dist,
            df_t=df_t,
            seed=seed,
            portfolio_slug=portfolio_slug,
        )

    def run_validation(
        self,
        *,
        compare_path: str | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        return self.analytics.run_validation(compare_path=compare_path, alpha=alpha)

    def rebalance_capital(
        self,
        *,
        portfolio_slug: str | None = None,
        total_budget_eur: float | None = None,
        reserve_ratio: float | None = None,
        reference_model: str | None = None,
        symbol_weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        return self.capital.rebalance_capital(
            portfolio_slug=portfolio_slug,
            total_budget_eur=total_budget_eur,
            reserve_ratio=reserve_ratio,
            reference_model=reference_model,
            symbol_weights=symbol_weights,
        )

    def run_report(self, *, compare_path: str | None = None, portfolio_slug: str | None = None) -> dict[str, Any]:
        return self.analytics.run_report(compare_path=compare_path, portfolio_slug=portfolio_slug)

    def latest_backtest_frame(self, *, limit: int = 400, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_backtest_frame(limit=limit, portfolio_slug=portfolio_slug)

    def latest_report_content(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_report_content(portfolio_slug=portfolio_slug)
