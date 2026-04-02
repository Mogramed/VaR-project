from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from var_project.desk.overview import build_desk_snapshot
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
        try:
            live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        except Exception:
            live_state = None
        if live_state is not None and live_state.get("capital_usage") is not None:
            return dict(live_state["capital_usage"])
        return self.reads.latest_capital(portfolio_slug=portfolio_slug)

    def capital_history(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def recent_execution_results(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_execution_results(limit=limit, portfolio_slug=portfolio_slug)

    def recent_execution_fills(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_execution_fills(limit=limit, portfolio_slug=portfolio_slug)

    def recent_audit_events(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_audit_events(limit=limit, portfolio_slug=portfolio_slug)

    def report_decision_history(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.report_decision_history(limit=limit, portfolio_slug=portfolio_slug)

    def report_capital_history(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.report_capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def portfolio_capital(self, portfolio_slug: str) -> dict[str, Any]:
        return self.reads.portfolio_capital(portfolio_slug)

    def desk_overview(self, *, desk_slug: str | None = None) -> dict[str, Any]:
        if desk_slug is not None and desk_slug != self.desk.slug:
            raise ValueError(f"Unknown desk '{desk_slug}'.")
        portfolio_map = {portfolio["slug"]: portfolio for portfolio in self.portfolios}
        snapshots: list[dict[str, Any]] = []
        alert_counts = self.runtime._alert_counts_by_portfolio()
        for portfolio in self.portfolios:
            live_alert_count = 0
            try:
                live_state = self.mt5.live_state(portfolio_slug=portfolio["slug"])
            except Exception:
                live_state = None
            if live_state is not None:
                live_alert_count = len(list(live_state.get("operator_alerts") or []))
                if live_state.get("capital_usage") is not None:
                    snapshots.append(dict(live_state["capital_usage"]))
                    if live_alert_count:
                        alert_counts[portfolio["slug"]] = int(alert_counts.get(portfolio["slug"], 0) + live_alert_count)
                    continue
            snapshots.append(self.reads.latest_capital(portfolio_slug=portfolio["slug"]))
            if live_alert_count:
                alert_counts[portfolio["slug"]] = int(alert_counts.get(portfolio["slug"], 0) + live_alert_count)
        return build_desk_snapshot(self.desk.to_dict(), snapshots, portfolio_map, alerts_by_portfolio=alert_counts).to_dict()

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
        live_state = self.mt5.live_state(portfolio_slug=self.portfolio["slug"])
        if live_state.get("terminal_status") is not None:
            return dict(live_state["terminal_status"])
        return self.mt5.mt5_status()

    def mt5_account(self) -> dict[str, Any]:
        live_state = self.mt5.live_state(portfolio_slug=self.portfolio["slug"])
        if live_state.get("account") is not None:
            return dict(live_state["account"])
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
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        live_orders = list(live_state.get("order_history") or [])
        if live_orders:
            return live_orders[: max(int(limit), 1)]
        return self.market.mt5_order_history(portfolio_slug=portfolio_slug, limit=limit)

    def mt5_history_deals(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        live_deals = list(live_state.get("deal_history") or [])
        if live_deals:
            return live_deals[: max(int(limit), 1)]
        return self.market.mt5_deal_history(portfolio_slug=portfolio_slug, limit=limit)

    def list_instruments(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.market.list_instruments(portfolio_slug=portfolio_slug)

    def live_holdings(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        live_holdings = list(live_state.get("holdings") or [])
        if live_holdings:
            return live_holdings
        return self.market.live_holdings(portfolio_slug=portfolio_slug)

    def live_exposure(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        if live_state.get("exposure") is not None:
            return dict(live_state["exposure"])
        return self.market.live_exposure(portfolio_slug=portfolio_slug)

    def risk_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        try:
            live_state = self.mt5.live_state(portfolio_slug=portfolio["slug"])
        except Exception:
            live_state = None
        if live_state is not None and live_state.get("risk_summary") is not None:
            return dict(live_state["risk_summary"])

        snapshot = None
        for source in ("mt5_live_bridge", "mt5_live", "historical"):
            snapshot = self.reads.latest_snapshot(source=source, portfolio_slug=portfolio["slug"])
            if snapshot is not None:
                break
        if snapshot is None:
            return None
        payload = dict(snapshot.get("payload") or {})
        return {
            "generated_at": payload.get("time_utc") or snapshot.get("created_at"),
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "source": snapshot.get("source") or "historical",
            "reference_model": payload.get("model") or self.runtime._decision_reference_model(portfolio["slug"]),
            "preferred_model": payload.get("preferred_model"),
            "alpha": payload.get("alpha") or self.runtime.risk_defaults["alpha"],
            "sample_size": int(payload.get("sample_size") or 0),
            "timeframe": payload.get("timeframe"),
            "days": payload.get("days"),
            "window": payload.get("window"),
            "latest_observation": payload.get("time_utc"),
            "var": dict(payload.get("var") or {}),
            "es": dict(payload.get("es") or {}),
            "risk_surface": dict(payload.get("risk_surface") or {}),
            "headline_risk": list(payload.get("headline_risk") or []),
            "stress_surface": dict(payload.get("stress_surface") or {}),
            "data_quality": dict(payload.get("data_quality") or {}),
            "model_diagnostics": dict(payload.get("model_diagnostics") or {}),
            "risk_nowcast": dict(payload.get("risk_nowcast") or {}),
            "microstructure": dict(payload.get("microstructure") or {}),
            "tick_quality": dict(payload.get("tick_quality") or {}),
            "pnl_explain": dict(payload.get("pnl_explain") or {}),
        }

    def risk_contributions(self, *, portfolio_slug: str | None = None, source: str | None = None) -> dict[str, Any] | None:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if source:
            return self.reads.latest_risk_attribution(source=source, portfolio_slug=portfolio["slug"])
        for candidate in ("mt5_live_bridge", "mt5_live", "historical"):
            payload = self.reads.latest_risk_attribution(source=candidate, portfolio_slug=portfolio["slug"])
            if payload is not None:
                return payload
        return None

    def market_data_status(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        payload = self.market.market_data_status(portfolio_slug=portfolio_slug)
        try:
            live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        except Exception:
            live_state = None
        if live_state is not None:
            payload = {
                **payload,
                "live_bridge_status": live_state.get("status"),
                "live_bridge_connected": live_state.get("connected"),
                "live_bridge_stale": live_state.get("stale"),
                "live_bridge_generated_at": live_state.get("generated_at"),
                "live_bridge_sequence": live_state.get("sequence"),
            }
        return payload

    def sync_market_data(
        self,
        *,
        portfolio_slug: str | None = None,
        days: int | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.market.sync_market_data(portfolio_slug=portfolio_slug, days=days, timeframes=timeframes)

    def reconciliation_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        if live_state.get("reconciliation") is not None:
            return dict(live_state["reconciliation"])
        return self.market.reconciliation_summary(portfolio_slug=portfolio_slug)

    def preview_execution(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        return self.mt5.preview_execution(
            symbol=symbol,
            exposure_change=exposure_change,
            delta_position_eur=delta_position_eur,
            note=note,
            portfolio_slug=portfolio_slug,
        )

    def submit_execution(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        return self.mt5.submit_execution(
            symbol=symbol,
            exposure_change=exposure_change,
            delta_position_eur=delta_position_eur,
            note=note,
            portfolio_slug=portfolio_slug,
        )

    def evaluate_trade_decision(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        return self.trading.evaluate_trade_decision(
            symbol=symbol,
            exposure_change=exposure_change,
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

    def acknowledge_reconciliation_mismatch(
        self,
        *,
        portfolio_slug: str | None = None,
        symbol: str,
        reason: str = "",
        operator_note: str = "",
        incident_status: str | None = None,
        resolution_note: str = "",
    ) -> dict[str, Any]:
        return self.update_reconciliation_incident(
            portfolio_slug=portfolio_slug,
            symbol=symbol,
            reason=reason,
            operator_note=operator_note,
            incident_status=incident_status or "acknowledged",
            resolution_note=resolution_note,
            audit_action="reconciliation.acknowledge",
        )

    def reconciliation_incidents(
        self,
        *,
        portfolio_slug: str | None = None,
        symbol: str | None = None,
        incident_status: str | None = None,
        include_resolved: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        normalized_symbol = None if symbol is None else str(symbol).upper().strip()
        normalized_status = None if incident_status is None else str(incident_status).strip().lower()
        if normalized_status is not None and normalized_status not in {"acknowledged", "investigating", "resolved"}:
            raise ValueError(f"Unsupported reconciliation incident status '{incident_status}'.")
        return self.storage.reconciliation_acknowledgements(
            portfolio_slug=portfolio["slug"],
            symbol=normalized_symbol,
            incident_status=normalized_status,
            include_resolved=include_resolved,
            limit=limit,
        )

    def update_reconciliation_incident(
        self,
        *,
        portfolio_slug: str | None = None,
        symbol: str,
        reason: str = "",
        operator_note: str = "",
        incident_status: str = "acknowledged",
        resolution_note: str = "",
        audit_action: str = "reconciliation.incident.update",
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.portfolio_ids.get(portfolio["slug"])
        normalized_symbol = str(symbol or "").upper().strip()
        normalized_incident_status = str(incident_status or "acknowledged").strip().lower()
        if normalized_incident_status not in {"acknowledged", "investigating", "resolved"}:
            raise ValueError(f"Unsupported reconciliation incident status '{incident_status}'.")
        summary = self.reconciliation_summary(portfolio_slug=portfolio["slug"])
        mismatch = next(
            (
                item
                for item in list(summary.get("mismatches") or [])
                if str(item.get("symbol") or "").upper() == normalized_symbol
            ),
            None,
        )
        existing_incident = next(
            (
                item
                for item in self.storage.reconciliation_acknowledgements(portfolio_slug=portfolio["slug"])
                if str(item.get("symbol") or "").upper() == normalized_symbol
            ),
            None,
        )
        if mismatch is None and existing_incident is None:
            raise ValueError(f"No reconciliation entry exists for symbol {normalized_symbol}.")
        if mismatch is not None and str(mismatch.get("status") or "").lower() == "match" and existing_incident is None:
            raise ValueError(f"Symbol {normalized_symbol} is already reconciled.")

        acknowledgement_id = self.storage.upsert_reconciliation_acknowledgement(
            portfolio_id=portfolio_id,
            symbol=normalized_symbol,
            reason=reason,
            operator_note=operator_note,
            mismatch_status=None if mismatch is None else str(mismatch.get("status") or ""),
            incident_status=normalized_incident_status,
            resolution_note=resolution_note,
            payload={
                "portfolio_slug": portfolio["slug"],
                "desk_exposure_eur": None if mismatch is None else mismatch.get("desk_exposure_eur"),
                "live_exposure_eur": None if mismatch is None else mismatch.get("live_exposure_eur"),
                "difference_eur": None if mismatch is None else mismatch.get("difference_eur"),
                "order_ticket": None if mismatch is None else mismatch.get("order_ticket"),
                "deal_ticket": None if mismatch is None else mismatch.get("deal_ticket"),
                "position_id": None if mismatch is None else mismatch.get("position_id"),
            },
        )
        audit_id = self.storage.record_audit_event(
            actor="operator",
            action_type=audit_action,
            object_type="reconciliation_mismatch",
            payload={
                "symbol": normalized_symbol,
                "reason": reason,
                "operator_note": operator_note,
                "incident_status": normalized_incident_status,
                "resolution_note": resolution_note,
                "portfolio_slug": portfolio["slug"],
                "mismatch_status": None if mismatch is None else mismatch.get("status"),
                "acknowledgement_id": acknowledgement_id,
            },
            portfolio_id=portfolio_id,
        )
        baseline_snapshot_id = None
        if normalized_incident_status == "resolved":
            baseline_snapshot_id = self.mt5.accept_reconciliation_baseline(
                portfolio_slug=portfolio["slug"],
                symbol=normalized_symbol,
                reason=reason,
                operator_note=resolution_note or operator_note,
            )
        acknowledgement = next(
            (
                item
                for item in self.storage.reconciliation_acknowledgements(portfolio_slug=portfolio["slug"])
                if str(item.get("symbol") or "").upper() == normalized_symbol
            ),
            None,
        )
        return {
            "acknowledged": True,
            "symbol": normalized_symbol,
            "audit_event_id": audit_id,
            "acknowledgement_id": acknowledgement_id,
            "mismatch_status": None if mismatch is None else mismatch.get("status"),
            "incident_status": normalized_incident_status,
            "baseline_snapshot_id": baseline_snapshot_id,
            "acknowledgement": acknowledgement,
        }

    def run_stress_test(
        self,
        *,
        portfolio_slug: str | None = None,
        scenarios: list[dict[str, Any]] | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        return self.analytics.run_stress_test(
            portfolio_slug=portfolio_slug,
            scenarios=scenarios,
            alpha=alpha,
        )

    def latest_backtest_frame(self, *, limit: int = 400, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_backtest_frame(limit=limit, portfolio_slug=portfolio_slug)

    def latest_report_content(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_report_content(portfolio_slug=portfolio_slug)
