from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from var_project.alerts.engine import (
    alerts_from_capital_snapshot,
    alerts_from_execution_result,
    alerts_from_live_operator_state,
    alerts_from_risk_budget,
)
from var_project.api.services.runtime import DeskServiceRuntime
from var_project.core.exceptions import MT5ConnectionError
from var_project.execution.mt5_bridge import build_empty_live_state, collect_live_state_from_connector
from var_project.execution.mt5_live import ExecutionPreview, ExecutionResult, MT5TerminalStatus


class DeskMt5Service:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime
        self._startup_import_done = False

    def _normalize_live_state(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        payload = dict(raw_state)
        payload["sequence"] = int(payload.get("sequence") or 0)
        payload["status"] = str(payload.get("status") or "ok")
        payload["connected"] = bool(payload.get("connected", True))
        payload["degraded"] = bool(payload.get("degraded", False))
        payload["stale"] = bool(payload.get("stale", False))
        payload["poll_interval_seconds"] = float(
            payload.get("poll_interval_seconds") or self.runtime.mt5_config.live_poll_seconds
        )
        payload["history_poll_interval_seconds"] = float(
            payload.get("history_poll_interval_seconds") or self.runtime.mt5_config.live_history_poll_seconds
        )
        payload["history_lookback_minutes"] = int(
            payload.get("history_lookback_minutes") or self.runtime.mt5_config.live_history_lookback_minutes
        )
        payload["generated_at"] = str(payload.get("generated_at") or datetime.now(timezone.utc).isoformat())
        payload["last_success_at"] = payload.get("last_success_at") or payload["generated_at"]
        payload["source"] = str(payload.get("source") or "mt5_agent_bridge")
        return payload

    def _check_startup_import(self, raw_state: dict[str, Any], *, portfolio_slug: str | None = None) -> None:
        if self._startup_import_done:
            return
        if not raw_state.get("connected"):
            return
        live_holdings = list(raw_state.get("holdings") or [])
        if not live_holdings:
            return
        self._startup_import_done = True

        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime.portfolio_ids.get(portfolio["slug"])
        desk_symbols = {
            str(s).upper()
            for s in (portfolio.get("watchlist_symbols") or portfolio["symbols"] or [])
        }
        imported_symbols: list[str] = []
        for holding in live_holdings:
            symbol = str((holding if isinstance(holding, dict) else {}).get("symbol") or "").upper()
            if symbol and symbol not in desk_symbols:
                imported_symbols.append(symbol)

        if imported_symbols:
            self.runtime.storage.record_audit_event(
                actor="system",
                action_type="mt5.startup_import",
                object_type="portfolio",
                payload={
                    "portfolio_slug": portfolio["slug"],
                    "imported_symbols": imported_symbols,
                    "total_live_holdings": len(live_holdings),
                    "desk_symbols": sorted(desk_symbols),
                },
                portfolio_id=portfolio_id,
            )

    def _live_portfolio_scope(
        self,
        raw_state: dict[str, Any],
        *,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        symbols = {
            str(symbol).upper()
            for symbol in list(portfolio.get("watchlist_symbols") or portfolio["symbols"] or [])
            if str(symbol).strip()
        }
        for symbol in list(raw_state.get("symbols") or []):
            normalized = str(symbol).upper()
            if normalized:
                symbols.add(normalized)
        for section in ("holdings", "pending_orders", "order_history", "deal_history"):
            for item in list(raw_state.get(section) or []):
                normalized = str((item or {}).get("symbol") or "").upper()
                if normalized:
                    symbols.add(normalized)
        scoped_symbols = sorted(symbols)
        return {
            **dict(portfolio),
            "watchlist_symbols": scoped_symbols,
            "symbols": scoped_symbols,
        }

    def _build_live_analytics(
        self,
        raw_state: dict[str, Any],
        *,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        holdings = list(raw_state.get("holdings") or [])
        if not holdings:
            return None

        portfolio = self._live_portfolio_scope(raw_state, portfolio_slug=portfolio_slug)
        bundle = self.runtime._compute_portfolio_state_for_holdings(
            portfolio=portfolio,
            holdings=holdings,
            timeframe=self.runtime._default_timeframe(),
            days=self.runtime._default_days(),
            min_coverage=float(self.runtime.data_defaults["min_coverage"]),
            config=self.runtime._build_risk_model_config(None, None, None, None, None),
            window=int(self.runtime.risk_defaults["window"]),
            snapshot_source="mt5_live_bridge",
            snapshot_timestamp=str(raw_state.get("generated_at") or datetime.now(timezone.utc).isoformat()),
        )
        sample = bundle["sample"]
        risk_budget = bundle["risk_budget"].to_dict()
        risk_budget["snapshot_source"] = "mt5_live_bridge"
        risk_budget["snapshot_timestamp"] = str(raw_state.get("generated_at") or sample.index[-1].isoformat())
        capital_usage = dict(bundle["capital"])
        capital_usage["snapshot_source"] = "mt5_live_bridge"
        capital_usage["snapshot_timestamp"] = str(raw_state.get("generated_at") or sample.index[-1].isoformat())
        return {
            "bundle": bundle,
            "risk_summary": {
                "generated_at": str(raw_state.get("generated_at") or datetime.now(timezone.utc).isoformat()),
                "portfolio_slug": portfolio["slug"],
                "portfolio_mode": portfolio.get("mode"),
                "source": "mt5_live_bridge",
                "reference_model": self.runtime._decision_reference_model(portfolio["slug"]),
                "preferred_model": self.runtime._preferred_model(portfolio["slug"]),
                "alpha": float(bundle["config"].alpha),
                "sample_size": int(bundle["snapshot"].sample_size),
                "timeframe": bundle["timeframe"],
                "days": int(bundle["days"]),
                "window": int(bundle["window"]),
                "latest_observation": sample.index[-1].isoformat(),
                "var": bundle["snapshot"].vars_dict(),
                "es": bundle["snapshot"].es_dict(),
            },
            "risk_budget": risk_budget,
            "capital_usage": capital_usage,
            "alerts": [
                *[alert.to_dict() for alert in alerts_from_risk_budget(risk_budget)],
                *[alert.to_dict() for alert in alerts_from_capital_snapshot(capital_usage)],
            ],
        }

    def _dedupe_alerts(self, alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for alert in alerts:
            normalized = dict(alert)
            key = (
                f"{normalized.get('source')}|{normalized.get('severity')}|"
                f"{normalized.get('code')}|{normalized.get('message')}|"
                f"{normalized.get('context')}"
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _live_persistence_key(self, raw_state: dict[str, Any]) -> str:
        sequence = int(raw_state.get("sequence") or 0)
        if sequence > 0:
            return f"sequence:{sequence}"
        payload = {
            "status": raw_state.get("status"),
            "connected": bool(raw_state.get("connected", False)),
            "stale": bool(raw_state.get("stale", False)),
            "symbols": list(raw_state.get("symbols") or []),
            "holdings": list(raw_state.get("holdings") or []),
            "pending_orders": list(raw_state.get("pending_orders") or []),
            "order_history": list(raw_state.get("order_history") or []),
            "deal_history": list(raw_state.get("deal_history") or []),
        }
        return f"state:{json.dumps(payload, sort_keys=True, default=str)}"

    def _persist_live_analytics_if_needed(
        self,
        *,
        raw_state: dict[str, Any],
        analytics: dict[str, Any] | None,
        portfolio_slug: str | None = None,
    ) -> None:
        if analytics is None or not self.runtime.storage_ready:
            return
        if not bool(raw_state.get("connected", False)) or bool(raw_state.get("stale", False)):
            return

        sequence = int(raw_state.get("sequence") or 0)
        persistence_key = self._live_persistence_key(raw_state)

        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        latest = self.runtime.storage.latest_snapshot(source="mt5_live_bridge", portfolio_slug=portfolio["slug"])
        latest_payload = {} if latest is None else dict(latest.get("payload") or latest)
        latest_metadata = dict(latest_payload.get("metadata") or {})
        latest_key = str(
            latest_metadata.get("live_persistence_key")
            or latest_payload.get("live_persistence_key")
            or ""
        )
        if latest_key == persistence_key:
            if latest is not None:
                self.runtime.governance.refresh_live_report_if_needed(
                    portfolio=portfolio,
                    portfolio_id=portfolio_id,
                    snapshot_payload=latest_payload,
                    snapshot_id=int(latest["id"]),
                    source="mt5_live_bridge",
                    metadata=latest_metadata,
                )
            return

        self.runtime._persist_live_bundle(
            bundle=analytics["bundle"],
            portfolio_id=portfolio_id,
            source="mt5_live_bridge",
            metadata={
                "live_sequence": sequence,
                "live_persistence_key": persistence_key,
                "bridge_status": raw_state.get("status"),
                "bridge_generated_at": raw_state.get("generated_at"),
                "bridge_source": raw_state.get("source"),
            },
            persist_alerts=False,
            persist_audit=False,
        )

    def _build_live_state_direct(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        connector = self.runtime._build_mt5_connector()
        try:
            connector.init()
            raw_state = collect_live_state_from_connector(
                connector,
                config=self.runtime.mt5_config,
                base_currency=str(portfolio["base_currency"]),
                seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                history_lookback_minutes=int(self.runtime.mt5_config.live_history_lookback_minutes),
            )
        finally:
            connector.shutdown()
        return self._normalize_live_state(raw_state)

    def _enrich_live_state(self, raw_state: dict[str, Any], *, portfolio_slug: str | None = None) -> dict[str, Any]:
        self._check_startup_import(raw_state, portfolio_slug=portfolio_slug)
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        exposure = self.runtime.market_data.exposure_from_holdings(
            list(raw_state.get("holdings") or []),
            portfolio_slug=portfolio["slug"],
        )
        reconciliation = self.runtime.market_data.reconciliation_summary_from_live_state(
            raw_state,
            portfolio_slug=portfolio["slug"],
        )
        enriched = {
            **dict(raw_state),
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "exposure": exposure,
            "reconciliation": reconciliation,
        }
        operator_alerts = [alert.to_dict() for alert in alerts_from_live_operator_state(enriched)]
        try:
            analytics = self._build_live_analytics(enriched, portfolio_slug=portfolio["slug"])
        except Exception as exc:
            analytics = None
            operator_alerts.append(
                {
                    "source": "risk_live",
                    "severity": "WARN",
                    "code": "LIVE_ANALYTICS_UNAVAILABLE",
                    "message": "Live risk/capital recalculation is currently unavailable.",
                    "context": {
                        "detail": str(exc),
                        "portfolio_slug": portfolio["slug"],
                    },
                }
            )
        if analytics is not None:
            enriched["risk_summary"] = analytics["risk_summary"]
            enriched["risk_budget"] = analytics["risk_budget"]
            enriched["capital_usage"] = analytics["capital_usage"]
            operator_alerts.extend(list(analytics["alerts"]))
            self._persist_live_analytics_if_needed(
                raw_state=enriched,
                analytics=analytics,
                portfolio_slug=portfolio["slug"],
            )
        enriched["operator_alerts"] = self._dedupe_alerts(operator_alerts)
        return enriched

    def live_state(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        connector = self.runtime._build_mt5_connector()
        try:
            connector.init()
            if hasattr(connector, "live_state"):
                raw_state = self._normalize_live_state(dict(connector.live_state()))
            else:
                raw_state = self._normalize_live_state(
                    collect_live_state_from_connector(
                        connector,
                        config=self.runtime.mt5_config,
                        base_currency=str(portfolio["base_currency"]),
                        seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                        history_lookback_minutes=int(self.runtime.mt5_config.live_history_lookback_minutes),
                    )
                )
        except Exception as exc:
            raw_state = build_empty_live_state(
                config=self.runtime.mt5_config,
                seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                status="degraded",
                connected=False,
                degraded=True,
                stale=True,
                last_error=str(exc),
            )
        finally:
            connector.shutdown()
        return self._enrich_live_state(raw_state, portfolio_slug=portfolio["slug"])

    def live_events(
        self,
        *,
        portfolio_slug: str | None = None,
        after: int = 0,
        limit: int = 100,
        wait_seconds: float = 15.0,
    ) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        connector = self.runtime._build_mt5_connector()
        try:
            connector.init()
            if hasattr(connector, "live_events"):
                events = connector.live_events(after=after, limit=limit, wait_seconds=wait_seconds)
            else:
                events = [
                    {
                        "sequence": int(after) + 1,
                        "kind": "snapshot",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "change_summary": {},
                        "state": collect_live_state_from_connector(
                            connector,
                            config=self.runtime.mt5_config,
                            base_currency=str(portfolio["base_currency"]),
                            seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                            history_lookback_minutes=int(self.runtime.mt5_config.live_history_lookback_minutes),
                        ),
                    }
                ]
        except Exception as exc:
            events = [
                {
                    "sequence": int(after) + 1,
                    "kind": "connection_error",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "change_summary": {},
                    "state": build_empty_live_state(
                        config=self.runtime.mt5_config,
                        seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                        status="degraded",
                        connected=False,
                        degraded=True,
                        stale=True,
                        last_error=str(exc),
                    ),
                }
            ]
        finally:
            connector.shutdown()
        enriched = []
        for event in events[: max(int(limit), 1)]:
            payload = dict(event)
            payload["state"] = self._enrich_live_state(
                self._normalize_live_state(dict(event.get("state") or {})),
                portfolio_slug=portfolio["slug"],
            )
            enriched.append(payload)
        return enriched

    def mt5_status(self) -> dict[str, Any]:
        try:
            with self.runtime._mt5_gateway() as live:
                return live.terminal_status().to_dict()
        except MT5ConnectionError as exc:
            return MT5TerminalStatus(
                connected=False,
                ready=False,
                execution_enabled=bool(self.runtime.mt5_config.execution_enabled),
                trade_allowed=None,
                tradeapi_disabled=None,
                company=None,
                terminal_path=self.runtime.mt5_config.path,
                data_path=None,
                commondata_path=None,
                message=str(exc),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                raw={},
            ).to_dict()

    def mt5_account(self) -> dict[str, Any]:
        with self.runtime._mt5_gateway() as live:
            return live.account_snapshot().to_dict()

    def mt5_positions(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        portfolio = None if portfolio_slug is None else self.runtime._resolve_portfolio_context(portfolio_slug)
        with self.runtime._mt5_gateway() as live:
            symbols = None
            if portfolio is not None and not self.runtime.market_data.should_use_mt5_market_data(portfolio):
                symbols = portfolio["symbols"]
            return [item.to_dict() for item in live.positions(symbols=symbols)]

    def mt5_orders(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        portfolio = None if portfolio_slug is None else self.runtime._resolve_portfolio_context(portfolio_slug)
        with self.runtime._mt5_gateway() as live:
            symbols = None
            if portfolio is not None and not self.runtime.market_data.should_use_mt5_market_data(portfolio):
                symbols = portfolio["symbols"]
            return [item.to_dict() for item in live.pending_orders(symbols=symbols)]

    def preview_execution(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        normalized_exposure_change = (
            float(exposure_change) if exposure_change is not None else float(delta_position_eur or 0.0)
        )
        with self.runtime._mt5_gateway() as live:
            terminal_status = live.terminal_status()
            account = live.account_snapshot()
            live_positions = live.positions()
            pending_orders = live.pending_orders()
            bundle = self.runtime._compute_live_portfolio_state(portfolio=portfolio, live=live)
            decision = self.runtime._evaluate_trade_decision_from_bundle(
                bundle=bundle,
                symbol=symbol,
                exposure_change=normalized_exposure_change,
                note=note,
                persist=False,
            )
            post_capital = self.runtime._post_capital_after_trade(
                bundle=bundle,
                symbol=symbol,
                approved_exposure_change=float(
                    decision.get("approved_exposure_change", decision.get("approved_delta_position_eur", 0.0))
                ),
                snapshot_source="execution_preview",
            )
            guard, order_request, order_check = self.runtime._build_execution_guard(
                live=live,
                terminal_status=terminal_status,
                account=account.to_dict(),
                symbol=symbol,
                note=note,
                decision=decision,
            )
            preview = ExecutionPreview(
                time_utc=datetime.now(timezone.utc).isoformat(),
                portfolio_slug=portfolio["slug"],
                symbol=str(symbol).upper(),
                terminal_status=terminal_status,
                account=account,
                live_positions=live_positions,
                pending_orders=pending_orders,
                risk_decision=decision,
                guard=guard,
                order_request=order_request,
                order_check=order_check,
                pre_capital=dict(bundle["capital"]),
                post_capital=post_capital,
            ).to_dict()
            self.runtime.storage.record_audit_event(
                actor="api",
                action_type="execution.preview",
                object_type="execution_preview",
                payload={
                    "portfolio_slug": portfolio["slug"],
                    "symbol": str(symbol).upper(),
                    "decision": guard.decision,
                    "risk_decision": decision.get("decision"),
                },
                portfolio_id=self.runtime._resolve_portfolio_id(portfolio["slug"]),
            )
            return preview

    def submit_execution(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        normalized_exposure_change = (
            float(exposure_change) if exposure_change is not None else float(delta_position_eur or 0.0)
        )
        with self.runtime._mt5_gateway() as live:
            terminal_status = live.terminal_status()
            account_before = live.account_snapshot()
            bundle = self.runtime._compute_live_portfolio_state(portfolio=portfolio, live=live)
            decision = self.runtime._evaluate_trade_decision_from_bundle(
                bundle=bundle,
                symbol=symbol,
                exposure_change=normalized_exposure_change,
                note=note,
                persist=True,
                audit_action="execution.guard",
            )
            guard, order_request, order_check = self.runtime._build_execution_guard(
                live=live,
                terminal_status=terminal_status,
                account=account_before.to_dict(),
                symbol=symbol,
                note=note,
                decision=decision,
            )

            executed_delta = 0.0
            mt5_result: dict[str, Any] = {}
            status = "BLOCKED" if guard.risk_decision != "REJECT" else "REJECTED"
            account_after = account_before
            positions_after = live.positions()
            fills: list[dict[str, Any]] = []
            requested_volume_lots = float(order_request.get("volume") or 0.0)
            approved_volume_lots = float(guard.volume_lots or 0.0)
            submitted_volume_lots = 0.0
            filled_volume_lots = 0.0
            remaining_volume_lots = approved_volume_lots
            fill_ratio = 0.0
            broker_status = "risk_blocked" if status == "BLOCKED" else "risk_rejected"
            position_id = None
            slippage_points = None
            reconciliation_status = "rejected_by_broker" if status == "REJECTED" else "pending_broker"
            post_capital = self.runtime._post_capital_after_trade(
                bundle=bundle,
                symbol=symbol,
                approved_exposure_change=float(
                    decision.get("approved_exposure_change", decision.get("approved_delta_position_eur", 0.0))
                ),
                snapshot_source="execution_submit",
            )

            if guard.submit_allowed:
                mt5_result = live.connector.order_send(order_request)
                retcode = int(mt5_result.get("retcode", -1))
                submitted_volume_lots = approved_volume_lots
                if retcode in {10009, 10008}:
                    status = "EXECUTED" if retcode == 10009 else "PLACED"
                    executed_delta = float(
                        getattr(guard, "executable_exposure_change", None)
                        or getattr(guard, "executable_delta_position_eur", 0.0)
                    )
                    broker_status = "filled" if retcode == 10009 else "placed"
                else:
                    status = "FAILED"
                    broker_status = "rejected"

                account_after = live.account_snapshot()
                positions_after = live.positions()

                order_ticket = mt5_result.get("order")
                deal_ticket = mt5_result.get("deal")
                history_from = datetime.now(timezone.utc) - timedelta(minutes=30)
                history_to = datetime.now(timezone.utc) + timedelta(minutes=1)
                try:
                    live_deals = [
                        item.to_dict()
                        for item in live.deal_history(
                            date_from=history_from,
                            date_to=history_to,
                            symbols=[str(symbol).upper()],
                        )
                    ]
                except Exception:
                    live_deals = []
                for deal in live_deals:
                    same_order = order_ticket is not None and int(deal.get("order_ticket") or -1) == int(order_ticket)
                    same_ticket = deal_ticket is not None and int(deal.get("ticket") or -1) == int(deal_ticket)
                    if same_order or same_ticket:
                        fills.append(
                            {
                                "symbol": str(symbol).upper(),
                                "order_ticket": order_ticket,
                                "deal_ticket": deal.get("ticket"),
                                "position_id": deal.get("position_id"),
                                "side": deal.get("side"),
                                "entry": deal.get("entry"),
                                "volume_lots": float(deal.get("volume") or 0.0),
                                "price": None if deal.get("price") is None else float(deal.get("price")),
                                "profit": None if deal.get("profit") is None else float(deal.get("profit")),
                                "commission": None if deal.get("commission") is None else float(deal.get("commission")),
                                "swap": None if deal.get("swap") is None else float(deal.get("swap")),
                                "fee": None if deal.get("fee") is None else float(deal.get("fee")),
                                "reason": deal.get("reason"),
                                "comment": deal.get("comment"),
                                "is_manual": bool(deal.get("is_manual")),
                                "time_utc": deal.get("time_utc"),
                                "raw": dict(deal),
                            }
                        )

                filled_volume_lots = float(sum(float(item.get("volume_lots") or 0.0) for item in fills))
                remaining_volume_lots = max(submitted_volume_lots - filled_volume_lots, 0.0)
                fill_ratio = 0.0 if submitted_volume_lots <= 1e-9 else filled_volume_lots / submitted_volume_lots
                if fills:
                    position_id = fills[0].get("position_id")
                    deal_price = fills[0].get("price")
                    request_price = None if order_request.get("price") is None else float(order_request.get("price"))
                    try:
                        tick_size = float(live.instrument_definition(str(symbol).upper()).tick_size or 0.0)
                    except Exception:
                        tick_size = 0.0
                    if deal_price is not None and request_price is not None:
                        raw_slippage = abs(float(deal_price) - request_price)
                        slippage_points = raw_slippage if tick_size <= 0 else raw_slippage / tick_size

                if status == "FAILED":
                    reconciliation_status = "rejected_by_broker"
                elif fill_ratio <= 1e-9:
                    reconciliation_status = "pending_broker"
                elif fill_ratio < 0.999:
                    reconciliation_status = "partial_fill"
                elif fill_ratio > 1.001:
                    reconciliation_status = "overfill_or_volume_drift"
                else:
                    reconciliation_status = "match"

                if status in {"EXECUTED", "PLACED"}:
                    live_bundle = self.runtime._compute_portfolio_state_for_holdings(
                        portfolio=portfolio,
                        holdings=[item.to_dict() for item in live.holdings(symbols=None)],
                        timeframe=bundle["timeframe"],
                        days=bundle["days"],
                        min_coverage=bundle["min_coverage"],
                        config=bundle["config"],
                        window=bundle["window"],
                        snapshot_source="mt5_live",
                        snapshot_timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    post_capital = dict(live_bundle["capital"])
                    self.runtime._persist_live_bundle(
                        bundle=live_bundle,
                        portfolio_id=portfolio_id,
                        source="mt5_live",
                    )

            result_payload = ExecutionResult(
                time_utc=datetime.now(timezone.utc).isoformat(),
                portfolio_slug=portfolio["slug"],
                symbol=str(symbol).upper(),
                status=status,
                requested_delta_position_eur=normalized_exposure_change,
                approved_delta_position_eur=float(
                    decision.get("approved_exposure_change", decision.get("approved_delta_position_eur", 0.0))
                ),
                executed_delta_position_eur=executed_delta,
                terminal_status=terminal_status,
                account_before=account_before,
                account_after=account_after,
                guard=guard,
                risk_decision=decision,
                order_request=order_request,
                order_check=order_check,
                mt5_result=mt5_result,
                positions_after=positions_after,
                post_capital=post_capital,
            ).to_dict()
            result_payload.update(
                {
                    "portfolio_id": portfolio_id,
                    "decision_id": decision.get("id"),
                    "requested_volume_lots": requested_volume_lots,
                    "approved_volume_lots": approved_volume_lots,
                    "submitted_volume_lots": submitted_volume_lots,
                    "filled_volume_lots": filled_volume_lots,
                    "remaining_volume_lots": remaining_volume_lots,
                    "fill_ratio": fill_ratio,
                    "broker_status": broker_status,
                    "position_id": position_id,
                    "slippage_points": slippage_points,
                    "reconciliation_status": reconciliation_status,
                    "mt5_order_ticket": mt5_result.get("order"),
                    "mt5_deal_ticket": mt5_result.get("deal"),
                    "fills": fills,
                }
            )
            execution_id = self.runtime.storage.record_execution_result(
                result_payload,
                portfolio_id=portfolio_id,
                decision_id=decision.get("id"),
            )
            result_payload["id"] = execution_id
            result_payload["created_at"] = result_payload["time_utc"]
            execution_alerts = alerts_from_execution_result(result_payload)
            if execution_alerts:
                self.runtime.storage.record_alerts(execution_alerts, portfolio_id=portfolio_id)
            self.runtime.storage.record_audit_event(
                actor="api",
                action_type="execution.submit",
                object_type="execution_result",
                object_id=execution_id,
                payload={
                    "portfolio_slug": portfolio["slug"],
                    "symbol": str(symbol).upper(),
                    "status": status,
                    "decision_id": decision.get("id"),
                    "mt5_result": mt5_result,
                },
                portfolio_id=portfolio_id,
            )
            return result_payload
