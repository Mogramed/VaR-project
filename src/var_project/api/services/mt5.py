from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from var_project.alerts.engine import alerts_from_execution_result
from var_project.api.services.runtime import DeskServiceRuntime
from var_project.core.exceptions import MT5ConnectionError
from var_project.execution.mt5_bridge import build_empty_live_state, collect_live_state_from_connector
from var_project.execution.mt5_live import ExecutionPreview, ExecutionResult, MT5TerminalStatus


class DeskMt5Service:
    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

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
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        exposure = self.runtime.market_data.exposure_from_holdings(
            list(raw_state.get("holdings") or []),
            portfolio_slug=portfolio["slug"],
        )
        reconciliation = self.runtime.market_data.reconciliation_summary_from_live_state(
            raw_state,
            portfolio_slug=portfolio["slug"],
        )
        return {
            **dict(raw_state),
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "exposure": exposure,
            "reconciliation": reconciliation,
        }

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
        delta_position_eur: float,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        with self.runtime._mt5_gateway() as live:
            terminal_status = live.terminal_status()
            account = live.account_snapshot()
            live_positions = live.positions()
            pending_orders = live.pending_orders()
            bundle = self.runtime._compute_live_portfolio_state(portfolio=portfolio, live=live)
            decision = self.runtime._evaluate_trade_decision_from_bundle(
                bundle=bundle,
                symbol=symbol,
                delta_position_eur=delta_position_eur,
                note=note,
                persist=False,
            )
            post_capital = self.runtime._post_capital_after_trade(
                bundle=bundle,
                symbol=symbol,
                approved_delta_position_eur=float(decision.get("approved_delta_position_eur", 0.0)),
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
        delta_position_eur: float,
        note: str | None = None,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        with self.runtime._mt5_gateway() as live:
            terminal_status = live.terminal_status()
            account_before = live.account_snapshot()
            bundle = self.runtime._compute_live_portfolio_state(portfolio=portfolio, live=live)
            decision = self.runtime._evaluate_trade_decision_from_bundle(
                bundle=bundle,
                symbol=symbol,
                delta_position_eur=delta_position_eur,
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
                approved_delta_position_eur=float(decision.get("approved_delta_position_eur", 0.0)),
                snapshot_source="execution_submit",
            )

            if guard.submit_allowed:
                mt5_result = live.connector.order_send(order_request)
                retcode = int(mt5_result.get("retcode", -1))
                submitted_volume_lots = approved_volume_lots
                if retcode in {10009, 10008}:
                    status = "EXECUTED" if retcode == 10009 else "PLACED"
                    executed_delta = float(guard.executable_delta_position_eur)
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
                requested_delta_position_eur=float(delta_position_eur),
                approved_delta_position_eur=float(decision.get("approved_delta_position_eur", 0.0)),
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
