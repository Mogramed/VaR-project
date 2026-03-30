from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd

from var_project.execution.mt5_live import MT5LiveGateway
from var_project.market_data.transforms import compute_log_returns, intraday_to_daily_log_returns
from var_project.portfolio.holdings import aggregate_exposure_by_symbol, gross_exposure_base_ccy, normalize_holdings

if TYPE_CHECKING:
    from var_project.api.services.runtime import DeskServiceRuntime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeskMarketDataService:
    def __init__(self, runtime: "DeskServiceRuntime"):
        self.runtime = runtime
        self._direct_mt5_probe_ok: bool | None = None

    def mt5_configured(self) -> bool:
        config = self.runtime.mt5_config
        explicit = bool(
            self.runtime._has_custom_mt5_factory
            or config.agent_base_url
            or config.path
            or config.login
            or config.server
        )
        if explicit:
            return True
        if self._direct_mt5_probe_ok is not None:
            return self._direct_mt5_probe_ok
        try:
            with self.runtime._mt5_gateway() as live:
                status = live.terminal_status()
            self._direct_mt5_probe_ok = bool(status.connected)
        except Exception:
            self._direct_mt5_probe_ok = False
        return self._direct_mt5_probe_ok

    def should_use_mt5_market_data(self, portfolio: Mapping[str, Any]) -> bool:
        mode = str(portfolio.get("mode") or "offline_fixture").lower()
        if mode == "offline_fixture":
            return False
        if mode in {"live_mt5", "hybrid"}:
            return self.mt5_configured()
        return False

    def _portfolio_symbols_from_details(self, portfolio: Mapping[str, Any], details: Mapping[str, Any] | None = None) -> list[str]:
        collected = {
            str(symbol).upper()
            for symbol in list((portfolio.get("watchlist_symbols") or portfolio["symbols"]) or [])
            if str(symbol).strip()
        }
        for section in ("open_positions", "pending_orders", "order_history", "deal_history"):
            for item in list((details or {}).get(section) or []):
                symbol = str((item or {}).get("symbol") or "").upper()
                if symbol:
                    collected.add(symbol)
        return sorted(collected)

    def live_holdings(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if not self.should_use_mt5_market_data(portfolio):
            return list(portfolio.get("configured_holdings") or [])
        with self.runtime._mt5_gateway() as live:
            return [item.to_dict() for item in live.holdings(symbols=None)]

    def exposure_from_holdings(
        self,
        holdings: list[Mapping[str, Any]] | list[dict[str, Any]],
        *,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        normalized = normalize_holdings(holdings, base_currency=str(portfolio["base_currency"]))
        exposure_by_symbol = aggregate_exposure_by_symbol(normalized, base_currency=str(portfolio["base_currency"]))
        gross_exposure = gross_exposure_base_ccy(normalized, base_currency=str(portfolio["base_currency"]))
        items = []
        for symbol in sorted(exposure_by_symbol):
            matching = next((holding for holding in normalized if holding.symbol == symbol), None)
            exposure = float(exposure_by_symbol.get(symbol, 0.0))
            share = None if gross_exposure <= 1e-9 else abs(exposure) / gross_exposure
            items.append(
                {
                    "symbol": symbol,
                    "asset_class": None if matching is None else matching.asset_class,
                    "exposure_base_ccy": exposure,
                    "signed_position_eur": exposure,
                    "gross_exposure_share": share,
                }
            )
        return {
            "generated_at": _utcnow().isoformat(),
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "base_currency": str(portfolio["base_currency"]),
            "gross_exposure_base_ccy": gross_exposure,
            "items": items,
        }

    def live_exposure(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        holdings = self.live_holdings(portfolio_slug=portfolio["slug"])
        return self.exposure_from_holdings(holdings, portfolio_slug=portfolio["slug"])

    def list_instruments(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        status = self.market_data_status(portfolio_slug=portfolio["slug"])
        cached = self.runtime.storage.list_instruments(symbols=status["symbols"])
        if cached or not self.should_use_mt5_market_data(portfolio):
            return cached
        self.sync_market_data(portfolio_slug=portfolio["slug"])
        refreshed = self.market_data_status(portfolio_slug=portfolio["slug"])
        return self.runtime.storage.list_instruments(symbols=refreshed["symbols"])

    def market_data_status(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        timeframe = self.runtime._default_timeframe()
        latest_sync = self.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio["slug"])
        details = {} if latest_sync is None else dict(latest_sync.get("details") or {})
        tracked_symbols = self._portfolio_symbols_from_details(portfolio, details)
        instruments = self.runtime.storage.list_instruments(symbols=tracked_symbols)
        latest_bar_times = self.runtime.storage.latest_market_bar_times(symbols=tracked_symbols, timeframe=timeframe)
        missing_symbols = [symbol for symbol in tracked_symbols if symbol.upper() not in {item["symbol"] for item in instruments}]
        missing_bars = [symbol for symbol, ts in latest_bar_times.items() if ts is None]
        configured = self.should_use_mt5_market_data(portfolio)
        if not configured:
            status = "offline_fixture"
        elif latest_sync is None or missing_symbols or missing_bars:
            status = "incomplete"
        else:
            status = str(latest_sync.get("status") or "ok")
        return {
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "status": status,
            "configured": configured,
            "timeframe": timeframe,
            "symbols": tracked_symbols,
            "instrument_count": len(instruments),
            "latest_sync_at": None if latest_sync is None else latest_sync.get("synced_at"),
            "latest_bar_times": latest_bar_times,
            "missing_symbols": missing_symbols,
            "missing_bars": missing_bars,
            "open_positions": list(details.get("open_positions") or []),
            "pending_orders": list(details.get("pending_orders") or []),
            "details": details,
        }

    def sync_market_data(
        self,
        *,
        portfolio_slug: str | None = None,
        days: int | None = None,
        timeframes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if not self.should_use_mt5_market_data(portfolio):
            return self.market_data_status(portfolio_slug=portfolio["slug"])

        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        selected_days = int(days or self.runtime._default_days())
        selected_timeframes = [str(item).upper() for item in (timeframes or self.runtime.data_defaults["timeframes"] or [self.runtime._default_timeframe()])]
        started_at = _utcnow()
        sync_run_id = self.runtime.storage.record_market_data_sync(
            portfolio_id=portfolio_id,
            portfolio_slug=portfolio["slug"],
            mode=str(portfolio.get("mode") or "hybrid"),
            status="running",
            details={
                "symbols": self._portfolio_symbols_from_details(portfolio),
                "timeframes": selected_timeframes,
                "days": selected_days,
            },
        )

        errors: list[dict[str, Any]] = []
        instrument_payloads: list[dict[str, Any]] = []
        order_payloads: list[dict[str, Any]] = []
        deal_payloads: list[dict[str, Any]] = []
        open_positions: list[dict[str, Any]] = []
        pending_orders: list[dict[str, Any]] = []
        coverage: dict[str, dict[str, Any]] = {}

        with self.runtime._mt5_gateway() as live:
            open_positions = [item.to_dict() for item in live.holdings(symbols=None)]
            pending_orders = [item.to_dict() for item in live.pending_orders(symbols=None)]
            date_from = started_at - timedelta(days=selected_days)

            tracked_symbols = self._portfolio_symbols_from_details(
                portfolio,
                {
                    "open_positions": open_positions,
                    "pending_orders": pending_orders,
                },
            )

            for symbol in tracked_symbols:
                try:
                    instrument = live.instrument_definition(symbol).to_dict()
                    self.runtime.storage.upsert_instrument(instrument, source="mt5")
                    instrument_payloads.append(instrument)
                except Exception as exc:
                    errors.append({"scope": "instrument", "symbol": symbol, "detail": str(exc)})
                    continue

                coverage[symbol] = {}
                for timeframe in selected_timeframes:
                    try:
                        bars_per_day = live.connector.bars_per_day(timeframe)
                        bars = live.connector.fetch_last_n_bars(
                            str(symbol).upper(),
                            timeframe,
                            max(int(selected_days * bars_per_day) + int(bars_per_day), 10),
                        )
                        rows = bars.to_dict(orient="records")
                        self.runtime.storage.sync_market_bars(
                            symbol=symbol,
                            timeframe=timeframe,
                            bars=rows,
                            sync_run_id=sync_run_id,
                            source="mt5",
                        )
                        latest_bar_time = None if bars.empty else pd.to_datetime(bars["time"].iloc[-1], utc=True).isoformat()
                        coverage[symbol][timeframe] = {
                            "bars": int(len(bars)),
                            "latest_bar_time": latest_bar_time,
                        }
                    except Exception as exc:
                        errors.append({"scope": "bars", "symbol": symbol, "timeframe": timeframe, "detail": str(exc)})
                        coverage[symbol][timeframe] = {"bars": 0, "latest_bar_time": None}

            try:
                order_payloads = [item.to_dict() for item in live.order_history(date_from=date_from, date_to=started_at, symbols=None)]
                self.runtime.storage.sync_mt5_order_history(
                    order_payloads,
                    sync_run_id=sync_run_id,
                    portfolio_id=portfolio_id,
                    source="mt5",
                )
            except Exception as exc:
                errors.append({"scope": "history_orders", "detail": str(exc)})

            try:
                deal_payloads = [item.to_dict() for item in live.deal_history(date_from=date_from, date_to=started_at, symbols=None)]
                self.runtime.storage.sync_mt5_deal_history(
                    deal_payloads,
                    sync_run_id=sync_run_id,
                    portfolio_id=portfolio_id,
                    source="mt5",
                )
            except Exception as exc:
                errors.append({"scope": "history_deals", "detail": str(exc)})

            try:
                tracked_symbols = self._portfolio_symbols_from_details(
                    portfolio=portfolio,
                    details={
                        "open_positions": open_positions,
                        "pending_orders": pending_orders,
                        "order_history": order_payloads,
                        "deal_history": deal_payloads,
                    },
                )
                bundle = self.runtime._compute_portfolio_state_for_holdings(
                    portfolio={**dict(portfolio), "watchlist_symbols": tracked_symbols, "symbols": tracked_symbols},
                    holdings=open_positions,
                    timeframe=self.runtime._default_timeframe(),
                    days=selected_days,
                    min_coverage=float(self.runtime.data_defaults["min_coverage"]),
                    config=self.runtime._build_risk_model_config(None, None, None, None, None),
                    window=int(self.runtime.risk_defaults["window"]),
                    snapshot_source="mt5_live",
                    snapshot_timestamp=started_at.isoformat(),
                )
                self.runtime._persist_live_bundle(bundle=bundle, portfolio_id=portfolio_id, source="mt5_live")
            except Exception as exc:
                errors.append({"scope": "live_snapshot", "detail": str(exc)})

        completed_status = "ok" if not errors else "incomplete"
        self.runtime.storage.record_market_data_sync(
            portfolio_id=portfolio_id,
            portfolio_slug=portfolio["slug"],
            mode=str(portfolio.get("mode") or "hybrid"),
            status=completed_status,
            details={
                "symbols": tracked_symbols if "tracked_symbols" in locals() else self._portfolio_symbols_from_details(portfolio),
                "timeframes": selected_timeframes,
                "days": selected_days,
                "coverage": coverage,
                "instruments": len(instrument_payloads),
                "orders": len(order_payloads),
                "deals": len(deal_payloads),
                "open_positions": open_positions,
                "pending_orders": pending_orders,
                "order_history": order_payloads[:200],
                "deal_history": deal_payloads[:200],
                "errors": errors,
            },
        )
        return self.market_data_status(portfolio_slug=portfolio["slug"])

    def load_daily_returns_for_portfolio(
        self,
        portfolio: Mapping[str, Any],
        *,
        timeframe: str,
        days: int,
        min_coverage: float,
        ensure_sync: bool = False,
        symbols: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        if ensure_sync:
            self.sync_market_data(portfolio_slug=str(portfolio["slug"]), days=days, timeframes=[timeframe])

        frames: list[pd.DataFrame] = []
        since = _utcnow() - timedelta(days=int(days) + 5)
        selected_symbols = [str(symbol).upper() for symbol in (symbols or portfolio["symbols"])]
        for symbol in selected_symbols:
            bars = self.runtime.storage.market_bars(symbol=symbol, timeframe=timeframe, since=since)
            if not bars:
                raise RuntimeError(
                    f"MT5 market data is incomplete for {symbol} {timeframe}. Run /market-data/sync before analytics."
                )
            frame = pd.DataFrame(bars)
            frame["time"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
            intraday = compute_log_returns(frame[["time", "close"]], price_col="close")
            daily = intraday_to_daily_log_returns(intraday[["time", "log_return"]], timeframe=timeframe, min_coverage=min_coverage)
            if daily.empty:
                raise RuntimeError(
                    f"MT5 market data coverage is insufficient for {symbol} {timeframe}. Run /market-data/sync again."
                )
            daily[f"{symbol}_ret"] = np.expm1(daily["daily_log_return"].astype(float))
            frames.append(daily[["date", f"{symbol}_ret"]])

        merged: pd.DataFrame | None = None
        for frame in frames:
            merged = frame if merged is None else merged.merge(frame, on="date", how="inner")
        if merged is None:
            return pd.DataFrame(columns=["date", *selected_symbols])

        merged = merged.sort_values("date").reset_index(drop=True)
        out = pd.DataFrame({"date": merged["date"]})
        for symbol in selected_symbols:
            out[str(symbol)] = merged[f"{symbol}_ret"].astype(float)
        return out

    def mt5_order_history(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        rows = self.runtime.storage.recent_mt5_order_history(limit=limit, portfolio_slug=portfolio["slug"])
        if rows or not self.should_use_mt5_market_data(portfolio):
            return rows
        self.sync_market_data(portfolio_slug=portfolio["slug"])
        return self.runtime.storage.recent_mt5_order_history(limit=limit, portfolio_slug=portfolio["slug"])

    def mt5_deal_history(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        rows = self.runtime.storage.recent_mt5_deal_history(limit=limit, portfolio_slug=portfolio["slug"])
        if rows or not self.should_use_mt5_market_data(portfolio):
            return rows
        self.sync_market_data(portfolio_slug=portfolio["slug"])
        return self.runtime.storage.recent_mt5_deal_history(limit=limit, portfolio_slug=portfolio["slug"])

    def _build_reconciliation_summary(
        self,
        *,
        portfolio: Mapping[str, Any],
        market_status: Mapping[str, Any],
        holdings: list[dict[str, Any]],
        pending_orders: list[dict[str, Any]],
        order_history: list[dict[str, Any]],
        deal_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        exposure_payload = self.exposure_from_holdings(holdings, portfolio_slug=portfolio["slug"])
        live_exposure = {
            str(item["symbol"]).upper(): float(item["exposure_base_ccy"])
            for item in list(exposure_payload.get("items") or [])
        }
        latest_snapshot = (
            self.runtime.storage.latest_snapshot(source="mt5_live", portfolio_slug=portfolio["slug"])
            or self.runtime.storage.latest_snapshot(source="historical", portfolio_slug=portfolio["slug"])
        )
        snapshot_payload = {} if latest_snapshot is None else dict(latest_snapshot.get("payload") or {})
        desk_holdings = list(snapshot_payload.get("holdings") or portfolio.get("configured_holdings") or [])
        desk_exposure = dict(
            snapshot_payload.get("exposure_by_symbol")
            or snapshot_payload.get("positions_eur")
            or aggregate_exposure_by_symbol(desk_holdings, base_currency=str(portfolio["base_currency"]))
        )

        executions = self.runtime.storage.recent_execution_results(limit=50, portfolio_slug=portfolio["slug"])
        fills = self.runtime.storage.recent_execution_fills(limit=200, portfolio_slug=portfolio["slug"])
        acknowledgement_map = {
            str(item.get("symbol") or "").upper(): item
            for item in self.runtime.storage.reconciliation_acknowledgements(portfolio_slug=portfolio["slug"])
        }
        seen_order_tickets = {int(item["ticket"]) for item in order_history if item.get("ticket") is not None}
        seen_deal_tickets = {int(item["ticket"]) for item in deal_history if item.get("ticket") is not None}
        manual_events = sum(1 for item in order_history if item.get("is_manual")) + sum(
            1 for item in deal_history if item.get("is_manual")
        )
        unmatched_execution_count = 0
        status_counts: dict[str, int] = {}
        for execution in executions:
            order_ticket = execution.get("mt5_result", {}).get("order") or execution.get("mt5_order_ticket")
            deal_ticket = execution.get("mt5_result", {}).get("deal") or execution.get("mt5_deal_ticket")
            reconciliation_status = str(execution.get("reconciliation_status") or "")
            if not reconciliation_status:
                if order_ticket is None and deal_ticket is None:
                    reconciliation_status = "pending_broker"
                elif order_ticket is not None and int(order_ticket) not in seen_order_tickets and int(deal_ticket or -1) not in seen_deal_tickets:
                    reconciliation_status = "pending_broker"
                else:
                    fill_ratio = float(execution.get("fill_ratio") or 0.0)
                    reconciliation_status = "partial_fill" if 0.0 < fill_ratio < 0.999 else "match"
                execution["reconciliation_status"] = reconciliation_status
            status_counts[reconciliation_status] = int(status_counts.get(reconciliation_status, 0) + 1)
            if reconciliation_status != "match":
                unmatched_execution_count += 1

        symbol_set = {
            str(symbol).upper()
            for symbol in (
                list(desk_exposure.keys())
                + list(live_exposure.keys())
                + [item.get("symbol") for item in pending_orders]
                + [item.get("symbol") for item in order_history]
                + [item.get("symbol") for item in deal_history]
                + list(portfolio.get("watchlist_symbols") or portfolio["symbols"])
            )
            if str(symbol or "").strip()
        }
        live_volume_by_symbol: dict[str, float] = {}
        live_asset_class: dict[str, str | None] = {}
        for item in holdings:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            signed_lots = float(item.get("volume_lots") or 0.0)
            if str(item.get("side") or "").upper() == "SELL":
                signed_lots *= -1.0
            live_volume_by_symbol[symbol] = float(live_volume_by_symbol.get(symbol, 0.0) + signed_lots)
            live_asset_class[symbol] = item.get("asset_class")
        desk_volume_by_symbol: dict[str, float] = {}
        for item in desk_holdings:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            signed_lots = float(item.get("volume_lots") or 0.0)
            if str(item.get("side") or "").upper() == "SELL":
                signed_lots *= -1.0
            desk_volume_by_symbol[symbol] = float(desk_volume_by_symbol.get(symbol, 0.0) + signed_lots)

        mismatches = []
        for symbol in sorted(symbol_set):
            desk_value = float(desk_exposure.get(symbol, 0.0))
            live_value = float(live_exposure.get(symbol, 0.0))
            diff = live_value - desk_value
            pending = next((item for item in pending_orders if str(item.get("symbol") or "").upper() == symbol), None)
            order = next((item for item in order_history if str(item.get("symbol") or "").upper() == symbol), None)
            deal = next((item for item in deal_history if str(item.get("symbol") or "").upper() == symbol), None)
            if abs(diff) <= 1e-6 and pending is None:
                mismatch_status = "match"
                reason = "Desk exposure matches MT5 live holdings."
            elif abs(desk_value) <= 1e-6 and abs(live_value) > 1e-6:
                mismatch_status = "orphan_live_position"
                reason = "Live MT5 position exists without a matching desk exposure."
            elif abs(live_value) <= 1e-6 and pending is not None:
                mismatch_status = "orphan_live_order"
                reason = "MT5 has a pending order without a corresponding live position."
            else:
                mismatch_status = "desk_vs_broker_drift"
                reason = "Desk exposure and broker live exposure diverge."
            status_counts[mismatch_status] = int(status_counts.get(mismatch_status, 0) + 1)
            acknowledgement = None if mismatch_status == "match" else acknowledgement_map.get(symbol)
            mismatches.append(
                {
                    "symbol": symbol,
                    "asset_class": live_asset_class.get(symbol),
                    "desk_exposure_eur": desk_value,
                    "live_exposure_eur": live_value,
                    "difference_eur": diff,
                    "desk_volume_lots": desk_volume_by_symbol.get(symbol),
                    "live_volume_lots": live_volume_by_symbol.get(symbol),
                    "order_ticket": None if order is None else order.get("ticket"),
                    "deal_ticket": None if deal is None else deal.get("ticket"),
                    "position_id": None if deal is None else deal.get("position_id"),
                    "reason": reason,
                    "status": mismatch_status,
                    "acknowledged": acknowledgement is not None,
                    "acknowledged_at": None if acknowledgement is None else acknowledgement.get("acknowledged_at"),
                    "acknowledged_reason": None if acknowledgement is None else acknowledgement.get("reason"),
                    "acknowledged_note": None if acknowledgement is None else acknowledgement.get("operator_note"),
                }
            )

        return {
            "generated_at": _utcnow().isoformat(),
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "market_data_status": str(market_status.get("status") or "unknown"),
            "latest_sync_at": market_status.get("latest_sync_at"),
            "open_positions_count": len(holdings),
            "pending_orders_count": len(pending_orders),
            "manual_event_count": int(manual_events),
            "unmatched_execution_count": int(unmatched_execution_count),
            "status_counts": status_counts,
            "holdings": holdings,
            "mismatches": mismatches,
            "recent_execution_attempts": executions[:10],
            "recent_fills": fills[:20],
        }

    def reconciliation_summary_from_live_state(
        self,
        live_state: Mapping[str, Any],
        *,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        market_status = self.market_data_status(portfolio_slug=portfolio["slug"])
        market_status = {
            **market_status,
            "status": str(live_state.get("status") or market_status.get("status") or "unknown"),
        }
        return self._build_reconciliation_summary(
            portfolio=portfolio,
            market_status=market_status,
            holdings=list(live_state.get("holdings") or []),
            pending_orders=list(live_state.get("pending_orders") or []),
            order_history=list(live_state.get("order_history") or []),
            deal_history=list(live_state.get("deal_history") or []),
        )

    def reconciliation_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        status = self.market_data_status(portfolio_slug=portfolio["slug"])
        holdings = self.live_holdings(portfolio_slug=portfolio["slug"])
        pending_orders: list[dict[str, Any]] = list(status.get("pending_orders") or [])
        if self.should_use_mt5_market_data(portfolio):
            with self.runtime._mt5_gateway() as live:
                pending_orders = [item.to_dict() for item in live.pending_orders(symbols=None)]
        order_history = self.mt5_order_history(portfolio_slug=portfolio["slug"], limit=200)
        deal_history = self.mt5_deal_history(portfolio_slug=portfolio["slug"], limit=200)
        return self._build_reconciliation_summary(
            portfolio=portfolio,
            market_status=status,
            holdings=holdings,
            pending_orders=pending_orders,
            order_history=order_history,
            deal_history=deal_history,
        )
