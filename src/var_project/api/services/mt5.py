from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

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
        self._last_tick_archive_sequence: dict[str, int] = {}

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
        pending_orders = list(raw_state.get("pending_orders") or [])
        order_history = list(raw_state.get("order_history") or [])
        deal_history = list(raw_state.get("deal_history") or [])
        discovered_symbols = {
            str(symbol).upper()
            for symbol in list(raw_state.get("symbols") or [])
            if str(symbol).strip()
        }
        for section in (live_holdings, pending_orders, order_history, deal_history):
            for item in section:
                symbol = str((item if isinstance(item, dict) else {}).get("symbol") or "").upper()
                if symbol:
                    discovered_symbols.add(symbol)
        if not discovered_symbols and not live_holdings:
            return
        self._startup_import_done = True

        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime.portfolio_ids.get(portfolio["slug"])
        desk_symbols = {
            str(s).upper()
            for s in (portfolio.get("watchlist_symbols") or portfolio["symbols"] or [])
        }
        imported_symbols = sorted(symbol for symbol in discovered_symbols if symbol not in desk_symbols)

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
        try:
            self.runtime.market_data.sync_market_data(
                portfolio_slug=portfolio["slug"],
                days=self.runtime.market_data.history_backfill_days(),
                timeframes=self.runtime.market_data.startup_sync_timeframes(),
            )
        except Exception as exc:
            if portfolio_id is not None:
                self.runtime.storage.record_audit_event(
                    actor="system",
                    action_type="mt5.startup_sync_failed",
                    object_type="portfolio",
                    payload={
                        "portfolio_slug": portfolio["slug"],
                        "detail": str(exc),
                        "imported_symbols": imported_symbols,
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
                "risk_surface": dict(bundle["risk_surface"]),
                "headline_risk": list(bundle["headline_risk"]),
                "stress_surface": dict(bundle["stress_surface"]),
                "data_quality": dict(bundle["data_quality"]),
                "model_diagnostics": dict(bundle["risk_surface"].get("model_diagnostics") or {}),
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

    def _archive_live_ticks_if_needed(
        self,
        raw_state: Mapping[str, Any],
        *,
        portfolio_slug: str,
    ) -> dict[str, Any]:
        symbols = sorted(
            {
                str(symbol).upper()
                for symbol in list(raw_state.get("symbols") or []) + list(dict(raw_state.get("ticks") or {}).keys())
                if str(symbol or "").strip()
            }
        )
        if not self.runtime.storage_ready:
            return self.runtime.market_data.tick_archive_summary(symbols=symbols)
        if not bool(raw_state.get("connected", False)):
            return self.runtime.market_data.tick_archive_summary(symbols=symbols)
        sequence = int(raw_state.get("sequence") or 0)
        if sequence > 0 and self._last_tick_archive_sequence.get(portfolio_slug) == sequence:
            return self.runtime.market_data.tick_archive_summary(symbols=symbols)
        archived = self.runtime.market_data.archive_live_ticks_from_state(raw_state, portfolio_slug=portfolio_slug)
        if sequence > 0:
            self._last_tick_archive_sequence[portfolio_slug] = sequence
        return dict(archived.get("summary") or {})

    def _headline_risk_point(
        self,
        headline_risk: list[Mapping[str, Any]] | None,
        *,
        alpha: float,
        horizon_days: int,
        is_stressed: bool | None = None,
    ) -> dict[str, Any] | None:
        candidates = [dict(item) for item in list(headline_risk or [])]
        for item in candidates:
            same_alpha = abs(float(item.get("alpha") or 0.0) - float(alpha)) < 1e-9
            same_horizon = int(item.get("horizon_days") or 0) == int(horizon_days)
            if not same_alpha or not same_horizon:
                continue
            if is_stressed is None or bool(item.get("is_stressed")) == bool(is_stressed):
                return item
        return None if not candidates else candidates[0]

    def _build_microstructure(
        self,
        raw_state: Mapping[str, Any],
        *,
        tick_archive: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ticks = dict(raw_state.get("ticks") or {})
        archive_items = {
            str(item.get("symbol") or "").upper(): dict(item)
            for item in list(tick_archive.get("symbols") or [])
            if str(item.get("symbol") or "").strip()
        }
        symbols = sorted(set(ticks.keys()) | set(archive_items.keys()))
        items: list[dict[str, Any]] = []
        spread_bps_values: list[float] = []
        vol_values: list[float] = []
        widest_symbol: str | None = None
        widest_spread_bps: float | None = None
        regime = str(dict(tick_archive.get("microstructure") or {}).get("regime") or "incomplete")
        healthy = 0
        stale = 0
        incomplete = 0

        for symbol in symbols:
            tick = dict(ticks.get(symbol) or {})
            archive_item = archive_items.get(symbol, {})
            bid = tick.get("bid")
            ask = tick.get("ask")
            last = tick.get("last")
            mid = archive_item.get("mid")
            spread = archive_item.get("spread")
            spread_bps = archive_item.get("spread_bps")
            if bid is not None and ask is not None:
                bid = float(bid)
                ask = float(ask)
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else mid
                spread = float(ask - bid)
                spread_bps = None if not mid else float((spread / mid) * 10_000.0)
            elif last is not None and mid is None:
                mid = float(last)
            if spread_bps is not None:
                spread_bps_values.append(float(spread_bps))
                if widest_spread_bps is None or float(spread_bps) > widest_spread_bps:
                    widest_spread_bps = float(spread_bps)
                    widest_symbol = symbol
            realized_vol_30m = archive_item.get("realized_vol_30m")
            if realized_vol_30m is not None:
                vol_values.append(float(realized_vol_30m))
            quality = str(archive_item.get("tick_quality") or "incomplete")
            if quality == "healthy":
                healthy += 1
            elif quality == "stale":
                stale += 1
            else:
                incomplete += 1
            item_regime = str(archive_item.get("regime") or "incomplete")
            if item_regime == "stressed":
                regime = "stressed"
            elif item_regime == "volatile" and regime != "stressed":
                regime = "volatile"
            elif item_regime == "normal" and regime == "incomplete":
                regime = "normal"
            items.append(
                {
                    "symbol": symbol,
                    "mid": None if mid is None else float(mid),
                    "spread": None if spread is None else float(spread),
                    "spread_bps": None if spread_bps is None else float(spread_bps),
                    "latest_tick_at": tick.get("time_utc") or archive_item.get("latest_tick_at"),
                    "tick_count_5m": int(archive_item.get("tick_count_5m") or 0),
                    "tick_count_30m": int(archive_item.get("tick_count_30m") or 0),
                    "tick_count_1h": int(archive_item.get("tick_count_1h") or 0),
                    "realized_vol_5m": archive_item.get("realized_vol_5m"),
                    "realized_vol_30m": archive_item.get("realized_vol_30m"),
                    "realized_vol_1h": archive_item.get("realized_vol_1h"),
                    "row_count": int(archive_item.get("row_count") or 0),
                    "partition_count": int(archive_item.get("partition_count") or 0),
                    "regime": item_regime,
                    "tick_quality": quality,
                }
            )

        tick_quality = {
            **dict(tick_archive.get("tick_quality") or {}),
            "status": "healthy" if items and healthy == len(items) else ("stale" if stale > 0 else "incomplete"),
            "healthy_symbols": healthy,
            "stale_symbols": stale,
            "incomplete_symbols": incomplete,
        }
        microstructure = {
            **dict(tick_archive.get("microstructure") or {}),
            "generated_at": str(raw_state.get("generated_at") or datetime.now(timezone.utc).isoformat()),
            "status": str(tick_archive.get("coverage_status") or "incomplete"),
            "regime": regime,
            "avg_spread_bps": None if not spread_bps_values else float(sum(spread_bps_values) / len(spread_bps_values)),
            "widest_spread_bps": widest_spread_bps,
            "widest_symbol": widest_symbol,
            "realized_vol_30m": None if not vol_values else float(sum(vol_values) / len(vol_values)),
            "retention_tiers": self.runtime.market_data.retention_days_by_timeframe(),
            "tick_archive_rows": int(tick_archive.get("row_count") or 0),
            "tick_archive_partitions": int(tick_archive.get("partition_count") or 0),
            "items": items,
        }
        return microstructure, tick_quality

    def _compute_risk_nowcast(
        self,
        risk_summary: Mapping[str, Any] | None,
        *,
        microstructure: Mapping[str, Any],
        tick_quality: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not risk_summary:
            return {}
        headline = list(risk_summary.get("headline_risk") or [])
        live_95 = self._headline_risk_point(headline, alpha=0.95, horizon_days=1, is_stressed=False)
        live_99 = self._headline_risk_point(headline, alpha=0.99, horizon_days=1, is_stressed=False)
        governance = self._headline_risk_point(headline, alpha=0.99, horizon_days=10, is_stressed=True)
        regime = str(microstructure.get("regime") or "incomplete")
        quality_status = str(tick_quality.get("status") or "incomplete")
        scale_factor = 1.0
        if regime == "volatile":
            scale_factor = 1.15
        elif regime == "stressed":
            scale_factor = 1.35
        avg_spread_bps = microstructure.get("avg_spread_bps")
        if isinstance(avg_spread_bps, (float, int)):
            if float(avg_spread_bps) >= 10.0:
                scale_factor += 0.15
            elif float(avg_spread_bps) >= 3.0:
                scale_factor += 0.05
        if quality_status != "healthy":
            scale_factor += 0.05
        scale_factor = min(max(scale_factor, 0.8), 1.65)

        def _scale(point: Mapping[str, Any] | None) -> dict[str, Any] | None:
            if point is None:
                return None
            var_value = point.get("var")
            es_value = point.get("es")
            return {
                "label": point.get("label"),
                "model": point.get("model"),
                "alpha": point.get("alpha"),
                "horizon_days": point.get("horizon_days"),
                "baseline_var": var_value,
                "baseline_es": es_value,
                "nowcast_var": None if var_value is None else float(var_value) * scale_factor,
                "nowcast_es": None if es_value is None else float(es_value) * scale_factor,
                "is_stressed": bool(point.get("is_stressed")),
                "scenario_name": point.get("scenario_name"),
            }

        status = "healthy"
        if quality_status != "healthy":
            status = "degraded"
        return {
            "status": status,
            "scale_factor": scale_factor,
            "regime": regime,
            "quality_status": quality_status,
            "reference_model": risk_summary.get("reference_model"),
            "live_1d_95": _scale(live_95),
            "live_1d_99": _scale(live_99),
            "governance_10d_99": _scale(governance),
        }

    def _instrument_metadata(self, symbol: str) -> dict[str, Any]:
        normalized = str(symbol).upper()
        instruments = self.runtime.storage.list_instruments(symbols=[normalized]) if self.runtime.storage_ready else []
        instrument = {} if not instruments else dict(instruments[0])
        if any(instrument.get(key) not in {None, 0, 0.0} for key in ("contract_size", "tick_size", "tick_value")):
            return instrument
        try:
            with self.runtime._mt5_gateway() as live:
                live_instrument = live.instrument_definition(normalized).to_dict()
        except Exception:
            return instrument
        merged = dict(live_instrument)
        merged.update({key: value for key, value in instrument.items() if value not in {None, "", 0, 0.0}})
        return merged

    def _estimate_spread_cost(
        self,
        *,
        symbol: str,
        volume_lots: float,
        spread: float | None,
    ) -> float | None:
        if spread is None or volume_lots == 0:
            return None
        instrument = self._instrument_metadata(symbol)
        tick_size = instrument.get("tick_size")
        tick_value = instrument.get("tick_value")
        contract_size = instrument.get("contract_size")
        try:
            tick_size_value = float(tick_size or 0.0)
            tick_value_value = float(tick_value or 0.0)
            if tick_size_value <= 0.0 or tick_value_value <= 0.0:
                contract_size_value = float(contract_size or 0.0)
                if contract_size_value <= 0.0:
                    return None
                return float(abs(volume_lots) * contract_size_value * float(spread))
            return float(abs(volume_lots) * tick_value_value * (float(spread) / tick_size_value))
        except (TypeError, ValueError):
            return None

    def _build_pnl_explain(
        self,
        raw_state: Mapping[str, Any],
        *,
        microstructure: Mapping[str, Any],
    ) -> dict[str, Any]:
        holdings = list(raw_state.get("holdings") or [])
        deals = list(raw_state.get("deal_history") or [])
        spreads = {
            str(item.get("symbol") or "").upper(): item
            for item in list(microstructure.get("items") or [])
        }
        realized = float(sum(float(item.get("profit") or 0.0) for item in deals))
        unrealized = float(sum(float(item.get("profit") or 0.0) for item in holdings))
        swap = float(sum(float(item.get("swap") or 0.0) for item in holdings) + sum(float(item.get("swap") or 0.0) for item in deals))
        commission = float(sum(float(item.get("commission") or 0.0) for item in deals))
        fee = float(sum(float(item.get("fee") or 0.0) for item in deals))
        estimated_spread_cost = 0.0
        for item in holdings:
            symbol = str(item.get("symbol") or "").upper()
            spread_item = spreads.get(symbol, {})
            estimate = self._estimate_spread_cost(
                symbol=symbol,
                volume_lots=float(item.get("volume_lots") or 0.0),
                spread=None if spread_item.get("spread") is None else float(spread_item.get("spread")),
            )
            if estimate is not None:
                estimated_spread_cost += estimate
        return {
            "realized": realized,
            "unrealized": unrealized,
            "swap": swap,
            "commission": commission,
            "fee": fee,
            "estimated_spread_cost": estimated_spread_cost,
            "net_total": realized + unrealized + swap + commission + fee - estimated_spread_cost,
        }

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
        tick_archive = self._archive_live_ticks_if_needed(enriched, portfolio_slug=portfolio["slug"])
        microstructure, tick_quality = self._build_microstructure(enriched, tick_archive=tick_archive)
        pnl_explain = self._build_pnl_explain(enriched, microstructure=microstructure)
        enriched["microstructure"] = microstructure
        enriched["tick_quality"] = tick_quality
        enriched["pnl_explain"] = pnl_explain
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
            risk_nowcast = self._compute_risk_nowcast(
                analytics["risk_summary"],
                microstructure=microstructure,
                tick_quality=tick_quality,
            )
            enriched["risk_nowcast"] = risk_nowcast
            enriched["risk_summary"]["risk_nowcast"] = dict(risk_nowcast)
            enriched["risk_summary"]["microstructure"] = dict(microstructure)
            enriched["risk_summary"]["tick_quality"] = dict(tick_quality)
            enriched["risk_summary"]["pnl_explain"] = dict(pnl_explain)
            enriched["risk_budget"] = analytics["risk_budget"]
            enriched["capital_usage"] = analytics["capital_usage"]
            analytics["bundle"] = {
                **dict(analytics["bundle"]),
                "risk_nowcast": dict(risk_nowcast),
                "microstructure": dict(microstructure),
                "tick_quality": dict(tick_quality),
                "pnl_explain": dict(pnl_explain),
            }
            operator_alerts.extend(list(analytics["alerts"]))
            self._persist_live_analytics_if_needed(
                raw_state=enriched,
                analytics=analytics,
                portfolio_slug=portfolio["slug"],
            )
        else:
            enriched["risk_nowcast"] = {}
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

    def accept_reconciliation_baseline(
        self,
        *,
        portfolio_slug: str | None = None,
        reason: str,
        symbol: str | None = None,
        operator_note: str | None = None,
    ) -> int | None:
        if not self.runtime.storage_ready:
            return None

        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        live_state = self.live_state(portfolio_slug=portfolio["slug"])
        if not bool(live_state.get("connected", False)) or bool(live_state.get("stale", False)):
            return None

        metadata = {
            "reconciliation_baseline_accepted": True,
            "reconciliation_baseline_reason": str(reason or "").strip(),
            "reconciliation_baseline_symbol": None if symbol is None else str(symbol).upper(),
            "reconciliation_baseline_note": str(operator_note or "").strip(),
            "live_sequence": int(live_state.get("sequence") or 0),
            "live_persistence_key": self._live_persistence_key(live_state),
            "bridge_status": live_state.get("status"),
            "bridge_generated_at": live_state.get("generated_at"),
            "bridge_source": live_state.get("source"),
        }

        try:
            analytics = self._build_live_analytics(live_state, portfolio_slug=portfolio["slug"])
        except Exception:
            analytics = None

        if analytics is not None:
            self.runtime._persist_live_bundle(
                bundle=analytics["bundle"],
                portfolio_id=portfolio_id,
                source="mt5_live_bridge",
                metadata=metadata,
                persist_alerts=False,
                persist_audit=False,
            )
            latest = self.runtime.storage.latest_snapshot(source="mt5_live_bridge", portfolio_slug=portfolio["slug"])
            return None if latest is None else int(latest["id"])

        holdings = list(live_state.get("holdings") or [])
        exposure = self.runtime.market_data.exposure_from_holdings(holdings, portfolio_slug=portfolio["slug"])
        exposure_by_symbol = {
            str(item.get("symbol") or "").upper(): float(item.get("exposure_base_ccy") or 0.0)
            for item in list(exposure.get("items") or [])
            if str(item.get("symbol") or "").strip()
        }
        snapshot_payload = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "source": "mt5_live_bridge",
            "timeframe": self.runtime._default_timeframe(),
            "days": self.runtime._default_days(),
            "window": int(self.runtime.risk_defaults["window"]),
            "holdings": holdings,
            "exposure_by_symbol": exposure_by_symbol,
            "metadata": metadata,
        }
        return int(
            self.runtime.storage.record_snapshot(
                snapshot_payload,
                portfolio_id=portfolio_id,
                source="mt5_live_bridge",
            )
        )

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

    def _compute_execution_bundle(
        self,
        *,
        portfolio: Mapping[str, Any],
        live,
    ) -> dict[str, Any]:
        try:
            return self.runtime._compute_live_portfolio_state(
                portfolio=portfolio,
                live=live,
                allow_auto_sync=False,
            )
        except RuntimeError:
            return self.runtime._compute_live_portfolio_state(
                portfolio=portfolio,
                live=live,
                allow_auto_sync=True,
            )

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
            bundle = self._compute_execution_bundle(portfolio=portfolio, live=live)
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
                approved_exposure_change=float(decision.get("approved_exposure_change", 0.0)),
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
            current_tick: dict[str, Any] = {}
            try:
                current_tick = dict(live.connector.symbol_info_tick(str(symbol).upper()))
            except Exception:
                current_tick = {}
            if current_tick:
                try:
                    self.runtime.market_data.archive_ticks(
                        symbol=str(symbol).upper(),
                        ticks=[current_tick],
                        portfolio_slug=portfolio["slug"],
                        source="execution_preview",
                    )
                except Exception:
                    pass
            tick_archive = self.runtime.market_data.tick_archive_summary(symbols=[str(symbol).upper()])
            microstructure, tick_quality = self._build_microstructure(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "symbols": [str(symbol).upper()],
                    "ticks": {str(symbol).upper(): current_tick} if current_tick else {},
                },
                tick_archive=tick_archive,
            )
            pnl_explain = self._build_pnl_explain(
                {
                    "holdings": [item.to_dict() for item in live_positions],
                    "deal_history": [],
                },
                microstructure=microstructure,
            )
            pre_risk_nowcast = self._compute_risk_nowcast(
                {
                    "headline_risk": list(dict(decision.get("pre_trade") or {}).get("headline_risk") or []),
                    "reference_model": decision.get("reference_model") or guard.model_used,
                },
                microstructure=microstructure,
                tick_quality=tick_quality,
            )
            post_risk_nowcast = self._compute_risk_nowcast(
                {
                    "headline_risk": list(dict(decision.get("post_trade") or {}).get("headline_risk") or []),
                    "reference_model": decision.get("reference_model") or guard.model_used,
                },
                microstructure=microstructure,
                tick_quality=tick_quality,
            )
            symbol_microstructure = next(
                (item for item in list(microstructure.get("items") or []) if item.get("symbol") == str(symbol).upper()),
                {},
            )
            live_spread = symbol_microstructure.get("spread")
            if live_spread is None and current_tick.get("bid") is not None and current_tick.get("ask") is not None:
                try:
                    live_spread = float(current_tick["ask"]) - float(current_tick["bid"])
                except (TypeError, ValueError):
                    live_spread = None
            estimated_spread_cost = self._estimate_spread_cost(
                symbol=str(symbol).upper(),
                volume_lots=float(guard.volume_lots or 0.0),
                spread=None if live_spread is None else float(live_spread),
            )
            expected_slippage_points = None
            instrument = self._instrument_metadata(str(symbol).upper())
            tick_size = instrument.get("tick_size")
            if live_spread is not None:
                multiplier = 1.0
                regime = str(symbol_microstructure.get("regime") or microstructure.get("regime") or "normal")
                if regime == "volatile":
                    multiplier = 1.25
                elif regime == "stressed":
                    multiplier = 1.5
                try:
                    tick_size_value = float(tick_size or 0.0)
                    if tick_size_value > 0.0:
                        expected_slippage_points = float(live_spread) / tick_size_value * multiplier
                    else:
                        expected_slippage_points = float(live_spread) * multiplier
                except (TypeError, ValueError, ZeroDivisionError):
                    expected_slippage_points = None
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
                microstructure=microstructure,
                risk_nowcast={
                    "pre_trade": pre_risk_nowcast,
                    "post_trade": post_risk_nowcast,
                    "tick_quality": tick_quality,
                },
                pnl_explain=pnl_explain,
                estimated_spread_cost=estimated_spread_cost,
                expected_slippage_points=expected_slippage_points,
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

    def _execution_fill_from_deal(
        self,
        *,
        symbol: str,
        order_ticket: int | None,
        deal: dict[str, Any],
    ) -> dict[str, Any]:
        return {
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
            "raw": {
                **dict(deal),
                "source": "broker_history",
            },
        }

    def _dedupe_execution_fills(self, fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fill in fills:
            key = json.dumps(
                {
                    "deal_ticket": fill.get("deal_ticket"),
                    "order_ticket": fill.get("order_ticket"),
                    "position_id": fill.get("position_id"),
                    "time_utc": fill.get("time_utc"),
                    "volume_lots": round(float(fill.get("volume_lots") or 0.0), 8),
                    "price": None if fill.get("price") is None else round(float(fill.get("price")), 10),
                    "reason": fill.get("reason"),
                },
                sort_keys=True,
                default=str,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fill)
        return deduped

    def _provisional_execution_fill(
        self,
        *,
        symbol: str,
        order_request: dict[str, Any],
        mt5_result: dict[str, Any],
        order_ticket: int | None,
        deal_ticket: int | None,
        position_id: int | None,
    ) -> dict[str, Any]:
        request_type = int(order_request.get("type") or 0)
        side = "BUY" if request_type == 0 else "SELL"
        price = mt5_result.get("price", order_request.get("price"))
        volume = mt5_result.get("volume", order_request.get("volume"))
        now_utc = datetime.now(timezone.utc).isoformat()
        return {
            "symbol": str(symbol).upper(),
            "order_ticket": order_ticket,
            "deal_ticket": deal_ticket,
            "position_id": position_id,
            "side": side,
            "entry": None,
            "volume_lots": float(volume or 0.0),
            "price": None if price is None else float(price),
            "profit": None,
            "commission": None,
            "swap": None,
            "fee": None,
            "reason": "broker_result_provisional",
            "comment": "Provisional fill awaiting broker history sync.",
            "is_manual": False,
            "time_utc": now_utc,
            "raw": {
                "source": "mt5_result_provisional",
                "mt5_result": dict(mt5_result),
                "order_request": dict(order_request),
            },
        }

    def _collect_execution_fills(
        self,
        *,
        live: Any,
        symbol: str,
        order_request: dict[str, Any],
        mt5_result: dict[str, Any],
        status: str,
    ) -> tuple[list[dict[str, Any]], int | None]:
        order_ticket = None if mt5_result.get("order") is None else int(mt5_result.get("order"))
        deal_ticket = None if mt5_result.get("deal") is None else int(mt5_result.get("deal"))
        history_from = datetime.now(timezone.utc) - timedelta(minutes=30)
        history_to = datetime.now(timezone.utc) + timedelta(minutes=1)
        fills: list[dict[str, Any]] = []
        position_id: int | None = None
        max_attempts = 5 if status in {"EXECUTED", "PLACED"} else 1

        for attempt in range(max_attempts):
            order_entries: list[dict[str, Any]] = []
            if order_ticket is not None:
                try:
                    order_entries.extend(
                        item.to_dict()
                        for item in live.order_history(
                            date_from=history_from,
                            date_to=history_to,
                            ticket=order_ticket,
                        )
                    )
                except Exception:
                    order_entries = []

            if position_id is None:
                for order_entry in order_entries:
                    candidate = order_entry.get("position_id")
                    if candidate is not None:
                        try:
                            position_id = int(candidate)
                        except (TypeError, ValueError):
                            position_id = None
                        if position_id is not None:
                            break

            live_deals: list[dict[str, Any]] = []
            if deal_ticket is not None:
                try:
                    live_deals.extend(
                        item.to_dict()
                        for item in live.deal_history(
                            date_from=history_from,
                            date_to=history_to,
                            ticket=deal_ticket,
                        )
                    )
                except Exception:
                    pass
            try:
                live_deals.extend(
                    item.to_dict()
                    for item in live.deal_history(
                        date_from=history_from,
                        date_to=history_to,
                        symbols=[str(symbol).upper()],
                    )
                )
            except Exception:
                pass
            if position_id is not None:
                try:
                    live_deals.extend(
                        item.to_dict()
                        for item in live.deal_history(
                            date_from=history_from,
                            date_to=history_to,
                            position=position_id,
                        )
                    )
                except Exception:
                    pass

            candidate_fills: list[dict[str, Any]] = []
            for deal in live_deals:
                deal_order_ticket = deal.get("order_ticket")
                same_order = order_ticket is not None and int(deal_order_ticket or -1) == int(order_ticket)
                same_ticket = deal_ticket is not None and int(deal.get("ticket") or -1) == int(deal_ticket)
                same_position = (
                    position_id is not None
                    and (order_ticket is None or deal_order_ticket is None)
                    and deal.get("position_id") is not None
                    and int(deal.get("position_id") or -1) == int(position_id)
                )
                if same_order or same_ticket or same_position:
                    candidate_fills.append(
                        self._execution_fill_from_deal(
                            symbol=str(symbol).upper(),
                            order_ticket=order_ticket,
                            deal=deal,
                        )
                    )

            fills = self._dedupe_execution_fills(candidate_fills)
            if fills:
                break
            if attempt < max_attempts - 1:
                time.sleep(0.2 * (attempt + 1))

        if not fills and status == "EXECUTED":
            fills = [
                self._provisional_execution_fill(
                    symbol=str(symbol).upper(),
                    order_request=order_request,
                    mt5_result=mt5_result,
                    order_ticket=order_ticket,
                    deal_ticket=deal_ticket,
                    position_id=position_id,
                )
            ]
        return fills, position_id

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
            bundle = self._compute_execution_bundle(portfolio=portfolio, live=live)
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

            executed_exposure_change = 0.0
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
                approved_exposure_change=float(decision.get("approved_exposure_change", 0.0)),
                snapshot_source="execution_submit",
            )

            if guard.submit_allowed:
                mt5_result = live.connector.order_send(order_request)
                retcode = int(mt5_result.get("retcode", -1))
                submitted_volume_lots = approved_volume_lots
                if retcode in {10009, 10008}:
                    status = "EXECUTED" if retcode == 10009 else "PLACED"
                    executed_exposure_change = float(getattr(guard, "executable_exposure_change", 0.0) or 0.0)
                    broker_status = "filled" if retcode == 10009 else "placed"
                else:
                    status = "FAILED"
                    broker_status = "rejected"

                account_after = live.account_snapshot()
                positions_after = live.positions()

                fills, position_id = self._collect_execution_fills(
                    live=live,
                    symbol=str(symbol).upper(),
                    order_request=order_request,
                    mt5_result=mt5_result,
                    status=status,
                )

                filled_volume_lots = float(sum(float(item.get("volume_lots") or 0.0) for item in fills))
                remaining_volume_lots = max(submitted_volume_lots - filled_volume_lots, 0.0)
                fill_ratio = 0.0 if submitted_volume_lots <= 1e-9 else filled_volume_lots / submitted_volume_lots
                if fills:
                    if position_id is None:
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
                requested_exposure_change=normalized_exposure_change,
                approved_exposure_change=float(decision.get("approved_exposure_change", 0.0)),
                executed_exposure_change=executed_exposure_change,
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
