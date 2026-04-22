from __future__ import annotations

import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread
from typing import Any, Literal, Mapping

from var_project.alerts.engine import (
    alerts_from_capital_snapshot,
    alerts_from_execution_result,
    alerts_from_live_operator_state,
    alerts_from_risk_budget,
    alerts_from_validation_summary,
)
from var_project.api.services.runtime import DeskServiceRuntime
from var_project.core.exceptions import MT5ConnectionError
from var_project.execution.mt5_bridge import build_empty_live_state, collect_live_state_from_connector
from var_project.execution.mt5_live import ExecutionPreview, ExecutionResult, MT5TerminalStatus
from var_project.validation.model_validation import ValidationSummary


class DeskMt5Service:
    _shared_startup_import_done: set[tuple[str, str]] = set()
    _shared_startup_sync_inflight: set[tuple[str, str]] = set()
    _shared_last_tick_archive_sequence: dict[tuple[str, str], int] = {}
    _shared_enriched_live_state_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    _shared_live_analytics_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    _shared_last_broker_analytics: dict[tuple[str, str], dict[str, Any]] = {}
    _shared_last_good_live_state: dict[tuple[str, str], dict[str, Any]] = {}
    _shared_live_state_response_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    _shared_last_execution_reconciliation_heal_at: dict[tuple[str, str], float] = {}
    _shared_reconciliation_match_streak: dict[tuple[str, str, str], int] = {}
    _shared_startup_sync_lock = Lock()
    _shared_live_state_compute_lock = Lock()

    def __init__(self, runtime: DeskServiceRuntime):
        self.runtime = runtime

    def _live_scope_key(
        self,
        portfolio_slug: str,
        *,
        account_id: str | None = None,
    ) -> tuple[str, str]:
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        return (str(self.runtime.root.resolve()), f"{portfolio_slug}::{resolved_account_id}")

    def _live_state_response_cache_key(
        self,
        *,
        portfolio_slug: str,
        detail_level: str,
        account_id: str | None = None,
    ) -> tuple[str, str, str]:
        return (*self._live_scope_key(portfolio_slug, account_id=account_id), str(detail_level))

    def _summary_cache_ttl_seconds(self) -> float:
        configured = max(float(self.runtime.mt5_config.live_poll_seconds), 0.5)
        return min(max(configured * 0.75, 0.5), 2.0)

    def _detail_cache_ttl_seconds(self, detail_level: str) -> float:
        normalized = str(detail_level or "full").strip().lower()
        if normalized == "summary":
            return self._summary_cache_ttl_seconds()
        configured = max(float(self.runtime.mt5_config.live_poll_seconds), 0.5)
        if normalized == "inspector":
            return min(max(configured * 0.20, 0.15), 0.50)
        return min(max(configured * 0.35, 0.20), 1.00)

    def _analytics_fallback_ttl_seconds(self) -> float:
        configured = max(float(self.runtime.mt5_config.live_history_poll_seconds), 5.0)
        return min(max(configured * 12.0, 300.0), 1800.0)

    def _last_good_live_state_ttl_seconds(self) -> float:
        configured = max(float(self.runtime.mt5_config.live_history_poll_seconds), 5.0)
        return min(max(configured * 20.0, 300.0), 3600.0)

    def _remember_last_broker_analytics(
        self,
        *,
        portfolio_slug: str,
        analytics: Mapping[str, Any],
        account_id: str | None = None,
    ) -> None:
        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        self._shared_last_broker_analytics[scope_key] = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "analytics": deepcopy(dict(analytics)),
        }

    def _latest_broker_analytics(
        self,
        *,
        portfolio_slug: str,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        cached = self._shared_last_broker_analytics.get(scope_key)
        if cached is None:
            return None
        captured_at = self._to_utc_datetime(cached.get("captured_at"))
        if captured_at is None:
            return None
        age_seconds = max((datetime.now(timezone.utc) - captured_at).total_seconds(), 0.0)
        if age_seconds > self._analytics_fallback_ttl_seconds():
            return None
        payload = dict(cached.get("analytics") or {})
        if not payload:
            return None
        return deepcopy(payload)

    def _remember_last_good_live_state(
        self,
        *,
        portfolio_slug: str,
        payload: Mapping[str, Any],
        account_id: str | None = None,
    ) -> None:
        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        self._shared_last_good_live_state[scope_key] = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "payload": deepcopy(dict(payload)),
        }

    def _latest_good_live_state(
        self,
        *,
        portfolio_slug: str,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        cached = self._shared_last_good_live_state.get(scope_key)
        if cached is None:
            return None
        captured_at = self._to_utc_datetime(cached.get("captured_at"))
        if captured_at is None:
            return None
        age_seconds = max((datetime.now(timezone.utc) - captured_at).total_seconds(), 0.0)
        if age_seconds > self._last_good_live_state_ttl_seconds():
            return None
        payload = dict(cached.get("payload") or {})
        if not payload:
            return None
        return deepcopy(payload)

    def _overlay_dynamic_live_fields(
        self,
        *,
        cached_state: Mapping[str, Any],
        raw_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = deepcopy(dict(cached_state))
        dynamic_fields = (
            "sequence",
            "source",
            "status",
            "connected",
            "degraded",
            "stale",
            "fallback_snapshot_used",
            "generated_at",
            "last_success_at",
            "last_error",
            "poll_interval_seconds",
            "history_poll_interval_seconds",
            "history_lookback_minutes",
            "effective_history_lookback_minutes",
            "market_closed",
            "market_closed_reason",
            "market_reference_timestamp",
            "market_reference_source",
            "bridge_consecutive_failures",
            "bridge_next_poll_delay_seconds",
            "bridge_last_error_at",
            "bridge_capture_duration_ms",
            "bridge_event_buffer_usage",
            "bridge_event_buffer_capacity",
            "bridge_last_event_kind",
            "symbols",
            "terminal_status",
            "account",
            "ticks",
            "holdings",
            "pending_orders",
            "order_history",
            "deal_history",
        )
        for field in dynamic_fields:
            if field in raw_state:
                merged[field] = deepcopy(raw_state.get(field))

        analytics_generated_at = (
            (dict(merged.get("risk_summary") or {}).get("generated_at"))
            or merged.get("generated_at")
        )
        analytics_timestamp = self._to_utc_datetime(analytics_generated_at)
        max_age_seconds = max(
            60.0,
            float(self.runtime.mt5_config.live_history_poll_seconds) * 3.0,
        )
        analytics_stale = True
        if analytics_timestamp is not None:
            analytics_stale = (
                max((datetime.now(timezone.utc) - analytics_timestamp).total_seconds(), 0.0)
                > max_age_seconds
            )
        merged["analytics_generated_at"] = analytics_generated_at
        merged["analytics_stale"] = bool(analytics_stale)
        merged["health"] = self._build_live_health(merged)
        return merged

    def cached_live_state(
        self,
        *,
        portfolio_slug: str | None = None,
        detail_level: Literal["summary", "full", "inspector"] = "summary",
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        if detail_level != "summary":
            return None
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        cache_key = self._live_state_response_cache_key(
            portfolio_slug=portfolio["slug"],
            detail_level=detail_level,
            account_id=resolved_account_id,
        )
        cached = self._shared_live_state_response_cache.get(cache_key)
        if cached is None:
            return None
        if float(cached.get("expires_at") or 0.0) <= time.monotonic():
            return None
        return deepcopy(dict(cached.get("payload") or {}))

    @staticmethod
    def _to_utc_datetime(value: Any) -> datetime | None:
        if value in {None, "", "null"}:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _live_analytics_cache_key(
        self,
        *,
        portfolio_slug: str,
        raw_state: Mapping[str, Any],
        account_id: str | None = None,
    ) -> tuple[str, str, str]:
        holdings = []
        for item in list(raw_state.get("holdings") or []):
            if not isinstance(item, Mapping):
                continue
            holdings.append(
                {
                    str(key): item.get(key)
                    for key in sorted(item.keys())
                }
            )
        holdings.sort(
            key=lambda item: (
                str(item.get("symbol") or ""),
                str(item.get("side") or ""),
                str(item.get("ticket") or item.get("position_id") or ""),
                float(item.get("volume_lots") or item.get("volume") or 0.0),
            )
        )
        latest_sync_marker = self._latest_market_sync_marker(portfolio_slug)
        latest_validation_marker = self._latest_validation_marker(portfolio_slug)
        return (
            *self._live_scope_key(portfolio_slug, account_id=account_id),
            json.dumps(
                {
                    "holdings": holdings,
                    "latest_sync_marker": latest_sync_marker,
                    "latest_validation_marker": latest_validation_marker,
                    "market_closed": bool(raw_state.get("market_closed", False)),
                    "market_reference_timestamp": raw_state.get("market_reference_timestamp"),
                },
                sort_keys=True,
                default=str,
            ),
        )

    def _enriched_live_state_cache_key(
        self,
        *,
        portfolio_slug: str,
        raw_state: Mapping[str, Any],
        account_id: str | None = None,
    ) -> tuple[str, str, str, str]:
        return (
            *self._live_scope_key(portfolio_slug, account_id=account_id),
            self._live_persistence_key(dict(raw_state)),
            self._latest_market_sync_marker(portfolio_slug),
        )

    def _latest_market_sync_marker(self, portfolio_slug: str) -> str:
        if not self.runtime.storage_ready:
            return ""
        latest_sync = self.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
        if latest_sync is None:
            return ""
        return str(latest_sync.get("synced_at") or latest_sync.get("updated_at") or "")

    def _latest_validation_marker(self, portfolio_slug: str) -> str:
        if not self.runtime.storage_ready:
            return ""
        latest_validation = self.runtime.storage.latest_validation_run(portfolio_slug=portfolio_slug)
        if latest_validation is None:
            return ""
        return str(
            latest_validation.get("updated_at")
            or latest_validation.get("created_at")
            or latest_validation.get("id")
            or ""
        )

    def _cached_enrich_live_state(
        self,
        raw_state: dict[str, Any],
        *,
        portfolio_slug: str,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        cache_key = self._enriched_live_state_cache_key(
            portfolio_slug=portfolio_slug,
            raw_state=raw_state,
            account_id=account_id,
        )
        cached = self._shared_enriched_live_state_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

        enriched = self._enrich_live_state(
            raw_state,
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )
        self._shared_enriched_live_state_cache[cache_key] = deepcopy(enriched)

        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        for key in list(self._shared_enriched_live_state_cache):
            if key[:2] == scope_key and key != cache_key:
                del self._shared_enriched_live_state_cache[key]

        return enriched

    def _cached_build_live_analytics(
        self,
        raw_state: dict[str, Any],
        *,
        portfolio_slug: str,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        cache_key = self._live_analytics_cache_key(
            portfolio_slug=portfolio_slug,
            raw_state=raw_state,
            account_id=account_id,
        )
        cached = self._shared_live_analytics_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

        analytics = self._build_live_analytics(
            raw_state,
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )
        if analytics is None:
            return None

        self._shared_live_analytics_cache[cache_key] = deepcopy(analytics)

        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        for key in list(self._shared_live_analytics_cache):
            if key[:2] == scope_key and key != cache_key:
                del self._shared_live_analytics_cache[key]

        return analytics

    def _schedule_startup_sync(
        self,
        *,
        portfolio: Mapping[str, Any],
        portfolio_id: int | None,
        imported_symbols: list[str],
        account_id: str | None = None,
    ) -> None:
        scope_key = self._live_scope_key(
            str(portfolio["slug"]),
            account_id=account_id,
        )
        with self._shared_startup_sync_lock:
            if scope_key in self._shared_startup_sync_inflight:
                return
            self._shared_startup_sync_inflight.add(scope_key)

        def _run() -> None:
            try:
                self.runtime.market_data.sync_market_data(
                    portfolio_slug=str(portfolio["slug"]),
                    account_id=account_id,
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
            finally:
                with self._shared_startup_sync_lock:
                    self._shared_startup_sync_inflight.discard(scope_key)

        connector_delay_seconds = self._coerce_float(
            getattr(self.runtime.mt5_connector_factory, "delay_seconds", None)
        )
        # Test connectors usually benefit from deterministic startup sync completion,
        # but keep slow connector scenarios non-blocking for readiness timeout behavior.
        if self.runtime._has_custom_mt5_factory and (
            connector_delay_seconds is None or connector_delay_seconds <= 0.2
        ):
            _run()
            return

        Thread(
            target=_run,
            name=f"mt5-startup-sync-{portfolio['slug']}",
            daemon=True,
        ).start()

    def _normalize_live_state(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        payload = dict(raw_state)
        payload["sequence"] = int(payload.get("sequence") or 0)
        payload["status"] = str(payload.get("status") or "ok")
        payload["connected"] = bool(payload.get("connected", True))
        payload["degraded"] = bool(payload.get("degraded", False))
        payload["stale"] = bool(payload.get("stale", False))
        payload["fallback_snapshot_used"] = bool(payload.get("fallback_snapshot_used", False))
        payload["poll_interval_seconds"] = float(
            payload.get("poll_interval_seconds") or self.runtime.mt5_config.live_poll_seconds
        )
        payload["history_poll_interval_seconds"] = float(
            payload.get("history_poll_interval_seconds") or self.runtime.mt5_config.live_history_poll_seconds
        )
        payload["history_lookback_minutes"] = int(
            payload.get("history_lookback_minutes") or self.runtime.mt5_config.live_history_lookback_minutes
        )
        payload["effective_history_lookback_minutes"] = int(
            payload.get("effective_history_lookback_minutes") or payload["history_lookback_minutes"]
        )
        payload["market_closed"] = bool(payload.get("market_closed", False))
        payload["market_closed_reason"] = payload.get("market_closed_reason")
        payload["market_reference_timestamp"] = payload.get("market_reference_timestamp")
        payload["market_reference_source"] = payload.get("market_reference_source")
        payload["generated_at"] = str(payload.get("generated_at") or datetime.now(timezone.utc).isoformat())
        payload["last_success_at"] = payload.get("last_success_at") or payload["generated_at"]
        payload["source"] = str(payload.get("source") or "mt5_agent_bridge")
        payload["bridge_consecutive_failures"] = int(self._coerce_int(payload.get("bridge_consecutive_failures")) or 0)
        payload["bridge_next_poll_delay_seconds"] = float(
            self._coerce_float(payload.get("bridge_next_poll_delay_seconds"))
            or payload["poll_interval_seconds"]
        )
        payload["bridge_last_error_at"] = payload.get("bridge_last_error_at")
        payload["bridge_capture_duration_ms"] = self._coerce_float(payload.get("bridge_capture_duration_ms"))
        event_buffer_capacity = int(
            self._coerce_int(payload.get("bridge_event_buffer_capacity"))
            or max(int(self.runtime.mt5_config.live_event_buffer_size), 10)
        )
        event_buffer_usage = int(self._coerce_int(payload.get("bridge_event_buffer_usage")) or 0)
        payload["bridge_event_buffer_capacity"] = max(event_buffer_capacity, 1)
        payload["bridge_event_buffer_usage"] = min(
            max(event_buffer_usage, 0),
            payload["bridge_event_buffer_capacity"],
        )
        payload["bridge_last_event_kind"] = (
            None
            if payload.get("bridge_last_event_kind") in {None, "", "null"}
            else str(payload.get("bridge_last_event_kind"))
        )
        payload["health"] = dict(payload.get("health") or {})
        return payload

    def _check_startup_import(
        self,
        raw_state: dict[str, Any],
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> None:
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

        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        scope_key = self._live_scope_key(portfolio["slug"], account_id=resolved_account_id)
        if scope_key in self._shared_startup_import_done:
            return
        self._shared_startup_import_done.add(scope_key)

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
        self._schedule_startup_sync(
            portfolio=portfolio,
            portfolio_id=portfolio_id,
            imported_symbols=imported_symbols,
            account_id=resolved_account_id,
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

    def _empty_book_live_analytics(
        self,
        *,
        portfolio: Mapping[str, Any],
        raw_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        snapshot_timestamp = str(
            raw_state.get("market_reference_timestamp")
            or raw_state.get("generated_at")
            or datetime.now(timezone.utc).isoformat()
        )
        reference_model = self.runtime._decision_reference_model(portfolio["slug"])
        preferred_model = self.runtime._preferred_model(portfolio["slug"]) or reference_model
        model_names = sorted(
            {
                *dict(self.runtime.limits_config.get("model_limits_eur") or {}).keys(),
                reference_model,
                preferred_model,
            }
        )
        zero_metrics = {str(model): 0.0 for model in model_names if str(model).strip()}
        live_portfolio = self.runtime.is_live_portfolio(portfolio)

        capital_template = None
        if self.runtime.storage_ready:
            source_candidates = ("mt5_live_bridge", "mt5_live") if live_portfolio else ("mt5_live_bridge", "mt5_live", "historical")
            for source in source_candidates:
                capital_template = self.runtime.storage.latest_capital_snapshot(
                    source=source,
                    portfolio_slug=portfolio["slug"],
                )
                if capital_template is not None:
                    break
        capital_template = {} if capital_template is None else dict(capital_template)

        total_budget = float(capital_template.get("total_capital_budget_eur") or 0.0)
        budget_template = dict(capital_template.get("budget") or {})
        model_template = dict(capital_template.get("models") or {})
        allocation_template = dict(capital_template.get("allocations") or {})

        capital_usage = {
            "portfolio_slug": portfolio["slug"],
            "base_currency": str(portfolio["base_currency"]),
            "reference_model": reference_model,
            "snapshot_source": "mt5_live_bridge",
            "snapshot_timestamp": snapshot_timestamp,
            "source": "mt5_live_bridge",
            "created_at": snapshot_timestamp,
            "total_capital_budget_eur": total_budget,
            "total_capital_consumed_eur": 0.0,
            "total_capital_reserved_eur": 0.0,
            "total_capital_remaining_eur": total_budget,
            "headroom_ratio": None if total_budget <= 0.0 else 1.0,
            "status": "OK",
            "budget": {
                "reference_model": reference_model,
                "total_budget_eur": total_budget,
                "reserve_ratio": float(
                    budget_template.get("reserve_ratio")
                    or dict(self.runtime.limits_config.get("capital_management") or {}).get("reserve_ratio")
                    or 0.0
                ),
                "reserved_capital_eur": 0.0,
                "model_budgets": dict(budget_template.get("model_budgets") or {}),
                "symbol_budgets": dict(budget_template.get("symbol_budgets") or {}),
            },
            "models": {},
            "allocations": {},
            "recommendations": [],
        }
        for model, template in model_template.items():
            payload = dict(template or {})
            budget_eur = float(payload.get("budget_eur") or 0.0)
            capital_usage["models"][str(model)] = {
                "model": str(model),
                "budget_eur": budget_eur,
                "consumed_eur": 0.0,
                "remaining_eur": budget_eur,
                "utilization": None if budget_eur <= 0.0 else 0.0,
                "status": "OK",
            }
        for symbol, template in allocation_template.items():
            payload = dict(template or {})
            target = float(
                payload.get("target_capital_eur")
                or payload.get("remaining_capital_eur")
                or 0.0
            )
            capital_usage["allocations"][str(symbol)] = {
                "symbol": str(symbol),
                "weight": float(payload.get("weight") or 0.0),
                "target_capital_eur": target,
                "consumed_capital_eur": 0.0,
                "reserved_capital_eur": 0.0,
                "remaining_capital_eur": target,
                "utilization": None if target <= 0.0 else 0.0,
                "action": "monitor",
                "status": "OK",
            }

        return {
            "bundle": None,
            "risk_summary": {
                "generated_at": snapshot_timestamp,
                "portfolio_slug": portfolio["slug"],
                "portfolio_mode": portfolio.get("mode"),
                "source": "mt5_live_bridge",
                "reference_model": reference_model,
                "preferred_model": preferred_model,
                "alpha": float(self.runtime.risk_defaults["alpha"]),
                "sample_size": 0,
                "timeframe": self.runtime._default_timeframe(),
                "days": int(self.runtime._default_days()),
                "window": int(self.runtime.risk_defaults["window"]),
                "latest_observation": raw_state.get("market_reference_timestamp"),
                "var": zero_metrics,
                "es": zero_metrics,
                "risk_surface": {},
                "headline_risk": [],
                "stress_surface": {},
                "data_quality": {
                    "status": "no_exposure",
                    "estimation_window_days": int(self.runtime.risk_defaults.get("estimation_window_days") or 0),
                    "minimum_valid_days": int(self.runtime.risk_defaults.get("minimum_valid_days") or 0),
                    "available_observations": 0,
                    "oldest_observation": None,
                    "latest_observation": raw_state.get("market_reference_timestamp"),
                    "horizon_observations": {},
                    "symbol_count": 0,
                    "gross_exposure_base_ccy": 0.0,
                    "gross_exposure_epsilon_base_ccy": 0.0,
                    "no_exposure_epsilon_by_symbol": {},
                },
                "model_diagnostics": {
                    "flat_book": True,
                    "no_exposure": True,
                    "market_closed": bool(raw_state.get("market_closed", False)),
                },
            },
            "risk_budget": {
                "alpha": float(self.runtime.risk_defaults["alpha"]),
                "sample_size": 0,
                "preferred_model": preferred_model,
                "snapshot_source": "mt5_live_bridge",
                "snapshot_timestamp": snapshot_timestamp,
                "models": {},
            },
            "capital_usage": capital_usage,
            "alerts": [],
        }

    def _build_live_analytics(
        self,
        raw_state: dict[str, Any],
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        portfolio = self._live_portfolio_scope(raw_state, portfolio_slug=portfolio_slug)
        if not list(raw_state.get("holdings") or []):
            return self._empty_book_live_analytics(portfolio=portfolio, raw_state=raw_state)
        bundle = self.runtime._compute_portfolio_state_for_holdings(
            portfolio=portfolio,
            holdings=list(raw_state.get("holdings") or []),
            timeframe=self.runtime._default_timeframe(),
            days=self.runtime._default_days(),
            min_coverage=float(self.runtime.data_defaults["min_coverage"]),
            config=self.runtime._build_risk_model_config(None, None, None, None, None),
            window=int(self.runtime.risk_defaults["window"]),
            snapshot_source="mt5_live_bridge",
            snapshot_timestamp=str(raw_state.get("generated_at") or datetime.now(timezone.utc).isoformat()),
        )
        sample = bundle["sample"]
        latest_observation = None if sample.empty else sample.index[-1].isoformat()
        snapshot_timestamp = str(
            raw_state.get("generated_at")
            or latest_observation
            or datetime.now(timezone.utc).isoformat()
        )
        risk_budget = bundle["risk_budget"].to_dict()
        risk_budget["snapshot_source"] = "mt5_live_bridge"
        risk_budget["snapshot_timestamp"] = snapshot_timestamp
        capital_usage = dict(bundle["capital"])
        capital_usage["snapshot_source"] = "mt5_live_bridge"
        capital_usage["snapshot_timestamp"] = snapshot_timestamp
        validation_alerts = self._validation_governance_alerts(portfolio_slug=portfolio["slug"])
        return {
            "bundle": bundle,
            "risk_summary": {
                "generated_at": snapshot_timestamp,
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
                "latest_observation": latest_observation,
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
                *validation_alerts,
            ],
        }

    def _validation_governance_alerts(self, *, portfolio_slug: str) -> list[dict[str, Any]]:
        if not self.runtime.storage_ready:
            return []
        latest_validation = self.runtime.storage.latest_validation_run(portfolio_slug=portfolio_slug)
        if latest_validation is None:
            return []
        try:
            summary = ValidationSummary.from_dict(
                dict(latest_validation.get("summary") or latest_validation)
            )
        except Exception:
            return []

        validation_run_id = latest_validation.get("id")
        payload: list[dict[str, Any]] = []
        suppressed_live_codes = {
            "VALIDATION_SURFACE_SAMPLE_THIN",
            "VALIDATION_HORIZON_SAMPLE_THIN",
        }
        for alert in alerts_from_validation_summary(summary):
            code = str(alert.code or "")
            if not code.startswith("VALIDATION_"):
                continue
            if code in suppressed_live_codes:
                # Thin-sample governance points are expected on short live windows and
                # create persistent operator noise; keep them in analytics payloads but
                # suppress them from real-time operator alerts.
                continue
            row = alert.to_dict()
            context = dict(row.get("context") or {})
            context.setdefault("portfolio_slug", portfolio_slug)
            if validation_run_id is not None:
                context.setdefault("validation_run_id", validation_run_id)
            row["context"] = context
            payload.append(row)
        return payload

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

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value in {None, "", "null"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value in {None, "", "null"}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_retryable_live_error(value: Any) -> bool | None:
        if value in {None, "", "null"}:
            return None
        message = str(value).lower()
        non_retryable_markers = (
            "invalid mt5 agent key",
            "unknown symbol",
            "timeframe inconnu",
            "not configured",
            "non-json response",
            "http 400",
            "http 401",
            "http 403",
            "http 404",
            "http 422",
        )
        if any(marker in message for marker in non_retryable_markers):
            return False
        retryable_markers = (
            "ipc",
            "connection",
            "timeout",
            "temporar",
            "busy",
            "network",
            "broken pipe",
            "econnreset",
            "econnrefused",
            "service unavailable",
            "unavailable",
        )
        return any(marker in message for marker in retryable_markers)

    def _build_live_health(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        connected = bool(payload.get("connected", False))
        degraded = bool(payload.get("degraded", False))
        stale = bool(payload.get("stale", False))
        market_closed = bool(payload.get("market_closed", False))
        analytics_stale = bool(payload.get("analytics_stale", False))
        fallback_snapshot_used = bool(payload.get("fallback_snapshot_used", False))
        last_error = None if payload.get("last_error") in {None, "", "null"} else str(payload.get("last_error"))
        truth_score = self._coerce_float(payload.get("truth_score"))
        operational_truth = str(payload.get("operational_truth") or "").strip().lower() or None
        tick_quality_status = str(dict(payload.get("tick_quality") or {}).get("status") or "").strip().lower() or None
        nowcast_status = str(dict(payload.get("risk_nowcast") or {}).get("status") or "").strip().lower() or None
        now = datetime.now(timezone.utc)
        generated_at = self._to_utc_datetime(payload.get("generated_at"))
        last_success_at = self._to_utc_datetime(payload.get("last_success_at"))
        generated_age_seconds = (
            None
            if generated_at is None
            else round(max((now - generated_at).total_seconds(), 0.0), 3)
        )
        last_success_age_seconds = (
            None
            if last_success_at is None
            else round(max((now - last_success_at).total_seconds(), 0.0), 3)
        )
        error_retryable = self._is_retryable_live_error(last_error)
        bridge_consecutive_failures = int(self._coerce_int(payload.get("bridge_consecutive_failures")) or 0)
        bridge_next_poll_delay_seconds = self._coerce_float(payload.get("bridge_next_poll_delay_seconds"))
        bridge_capture_duration_ms = self._coerce_float(payload.get("bridge_capture_duration_ms"))
        bridge_event_buffer_usage = self._coerce_int(payload.get("bridge_event_buffer_usage"))
        bridge_event_buffer_capacity = self._coerce_int(payload.get("bridge_event_buffer_capacity"))
        bridge_event_buffer_fill_ratio = None
        if (
            bridge_event_buffer_usage is not None
            and bridge_event_buffer_capacity is not None
            and bridge_event_buffer_capacity > 0
        ):
            bridge_event_buffer_fill_ratio = round(
                min(max(float(bridge_event_buffer_usage) / float(bridge_event_buffer_capacity), 0.0), 1.0),
                4,
            )

        status = "healthy"
        message = "Live bridge healthy."
        quality_degraded = tick_quality_status in {"stale", "incomplete"}
        if not connected:
            status = "offline"
            message = "Live bridge disconnected from MT5."
        elif stale:
            status = "stale"
            message = "Live state is stale while waiting for fresh MT5 evidence."
        elif market_closed:
            status = "market_closed"
            message = "Market is closed; live state is anchored to the latest known snapshot."
        elif (
            degraded
            or analytics_stale
            or quality_degraded
            or nowcast_status == "degraded"
            or bridge_consecutive_failures > 0
        ):
            status = "degraded"
            message = "Live bridge connected but running in degraded mode."

        if last_error and status in {"offline", "stale", "degraded"}:
            message = f"{message} Last error: {last_error}"
        if fallback_snapshot_used:
            base_message = "Serving last known broker snapshot while MT5 is unavailable."
            message = base_message if not last_error else f"{base_message} Last error: {last_error}"

        return {
            "status": status,
            "message": message,
            "connected": connected,
            "degraded": degraded,
            "stale": stale,
            "market_closed": market_closed,
            "analytics_stale": analytics_stale,
            "fallback_snapshot_used": fallback_snapshot_used,
            "generated_age_seconds": generated_age_seconds,
            "last_success_age_seconds": last_success_age_seconds,
            "tick_quality_status": tick_quality_status,
            "nowcast_status": nowcast_status,
            "operational_truth": operational_truth,
            "truth_score": truth_score,
            "error_retryable": error_retryable,
            "last_error": last_error,
            "bridge_consecutive_failures": bridge_consecutive_failures,
            "bridge_next_poll_delay_seconds": bridge_next_poll_delay_seconds,
            "bridge_capture_duration_ms": bridge_capture_duration_ms,
            "bridge_event_buffer_fill_ratio": bridge_event_buffer_fill_ratio,
        }

    def _build_quality_checks(self, payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
        checks: list[dict[str, Any]] = []
        total = 0.0
        score = 0.0

        def _append(
            *,
            check_id: str,
            label: str,
            status: str,
            message: str,
            actual: Any = None,
            expected: Any = None,
            hint: str | None = None,
        ) -> None:
            nonlocal total, score
            normalized = str(status or "fail").strip().lower()
            if normalized not in {"pass", "warn", "fail"}:
                normalized = "fail"
            total += 1.0
            if normalized == "pass":
                score += 1.0
            elif normalized == "warn":
                score += 0.5
            checks.append(
                {
                    "id": check_id,
                    "label": label,
                    "status": normalized,
                    "message": message,
                    "actual": actual,
                    "expected": expected,
                    "hint": hint,
                }
            )

        capital = dict(payload.get("capital_usage") or {})
        if capital:
            budget = self._coerce_float(capital.get("total_capital_budget_eur"))
            consumed = self._coerce_float(capital.get("total_capital_consumed_eur"))
            reserved = self._coerce_float(capital.get("total_capital_reserved_eur"))
            remaining = self._coerce_float(capital.get("total_capital_remaining_eur"))
            if None not in {budget, consumed, reserved, remaining}:
                lhs = float(consumed or 0.0) + float(reserved or 0.0) + float(remaining or 0.0)
                rhs = float(budget or 0.0)
                tolerance = max(abs(rhs) * 0.005, 1e-6)
                ok = abs(lhs - rhs) <= tolerance
                _append(
                    check_id="capital_identity",
                    label="Capital identity",
                    status="pass" if ok else "fail",
                    message=(
                        "Capital identity holds."
                        if ok
                        else "Consumed + reserved + remaining does not match total budget."
                    ),
                    actual=lhs,
                    expected=rhs,
                    hint="Verify capital snapshot aggregation if mismatch persists.",
                )
            headroom = self._coerce_float(capital.get("headroom_ratio"))
            if headroom is None:
                _append(
                    check_id="capital_headroom",
                    label="Headroom bounds",
                    status="warn",
                    message="Headroom ratio is missing.",
                    hint="Headroom should normally be in [0, 1].",
                )
            else:
                in_range = -0.01 <= float(headroom) <= 1.01
                _append(
                    check_id="capital_headroom",
                    label="Headroom bounds",
                    status="pass" if in_range else "fail",
                    message=(
                        "Headroom ratio is coherent."
                        if in_range
                        else "Headroom ratio is outside the expected [0, 1] range."
                    ),
                    actual=headroom,
                    expected="[0, 1]",
                )

        exposure = dict(payload.get("exposure") or {})
        holdings = list(payload.get("holdings") or [])
        if exposure or holdings:
            gross = self._coerce_float(exposure.get("gross_exposure_base_ccy"))
            from_holdings = float(
                sum(
                    abs(
                        float(
                            (item or {}).get("signed_exposure_base_ccy")
                            or (item or {}).get("exposure_base_ccy")
                            or 0.0
                        )
                    )
                    for item in holdings
                )
            )
            if gross is None:
                _append(
                    check_id="exposure_identity",
                    label="Exposure identity",
                    status="warn",
                    message="Gross exposure is missing in live payload.",
                    actual=from_holdings,
                    expected=None,
                )
            else:
                tolerance = max(1e-6, max(abs(gross), abs(from_holdings)) * 0.01)
                ok = abs(float(gross) - from_holdings) <= tolerance
                _append(
                    check_id="exposure_identity",
                    label="Exposure identity",
                    status="pass" if ok else "fail",
                    message=(
                        "Exposure from holdings matches aggregate exposure."
                        if ok
                        else "Exposure aggregate diverges from holdings-derived value."
                    ),
                    actual=float(gross),
                    expected=from_holdings,
                )

        risk_summary = dict(payload.get("risk_summary") or {})
        if capital and risk_summary:
            reference_model = str(
                capital.get("reference_model")
                or risk_summary.get("reference_model")
                or ""
            ).lower()
            var_map = {
                str(key).lower(): value
                for key, value in dict(risk_summary.get("var") or {}).items()
            }
            es_map = {
                str(key).lower(): value
                for key, value in dict(risk_summary.get("es") or {}).items()
            }
            model_capitals = {
                str(key).lower(): dict(value or {})
                for key, value in dict(capital.get("models") or {}).items()
            }
            expected_consumed = max(
                self._coerce_float(var_map.get(reference_model)) or 0.0,
                self._coerce_float(es_map.get(reference_model)) or 0.0,
            )
            actual_consumed = self._coerce_float(
                dict(model_capitals.get(reference_model) or {}).get("consumed_eur")
            )
            if actual_consumed is None:
                _append(
                    check_id="risk_capital_alignment",
                    label="Risk vs capital",
                    status="warn",
                    message="Model capital consumed value is missing.",
                    hint="Reference model capital should track max(VaR, ES).",
                )
            else:
                tolerance = max(1.0, abs(expected_consumed) * 0.10)
                ok = abs(float(actual_consumed) - float(expected_consumed)) <= tolerance
                _append(
                    check_id="risk_capital_alignment",
                    label="Risk vs capital",
                    status="pass" if ok else "fail",
                    message=(
                        "Reference model capital aligns with live risk metrics."
                        if ok
                        else "Reference model capital diverges from live risk metrics."
                    ),
                    actual=actual_consumed,
                    expected=expected_consumed,
                    hint="Check risk_summary vs capital_usage model mapping.",
                )

        reconciliation = dict(payload.get("reconciliation") or {})
        if reconciliation:
            status_counts = {
                str(key): int(value or 0)
                for key, value in dict(reconciliation.get("status_counts") or {}).items()
            }
            mismatch_status_counts = {
                str(key): int(value or 0)
                for key, value in dict(reconciliation.get("mismatch_status_counts") or {}).items()
            }
            execution_status_counts = {
                str(key): int(value or 0)
                for key, value in dict(reconciliation.get("execution_status_counts") or {}).items()
            }
            if not mismatch_status_counts and status_counts:
                mismatch_like = {
                    "match",
                    "live_base_incomplete",
                    "desk_vs_broker_drift",
                    "orphan_live_position",
                    "orphan_live_order",
                }
                mismatch_status_counts = {
                    key: value
                    for key, value in status_counts.items()
                    if str(key).lower() in mismatch_like
                }
            if not execution_status_counts and status_counts:
                execution_status_counts = {
                    key: value
                    for key, value in status_counts.items()
                    if key not in mismatch_status_counts
                }
            mismatch_non_match_total = int(
                sum(
                    int(value or 0)
                    for key, value in mismatch_status_counts.items()
                    if str(key).lower() != "match"
                )
            )
            mismatches = [dict(item) for item in list(reconciliation.get("mismatches") or [])]
            mismatch_count = len(mismatches)
            mismatch_active_count = int(
                sum(1 for item in mismatches if str(item.get("status") or "").lower() != "match")
            )
            market_closed = bool(reconciliation.get("market_closed", False))
            if mismatch_non_match_total <= 0 and mismatch_count <= 0:
                _append(
                    check_id="reconciliation_totals",
                    label="Reconciliation totals",
                    status="pass",
                    message="No active reconciliation mismatches.",
                    actual=0,
                    expected=0,
                )
            else:
                expected_active = mismatch_active_count if mismatch_count > 0 else 0
                consistent = expected_active == mismatch_non_match_total
                _append(
                    check_id="reconciliation_totals",
                    label="Reconciliation totals",
                    status=(
                        "pass"
                        if consistent
                        else "warn"
                        if market_closed
                        else "fail"
                    ),
                    message=(
                        "Reconciliation counters are coherent."
                        if consistent
                        else "Mismatch counters and mismatch rows are not aligned."
                    ),
                    actual=mismatch_non_match_total,
                    expected=expected_active,
                    hint="When market is closed, stale broker windows may temporarily skew counters.",
                )
            execution_non_match_total = int(
                sum(
                    int(value or 0)
                    for key, value in execution_status_counts.items()
                    if str(key).lower() != "match"
                )
            )
            expected_execution_non_match_total = int(
                reconciliation.get("unmatched_execution_count") or 0
            ) + int(reconciliation.get("history_window_expired_execution_count") or 0)
            if execution_status_counts or expected_execution_non_match_total > 0:
                execution_consistent = execution_non_match_total == expected_execution_non_match_total
                _append(
                    check_id="execution_reconciliation_totals",
                    label="Execution reconciliation totals",
                    status=(
                        "pass"
                        if execution_consistent
                        else "warn"
                        if market_closed
                        else "fail"
                    ),
                    message=(
                        "Execution reconciliation counters are coherent."
                        if execution_consistent
                        else "Execution status counters are not aligned with unmatched/expired totals."
                    ),
                    actual=execution_non_match_total,
                    expected=expected_execution_non_match_total,
                    hint="Verify execution lineage reconciliation rollups when this persists.",
                )
            portfolio = self.runtime._resolve_portfolio_context(payload.get("portfolio_slug"))
            if self.runtime.strict_live_required(portfolio):
                operational_truth = str(reconciliation.get("operational_truth") or "").lower()
                truth_ok = operational_truth == "broker"
                truth_degraded = operational_truth == "broker_delayed"
                _append(
                    check_id="strict_live_truth",
                    label="Strict live truth",
                    status=(
                        "pass"
                        if truth_ok
                        else "warn"
                        if truth_degraded or market_closed
                        else "fail"
                    ),
                    message=(
                        "Broker is the active source of operational truth."
                        if truth_ok
                        else "Broker evidence is temporarily delayed; using the latest broker-backed reference."
                        if truth_degraded
                        else "Operational truth is not broker-backed in strict live mode."
                    ),
                    actual=operational_truth or None,
                    expected="broker",
                    hint=(
                        "If market is closed or broker evidence is delayed, this may remain informational until the next broker session."
                    ),
                )

        truth_score = None if total <= 0.0 else round(float(score / total), 4)
        return checks, truth_score

    def _archive_live_ticks_if_needed(
        self,
        raw_state: Mapping[str, Any],
        *,
        portfolio_slug: str,
        account_id: str | None = None,
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
        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        if sequence > 0 and self._shared_last_tick_archive_sequence.get(scope_key) == sequence:
            return self.runtime.market_data.tick_archive_summary(symbols=symbols)
        archived = self.runtime.market_data.archive_live_ticks_from_state(raw_state, portfolio_slug=portfolio_slug)
        if sequence > 0:
            self._shared_last_tick_archive_sequence[scope_key] = sequence
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

    def analytics_series(
        self,
        *,
        portfolio_slug: str | None = None,
        window_minutes: int = 240,
        max_points: int = 300,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_window_minutes = max(int(window_minutes), 15)
        resolved_max_points = max(int(max_points), 50)
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=resolved_window_minutes)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        current_state = self.live_state(
            portfolio_slug=portfolio["slug"],
            detail_level="summary",
            account_id=resolved_account_id,
        )
        events = self.live_events(
            portfolio_slug=portfolio["slug"],
            after=0,
            limit=max(resolved_max_points * 4, 200),
            wait_seconds=0.0,
            detail_level="summary",
            account_id=resolved_account_id,
        )
        states = [dict(item.get("state") or {}) for item in events]
        states.append(dict(current_state))

        point_by_timestamp: dict[str, dict[str, Any]] = {}
        for state in states:
            account = dict(state.get("account") or {})
            if not account and not state:
                continue
            timestamp = self._to_utc_datetime(
                account.get("timestamp_utc")
                or dict(state.get("risk_summary") or {}).get("generated_at")
                or state.get("generated_at")
            )
            if timestamp is None:
                continue
            if timestamp < window_start:
                continue
            tick_reference = self._to_utc_datetime(
                dict(state.get("tick_quality") or {}).get("market_reference_timestamp")
                or dict(state.get("reconciliation") or {}).get("market_reference_timestamp")
                or state.get("market_reference_timestamp")
            )
            tick_age_seconds = None
            if tick_reference is not None:
                tick_age_seconds = max((timestamp - tick_reference).total_seconds(), 0.0)
            key = timestamp.isoformat()
            point_by_timestamp[key] = {
                "timestamp": key,
                "balance": self._coerce_float(account.get("balance")),
                "equity": self._coerce_float(account.get("equity")),
                "margin_free": self._coerce_float(account.get("margin_free")),
                "margin_level": self._coerce_float(account.get("margin_level")),
                "profit": self._coerce_float(account.get("profit")),
                "avg_spread_bps": self._coerce_float(
                    dict(state.get("microstructure") or {}).get("avg_spread_bps")
                ),
                "tick_age_seconds": tick_age_seconds,
                "tick_quality_status": dict(state.get("tick_quality") or {}).get("status"),
            }

        points = [point_by_timestamp[key] for key in sorted(point_by_timestamp.keys())]
        if len(points) > resolved_max_points:
            step = max(len(points) // resolved_max_points, 1)
            sampled = points[::step]
            if sampled[-1] != points[-1]:
                sampled.append(points[-1])
            points = sampled[-resolved_max_points:]

        return {
            "generated_at": now.isoformat(),
            "portfolio_slug": portfolio["slug"],
            "account_id": resolved_account_id,
            "window_minutes": resolved_window_minutes,
            "market_closed": bool(current_state.get("market_closed", False)),
            "market_reference_timestamp": current_state.get("market_reference_timestamp"),
            "points": points,
        }

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
        market_closed = bool(raw_state.get("market_closed", False))
        market_closed_reason = raw_state.get("market_closed_reason")
        market_reference_timestamp = raw_state.get("market_reference_timestamp") or tick_archive.get("latest_tick_at")
        market_reference_source = raw_state.get("market_reference_source") or (
            "tick_archive" if tick_archive.get("latest_tick_at") else None
        )
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

        status = "healthy" if items and healthy == len(items) else ("stale" if stale > 0 else "incomplete")
        if market_closed:
            status = "market_closed"
        resolved_regime = "closed" if market_closed else regime
        tick_quality = {
            **dict(tick_archive.get("tick_quality") or {}),
            "status": status,
            "healthy_symbols": healthy,
            "stale_symbols": stale,
            "incomplete_symbols": incomplete,
            "market_closed": market_closed,
            "market_closed_reason": market_closed_reason,
            "market_reference_timestamp": market_reference_timestamp,
            "market_reference_source": market_reference_source,
        }
        microstructure = {
            **dict(tick_archive.get("microstructure") or {}),
            "generated_at": str(raw_state.get("generated_at") or datetime.now(timezone.utc).isoformat()),
            "status": str(tick_archive.get("coverage_status") or "incomplete"),
            "regime": resolved_regime,
            "avg_spread_bps": None if not spread_bps_values else float(sum(spread_bps_values) / len(spread_bps_values)),
            "widest_spread_bps": widest_spread_bps,
            "widest_symbol": widest_symbol,
            "realized_vol_30m": None if not vol_values else float(sum(vol_values) / len(vol_values)),
            "retention_tiers": self.runtime.market_data.retention_days_by_timeframe(),
            "tick_archive_rows": int(tick_archive.get("row_count") or 0),
            "tick_archive_partitions": int(tick_archive.get("partition_count") or 0),
            "market_closed": market_closed,
            "market_closed_reason": market_closed_reason,
            "market_reference_timestamp": market_reference_timestamp,
            "market_reference_source": market_reference_source,
            "session_regime": regime,
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
        if quality_status not in {"healthy", "market_closed"}:
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
        if quality_status not in {"healthy", "market_closed"}:
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

    def _instrument_metadata(self, symbol: str, *, account_id: str | None = None) -> dict[str, Any]:
        normalized = str(symbol).upper()
        instruments = self.runtime.storage.list_instruments(symbols=[normalized]) if self.runtime.storage_ready else []
        instrument = {} if not instruments else dict(instruments[0])
        if any(instrument.get(key) not in {None, 0, 0.0} for key in ("contract_size", "tick_size", "tick_value")):
            return instrument
        try:
            with self.runtime._mt5_gateway(account_id=account_id) as live:
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
        account_id: str | None = None,
    ) -> float | None:
        if spread is None or volume_lots == 0:
            return None
        instrument = self._instrument_metadata(symbol, account_id=account_id)
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
            "ticks": {
                str(symbol).upper(): {
                    "bid": dict(tick or {}).get("bid"),
                    "ask": dict(tick or {}).get("ask"),
                    "last": dict(tick or {}).get("last"),
                    "time_utc": dict(tick or {}).get("time_utc"),
                }
                for symbol, tick in dict(raw_state.get("ticks") or {}).items()
                if str(symbol or "").strip()
            },
            "holdings": list(raw_state.get("holdings") or []),
            "pending_orders": list(raw_state.get("pending_orders") or []),
            "order_history": list(raw_state.get("order_history") or []),
            "deal_history": list(raw_state.get("deal_history") or []),
            "effective_history_lookback_minutes": raw_state.get("effective_history_lookback_minutes"),
            "market_closed": bool(raw_state.get("market_closed", False)),
            "market_closed_reason": raw_state.get("market_closed_reason"),
            "market_reference_timestamp": raw_state.get("market_reference_timestamp"),
            "market_reference_source": raw_state.get("market_reference_source"),
        }
        return f"state:{json.dumps(payload, sort_keys=True, default=str)}"

    def _persist_live_analytics_if_needed(
        self,
        *,
        raw_state: dict[str, Any],
        analytics: dict[str, Any] | None,
        portfolio_slug: str | None = None,
    ) -> None:
        if analytics is None or analytics.get("bundle") is None or not self.runtime.storage_ready:
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

    def _apply_live_detail_level(
        self,
        payload: Mapping[str, Any],
        *,
        detail_level: Literal["summary", "full", "inspector"] = "full",
    ) -> dict[str, Any]:
        if detail_level in {"full", "inspector"}:
            return deepcopy(dict(payload))

        compact = deepcopy(dict(payload))
        compact["order_history"] = []
        compact["deal_history"] = []

        reconciliation = dict(compact.get("reconciliation") or {})
        if reconciliation:
            reconciliation["mismatches"] = []
            reconciliation["incidents"] = []
            reconciliation["recent_execution_attempts"] = []
            reconciliation["recent_fills"] = []
            compact["reconciliation"] = reconciliation

        return compact

    def _build_live_state_direct(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        account_config = self.runtime.mt5_config_for_account(resolved_account_id)
        connector = self.runtime._build_mt5_connector(account_id=resolved_account_id)
        try:
            connector.init()
            raw_state = collect_live_state_from_connector(
                connector,
                config=account_config,
                base_currency=str(portfolio["base_currency"]),
                seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                history_lookback_minutes=int(account_config.live_history_lookback_minutes),
            )
        finally:
            connector.shutdown()
        payload = self._normalize_live_state(raw_state)
        payload["account_id"] = resolved_account_id
        return payload

    def _clear_scope_caches(self, *, portfolio_slug: str, account_id: str | None = None) -> None:
        scope_key = self._live_scope_key(portfolio_slug, account_id=account_id)
        for key in list(self._shared_enriched_live_state_cache):
            if key[:2] == scope_key:
                del self._shared_enriched_live_state_cache[key]
        for key in list(self._shared_live_state_response_cache):
            if key[:2] == scope_key:
                del self._shared_live_state_response_cache[key]
        if scope_key in self._shared_last_good_live_state:
            del self._shared_last_good_live_state[scope_key]

    def _reconciliation_heal_interval_seconds(self) -> float:
        configured = max(float(self.runtime.mt5_config.live_poll_seconds), 0.5)
        return min(max(configured * 20.0, 30.0), 180.0)

    def _reconciliation_auto_resolve_streak_threshold(self) -> int:
        return 2

    def _reconciliation_heal_window_days(self) -> int:
        return 30

    def _auto_heal_execution_reconciliation(self, *, portfolio_slug: str) -> dict[str, int] | None:
        if not self.runtime.storage_ready:
            return None
        scope_key = self._live_scope_key(portfolio_slug)
        now_monotonic = time.monotonic()
        last_run = float(self._shared_last_execution_reconciliation_heal_at.get(scope_key) or 0.0)
        if (now_monotonic - last_run) < self._reconciliation_heal_interval_seconds():
            return None
        self._shared_last_execution_reconciliation_heal_at[scope_key] = now_monotonic

        executions = self.runtime.storage.recent_execution_results(limit=500, portfolio_slug=portfolio_slug)
        candidates: list[dict[str, Any]] = []
        oldest_created_at: datetime | None = None
        for execution in executions:
            status = str(execution.get("reconciliation_status") or "").strip().lower()
            if status not in {"pending_broker", "history_window_expired"}:
                continue
            deal_ticket = self._coerce_int(execution.get("mt5_deal_ticket"))
            order_ticket = self._coerce_int(execution.get("mt5_order_ticket"))
            if deal_ticket is None and order_ticket is None:
                continue
            execution_id = self._coerce_int(execution.get("id"))
            if execution_id is None:
                continue
            created_at = self._to_utc_datetime(execution.get("created_at") or execution.get("time_utc"))
            if created_at is not None and (oldest_created_at is None or created_at < oldest_created_at):
                oldest_created_at = created_at
            candidates.append(
                {
                    "id": execution_id,
                    "payload": dict(execution),
                    "deal_ticket": deal_ticket,
                    "order_ticket": order_ticket,
                }
            )

        if not candidates:
            return {"checked": 0, "healed": 0}

        date_to = datetime.now(timezone.utc) + timedelta(minutes=5)
        floor_from = date_to - timedelta(days=self._reconciliation_heal_window_days())
        if oldest_created_at is None:
            date_from = floor_from
        else:
            date_from = max(oldest_created_at - timedelta(days=1), floor_from)

        checked = 0
        healed = 0
        with self.runtime._mt5_gateway() as live:
            for candidate in candidates:
                checked += 1
                payload = dict(candidate["payload"])
                submitted_volume_lots = float(
                    payload.get("submitted_volume_lots")
                    or payload.get("approved_volume_lots")
                    or payload.get("requested_volume_lots")
                    or 0.0
                )
                deals: list[dict[str, Any]] = []
                if candidate["deal_ticket"] is not None:
                    try:
                        deals = [
                            dict(item)
                            for item in live.connector.history_deals_get(
                                date_from,
                                date_to,
                                ticket=int(candidate["deal_ticket"]),
                            )
                        ]
                    except Exception:
                        deals = []
                if not deals and candidate["order_ticket"] is not None:
                    try:
                        deals = [
                            dict(item)
                            for item in live.connector.history_deals_get(
                                date_from,
                                date_to,
                                ticket=int(candidate["order_ticket"]),
                            )
                        ]
                    except Exception:
                        deals = []
                if not deals:
                    continue

                filled_volume_lots = float(
                    sum(
                        float(item.get("volume_lots") or item.get("volume") or 0.0)
                        for item in deals
                    )
                )
                if submitted_volume_lots <= 1e-9:
                    fill_ratio = 1.0 if filled_volume_lots > 0.0 else 0.0
                else:
                    fill_ratio = float(filled_volume_lots / submitted_volume_lots)
                if fill_ratio <= 1e-9:
                    reconciliation_status = "pending_broker"
                elif fill_ratio < 0.999:
                    reconciliation_status = "partial_fill"
                elif fill_ratio > 1.001:
                    reconciliation_status = "overfill_or_volume_drift"
                else:
                    reconciliation_status = "match"
                remaining_volume_lots = (
                    0.0
                    if submitted_volume_lots <= 1e-9
                    else max(submitted_volume_lots - filled_volume_lots, 0.0)
                )
                position_id = self._coerce_int(deals[0].get("position_id"))
                deal_ticket = self._coerce_int(deals[0].get("ticket")) or candidate["deal_ticket"]
                updated = self.runtime.storage.update_execution_reconciliation(
                    int(candidate["id"]),
                    reconciliation_status=reconciliation_status,
                    filled_volume_lots=filled_volume_lots,
                    remaining_volume_lots=remaining_volume_lots,
                    fill_ratio=fill_ratio,
                    broker_status="filled" if fill_ratio > 0.0 else "pending_broker",
                    position_id=position_id,
                    mt5_order_ticket=candidate["order_ticket"],
                    mt5_deal_ticket=deal_ticket,
                )
                if updated is not None and str(payload.get("reconciliation_status") or "").lower() != reconciliation_status:
                    healed += 1

        return {"checked": checked, "healed": healed}

    def _auto_resolve_reconciliation_incidents(
        self,
        *,
        portfolio_slug: str,
        summary: Mapping[str, Any],
    ) -> dict[str, int] | None:
        if not self.runtime.storage_ready:
            return None

        threshold = self._reconciliation_auto_resolve_streak_threshold()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        scope_key = self._live_scope_key(portfolio["slug"])

        mismatch_status_by_symbol = {
            str(item.get("symbol") or "").upper(): str(item.get("status") or "").strip().lower()
            for item in list(summary.get("mismatches") or [])
            if str(item.get("symbol") or "").strip()
        }
        incidents = list(summary.get("incidents") or [])
        resolved = 0
        checked = 0

        for incident in incidents:
            symbol = str(incident.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            checked += 1
            streak_key = (*scope_key, symbol)
            incident_status = str(incident.get("incident_status") or "acknowledged").strip().lower()
            if incident_status not in {"acknowledged", "investigating"}:
                self._shared_reconciliation_match_streak.pop(streak_key, None)
                continue

            mismatch_status = mismatch_status_by_symbol.get(symbol)
            if mismatch_status != "match":
                self._shared_reconciliation_match_streak.pop(streak_key, None)
                continue

            next_streak = int(self._shared_reconciliation_match_streak.get(streak_key) or 0) + 1
            self._shared_reconciliation_match_streak[streak_key] = next_streak
            if next_streak < threshold:
                continue

            resolution_note = (
                str(incident.get("resolution_note") or "").strip()
                or "Auto-resolved after 2 consecutive MATCH reconciliation cycles."
            )
            reason = str(incident.get("reason") or "resolved_after_reconciliation").strip() or "resolved_after_reconciliation"
            operator_note = str(incident.get("operator_note") or "").strip()
            payload = {
                "portfolio_slug": portfolio["slug"],
                "auto_resolved": True,
                "auto_resolve_streak": next_streak,
                "auto_resolved_at": datetime.now(timezone.utc).isoformat(),
                "live_window_minutes": summary.get("live_window_minutes"),
                "history_window_minutes": summary.get("history_window_minutes"),
                "heal_window_days": summary.get("heal_window_days"),
            }
            acknowledgement_id = self.runtime.storage.upsert_reconciliation_acknowledgement(
                portfolio_id=portfolio_id,
                symbol=symbol,
                reason=reason,
                operator_note=operator_note,
                mismatch_status="match",
                incident_status="resolved",
                resolution_note=resolution_note,
                payload=payload,
            )
            self.runtime.storage.record_audit_event(
                actor="system",
                action_type="reconciliation.auto_resolve",
                object_type="reconciliation_mismatch",
                object_id=acknowledgement_id,
                payload={
                    "portfolio_slug": portfolio["slug"],
                    "symbol": symbol,
                    "auto_resolve_streak": next_streak,
                    "resolution_note": resolution_note,
                },
                portfolio_id=portfolio_id,
            )
            self._shared_reconciliation_match_streak.pop(streak_key, None)
            resolved += 1

        return {"checked": checked, "resolved": resolved}

    def _enrich_live_state(
        self,
        raw_state: dict[str, Any],
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        self._check_startup_import(
            raw_state,
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        try:
            healed = self._auto_heal_execution_reconciliation(portfolio_slug=portfolio["slug"])
        except Exception:
            healed = None
        if healed is not None and int(healed.get("healed") or 0) > 0:
            self._clear_scope_caches(portfolio_slug=portfolio["slug"], account_id=resolved_account_id)
        exposure = self.runtime.market_data.exposure_from_holdings(
            list(raw_state.get("holdings") or []),
            portfolio_slug=portfolio["slug"],
        )
        reconciliation = self.runtime.market_data.reconciliation_summary_from_live_state(
            raw_state,
            portfolio_slug=portfolio["slug"],
        )
        try:
            autoresolve = self._auto_resolve_reconciliation_incidents(
                portfolio_slug=portfolio["slug"],
                summary=reconciliation,
            )
        except Exception:
            autoresolve = None
        if autoresolve is not None and int(autoresolve.get("resolved") or 0) > 0:
            self._clear_scope_caches(portfolio_slug=portfolio["slug"], account_id=resolved_account_id)
            reconciliation = self.runtime.market_data.reconciliation_summary_from_live_state(
                raw_state,
                portfolio_slug=portfolio["slug"],
            )
        reconciliation["autoresolved_count"] = int((autoresolve or {}).get("resolved") or 0)
        strict_live = self.runtime.strict_live_required(portfolio)
        live_base_ready = bool(reconciliation.get("live_base_ready", False))
        enriched = {
            **dict(raw_state),
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "account_id": resolved_account_id,
            "exposure": exposure,
            "reconciliation": reconciliation,
        }
        enriched["market_closed"] = bool(reconciliation.get("market_closed", enriched.get("market_closed", False)))
        enriched["market_closed_reason"] = reconciliation.get("market_closed_reason") or enriched.get("market_closed_reason")
        enriched["market_reference_timestamp"] = (
            reconciliation.get("market_reference_timestamp")
            or enriched.get("market_reference_timestamp")
        )
        enriched["market_reference_source"] = (
            reconciliation.get("market_reference_source")
            or enriched.get("market_reference_source")
        )
        enriched["effective_history_lookback_minutes"] = int(
            reconciliation.get("effective_history_lookback_minutes")
            or enriched.get("effective_history_lookback_minutes")
            or enriched.get("history_lookback_minutes")
            or self.runtime.mt5_config.live_history_lookback_minutes
        )
        tick_archive = self._archive_live_ticks_if_needed(
            enriched,
            portfolio_slug=portfolio["slug"],
            account_id=resolved_account_id,
        )
        microstructure, tick_quality = self._build_microstructure(enriched, tick_archive=tick_archive)
        pnl_explain = self._build_pnl_explain(enriched, microstructure=microstructure)
        enriched["microstructure"] = microstructure
        enriched["tick_quality"] = tick_quality
        enriched["pnl_explain"] = pnl_explain
        operator_alerts = [alert.to_dict() for alert in alerts_from_live_operator_state(enriched)]
        try:
            analytics = self._cached_build_live_analytics(
                enriched,
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
            )
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
        if analytics is not None and live_base_ready:
            self._remember_last_broker_analytics(
                portfolio_slug=portfolio["slug"],
                analytics=analytics,
                account_id=resolved_account_id,
            )

        using_cached_broker_fallback = False
        if strict_live and not live_base_ready:
            fallback_analytics = self._latest_broker_analytics(
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
            )
            if fallback_analytics is not None:
                analytics = fallback_analytics
                using_cached_broker_fallback = True
                fallback_generated_at = (
                    dict(fallback_analytics.get("risk_summary") or {}).get("generated_at")
                )
                operator_alerts.append(
                    {
                        "source": "risk_live",
                        "severity": "INFO",
                        "code": "LIVE_ANALYTICS_FALLBACK",
                        "message": (
                            "Broker evidence is temporarily delayed; showing the latest broker-backed "
                            "risk/capital analytics."
                        ),
                        "context": {
                            "portfolio_slug": portfolio["slug"],
                            "fallback_generated_at": fallback_generated_at,
                            "market_reference_timestamp": reconciliation.get("market_reference_timestamp"),
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
            if analytics.get("bundle") is not None:
                analytics["bundle"] = {
                    **dict(analytics["bundle"]),
                    "risk_nowcast": dict(risk_nowcast),
                    "microstructure": dict(microstructure),
                    "tick_quality": dict(tick_quality),
                    "pnl_explain": dict(pnl_explain),
                }
            operator_alerts.extend(list(analytics["alerts"]))
            if not using_cached_broker_fallback:
                self._persist_live_analytics_if_needed(
                    raw_state=enriched,
                    analytics=analytics,
                    portfolio_slug=portfolio["slug"],
                )
        else:
            enriched["risk_nowcast"] = {}
        analytics_generated_at = (
            (dict(enriched.get("risk_summary") or {}).get("generated_at"))
            or enriched.get("generated_at")
        )
        analytics_timestamp = self._to_utc_datetime(analytics_generated_at)
        max_age_seconds = max(
            60.0,
            float(self.runtime.mt5_config.live_history_poll_seconds) * 3.0,
        )
        analytics_stale = True
        if analytics_timestamp is not None:
            analytics_stale = (
                max((datetime.now(timezone.utc) - analytics_timestamp).total_seconds(), 0.0)
                > max_age_seconds
            )
        quality_checks, truth_score = self._build_quality_checks(enriched)
        enriched["operational_truth"] = dict(enriched.get("reconciliation") or {}).get("operational_truth")
        enriched["quality_checks"] = quality_checks
        enriched["truth_score"] = truth_score
        enriched["analytics_generated_at"] = analytics_generated_at
        enriched["analytics_stale"] = bool(analytics_stale)
        enriched["operator_alerts"] = self._dedupe_alerts(operator_alerts)
        enriched["health"] = self._build_live_health(enriched)
        return enriched

    def live_state(
        self,
        *,
        portfolio_slug: str | None = None,
        detail_level: Literal["summary", "full", "inspector"] = "full",
        force_refresh: bool = False,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        account_config = self.runtime.mt5_config_for_account(resolved_account_id)
        cache_key = self._live_state_response_cache_key(
            portfolio_slug=portfolio["slug"],
            detail_level=detail_level,
            account_id=resolved_account_id,
        )
        if not force_refresh:
            cached = self._shared_live_state_response_cache.get(cache_key)
            if cached is not None and float(cached.get("expires_at") or 0.0) > time.monotonic():
                return deepcopy(dict(cached.get("payload") or {}))
        with self._shared_live_state_compute_lock:
            if not force_refresh:
                cached = self._shared_live_state_response_cache.get(cache_key)
                if cached is not None and float(cached.get("expires_at") or 0.0) > time.monotonic():
                    return deepcopy(dict(cached.get("payload") or {}))
            connector = self.runtime._build_mt5_connector(account_id=resolved_account_id)
            try:
                connector.init()
                if hasattr(connector, "live_state"):
                    raw_state = self._normalize_live_state(dict(connector.live_state()))
                else:
                    raw_state = self._normalize_live_state(
                        collect_live_state_from_connector(
                            connector,
                            config=account_config,
                            base_currency=str(portfolio["base_currency"]),
                            seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                            history_lookback_minutes=int(account_config.live_history_lookback_minutes),
                        )
                    )
            except Exception as exc:
                fallback_state = self._latest_good_live_state(
                    portfolio_slug=portfolio["slug"],
                    account_id=resolved_account_id,
                )
                if fallback_state is not None:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    raw_state = self._normalize_live_state(dict(fallback_state))
                    raw_state.update(
                        {
                            "status": "degraded",
                            "connected": False,
                            "degraded": True,
                            "stale": True,
                            "fallback_snapshot_used": True,
                            "generated_at": now_iso,
                            "last_error": str(exc),
                            "bridge_last_error_at": now_iso,
                            "bridge_last_event_kind": "connection_error_fallback",
                            "bridge_consecutive_failures": max(
                                int(raw_state.get("bridge_consecutive_failures") or 0),
                                1,
                            ),
                        }
                    )
                    if raw_state.get("last_success_at") in {None, "", "null"}:
                        raw_state["last_success_at"] = str(fallback_state.get("generated_at") or now_iso)
                else:
                    raw_state = build_empty_live_state(
                        config=account_config,
                        seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                        status="degraded",
                        connected=False,
                        degraded=True,
                        stale=True,
                        last_error=str(exc),
                    )
            finally:
                connector.shutdown()

            cached_good_state = self._latest_good_live_state(
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
            )
            can_use_fast_overlay = (
                not force_refresh
                and cached_good_state is not None
            )

            if can_use_fast_overlay and cached_good_state is not None:
                enriched = self._overlay_dynamic_live_fields(
                    cached_state=cached_good_state,
                    raw_state=raw_state,
                )
            else:
                enriched = self._cached_enrich_live_state(
                    raw_state,
                    portfolio_slug=portfolio["slug"],
                    account_id=resolved_account_id,
                )
            enriched["account_id"] = resolved_account_id

            if bool(enriched.get("connected", False)) and not bool(enriched.get("stale", False)):
                self._remember_last_good_live_state(
                    portfolio_slug=portfolio["slug"],
                    payload=enriched,
                    account_id=resolved_account_id,
                )
            payload_summary = self._apply_live_detail_level(enriched, detail_level="summary")
            payload_full = self._apply_live_detail_level(enriched, detail_level="full")
            payload_inspector = self._apply_live_detail_level(enriched, detail_level="inspector")
            now_monotonic = time.monotonic()
            self._shared_live_state_response_cache[
                self._live_state_response_cache_key(
                    portfolio_slug=portfolio["slug"],
                    detail_level="summary",
                    account_id=resolved_account_id,
                )
            ] = {
                "expires_at": now_monotonic + self._detail_cache_ttl_seconds("summary"),
                "payload": deepcopy(payload_summary),
            }
            self._shared_live_state_response_cache[
                self._live_state_response_cache_key(
                    portfolio_slug=portfolio["slug"],
                    detail_level="full",
                    account_id=resolved_account_id,
                )
            ] = {
                "expires_at": now_monotonic + self._detail_cache_ttl_seconds("full"),
                "payload": deepcopy(payload_full),
            }
            self._shared_live_state_response_cache[
                self._live_state_response_cache_key(
                    portfolio_slug=portfolio["slug"],
                    detail_level="inspector",
                    account_id=resolved_account_id,
                )
            ] = {
                "expires_at": now_monotonic + self._detail_cache_ttl_seconds("inspector"),
                "payload": deepcopy(payload_inspector),
            }
            if detail_level == "summary":
                return payload_summary
            if detail_level == "inspector":
                return payload_inspector
            return payload_full

    def live_events(
        self,
        *,
        portfolio_slug: str | None = None,
        after: int = 0,
        limit: int = 100,
        wait_seconds: float = 15.0,
        detail_level: Literal["summary", "full", "inspector"] = "full",
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        account_config = self.runtime.mt5_config_for_account(resolved_account_id)
        connector = self.runtime._build_mt5_connector(account_id=resolved_account_id)
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
                            config=account_config,
                            base_currency=str(portfolio["base_currency"]),
                            seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                            history_lookback_minutes=int(account_config.live_history_lookback_minutes),
                        ),
                    }
                ]
        except Exception as exc:
            fallback_state = self._latest_good_live_state(
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
            )
            if fallback_state is not None:
                now_iso = datetime.now(timezone.utc).isoformat()
                degraded_state = self._normalize_live_state(dict(fallback_state))
                degraded_state.update(
                    {
                        "status": "degraded",
                        "connected": False,
                        "degraded": True,
                        "stale": True,
                        "fallback_snapshot_used": True,
                        "generated_at": now_iso,
                        "last_error": str(exc),
                        "bridge_last_error_at": now_iso,
                        "bridge_last_event_kind": "connection_error_fallback",
                        "bridge_consecutive_failures": max(
                            int(degraded_state.get("bridge_consecutive_failures") or 0),
                            1,
                        ),
                    }
                )
                if degraded_state.get("last_success_at") in {None, "", "null"}:
                    degraded_state["last_success_at"] = str(fallback_state.get("generated_at") or now_iso)
                error_state = degraded_state
            else:
                error_state = build_empty_live_state(
                    config=account_config,
                    seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                    status="degraded",
                    connected=False,
                    degraded=True,
                    stale=True,
                    last_error=str(exc),
                )
            events = [
                {
                    "sequence": int(after) + 1,
                    "kind": "connection_error",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "change_summary": {},
                    "state": error_state,
                }
            ]
        finally:
            connector.shutdown()
        enriched = []
        for event in events[: max(int(limit), 1)]:
            payload = dict(event)
            enriched_state = self._cached_enrich_live_state(
                self._normalize_live_state(dict(event.get("state") or {})),
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
            )
            enriched_state["account_id"] = resolved_account_id
            payload["state"] = self._apply_live_detail_level(
                enriched_state,
                detail_level=detail_level,
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

        if analytics is not None and analytics.get("bundle") is not None:
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

    def mt5_status(self, *, account_id: str | None = None) -> dict[str, Any]:
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        account_config = self.runtime.mt5_config_for_account(resolved_account_id)
        try:
            with self.runtime._mt5_gateway(account_id=resolved_account_id) as live:
                payload = live.terminal_status().to_dict()
                payload["account_id"] = resolved_account_id
                return payload
        except MT5ConnectionError as exc:
            payload = MT5TerminalStatus(
                connected=False,
                ready=False,
                execution_enabled=bool(account_config.execution_enabled),
                trade_allowed=None,
                tradeapi_disabled=None,
                company=None,
                terminal_path=account_config.path,
                data_path=None,
                commondata_path=None,
                message=str(exc),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                raw={},
            ).to_dict()
            payload["account_id"] = resolved_account_id
            return payload

    def mt5_account(self, *, account_id: str | None = None) -> dict[str, Any]:
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        with self.runtime._mt5_gateway(account_id=resolved_account_id) as live:
            payload = live.account_snapshot().to_dict()
            payload["account_id"] = resolved_account_id
            return payload

    def mt5_accounts(self) -> dict[str, Any]:
        accounts = self.runtime.list_mt5_accounts()
        return {
            "active_account_id": self.runtime.resolve_mt5_account_id(None),
            "accounts": accounts,
        }

    def mt5_positions(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        portfolio = None if portfolio_slug is None else self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        with self.runtime._mt5_gateway(account_id=resolved_account_id) as live:
            symbols = None
            if portfolio is not None and not self.runtime.market_data.should_use_mt5_market_data(portfolio):
                symbols = portfolio["symbols"]
            return [
                {
                    **item.to_dict(),
                    "account_id": resolved_account_id,
                }
                for item in live.positions(symbols=symbols)
            ]

    def mt5_orders(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        portfolio = None if portfolio_slug is None else self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        with self.runtime._mt5_gateway(account_id=resolved_account_id) as live:
            symbols = None
            if portfolio is not None and not self.runtime.market_data.should_use_mt5_market_data(portfolio):
                symbols = portfolio["symbols"]
            return [
                {
                    **item.to_dict(),
                    "account_id": resolved_account_id,
                }
                for item in live.pending_orders(symbols=symbols)
            ]

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

    def _preview_execution_result_payload(
        self,
        *,
        requested_exposure_change: float,
        portfolio_slug: str,
        account_id: str,
        symbol: str,
        timestamp_utc: str,
        terminal_status: MT5TerminalStatus,
        account: Any,
        live_positions: list[Any],
        guard: Any,
        decision: Mapping[str, Any],
        order_request: Mapping[str, Any],
        order_check: Mapping[str, Any],
        post_capital: Mapping[str, Any],
    ) -> dict[str, Any]:
        requested_volume_raw = order_request.get("volume")
        requested_volume_lots = None if requested_volume_raw is None else float(requested_volume_raw)
        approved_volume_lots = float(getattr(guard, "volume_lots", 0.0) or 0.0)
        broker_status = "preview_ready" if bool(getattr(guard, "submit_allowed", False)) else "preview_blocked"
        payload = ExecutionResult(
            time_utc=timestamp_utc,
            portfolio_slug=portfolio_slug,
            symbol=symbol,
            status="PREVIEW",
            requested_exposure_change=float(requested_exposure_change),
            approved_exposure_change=float(decision.get("approved_exposure_change", 0.0)),
            executed_exposure_change=0.0,
            terminal_status=terminal_status,
            account_before=account,
            account_after=None,
            guard=guard,
            risk_decision=dict(decision),
            order_request=dict(order_request),
            order_check=dict(order_check),
            mt5_result={},
            positions_after=list(live_positions),
            post_capital=dict(post_capital),
        ).to_dict()
        payload.update(
            {
                "account_id": account_id,
                "requested_volume_lots": requested_volume_lots,
                "approved_volume_lots": approved_volume_lots,
                "submitted_volume_lots": 0.0,
                "filled_volume_lots": 0.0,
                "remaining_volume_lots": approved_volume_lots,
                "fill_ratio": 0.0,
                "broker_status": broker_status,
                "position_id": None,
                "slippage_points": None,
                "reconciliation_status": "preview_only",
                "mt5_order_ticket": None,
                "mt5_deal_ticket": None,
                "fills": [],
                "preview_only": True,
                "created_at": timestamp_utc,
            }
        )
        return payload

    def preview_execution(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        normalized_exposure_change = (
            float(exposure_change) if exposure_change is not None else float(delta_position_eur or 0.0)
        )
        with self.runtime._mt5_gateway(account_id=resolved_account_id) as live:
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
                account_id=resolved_account_id,
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
                account_id=resolved_account_id,
            )
            expected_slippage_points = None
            instrument = self._instrument_metadata(str(symbol).upper(), account_id=resolved_account_id)
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
            preview_timestamp = datetime.now(timezone.utc).isoformat()
            preview = ExecutionPreview(
                time_utc=preview_timestamp,
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
            preview["account_id"] = resolved_account_id
            preview_result_payload = self._preview_execution_result_payload(
                requested_exposure_change=normalized_exposure_change,
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
                symbol=str(symbol).upper(),
                timestamp_utc=preview_timestamp,
                terminal_status=terminal_status,
                account=account,
                live_positions=live_positions,
                guard=guard,
                decision=decision,
                order_request=order_request,
                order_check=order_check,
                post_capital=post_capital,
            )
            preview_execution_id = self.runtime.storage.record_execution_result(
                preview_result_payload,
                portfolio_id=portfolio_id,
                decision_id=decision.get("id"),
            )
            self.runtime.storage.record_audit_event(
                actor="api",
                action_type="execution.preview",
                object_type="execution_result",
                object_id=preview_execution_id,
                payload={
                    "portfolio_slug": portfolio["slug"],
                    "account_id": resolved_account_id,
                    "symbol": str(symbol).upper(),
                    "decision": guard.decision,
                    "risk_decision": decision.get("decision"),
                    "execution_result_id": preview_execution_id,
                },
                portfolio_id=portfolio_id,
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
        account_id: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        normalized_exposure_change = (
            float(exposure_change) if exposure_change is not None else float(delta_position_eur or 0.0)
        )
        with self.runtime._mt5_gateway(account_id=resolved_account_id) as live:
            terminal_status = live.terminal_status()
            account_before = live.account_snapshot()
            bundle = self._compute_execution_bundle(portfolio=portfolio, live=live)
            decision = self.runtime._evaluate_trade_decision_from_bundle(
                bundle=bundle,
                symbol=symbol,
                exposure_change=normalized_exposure_change,
                note=note,
                account_id=resolved_account_id,
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
                    "account_id": resolved_account_id,
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
                    "account_id": resolved_account_id,
                    "symbol": str(symbol).upper(),
                    "status": status,
                    "decision_id": decision.get("id"),
                    "mt5_result": mt5_result,
                },
                portfolio_id=portfolio_id,
            )
            return result_payload
