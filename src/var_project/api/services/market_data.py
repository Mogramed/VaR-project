from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
import time
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd

from var_project.core.exceptions import MT5ConnectionError
from var_project.execution.mt5_live import MT5LiveGateway
from var_project.market_data.transforms import compute_log_returns, intraday_to_daily_log_returns
from var_project.portfolio.holdings import aggregate_exposure_by_symbol, gross_exposure_base_ccy, normalize_holdings
from var_project.storage.tick_archive import archive_ticks, summarize_tick_archive

if TYPE_CHECKING:
    from var_project.api.services.runtime import DeskServiceRuntime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _is_fx_weekend_closed(now: datetime | None = None) -> bool:
    current = _utcnow() if now is None else now.astimezone(timezone.utc)
    weekday = current.weekday()
    if weekday == 4 and current.hour >= 21:
        return True
    if weekday == 5:
        return True
    if weekday == 6 and current.hour < 21:
        return True
    return False


def _bars_per_day(timeframe: str) -> int:
    tf = str(timeframe).upper().strip()
    minutes_map = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M4": 4,
        "M5": 5,
        "M10": 10,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H2": 120,
        "H4": 240,
        "H6": 360,
        "H8": 480,
        "H12": 720,
        "D1": 1440,
    }
    minutes = minutes_map.get(tf, 60)
    return max(int(1440 / minutes), 1)


def _timeframe_minutes(timeframe: str) -> int:
    tf = str(timeframe).upper().strip()
    minutes_map = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M4": 4,
        "M5": 5,
        "M10": 10,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H2": 120,
        "H4": 240,
        "H6": 360,
        "H8": 480,
        "H12": 720,
        "D1": 1440,
    }
    return int(minutes_map.get(tf, 60))


class DeskMarketDataService:
    _shared_tick_archive_summary_cache: dict[tuple[str, str], dict[str, Any]] = {}
    _shared_market_status_cache: dict[tuple[str, str], dict[str, Any]] = {}
    _shared_status_autosync_attempt_at: dict[tuple[str, str], float] = {}
    _shared_market_status_cache_lock = Lock()
    _shared_market_status_compute_locks: dict[tuple[str, str], Lock] = {}
    _shared_status_autosync_lock = Lock()

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

    def history_backfill_days(self, requested_days: int | None = None) -> int:
        configured = int(self.runtime.data_defaults.get("market_history_days") or self.runtime._default_days())
        if requested_days is None:
            return max(configured, 1)
        return max(int(requested_days), configured, 1)

    def retention_days_by_timeframe(self) -> dict[str, int]:
        configured = {
            str(timeframe).upper(): int(days)
            for timeframe, days in dict(self.runtime.data_defaults.get("market_retention_days") or {}).items()
        }
        if configured:
            return configured
        fallback = self.history_backfill_days()
        return {"M1": min(fallback, 180), "H1": fallback, "D1": fallback}

    def startup_sync_timeframes(self) -> list[str]:
        configured = self.retention_days_by_timeframe()
        default_timeframe = str(self.runtime._default_timeframe()).upper()
        selected: list[str] = []
        for timeframe in (default_timeframe, "D1"):
            normalized = str(timeframe).upper()
            if normalized in configured and normalized not in selected:
                selected.append(normalized)
        if not selected:
            selected.append(default_timeframe)
        return selected

    def tick_retention_days(self) -> int:
        return max(int(self.runtime.data_defaults.get("tick_retention_days") or 30), 1)

    def tick_archive_root(self) -> Path:
        return self.runtime.storage.settings.tick_archive_dir

    def _tick_archive_summary_cache_key(self, *, symbols: Iterable[str] | None = None) -> tuple[str, str]:
        normalized_symbols = sorted(
            {
                str(symbol).upper()
                for symbol in symbols or []
                if str(symbol or "").strip()
            }
        )
        return (str(self.tick_archive_root().resolve()), ",".join(normalized_symbols))

    def _market_status_scope_key(self, *, portfolio_slug: str) -> tuple[str, str]:
        return (str(self.runtime.root.resolve()), str(portfolio_slug))

    def _market_status_cache_key(self, *, portfolio_slug: str) -> tuple[str, str]:
        return self._market_status_scope_key(portfolio_slug=portfolio_slug)

    @staticmethod
    def _market_status_cache_ttl_seconds() -> float:
        return 1.5

    def _cached_market_status(self, *, cache_key: tuple[str, str]) -> tuple[dict[str, Any] | None, bool]:
        with self._shared_market_status_cache_lock:
            cached = self._shared_market_status_cache.get(cache_key)
            if cached is None:
                return None, False
            expired = float(cached.get("expires_at") or 0.0) <= time.monotonic()
            payload = dict(cached.get("payload") or {})
        if not payload:
            return None, False
        return deepcopy(payload), bool(expired)

    def _store_market_status_cache(self, *, cache_key: tuple[str, str], payload: Mapping[str, Any]) -> None:
        with self._shared_market_status_cache_lock:
            self._shared_market_status_cache[cache_key] = {
                "expires_at": time.monotonic() + self._market_status_cache_ttl_seconds(),
                "payload": deepcopy(dict(payload)),
            }

    def _invalidate_market_status_cache(self, *, portfolio_slug: str | None = None) -> None:
        scope_root = str(self.runtime.root.resolve())
        with self._shared_market_status_cache_lock:
            if portfolio_slug is None:
                for key in list(self._shared_market_status_cache):
                    if key[0] == scope_root:
                        del self._shared_market_status_cache[key]
                return
            cache_key = (scope_root, str(portfolio_slug))
            self._shared_market_status_cache.pop(cache_key, None)

    def _market_status_lock(self, *, cache_key: tuple[str, str]) -> Lock:
        with self._shared_market_status_cache_lock:
            lock = self._shared_market_status_compute_locks.get(cache_key)
            if lock is None:
                lock = Lock()
                self._shared_market_status_compute_locks[cache_key] = lock
            return lock

    @staticmethod
    def _tick_archive_summary_cache_ttl_seconds() -> float:
        return 2.0

    @staticmethod
    def _market_status_autosync_cooldown_seconds() -> float:
        return 2.0

    def _should_autosync_market_status(
        self,
        *,
        portfolio_slug: str,
        configured: bool,
        status: str,
        latest_sync: Mapping[str, Any] | None,
        missing_symbols: list[str],
        missing_bars: list[str],
    ) -> bool:
        if not self.runtime._has_custom_mt5_factory:
            return False
        if not configured:
            return False
        if str(status).lower() != "incomplete":
            return False
        # Only opportunistically bootstrap once when no sync run exists yet.
        # Existing runs (including stale/incomplete diagnostics) should remain observable.
        if latest_sync is not None:
            return False
        if not missing_symbols and not missing_bars:
            return False

        scope_key = self._market_status_scope_key(portfolio_slug=portfolio_slug)
        now_monotonic = time.monotonic()
        cooldown = self._market_status_autosync_cooldown_seconds()
        with self._shared_status_autosync_lock:
            last_attempt = float(self._shared_status_autosync_attempt_at.get(scope_key) or 0.0)
            if last_attempt > 0.0 and (now_monotonic - last_attempt) < cooldown:
                return False
            self._shared_status_autosync_attempt_at[scope_key] = now_monotonic
        return True

    def _invalidate_tick_archive_summary_cache(self) -> None:
        root_key = str(self.tick_archive_root().resolve())
        for key in list(self._shared_tick_archive_summary_cache):
            if key[0] == root_key:
                del self._shared_tick_archive_summary_cache[key]

    def _selected_timeframe_days(
        self,
        *,
        requested_days: int | None = None,
        timeframes: Iterable[str] | None = None,
    ) -> dict[str, int]:
        retention = self.retention_days_by_timeframe()
        if timeframes:
            selected = [str(item).upper() for item in timeframes]
        else:
            selected = list(retention.keys())
        payload: dict[str, int] = {}
        for timeframe in selected:
            base_days = retention.get(timeframe, self.history_backfill_days(requested_days))
            if requested_days is not None:
                base_days = max(base_days, int(requested_days))
            payload[timeframe] = int(max(base_days, 1))
        return payload

    def _incremental_overlap_minutes(self, timeframe: str) -> int:
        return max(int(_timeframe_minutes(timeframe) * 3), 15)

    def _history_reconciliation_days(
        self,
        *,
        portfolio_slug: str,
        requested_days: int,
        now: datetime,
    ) -> int:
        base_days = min(max(int(requested_days or 0), 7), 30)
        if _is_fx_weekend_closed(now):
            base_days = max(base_days, 3)
        oldest_unresolved: datetime | None = None
        for execution in self.runtime.storage.recent_execution_results(limit=500, portfolio_slug=portfolio_slug):
            status = str(execution.get("reconciliation_status") or "").strip().lower()
            if status == "match":
                continue
            executed_at = _coerce_utc_datetime(execution.get("created_at") or execution.get("time_utc"))
            if executed_at is None:
                continue
            if oldest_unresolved is None or executed_at < oldest_unresolved:
                oldest_unresolved = executed_at
        if oldest_unresolved is not None:
            anchored_days = int(max((now - oldest_unresolved).total_seconds(), 0.0) // 86400) + 1
            base_days = max(base_days, min(anchored_days, 30))
        return min(max(int(base_days), 1), 30)

    def _register_tick_partition_artifact(self, path: Path, details: Mapping[str, Any]) -> None:
        if not self.runtime.storage_ready:
            return
        try:
            self.runtime.storage.register_artifact(
                path,
                artifact_type="market_tick_partition",
                format="parquet",
                details=details,
            )
        except Exception:
            # Tick partition registration is best-effort and should not block live-state delivery.
            return

    def tick_archive_summary(self, *, symbols: Iterable[str] | None = None) -> dict[str, Any]:
        cache_key = self._tick_archive_summary_cache_key(symbols=symbols)
        cached = self._shared_tick_archive_summary_cache.get(cache_key)
        now_monotonic = time.monotonic()
        if cached is not None and float(cached.get("expires_at") or 0.0) > now_monotonic:
            return deepcopy(dict(cached.get("payload") or {}))
        payload = summarize_tick_archive(self.tick_archive_root(), symbols=symbols)
        self._shared_tick_archive_summary_cache[cache_key] = {
            "expires_at": now_monotonic + self._tick_archive_summary_cache_ttl_seconds(),
            "payload": deepcopy(payload),
        }
        return payload

    def archive_ticks(
        self,
        *,
        symbol: str,
        ticks: Iterable[Mapping[str, Any]],
        portfolio_slug: str | None = None,
        source: str = "mt5",
    ) -> dict[str, Any]:
        archived = archive_ticks(
            root=self.tick_archive_root(),
            symbol=symbol,
            ticks=ticks,
            retention_days=self.tick_retention_days(),
            register_artifact=self._register_tick_partition_artifact,
            artifact_base_details={
                "portfolio_slug": portfolio_slug,
                "source": source,
            },
        )
        self._invalidate_tick_archive_summary_cache()
        return archived

    def archive_live_ticks_from_state(
        self,
        live_state: Mapping[str, Any],
        *,
        portfolio_slug: str | None = None,
    ) -> dict[str, Any]:
        tick_map = dict(live_state.get("ticks") or {})
        archived: list[dict[str, Any]] = []
        for symbol, payload in tick_map.items():
            archived.append(
                self.archive_ticks(
                    symbol=str(symbol).upper(),
                    ticks=[payload],
                    portfolio_slug=portfolio_slug,
                    source="mt5_live_bridge",
                )
            )
        return {
            "retention_days": self.tick_retention_days(),
            "symbols": archived,
            "summary": self.tick_archive_summary(symbols=tick_map.keys()),
        }

    def sync_market_data_if_stale(
        self,
        *,
        portfolio_slug: str | None = None,
        max_age_seconds: float = 900.0,
        days: int | None = None,
        timeframes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if not self.should_use_mt5_market_data(portfolio):
            return self.market_data_status(portfolio_slug=portfolio["slug"])
        if not self.runtime.storage_ready:
            return self.market_data_status(portfolio_slug=portfolio["slug"])

        status = self.market_data_status(portfolio_slug=portfolio["slug"])
        latest_sync = self.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio["slug"])
        needs_refresh = status["status"] != "ok" or latest_sync is None

        if not needs_refresh and latest_sync is not None:
            synced_at = pd.to_datetime(latest_sync.get("synced_at"), utc=True, errors="coerce")
            if pd.isna(synced_at):
                needs_refresh = True
            else:
                age_seconds = max(0.0, (_utcnow() - synced_at.to_pydatetime()).total_seconds())
                needs_refresh = age_seconds >= max(float(max_age_seconds), 1.0)

        if not needs_refresh:
            return status

        return self.sync_market_data(
            portfolio_slug=portfolio["slug"],
            days=days,
            timeframes=timeframes,
        )

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

    def _market_sync_running_ttl_seconds(self) -> float:
        poll_interval = max(float(self.runtime.mt5_config.live_history_poll_seconds or 0.0), 1.0)
        return max(300.0, poll_interval * 10.0)

    def _close_stale_running_sync(self, *, portfolio_slug: str) -> dict[str, Any] | None:
        latest_sync = self.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
        if latest_sync is None:
            return None
        if str(latest_sync.get("status") or "").lower() != "running":
            return latest_sync

        synced_at = _coerce_utc_datetime(latest_sync.get("synced_at"))
        if synced_at is None:
            return latest_sync
        stale_after_seconds = self._market_sync_running_ttl_seconds()
        age_seconds = max((_utcnow() - synced_at).total_seconds(), 0.0)
        if age_seconds <= stale_after_seconds:
            return latest_sync

        details = dict(latest_sync.get("details") or {})
        errors = list(details.get("errors") or [])
        errors.append(
            {
                "scope": "market_data_sync",
                "code": "stale_market_sync_run",
                "detail": (
                    "A previous market-data sync run remained in running state past its TTL "
                    "and was automatically closed as incomplete."
                ),
                "age_seconds": round(float(age_seconds), 3),
                "ttl_seconds": round(float(stale_after_seconds), 3),
            }
        )
        details["errors"] = errors[-25:]
        details["stale_closed_at"] = _utcnow().isoformat()

        return self.runtime.storage.update_market_data_sync(
            int(latest_sync["id"]),
            status="incomplete",
            details=details,
            synced_at=_utcnow(),
        )

    def live_holdings(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if not self.should_use_mt5_market_data(portfolio):
            if self.runtime.is_live_portfolio(portfolio):
                raise self.runtime.strict_live_unavailable_error(portfolio=portfolio)
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
                    "signed_exposure_base_ccy": exposure,
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

    def market_data_status(
        self,
        *,
        portfolio_slug: str | None = None,
        _allow_autosync: bool = True,
    ) -> dict[str, Any]:
        if not _allow_autosync:
            return self._compute_market_data_status(
                portfolio_slug=portfolio_slug,
                _allow_autosync=False,
            )

        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        cache_key = self._market_status_cache_key(portfolio_slug=portfolio["slug"])
        cached, expired = self._cached_market_status(cache_key=cache_key)
        if cached is not None and not expired:
            return cached

        compute_lock = self._market_status_lock(cache_key=cache_key)
        with compute_lock:
            cached, expired = self._cached_market_status(cache_key=cache_key)
            if cached is not None and not expired:
                return cached
            payload = self._compute_market_data_status(
                portfolio_slug=portfolio["slug"],
                _allow_autosync=True,
            )
            self._store_market_status_cache(cache_key=cache_key, payload=payload)
            return dict(payload)

    def _compute_market_data_status(
        self,
        *,
        portfolio_slug: str | None = None,
        _allow_autosync: bool = True,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        timeframe = self.runtime._default_timeframe()
        self._close_stale_running_sync(portfolio_slug=portfolio["slug"])
        latest_sync = self.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio["slug"])
        details = {} if latest_sync is None else dict(latest_sync.get("details") or {})
        tracked_symbols = self._portfolio_symbols_from_details(portfolio, details)
        instruments = self.runtime.storage.list_instruments(symbols=tracked_symbols)
        latest_bar_times = self.runtime.storage.latest_market_bar_times(symbols=tracked_symbols, timeframe=timeframe)
        missing_symbols = [symbol for symbol in tracked_symbols if symbol.upper() not in {item["symbol"] for item in instruments}]
        missing_bars = [symbol for symbol, ts in latest_bar_times.items() if ts is None]
        configured = self.should_use_mt5_market_data(portfolio)
        retention_tiers = self.retention_days_by_timeframe()
        tick_archive = self.tick_archive_summary(symbols=tracked_symbols)
        coverage = dict(details.get("coverage") or {})
        thin_history = False
        for symbol, by_timeframe in coverage.items():
            for tracked_timeframe, stats in dict(by_timeframe or {}).items():
                target_days = int(
                    dict(stats or {}).get("expected_history_days")
                    or dict(stats or {}).get("stored_history_days")
                    or retention_tiers.get(str(tracked_timeframe).upper(), 0)
                    or 0
                )
                bars_per_day = _bars_per_day(str(tracked_timeframe).upper())
                expected = max(int(target_days * bars_per_day * 0.8), 1) if target_days > 0 else 0
                observed = int(dict(stats or {}).get("bars") or 0)
                observed = int(dict(stats or {}).get("stored_bars") or observed)
                if expected > 0 and observed < expected:
                    thin_history = True
                    break
            if thin_history:
                break
        if not configured:
            status = "not_configured" if self.runtime.is_live_portfolio(portfolio) else "offline_fixture"
        elif latest_sync is None or missing_symbols or missing_bars:
            status = "incomplete"
        else:
            status = str(latest_sync.get("status") or "ok")
            if status.lower() == "running":
                synced_at = _coerce_utc_datetime(latest_sync.get("synced_at"))
                if synced_at is not None:
                    age_seconds = max((_utcnow() - synced_at).total_seconds(), 0.0)
                    if age_seconds > self._market_sync_running_ttl_seconds():
                        status = "incomplete"
        if _allow_autosync and self._should_autosync_market_status(
            portfolio_slug=portfolio["slug"],
            configured=configured,
            status=status,
            latest_sync=latest_sync,
            missing_symbols=missing_symbols,
            missing_bars=missing_bars,
        ):
            try:
                self.sync_market_data(
                    portfolio_slug=portfolio["slug"],
                    days=self.history_backfill_days(),
                    timeframes=self.startup_sync_timeframes(),
                )
            except Exception:
                # Autosync is best-effort; preserve current status payload on failures.
                pass
            return self.market_data_status(
                portfolio_slug=portfolio["slug"],
                _allow_autosync=False,
            )
        coverage_status = "healthy"
        if status in {"offline_fixture", "not_configured"}:
            coverage_status = status
        elif status == "incomplete":
            coverage_status = "incomplete"
        elif tick_archive.get("coverage_status") == "stale":
            coverage_status = "stale"
        elif thin_history or tick_archive.get("coverage_status") == "thin_history":
            coverage_status = "thin_history"
        market_closed = bool(details.get("market_closed", False)) or _is_fx_weekend_closed()
        return {
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "status": status,
            "coverage_status": coverage_status,
            "configured": configured,
            "timeframe": timeframe,
            "symbols": tracked_symbols,
            "instrument_count": len(instruments),
            "stored_history_days": int(
                details.get("stored_history_days")
                or self.runtime.data_defaults.get("market_history_days")
                or self.runtime._default_days()
            ),
            "market_closed": market_closed,
            "market_closed_reason": (
                details.get("market_closed_reason")
                or ("weekend" if market_closed else None)
            ),
            "retention_tiers": retention_tiers,
            "latest_sync_at": None if latest_sync is None else latest_sync.get("synced_at"),
            "latest_bar_times": latest_bar_times,
            "missing_symbols": missing_symbols,
            "missing_bars": missing_bars,
            "tick_archive": tick_archive,
            "tick_quality": dict(tick_archive.get("tick_quality") or {}),
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
        self._invalidate_market_status_cache(portfolio_slug=portfolio["slug"])
        if not self.should_use_mt5_market_data(portfolio):
            if self.runtime.strict_live_required(portfolio):
                raise MT5ConnectionError(str(self.runtime.strict_live_unavailable_error(portfolio=portfolio)))
            return self.market_data_status(portfolio_slug=portfolio["slug"])
        self._close_stale_running_sync(portfolio_slug=portfolio["slug"])

        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        requested_days = int(days or self.runtime._default_days())
        selected_timeframes = list(timeframes) if timeframes else self.startup_sync_timeframes()
        timeframe_days = self._selected_timeframe_days(
            requested_days=requested_days,
            timeframes=selected_timeframes,
        )
        selected_timeframes = list(timeframe_days.keys())
        stored_history_days = (
            max(min(int(value), requested_days) for value in timeframe_days.values())
            if timeframe_days
            else self.history_backfill_days(requested_days)
        )
        started_at = _utcnow()
        history_reconciliation_days = self._history_reconciliation_days(
            portfolio_slug=portfolio["slug"],
            requested_days=requested_days,
            now=started_at,
        )
        market_closed = _is_fx_weekend_closed(started_at)
        sync_run_id = self.runtime.storage.record_market_data_sync(
            portfolio_id=portfolio_id,
            portfolio_slug=portfolio["slug"],
            mode=str(portfolio.get("mode") or "hybrid"),
            status="running",
            details={
                "symbols": self._portfolio_symbols_from_details(portfolio),
                "timeframes": selected_timeframes,
                "requested_days": requested_days,
                "stored_history_days": stored_history_days,
                "history_reconciliation_days": history_reconciliation_days,
                "retention_tiers": timeframe_days,
                "market_closed": market_closed,
                "sync_strategy": "incremental",
            },
        )

        errors: list[dict[str, Any]] = []
        instrument_payloads: list[dict[str, Any]] = []
        order_payloads: list[dict[str, Any]] = []
        deal_payloads: list[dict[str, Any]] = []
        open_positions: list[dict[str, Any]] = []
        pending_orders: list[dict[str, Any]] = []
        coverage: dict[str, dict[str, Any]] = {}
        tick_archives: list[dict[str, Any]] = []
        tracked_symbols = self._portfolio_symbols_from_details(portfolio)
        fatal_exception: Exception | None = None

        try:
            with self.runtime._mt5_gateway() as live:
                open_positions = [item.to_dict() for item in live.holdings(symbols=None)]
                pending_orders = [item.to_dict() for item in live.pending_orders(symbols=None)]
                date_from = started_at - timedelta(days=history_reconciliation_days)

                tracked_symbols = self._portfolio_symbols_from_details(
                    portfolio,
                    {
                        "open_positions": open_positions,
                        "pending_orders": pending_orders,
                    },
                )
                latest_bar_times_by_timeframe = {
                    timeframe: self.runtime.storage.latest_market_bar_times(
                        symbols=tracked_symbols,
                        timeframe=timeframe,
                    )
                    for timeframe in selected_timeframes
                }

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
                        timeframe_days_value = int(timeframe_days.get(timeframe, stored_history_days))
                        bars_per_day = live.connector.bars_per_day(timeframe)
                        latest_bar_time = _coerce_utc_datetime(
                            latest_bar_times_by_timeframe.get(timeframe, {}).get(str(symbol).upper())
                        )
                        range_start = None if latest_bar_time is None else latest_bar_time - timedelta(
                            minutes=self._incremental_overlap_minutes(timeframe)
                        )
                        effective_history_days = max(min(timeframe_days_value, requested_days), 1)
                        fetch_mode = "bootstrap"
                        try:
                            if range_start is not None:
                                fetch_mode = "incremental"
                                bars = live.connector.fetch_bars_range(
                                    str(symbol).upper(),
                                    timeframe,
                                    range_start,
                                    started_at,
                                )
                            else:
                                bootstrap_days = max(min(timeframe_days_value, requested_days), 1)
                                bars = live.connector.fetch_last_n_bars(
                                    str(symbol).upper(),
                                    timeframe,
                                    max(int(bootstrap_days * bars_per_day) + int(bars_per_day), 10),
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
                            stored_bars = len(
                                self.runtime.storage.market_bars(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    since=started_at - timedelta(days=timeframe_days_value),
                                )
                            )
                            coverage[symbol][timeframe] = {
                                "bars": int(len(bars)),
                                "stored_bars": int(stored_bars),
                                "latest_bar_time": latest_bar_time,
                                "stored_history_days": effective_history_days,
                                "expected_history_days": effective_history_days,
                                "fetch_mode": fetch_mode,
                                "range_start": None if range_start is None else range_start.isoformat(),
                                "range_end": started_at.isoformat(),
                            }
                        except Exception as exc:
                            try:
                                fetch_mode = "fallback_last_n"
                                bootstrap_days = max(min(timeframe_days_value, requested_days), 1)
                                bars = live.connector.fetch_last_n_bars(
                                    str(symbol).upper(),
                                    timeframe,
                                    max(int(bootstrap_days * bars_per_day) + int(bars_per_day), 10),
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
                                stored_bars = len(
                                    self.runtime.storage.market_bars(
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        since=started_at - timedelta(days=timeframe_days_value),
                                    )
                                )
                                coverage[symbol][timeframe] = {
                                    "bars": int(len(bars)),
                                    "stored_bars": int(stored_bars),
                                    "latest_bar_time": latest_bar_time,
                                    "stored_history_days": effective_history_days,
                                    "expected_history_days": effective_history_days,
                                    "fetch_mode": fetch_mode,
                                    "range_start": None if range_start is None else range_start.isoformat(),
                                    "range_end": started_at.isoformat(),
                                }
                                errors.append(
                                    {
                                        "scope": "bars_range",
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "detail": str(exc),
                                        "fallback": "last_n",
                                    }
                                )
                            except Exception as fallback_exc:
                                errors.append(
                                    {
                                        "scope": "bars",
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "detail": str(fallback_exc),
                                        "range_error": str(exc),
                                    }
                                )
                                coverage[symbol][timeframe] = {
                                    "bars": 0,
                                    "stored_bars": 0,
                                    "latest_bar_time": None,
                                    "stored_history_days": effective_history_days,
                                    "expected_history_days": effective_history_days,
                                    "fetch_mode": "failed",
                                    "range_start": None if range_start is None else range_start.isoformat(),
                                    "range_end": started_at.isoformat(),
                                }

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
                    existing_tick_summary = self.tick_archive_summary(symbols=tracked_symbols)
                    existing_tick_map = {
                        str(item.get("symbol") or "").upper(): item
                        for item in list(existing_tick_summary.get("symbols") or [])
                    }
                    for symbol in tracked_symbols:
                        try:
                            latest_tick_at = pd.to_datetime(
                                existing_tick_map.get(symbol, {}).get("latest_tick_at"),
                                utc=True,
                                errors="coerce",
                            )
                            if pd.isna(latest_tick_at):
                                tick_date_from = started_at - timedelta(hours=(24 if market_closed else 6))
                            else:
                                tick_date_from = latest_tick_at.to_pydatetime() - timedelta(minutes=5)
                            if tick_date_from >= started_at:
                                tick_date_from = started_at - timedelta(hours=(24 if market_closed else 6))
                            tick_frame = live.connector.fetch_ticks_range(
                                str(symbol).upper(),
                                tick_date_from,
                                started_at,
                            )
                            tick_archives.append(
                                self.archive_ticks(
                                    symbol=str(symbol).upper(),
                                    ticks=tick_frame.to_dict(orient="records"),
                                    portfolio_slug=portfolio["slug"],
                                    source="mt5",
                                )
                            )
                        except Exception as exc:
                            errors.append({"scope": "ticks", "symbol": symbol, "detail": str(exc)})
                    bundle = self.runtime._compute_portfolio_state_for_holdings(
                        portfolio={**dict(portfolio), "watchlist_symbols": tracked_symbols, "symbols": tracked_symbols},
                        holdings=open_positions,
                        timeframe=self.runtime._default_timeframe(),
                        days=requested_days,
                        min_coverage=float(self.runtime.data_defaults["min_coverage"]),
                        config=self.runtime._build_risk_model_config(None, None, None, None, None),
                        window=int(self.runtime.risk_defaults["window"]),
                        allow_auto_sync=False,
                        snapshot_source="mt5_live",
                        snapshot_timestamp=started_at.isoformat(),
                    )
                    self.runtime._persist_live_bundle(bundle=bundle, portfolio_id=portfolio_id, source="mt5_live")
                except Exception as exc:
                    errors.append({"scope": "live_snapshot", "detail": str(exc)})
        except Exception as exc:
            fatal_exception = exc
            errors.append({"scope": "sync_market_data", "detail": str(exc)})

        completed_status = "ok" if not errors else "incomplete"
        final_details = {
            "symbols": tracked_symbols,
            "timeframes": selected_timeframes,
            "requested_days": requested_days,
            "stored_history_days": stored_history_days,
            "history_reconciliation_days": history_reconciliation_days,
            "coverage": coverage,
            "retention_tiers": timeframe_days,
            "market_closed": market_closed,
            "sync_strategy": "incremental",
            "instruments": len(instrument_payloads),
            "orders": len(order_payloads),
            "deals": len(deal_payloads),
            "tick_archive": {
                "retention_days": self.tick_retention_days(),
                "symbols": tick_archives,
                "summary": self.tick_archive_summary(symbols=tracked_symbols),
            },
            "open_positions": open_positions,
            "pending_orders": pending_orders,
            "order_history": order_payloads[:200],
            "deal_history": deal_payloads[:200],
            "errors": errors,
        }
        updated_sync = self.runtime.storage.update_market_data_sync(
            sync_run_id,
            status=completed_status,
            details=final_details,
            synced_at=_utcnow(),
        )
        if updated_sync is None:
            self.runtime.storage.record_market_data_sync(
                portfolio_id=portfolio_id,
                portfolio_slug=portfolio["slug"],
                mode=str(portfolio.get("mode") or "hybrid"),
                status=completed_status,
                details=final_details,
        )
        if fatal_exception is not None:
            raise fatal_exception
        return self.market_data_status(
            portfolio_slug=portfolio["slug"],
            _allow_autosync=False,
        )

    def load_daily_returns_for_portfolio(
        self,
        portfolio: Mapping[str, Any],
        *,
        timeframe: str,
        days: int,
        min_coverage: float,
        ensure_sync: bool = False,
        allow_auto_sync: bool = True,
        symbols: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        sync_attempted = False
        if ensure_sync:
            self.sync_market_data(portfolio_slug=str(portfolio["slug"]), days=days, timeframes=[timeframe])
            sync_attempted = True

        since = _utcnow() - timedelta(days=int(days) + 5)
        selected_symbols = [str(symbol).upper() for symbol in (symbols or portfolio["symbols"])]
        minimum_daily_observations = max(int(days // 2), 10)
        target_daily_observations = max(int(days), minimum_daily_observations)

        def _daily_returns_from_bars(rows: list[dict[str, Any]]) -> pd.DataFrame | None:
            if not rows:
                return None
            frame = pd.DataFrame(rows)
            frame["time"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
            intraday = compute_log_returns(frame[["time", "close"]], price_col="close")
            daily_frame = intraday_to_daily_log_returns(
                intraday[["time", "log_return"]],
                timeframe=timeframe,
                min_coverage=min_coverage,
            )
            if daily_frame.empty:
                return None
            return daily_frame

        while True:
            frames: list[pd.DataFrame] = []
            missing_symbol: str | None = None
            insufficient_symbol: str | None = None

            for symbol in selected_symbols:
                recent_rows = self.runtime.storage.market_bars(symbol=symbol, timeframe=timeframe, since=since)
                daily = _daily_returns_from_bars(recent_rows)
                recent_count = 0 if daily is None else int(len(daily))
                has_any_rows = bool(recent_rows)

                if daily is None or recent_count < minimum_daily_observations:
                    full_rows = self.runtime.storage.market_bars(symbol=symbol, timeframe=timeframe, since=None)
                    has_any_rows = has_any_rows or bool(full_rows)
                    full_daily = _daily_returns_from_bars(full_rows)
                    full_count = 0 if full_daily is None else int(len(full_daily))
                    if full_daily is not None and full_count > recent_count:
                        daily = full_daily

                if daily is None:
                    if has_any_rows:
                        insufficient_symbol = symbol
                    else:
                        missing_symbol = symbol
                    break
                daily = daily.tail(target_daily_observations).reset_index(drop=True)
                daily[f"{symbol}_ret"] = np.expm1(daily["daily_log_return"].astype(float))
                frames.append(daily[["date", f"{symbol}_ret"]])

            if missing_symbol is None and insufficient_symbol is None:
                break

            if not sync_attempted:
                if not allow_auto_sync:
                    if missing_symbol is not None:
                        raise RuntimeError(
                            f"MT5 market data is missing for {missing_symbol} {timeframe} and fast execution mode avoids a blocking sync."
                        )
                    raise RuntimeError(
                        f"MT5 market data coverage is insufficient for {insufficient_symbol} {timeframe} and fast execution mode avoids a blocking sync."
                    )
                self.sync_market_data(
                    portfolio_slug=str(portfolio["slug"]),
                    days=days,
                    timeframes=[timeframe],
                )
                sync_attempted = True
                continue

            if missing_symbol is not None:
                raise RuntimeError(
                    f"MT5 market data is still incomplete for {missing_symbol} {timeframe} after automatic sync."
                )
            raise RuntimeError(
                f"MT5 market data coverage is still insufficient for {insufficient_symbol} {timeframe} after automatic sync."
            )

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
        ticks: Mapping[str, Mapping[str, Any]] | None = None,
        effective_history_lookback_minutes: int | None = None,
    ) -> dict[str, Any]:
        exposure_payload = self.exposure_from_holdings(holdings, portfolio_slug=portfolio["slug"])
        live_exposure = {
            str(item["symbol"]).upper(): float(item["exposure_base_ccy"])
            for item in list(exposure_payload.get("items") or [])
        }
        def _snapshot_exposure(snapshot: Mapping[str, Any] | None) -> dict[str, float]:
            snapshot_payload = {} if snapshot is None else dict(snapshot.get("payload") or {})
            snapshot_holdings = (
                list(snapshot_payload.get("holdings") or [])
                if "holdings" in snapshot_payload
                else []
            )
            if "exposure_by_symbol" in snapshot_payload:
                exposure = dict(snapshot_payload.get("exposure_by_symbol") or {})
            elif snapshot_payload.get("positions_eur") is not None:
                exposure = dict(snapshot_payload.get("positions_eur") or {})
            else:
                exposure = aggregate_exposure_by_symbol(snapshot_holdings, base_currency=str(portfolio["base_currency"]))
            return {
                str(symbol).upper(): float(value)
                for symbol, value in exposure.items()
                if str(symbol or "").strip()
            }

        def _matches_live(snapshot: Mapping[str, Any] | None) -> bool:
            snapshot_exposure = _snapshot_exposure(snapshot)
            symbols = set(snapshot_exposure) | set(live_exposure)
            if not symbols:
                return False
            return all(abs(float(snapshot_exposure.get(symbol, 0.0)) - float(live_exposure.get(symbol, 0.0))) <= 1e-6 for symbol in symbols)

        def _accepted_live_baseline(snapshot: Mapping[str, Any] | None) -> bool:
            snapshot_payload = {} if snapshot is None else dict(snapshot.get("payload") or {})
            snapshot_metadata = dict(snapshot_payload.get("metadata") or {})
            return bool(snapshot_metadata.get("reconciliation_baseline_accepted"))

        live_portfolio = self.runtime.is_live_portfolio(portfolio)
        latest_snapshot = None
        for source in ("mt5_live_bridge", "mt5_live"):
            candidates = self.runtime.storage.recent_snapshots(limit=2, source=source, portfolio_slug=portfolio["slug"])
            if not candidates:
                continue
            latest_snapshot = candidates[0]
            if len(candidates) > 1 and _matches_live(latest_snapshot) and not _accepted_live_baseline(latest_snapshot):
                latest_snapshot = candidates[1]
            break
        if latest_snapshot is None and not live_portfolio:
            latest_snapshot = self.runtime.storage.latest_snapshot(source="historical", portfolio_slug=portfolio["slug"])
        snapshot_payload = {} if latest_snapshot is None else dict(latest_snapshot.get("payload") or {})
        if "holdings" in snapshot_payload:
            desk_holdings = list(snapshot_payload.get("holdings") or [])
        elif live_portfolio:
            desk_holdings = []
        else:
            desk_holdings = list(portfolio.get("configured_holdings") or [])
        if "exposure_by_symbol" in snapshot_payload:
            desk_exposure = dict(snapshot_payload.get("exposure_by_symbol") or {})
        elif snapshot_payload.get("positions_eur") is not None:
            desk_exposure = dict(snapshot_payload.get("positions_eur") or {})
        else:
            desk_exposure = aggregate_exposure_by_symbol(desk_holdings, base_currency=str(portfolio["base_currency"]))
        live_evidence_counts = {
            "holdings": sum(1 for item in holdings if str(item.get("symbol") or "").strip()),
            "pending_orders": sum(1 for item in pending_orders if str(item.get("symbol") or "").strip()),
            "order_history": sum(1 for item in order_history if str(item.get("symbol") or "").strip()),
            "deal_history": sum(1 for item in deal_history if str(item.get("symbol") or "").strip()),
        }
        tick_payload = {
            str(symbol).upper(): dict(payload or {})
            for symbol, payload in dict(ticks or {}).items()
            if str(symbol or "").strip()
        }
        tick_timestamps = [
            parsed
            for parsed in (
                _coerce_utc_datetime(payload.get("time_utc"))
                for payload in tick_payload.values()
            )
            if parsed is not None
        ]
        live_window_minutes = max(int(self.runtime.mt5_config.live_history_lookback_minutes or 0), 1)
        history_window_minutes = max(
            int(effective_history_lookback_minutes or self.runtime.mt5_config.live_history_lookback_minutes or 0),
            1,
        )
        heal_window_days = 30
        history_backfill_applied = history_window_minutes > live_window_minutes
        history_window_start = _utcnow() - timedelta(minutes=history_window_minutes)
        fresh_tick_count = sum(
            1
            for tick_time in tick_timestamps
            if tick_time >= history_window_start
        )
        live_evidence_counts["tick_symbols"] = len(tick_payload)
        live_evidence_counts["ticks_with_timestamp"] = len(tick_timestamps)
        live_evidence_counts["fresh_ticks"] = int(fresh_tick_count)
        live_evidence_present = any(
            int(live_evidence_counts.get(key) or 0) > 0
            for key in ("holdings", "pending_orders", "order_history", "deal_history")
        ) or fresh_tick_count > 0

        executions = self.runtime.storage.recent_execution_results(limit=50, portfolio_slug=portfolio["slug"])
        fills = self.runtime.storage.recent_execution_fills(limit=200, portfolio_slug=portfolio["slug"])
        incident_records = self.runtime.storage.reconciliation_acknowledgements(portfolio_slug=portfolio["slug"])
        acknowledgement_map = {
            str(item.get("symbol") or "").upper(): item
            for item in incident_records
        }
        incident_status_counts: dict[str, int] = {}
        incidents: list[dict[str, Any]] = []
        for item in incident_records:
            incident_status = str(item.get("incident_status") or "acknowledged")
            incident_status_counts[incident_status] = int(incident_status_counts.get(incident_status, 0) + 1)
            incidents.append(
                {
                    "id": item.get("id"),
                    "portfolio_id": item.get("portfolio_id"),
                    "symbol": item.get("symbol"),
                    "reason": item.get("reason"),
                    "operator_note": item.get("operator_note"),
                    "mismatch_status": item.get("mismatch_status"),
                    "incident_status": incident_status,
                    "resolution_note": item.get("resolution_note"),
                    "acknowledged_at": item.get("acknowledged_at"),
                    "resolved_at": item.get("resolved_at"),
                    "updated_at": item.get("updated_at"),
                }
            )
        seen_order_tickets = {int(item["ticket"]) for item in order_history if item.get("ticket") is not None}
        seen_deal_tickets = {int(item["ticket"]) for item in deal_history if item.get("ticket") is not None}
        manual_events = sum(1 for item in order_history if item.get("is_manual")) + sum(
            1 for item in deal_history if item.get("is_manual")
        )
        unmatched_execution_count = 0
        history_window_expired_execution_count = 0
        execution_status_counts: dict[str, int] = {}
        for execution in executions:
            order_ticket = execution.get("mt5_result", {}).get("order") or execution.get("mt5_order_ticket")
            deal_ticket = execution.get("mt5_result", {}).get("deal") or execution.get("mt5_deal_ticket")
            execution_timestamp = pd.to_datetime(
                execution.get("created_at") or execution.get("time_utc"),
                utc=True,
                errors="coerce",
            )
            within_history_window = True
            if not pd.isna(execution_timestamp):
                within_history_window = execution_timestamp.to_pydatetime() >= history_window_start
            reconciliation_status = str(execution.get("reconciliation_status") or "")
            if not reconciliation_status:
                if order_ticket is None and deal_ticket is None:
                    reconciliation_status = "pending_broker" if within_history_window else "history_window_expired"
                elif order_ticket is not None and int(order_ticket) not in seen_order_tickets and int(deal_ticket or -1) not in seen_deal_tickets:
                    reconciliation_status = "pending_broker" if within_history_window else "history_window_expired"
                else:
                    fill_ratio = float(execution.get("fill_ratio") or 0.0)
                    reconciliation_status = "partial_fill" if 0.0 < fill_ratio < 0.999 else "match"
                execution["reconciliation_status"] = reconciliation_status
            elif reconciliation_status == "pending_broker":
                matched_order = order_ticket is not None and int(order_ticket) in seen_order_tickets
                matched_deal = deal_ticket is not None and int(deal_ticket) in seen_deal_tickets
                if matched_order or matched_deal:
                    fill_ratio = float(execution.get("fill_ratio") or 0.0)
                    reconciliation_status = "partial_fill" if 0.0 < fill_ratio < 0.999 else "match"
                    execution["reconciliation_status"] = reconciliation_status
                elif not within_history_window:
                    reconciliation_status = "history_window_expired"
                    execution["reconciliation_status"] = reconciliation_status
            execution_status_counts[reconciliation_status] = int(
                execution_status_counts.get(reconciliation_status, 0) + 1
            )
            if reconciliation_status == "history_window_expired":
                history_window_expired_execution_count += 1
            elif reconciliation_status != "match":
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

        mismatch_status_counts: dict[str, int] = {}
        mismatches = []
        for symbol in sorted(symbol_set):
            desk_value = float(desk_exposure.get(symbol, 0.0))
            live_value = float(live_exposure.get(symbol, 0.0))
            diff = live_value - desk_value
            pending = next((item for item in pending_orders if str(item.get("symbol") or "").upper() == symbol), None)
            order = next((item for item in order_history if str(item.get("symbol") or "").upper() == symbol), None)
            deal = next((item for item in deal_history if str(item.get("symbol") or "").upper() == symbol), None)
            if not live_evidence_present and abs(desk_value) > 1e-6:
                mismatch_status = "live_base_incomplete"
                reason = (
                    "The MT5 bridge is connected, but the broker live book/history is empty for the current "
                    f"{history_window_minutes} minute reconciliation window. Desk exposure cannot be confirmed yet."
                )
            elif abs(diff) <= 1e-6 and pending is None:
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
            mismatch_status_counts[mismatch_status] = int(mismatch_status_counts.get(mismatch_status, 0) + 1)
            acknowledgement = acknowledgement_map.get(symbol)
            if acknowledgement is not None and mismatch_status == "match":
                if str(acknowledgement.get("incident_status") or "").lower() == "resolved":
                    acknowledgement = None
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
                    "incident_id": None if acknowledgement is None else acknowledgement.get("id"),
                    "incident_status": None if acknowledgement is None else acknowledgement.get("incident_status"),
                    "incident_reason": None if acknowledgement is None else acknowledgement.get("reason"),
                    "incident_note": None if acknowledgement is None else acknowledgement.get("operator_note"),
                    "incident_updated_at": None if acknowledgement is None else acknowledgement.get("updated_at"),
                    "resolution_note": None if acknowledgement is None else acknowledgement.get("resolution_note"),
                    "resolved_at": None if acknowledgement is None else acknowledgement.get("resolved_at"),
                    "acknowledged_at": None if acknowledgement is None else acknowledgement.get("acknowledged_at"),
                    "acknowledged_reason": None if acknowledgement is None else acknowledgement.get("reason"),
                    "acknowledged_note": None if acknowledgement is None else acknowledgement.get("operator_note"),
                }
            )

        status_counts: dict[str, int] = {}
        for counts in (execution_status_counts, mismatch_status_counts):
            for status, count in counts.items():
                status_counts[status] = int(status_counts.get(status, 0) + int(count or 0))

        active_incident_count = 0
        for mismatch in mismatches:
            mismatch_status = str(mismatch.get("status") or "").lower()
            if mismatch_status == "match":
                continue
            incident_status = str(mismatch.get("incident_status") or "").strip().lower()
            if incident_status != "resolved":
                active_incident_count += 1
        resolved_incident_count = sum(
            1
            for incident in incidents
            if str(incident.get("incident_status") or "").strip().lower() == "resolved"
        )

        market_closed = bool(market_status.get("market_closed")) or _is_fx_weekend_closed()
        live_portfolio = self.runtime.is_live_portfolio(portfolio)
        if live_evidence_present or market_closed:
            operational_truth = "broker"
        elif live_portfolio:
            operational_truth = "broker_delayed"
        else:
            operational_truth = "target_fallback"

        return {
            "generated_at": _utcnow().isoformat(),
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "market_data_status": str(market_status.get("status") or "unknown"),
            "latest_sync_at": market_status.get("latest_sync_at"),
            "operational_truth": operational_truth,
            "target_exposure_by_symbol": {
                str(symbol).upper(): float(value)
                for symbol, value in desk_exposure.items()
                if str(symbol or "").strip()
            },
            "broker_exposure_by_symbol": {
                str(symbol).upper(): float(value)
                for symbol, value in live_exposure.items()
                if str(symbol or "").strip()
            },
            "staleness_reason": (
                "market_closed"
                if market_closed
                else "live_evidence_missing"
                if not live_evidence_present
                else "broker_history_stale"
                if history_window_expired_execution_count > 0
                else None
            ),
            "open_positions_count": len(holdings),
            "pending_orders_count": len(pending_orders),
            "live_window_minutes": int(live_window_minutes),
            "history_window_minutes": history_window_minutes,
            "effective_history_lookback_minutes": history_window_minutes,
            "heal_window_days": int(heal_window_days),
            "history_backfill_applied": bool(history_backfill_applied),
            "manual_event_count": int(manual_events),
            "unmatched_execution_count": int(unmatched_execution_count),
            "history_window_expired_execution_count": int(history_window_expired_execution_count),
            "active_incident_count": int(active_incident_count),
            "resolved_incident_count": int(resolved_incident_count),
            "autoresolved_count": 0,
            "live_evidence_present": bool(live_evidence_present),
            "live_evidence_counts": live_evidence_counts,
            "bridge_connected": None,
            "live_base_ready": bool(live_evidence_present),
            "bridge_status": None,
            "diagnostic_code": None,
            "diagnostic_message": None,
            "suppressed_status_counts": {},
            "status_counts": status_counts,
            "execution_status_counts": execution_status_counts,
            "mismatch_status_counts": mismatch_status_counts,
            "incident_status_counts": incident_status_counts,
            "holdings": holdings,
            "mismatches": mismatches,
            "incidents": incidents,
            "recent_execution_attempts": executions[:10],
            "recent_fills": fills[:20],
        }

    def _market_session_context(
        self,
        *,
        live_state: Mapping[str, Any],
        market_status: Mapping[str, Any],
    ) -> dict[str, Any]:
        reference_timestamp: datetime | None = None
        reference_source: str | None = None

        live_reference = _coerce_utc_datetime(live_state.get("market_reference_timestamp"))
        if live_reference is not None:
            reference_timestamp = live_reference
            reference_source = str(live_state.get("market_reference_source") or "bridge_state")

        for symbol, tick in dict(live_state.get("ticks") or {}).items():
            tick_time = _coerce_utc_datetime(dict(tick or {}).get("time_utc"))
            if tick_time is None:
                continue
            if reference_timestamp is None or tick_time > reference_timestamp:
                reference_timestamp = tick_time
                reference_source = f"tick:{str(symbol).upper()}"

        for symbol, value in dict(market_status.get("latest_bar_times") or {}).items():
            bar_time = _coerce_utc_datetime(value)
            if bar_time is None:
                continue
            if reference_timestamp is None or bar_time > reference_timestamp:
                reference_timestamp = bar_time
                reference_source = f"bar:{str(symbol).upper()}"

        archive_latest = _coerce_utc_datetime(dict(market_status.get("tick_archive") or {}).get("latest_tick_at"))
        if archive_latest is not None and (reference_timestamp is None or archive_latest > reference_timestamp):
            reference_timestamp = archive_latest
            reference_source = "tick_archive"

        latest_sync = _coerce_utc_datetime(market_status.get("latest_sync_at"))
        if latest_sync is not None and reference_timestamp is None:
            reference_timestamp = latest_sync
            reference_source = "market_sync"

        market_closed = bool(live_state.get("market_closed", False)) or _is_fx_weekend_closed()
        return {
            "market_closed": market_closed,
            "market_closed_reason": live_state.get("market_closed_reason") or ("weekend" if market_closed else None),
            "market_reference_timestamp": None if reference_timestamp is None else reference_timestamp.isoformat(),
            "market_reference_source": reference_source,
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
        session_context = self._market_session_context(live_state=live_state, market_status=market_status)
        effective_history_lookback_minutes = int(
            live_state.get("effective_history_lookback_minutes")
            or self.runtime.mt5_config.live_history_lookback_minutes
            or 0
        )
        if bool(session_context.get("market_closed")):
            effective_history_lookback_minutes = max(effective_history_lookback_minutes, 72 * 60)
        summary = self._build_reconciliation_summary(
            portfolio=portfolio,
            market_status=market_status,
            holdings=list(live_state.get("holdings") or []),
            pending_orders=list(live_state.get("pending_orders") or []),
            order_history=list(live_state.get("order_history") or []),
            deal_history=list(live_state.get("deal_history") or []),
            ticks=dict(live_state.get("ticks") or {}),
            effective_history_lookback_minutes=effective_history_lookback_minutes,
        )
        bridge_connected = bool(live_state.get("connected"))
        live_base_ready = bridge_connected and bool(summary.get("live_evidence_present"))
        suppressed_status_counts = {
            str(key): int(value)
            for key, value in dict(summary.get("status_counts") or {}).items()
            if str(key) != "match" and int(value or 0) > 0
        }
        summary.update(session_context)
        live_portfolio = self.runtime.is_live_portfolio(portfolio)
        if bool(summary.get("market_closed")) and live_portfolio:
            summary["operational_truth"] = "broker"
            if summary.get("staleness_reason") in {None, "", "live_evidence_missing"}:
                summary["staleness_reason"] = "market_closed"
        elif live_portfolio:
            if bridge_connected and not live_base_ready:
                summary["operational_truth"] = "broker_delayed"
                if summary.get("staleness_reason") in {None, ""}:
                    summary["staleness_reason"] = "live_evidence_missing"
            elif not bridge_connected:
                summary["operational_truth"] = "broker_unavailable"
                if summary.get("staleness_reason") in {None, ""}:
                    summary["staleness_reason"] = "live_bridge_disconnected"
        summary["bridge_connected"] = bridge_connected
        summary["live_base_ready"] = live_base_ready
        summary["bridge_status"] = str(live_state.get("status") or "unknown")
        summary["suppressed_status_counts"] = suppressed_status_counts if not live_base_ready else {}
        if bridge_connected and not live_base_ready and (
            bool(suppressed_status_counts)
            or int(summary.get("history_window_expired_execution_count") or 0) > 0
        ):
            if bool(summary.get("market_closed")):
                reference = summary.get("market_reference_timestamp")
                reference_text = (
                    f" The latest broker/market reference is {reference}."
                    if reference
                    else ""
                )
                summary["diagnostic_code"] = "MT5_MARKET_CLOSED"
                summary["diagnostic_message"] = (
                    "The FX market is currently closed, so the bridge cannot refresh fresh broker evidence until "
                    f"the next session opens.{reference_text} Derived drift alerts stay suppressed while the market is closed."
                )
            else:
                expired_count = int(summary.get("history_window_expired_execution_count") or 0)
                expired_suffix = (
                    f" {expired_count} stored desk execution(s) are also outside the broker history window."
                    if expired_count > 0
                    else ""
                )
                summary["diagnostic_code"] = "MT5_RECONCILIATION_INCOMPLETE"
                summary["diagnostic_message"] = (
                    "The MT5 bridge is connected, but the broker live book/history is empty for the current "
                    "reconciliation window. Derived drift and unmatched-execution alerts are withheld until live "
                    f"broker evidence is available.{expired_suffix}"
                )
        return summary

    def reconciliation_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        status = self.market_data_status(portfolio_slug=portfolio["slug"])
        holdings = self.live_holdings(portfolio_slug=portfolio["slug"])
        pending_orders: list[dict[str, Any]] = list(status.get("pending_orders") or [])
        effective_history_lookback_minutes = int(self.runtime.mt5_config.live_history_lookback_minutes or 0)
        if _is_fx_weekend_closed():
            effective_history_lookback_minutes = max(effective_history_lookback_minutes, 72 * 60)
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
            ticks={},
            effective_history_lookback_minutes=effective_history_lookback_minutes,
        )
