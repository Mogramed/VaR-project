from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd

from var_project.execution.mt5_live import MT5LiveGateway
from var_project.market_data.transforms import compute_log_returns, intraday_to_daily_log_returns
from var_project.portfolio.holdings import aggregate_exposure_by_symbol, gross_exposure_base_ccy, normalize_holdings
from var_project.storage.tick_archive import archive_ticks, summarize_tick_archive

if TYPE_CHECKING:
    from var_project.api.services.runtime import DeskServiceRuntime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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

    def _register_tick_partition_artifact(self, path: Path, details: Mapping[str, Any]) -> None:
        if not self.runtime.storage_ready:
            return
        self.runtime.storage.register_artifact(
            path,
            artifact_type="market_tick_partition",
            format="parquet",
            details=details,
        )

    def tick_archive_summary(self, *, symbols: Iterable[str] | None = None) -> dict[str, Any]:
        return summarize_tick_archive(self.tick_archive_root(), symbols=symbols)

    def archive_ticks(
        self,
        *,
        symbol: str,
        ticks: Iterable[Mapping[str, Any]],
        portfolio_slug: str | None = None,
        source: str = "mt5",
    ) -> dict[str, Any]:
        return archive_ticks(
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
        retention_tiers = self.retention_days_by_timeframe()
        tick_archive = self.tick_archive_summary(symbols=tracked_symbols)
        coverage = dict(details.get("coverage") or {})
        thin_history = False
        for symbol, by_timeframe in coverage.items():
            for tracked_timeframe, stats in dict(by_timeframe or {}).items():
                target_days = int(retention_tiers.get(str(tracked_timeframe).upper(), 0) or 0)
                bars_per_day = _bars_per_day(str(tracked_timeframe).upper())
                expected = max(int(target_days * bars_per_day * 0.8), 1) if target_days > 0 else 0
                observed = int(dict(stats or {}).get("bars") or 0)
                if expected > 0 and observed < expected:
                    thin_history = True
                    break
            if thin_history:
                break
        if not configured:
            status = "offline_fixture"
        elif latest_sync is None or missing_symbols or missing_bars:
            status = "incomplete"
        else:
            status = str(latest_sync.get("status") or "ok")
        coverage_status = "healthy"
        if status == "offline_fixture":
            coverage_status = "offline_fixture"
        elif status == "incomplete":
            coverage_status = "incomplete"
        elif tick_archive.get("coverage_status") == "stale":
            coverage_status = "stale"
        elif thin_history or tick_archive.get("coverage_status") == "thin_history":
            coverage_status = "thin_history"
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
        if not self.should_use_mt5_market_data(portfolio):
            return self.market_data_status(portfolio_slug=portfolio["slug"])

        portfolio_id = self.runtime._resolve_portfolio_id(portfolio["slug"])
        requested_days = int(days or self.runtime._default_days())
        timeframe_days = self._selected_timeframe_days(requested_days=requested_days, timeframes=timeframes)
        selected_timeframes = list(timeframe_days.keys())
        stored_history_days = max(timeframe_days.values()) if timeframe_days else self.history_backfill_days(requested_days)
        started_at = _utcnow()
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
                "retention_tiers": timeframe_days,
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

        with self.runtime._mt5_gateway() as live:
            open_positions = [item.to_dict() for item in live.holdings(symbols=None)]
            pending_orders = [item.to_dict() for item in live.pending_orders(symbols=None)]
            date_from = started_at - timedelta(days=stored_history_days)

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
                        timeframe_days_value = int(timeframe_days.get(timeframe, stored_history_days))
                        bars_per_day = live.connector.bars_per_day(timeframe)
                        bars = live.connector.fetch_last_n_bars(
                            str(symbol).upper(),
                            timeframe,
                            max(int(timeframe_days_value * bars_per_day) + int(bars_per_day), 10),
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
                            "stored_history_days": timeframe_days_value,
                        }
                    except Exception as exc:
                        errors.append({"scope": "bars", "symbol": symbol, "timeframe": timeframe, "detail": str(exc)})
                        coverage[symbol][timeframe] = {
                            "bars": 0,
                            "latest_bar_time": None,
                            "stored_history_days": int(timeframe_days.get(timeframe, stored_history_days)),
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
                            tick_date_from = started_at - timedelta(days=self.tick_retention_days())
                        else:
                            tick_date_from = latest_tick_at.to_pydatetime() - timedelta(minutes=5)
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
                "requested_days": requested_days,
                "stored_history_days": stored_history_days,
                "coverage": coverage,
                "retention_tiers": timeframe_days,
                "instruments": len(instrument_payloads),
                "orders": len(order_payloads),
                "deals": len(deal_payloads),
                "tick_archive": {
                    "retention_days": self.tick_retention_days(),
                    "symbols": tick_archives,
                    "summary": self.tick_archive_summary(symbols=(tracked_symbols if "tracked_symbols" in locals() else None)),
                },
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
        allow_auto_sync: bool = True,
        symbols: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        sync_attempted = False
        if ensure_sync:
            self.sync_market_data(portfolio_slug=str(portfolio["slug"]), days=days, timeframes=[timeframe])
            sync_attempted = True

        since = _utcnow() - timedelta(days=int(days) + 5)
        selected_symbols = [str(symbol).upper() for symbol in (symbols or portfolio["symbols"])]

        while True:
            frames: list[pd.DataFrame] = []
            missing_symbol: str | None = None
            insufficient_symbol: str | None = None

            for symbol in selected_symbols:
                bars = self.runtime.storage.market_bars(symbol=symbol, timeframe=timeframe, since=since)
                if not bars:
                    missing_symbol = symbol
                    break
                frame = pd.DataFrame(bars)
                frame["time"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
                frame = frame.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
                intraday = compute_log_returns(frame[["time", "close"]], price_col="close")
                daily = intraday_to_daily_log_returns(
                    intraday[["time", "log_return"]],
                    timeframe=timeframe,
                    min_coverage=min_coverage,
                )
                if daily.empty:
                    insufficient_symbol = symbol
                    break
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

        latest_snapshot = None
        for source in ("mt5_live_bridge", "mt5_live"):
            candidates = self.runtime.storage.recent_snapshots(limit=2, source=source, portfolio_slug=portfolio["slug"])
            if not candidates:
                continue
            latest_snapshot = candidates[0]
            if len(candidates) > 1 and _matches_live(latest_snapshot) and not _accepted_live_baseline(latest_snapshot):
                latest_snapshot = candidates[1]
            break
        if latest_snapshot is None:
            latest_snapshot = self.runtime.storage.latest_snapshot(source="historical", portfolio_slug=portfolio["slug"])
        snapshot_payload = {} if latest_snapshot is None else dict(latest_snapshot.get("payload") or {})
        desk_holdings = (
            list(snapshot_payload.get("holdings") or [])
            if "holdings" in snapshot_payload
            else list(portfolio.get("configured_holdings") or [])
        )
        if "exposure_by_symbol" in snapshot_payload:
            desk_exposure = dict(snapshot_payload.get("exposure_by_symbol") or {})
        elif snapshot_payload.get("positions_eur") is not None:
            desk_exposure = dict(snapshot_payload.get("positions_eur") or {})
        else:
            desk_exposure = aggregate_exposure_by_symbol(desk_holdings, base_currency=str(portfolio["base_currency"]))

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
            "incident_status_counts": incident_status_counts,
            "holdings": holdings,
            "mismatches": mismatches,
            "incidents": incidents,
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
