from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import StringIO
import math
import os
from pathlib import Path
from threading import Lock, Thread
import time
from typing import Any, Mapping
from uuid import uuid4

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
from var_project.storage.serialization import coerce_datetime, jsonable, utcnow

TRADE_EXPOSURE_STEP_EUR = 1000.0
TRADE_EXPOSURE_MIN_EUR = 1000.0
TRADE_EXPOSURE_EPSILON = 1e-6
TRADE_EXPOSURE_RULE_MESSAGE = (
    "Exposure must be at least 1,000 EUR in absolute value and use 1,000 EUR increments."
)


class DeskApiService:
    _shared_desk_overview_cache: dict[tuple[str, str], dict[str, Any]] = {}
    _shared_desk_overview_refresh_inflight: set[tuple[str, str]] = set()
    _shared_desk_overview_cache_lock = Lock()
    _shared_desk_overview_compute_lock = Lock()

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
        self._live_read_executor = ThreadPoolExecutor(
            max_workers=self._live_read_workers(),
            thread_name_prefix="desk-live-read",
        )

    @staticmethod
    def _bounded_timeout_ms(value: Any, *, default: int = 1200, min_value: int = 100, max_value: int = 15000) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        if parsed < int(min_value):
            return int(min_value)
        if parsed > int(max_value):
            return int(max_value)
        return int(parsed)

    def _live_read_timeout_ms(self) -> int:
        return self._bounded_timeout_ms(
            os.getenv("VAR_PROJECT_API_LIVE_READ_TIMEOUT_MS"),
            default=1200,
            min_value=100,
            max_value=15000,
        )

    def _live_read_workers(self) -> int:
        workers = self._bounded_timeout_ms(
            os.getenv("VAR_PROJECT_API_LIVE_READ_WORKERS"),
            default=2,
            min_value=1,
            max_value=8,
        )
        return int(workers)

    def _desk_overview_cache_ttl_seconds(self) -> float:
        configured = max(float(self.mt5_config.live_poll_seconds), 0.5)
        return min(max(configured * 5.0, 2.0), 10.0)

    def _desk_overview_cache_key(self, *, desk_slug: str) -> tuple[str, str]:
        return (str(self.root.resolve()), str(desk_slug))

    def _cached_desk_overview(self, *, cache_key: tuple[str, str]) -> tuple[dict[str, Any] | None, bool]:
        with self._shared_desk_overview_cache_lock:
            cached = self._shared_desk_overview_cache.get(cache_key)
            if cached is None:
                return None, False
            expired = float(cached.get("expires_at") or 0.0) <= time.monotonic()
            payload = dict(cached.get("payload") or {})
        if not payload:
            return None, False
        return deepcopy(payload), bool(expired)

    def _store_desk_overview_cache(self, *, cache_key: tuple[str, str], payload: Mapping[str, Any]) -> None:
        with self._shared_desk_overview_cache_lock:
            self._shared_desk_overview_cache[cache_key] = {
                "expires_at": time.monotonic() + self._desk_overview_cache_ttl_seconds(),
                "payload": deepcopy(dict(payload)),
            }

    def _refresh_desk_overview_cache_async(self, *, cache_key: tuple[str, str], desk_slug: str) -> None:
        with self._shared_desk_overview_cache_lock:
            if cache_key in self._shared_desk_overview_refresh_inflight:
                return
            self._shared_desk_overview_refresh_inflight.add(cache_key)

        def _worker() -> None:
            try:
                payload = self._compute_desk_overview_payload(desk_slug=desk_slug)
                self._store_desk_overview_cache(cache_key=cache_key, payload=payload)
            except Exception:
                # Keep async refresh best-effort; stale cache remains available.
                return
            finally:
                with self._shared_desk_overview_cache_lock:
                    self._shared_desk_overview_refresh_inflight.discard(cache_key)

        Thread(
            target=_worker,
            name=f"desk-overview-refresh-{desk_slug}",
            daemon=True,
        ).start()

    def _latest_capital_snapshot_fast(self, *, portfolio_slug: str) -> dict[str, Any] | None:
        if not self.runtime.storage_ready:
            return None
        for candidate_source in self.reads._preferred_snapshot_sources(  # noqa: SLF001 - bounded internal use for fast path
            portfolio_slug=portfolio_slug,
            source="auto",
        ):
            payload = self.storage.latest_capital_snapshot(
                source=candidate_source,
                portfolio_slug=portfolio_slug,
            )
            if payload is not None:
                return dict(payload)
        return None

    def _compute_desk_overview_payload(self, *, desk_slug: str) -> dict[str, Any]:
        portfolio_map = {portfolio["slug"]: portfolio for portfolio in self.portfolios}
        snapshots: list[dict[str, Any]] = []
        alert_counts: dict[str, int] = {}
        for portfolio in self.portfolios:
            live_alert_count = 0
            live_state = self.mt5.cached_live_state(portfolio_slug=portfolio["slug"], detail_level="summary")
            if live_state is not None:
                live_alert_count = len(list(live_state.get("operator_alerts") or []))
                if live_state.get("capital_usage") is not None:
                    snapshots.append(dict(live_state["capital_usage"]))
                    if live_alert_count:
                        alert_counts[portfolio["slug"]] = int(live_alert_count)
                    continue
            fallback_capital = self._latest_capital_snapshot_fast(portfolio_slug=portfolio["slug"])
            if fallback_capital is None:
                fallback_capital = {
                    "portfolio_slug": portfolio["slug"],
                    "reference_model": "hist",
                    "total_capital_budget_eur": 0.0,
                    "total_capital_consumed_eur": 0.0,
                    "total_capital_reserved_eur": 0.0,
                    "total_capital_remaining_eur": 0.0,
                    "status": "PENDING",
                    "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "none",
                }
            snapshots.append(dict(fallback_capital))
            if live_alert_count:
                alert_counts[portfolio["slug"]] = int(live_alert_count)
        return build_desk_snapshot(
            self.desk.to_dict(),
            snapshots,
            portfolio_map,
            alerts_by_portfolio=alert_counts,
        ).to_dict()

    def _safe_live_state_summary(
        self,
        *,
        portfolio_slug: str,
        force_refresh: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any] | None:
        if not force_refresh:
            cached = self.mt5.cached_live_state(portfolio_slug=portfolio_slug, detail_level="summary")
            if cached is not None:
                return cached
        wait_ms = self._bounded_timeout_ms(
            timeout_ms if timeout_ms is not None else self._live_read_timeout_ms(),
            default=1200,
            min_value=100,
            max_value=15000,
        )
        future = self._live_read_executor.submit(
            self.mt5.live_state,
            portfolio_slug=portfolio_slug,
            detail_level="summary",
            force_refresh=bool(force_refresh),
        )
        try:
            return future.result(timeout=max(float(wait_ms) / 1000.0, 0.1))
        except FuturesTimeoutError:
            return None
        except Exception:
            return None
        finally:
            if not future.done():
                future.cancel()

    def close(self) -> None:
        try:
            self._live_read_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def health(self) -> dict[str, Any]:
        return self.reads.health()

    def health_dependencies(self) -> dict[str, Any]:
        return self.reads.health_dependencies()

    def health_readiness(
        self,
        *,
        portfolio_slug: str | None = None,
        refresh_live: bool = False,
        max_wait_ms: int = 1200,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        generated_at = datetime.now(timezone.utc).isoformat()
        strict_live_required = bool(self.runtime.strict_live_required(portfolio))
        schema_status = self.runtime.refresh_storage_schema_status()
        storage_ready = bool(schema_status.get("ready"))
        mt5_configured = bool(
            self.runtime._has_custom_mt5_factory
            or self.runtime.mt5_config.agent_base_url
            or self.runtime.mt5_config.path
            or self.runtime.mt5_config.login
            or self.runtime.mt5_config.server
        )

        live_state = self.mt5.cached_live_state(portfolio_slug=portfolio["slug"], detail_level="summary")
        live_fetch_timed_out = False
        live_fetch_error: str | None = None

        should_fetch_live = bool(refresh_live or strict_live_required or live_state is None)
        if should_fetch_live:
            wait_ms = self._bounded_timeout_ms(max_wait_ms, default=1200, min_value=100, max_value=15000)
            timeout_seconds = max(float(wait_ms) / 1000.0, 0.1)
            future = self._live_read_executor.submit(
                self.mt5.live_state,
                portfolio_slug=portfolio["slug"],
                detail_level="summary",
                force_refresh=bool(refresh_live),
            )
            try:
                live_state = future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                live_fetch_timed_out = True
                live_fetch_error = f"Live state fetch timed out after {int(wait_ms)} ms."
                live_state = None
            except Exception as exc:
                live_fetch_error = str(exc)
                live_state = None
            finally:
                if not future.done():
                    future.cancel()

        live_connected = bool((live_state or {}).get("connected", False))
        live_degraded = bool((live_state or {}).get("degraded", False))
        live_stale = bool((live_state or {}).get("stale", False))
        fallback_snapshot_used = bool((live_state or {}).get("fallback_snapshot_used", False))
        live_health = dict((live_state or {}).get("health") or {})
        live_health_status = str(
            live_health.get("status")
            or (live_state or {}).get("status")
            or "unknown"
        ).strip().lower()

        checks: dict[str, dict[str, Any]] = {
            "database": {
                "status": "ready" if storage_ready else "not_ready",
                "required": True,
                "detail": str(schema_status.get("detail") or "Database schema status unavailable."),
                "value": {
                    "schema_ready": storage_ready,
                    "target": self.runtime.storage.settings.database_url,
                    "issues": list(schema_status.get("issues") or []),
                    "current_revision": schema_status.get("current_revision"),
                    "expected_revision": schema_status.get("expected_revision"),
                },
            },
            "mt5_config": {
                "status": "ready" if mt5_configured else ("not_ready" if strict_live_required else "degraded"),
                "required": bool(strict_live_required),
                "detail": (
                    "MT5 connectivity is configured."
                    if mt5_configured
                    else "MT5 connectivity is not configured in this API process."
                ),
                "value": {
                    "agent_base_url": self.runtime.mt5_config.agent_base_url,
                    "terminal_path": self.runtime.mt5_config.path,
                },
            },
        }

        live_required = bool(strict_live_required)
        if live_state is None:
            live_status = "not_ready" if live_required else "unknown"
            live_detail = (
                live_fetch_error
                or (
                    "No cached live state is available yet."
                    if not live_required
                    else "Live state is unavailable."
                )
            )
        else:
            if live_connected and not live_stale and not live_degraded and not fallback_snapshot_used:
                live_status = "ready"
                live_detail = "Live bridge is connected with fresh broker evidence."
            elif fallback_snapshot_used or live_connected or live_health_status in {"degraded", "stale", "market_closed"}:
                live_status = "degraded"
                live_detail = str(
                    live_health.get("message")
                    or "Live bridge is running in degraded mode but remains usable for demo continuity."
                )
            else:
                live_status = "not_ready" if live_required else "degraded"
                live_detail = str(
                    live_health.get("message")
                    or live_fetch_error
                    or (
                        "Live MT5 evidence is unavailable. Live portfolios are MT5-only and do not use configured fallback exposure."
                        if live_required
                        else "Live MT5 evidence is unavailable."
                    )
                )
        if live_fetch_timed_out:
            live_detail = (f"{live_detail} " if live_detail else "") + "Request timed out before MT5 returned."
        checks["mt5_live"] = {
            "status": live_status,
            "required": live_required,
            "detail": live_detail,
            "value": {
                "connected": live_connected,
                "degraded": live_degraded,
                "stale": live_stale,
                "fallback_snapshot_used": fallback_snapshot_used,
                "status": (live_state or {}).get("status"),
                "health_status": live_health_status,
                "sequence": (live_state or {}).get("sequence"),
                "generated_at": (live_state or {}).get("generated_at"),
                "last_success_at": (live_state or {}).get("last_success_at"),
                "timed_out": live_fetch_timed_out,
            },
        }

        required_not_ready = [
            name for name, check in checks.items() if check["required"] and check["status"] == "not_ready"
        ]
        degraded_checks = [
            name for name, check in checks.items() if check["status"] == "degraded"
        ]

        if required_not_ready:
            status = "not_ready"
            summary = "Platform not ready for live demo. Resolve required checks first."
        elif degraded_checks:
            status = "degraded"
            summary = "Platform is usable for demo but currently running in degraded mode."
        else:
            status = "ready"
            summary = "Platform ready for demo."

        recommendations: list[str] = []
        if not storage_ready:
            recommendations.append(
                "Run `var-project db upgrade` (or `alembic upgrade head`) and re-check `/health` before starting the demo."
            )
        if strict_live_required and not mt5_configured:
            recommendations.append(
                "Configure `VAR_PROJECT_MT5_AGENT_BASE_URL` (and API key if enabled) in the API process."
            )
        if live_fetch_timed_out:
            recommendations.append("Increase `max_wait_ms` or verify MT5 agent responsiveness.")
        if fallback_snapshot_used:
            recommendations.append("MT5 is disconnected; demo currently uses the last known broker snapshot.")
        if live_state is None and strict_live_required:
            recommendations.append(
                "Start the MT5 agent and verify `/health` and `/live/state` endpoints. "
                "Live compute remains blocked while broker evidence is unavailable."
            )

        return {
            "status": status,
            "generated_at": generated_at,
            "portfolio_slug": portfolio["slug"],
            "portfolio_mode": portfolio.get("mode"),
            "strict_live_required": strict_live_required,
            "summary": summary,
            "checks": checks,
            "recommendations": recommendations,
        }

    def jobs_status(self) -> dict[str, Any]:
        return self.reads.jobs_status()

    def list_portfolios(self) -> list[dict[str, Any]]:
        runtime_portfolios = [dict(item) for item in self.portfolios]
        if not runtime_portfolios:
            return self.reads.list_portfolios()

        database_portfolios = self.reads.list_portfolios()
        if not database_portfolios:
            return runtime_portfolios

        by_slug = {str(item.get("slug") or ""): dict(item) for item in database_portfolios}
        by_slug_lower = {slug.lower(): payload for slug, payload in by_slug.items() if slug}

        legacy_singleton_record: dict[str, Any] | None = None
        if len(runtime_portfolios) == 1 and len(database_portfolios) == 1:
            runtime_slug = str(runtime_portfolios[0].get("slug") or "").strip().lower().replace("-", "_")
            database_slug = str(database_portfolios[0].get("slug") or "").strip().lower().replace("-", "_")
            runtime_mode = str(runtime_portfolios[0].get("mode") or "").strip().lower()
            if runtime_mode == "live_mt5" and runtime_slug == "mt5_live_portfolio" and database_slug == "fx_eur_20k":
                legacy_singleton_record = dict(database_portfolios[0])

        normalized: list[dict[str, Any]] = []
        for runtime_portfolio in runtime_portfolios:
            slug = str(runtime_portfolio.get("slug") or "")
            database_record = by_slug.get(slug) or by_slug_lower.get(slug.lower())
            if database_record is None and legacy_singleton_record is not None:
                database_record = legacy_singleton_record
            merged = dict(runtime_portfolio)
            if database_record is not None:
                merged["id"] = database_record.get("id")
                merged["created_at"] = database_record.get("created_at")
                merged["updated_at"] = database_record.get("updated_at")
            normalized.append(merged)
        return normalized

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

    def active_alerts(
        self,
        *,
        limit: int = 25,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        if portfolio_slug:
            portfolios = [self.runtime._resolve_portfolio_context(portfolio_slug)]
        else:
            portfolios = [dict(item) for item in self.portfolios]
        now_iso = utcnow().isoformat()
        payload: list[dict[str, Any]] = []
        synthetic_id = -1
        for portfolio in portfolios:
            slug = str(portfolio["slug"])
            portfolio_id = self.runtime.portfolio_ids.get(slug)
            live_state = self._safe_live_state_summary(portfolio_slug=slug)
            if live_state is None:
                continue
            generated_at = str(live_state.get("generated_at") or now_iso)
            for alert in list(live_state.get("operator_alerts") or []):
                item = dict(alert)
                context = dict(item.get("context") or {})
                context.setdefault("portfolio_slug", slug)
                payload.append(
                    {
                        "id": synthetic_id,
                        "portfolio_id": portfolio_id,
                        "snapshot_id": None,
                        "validation_run_id": None,
                        "is_active": True,
                        "source": str(item.get("source") or "live_operator"),
                        "severity": str(item.get("severity") or "INFO"),
                        "code": str(item.get("code") or "UNKNOWN"),
                        "message": str(item.get("message") or ""),
                        "context": context,
                        "created_at": generated_at,
                    }
                )
                synthetic_id -= 1
        def _code_priority(code: str) -> int:
            normalized = str(code or "").upper()
            if (
                "VALIDATION_GOVERNANCE_FAIL" in normalized
                or "VALIDATION_SURFACE_COVERAGE_FAIL" in normalized
                or "VALIDATION_SURFACE_CONDITIONAL_FAIL" in normalized
                or "VALIDATION_HORIZON_FAIL" in normalized
                or "VALIDATION_ES_SHORTFALL_BREACH" in normalized
                or "VALIDATION_ES_BREACH_RATE_BREACH" in normalized
                or "BROKER_REJECTION" in normalized
                or "DESK_BROKER_DRIFT" in normalized
                or "PNL_DRIFT" in normalized
                or "ORPHAN_LIVE_POSITION" in normalized
                or "OVERFILL_OR_VOLUME_DRIFT" in normalized
            ):
                return 0
            if "RECONCILIATION_INCOMPLETE" in normalized:
                return 1
            if (
                "VALIDATION_GOVERNANCE_WARN" in normalized
                or "VALIDATION_SURFACE_INDEPENDENCE_FAIL" in normalized
                or "VALIDATION_HORIZON_WARN" in normalized
                or "VALIDATION_SURFACE_SAMPLE_THIN" in normalized
                or "VALIDATION_HORIZON_SAMPLE_THIN" in normalized
                or "VALIDATION_ES_SHORTFALL_WARN" in normalized
                or "VALIDATION_ES_BREACH_RATE_WARN" in normalized
                or "WINDOW_EXPIRED" in normalized
            ):
                return 2
            if "PARTIAL_FILL" in normalized:
                return 3
            if "PENDING_BROKER" in normalized:
                return 4
            if "MANUAL_TRADE" in normalized or "MANUAL_EVENTS" in normalized:
                return 5
            if "UNMATCHED" in normalized:
                return 6
            return 7

        severity_rank = {"breach": 0, "critical": 0, "warn": 1, "warning": 1, "info": 2, "ok": 3}
        payload.sort(
            key=lambda item: (
                int(severity_rank.get(str(item.get("severity", "")).lower(), 4)),
                int(_code_priority(str(item.get("code") or ""))),
                str(item.get("code") or ""),
                str(item.get("message") or ""),
            )
        )
        return payload[: max(int(limit), 1)]

    def recent_decisions(
        self,
        *,
        limit: int = 25,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.recent_decisions(limit=limit, portfolio_slug=portfolio_slug, account_id=account_id)

    def latest_capital(
        self,
        *,
        portfolio_slug: str | None = None,
        source: str | None = "auto",
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        normalized_source = str(source or "auto").strip().lower()
        live_source_requested = normalized_source in {"", "auto", "mt5_live_bridge", "mt5_live"}
        # Only live MT5 portfolios should consume MT5 live-state fast paths for capital.
        # Offline portfolios must remain deterministic and use the historical read path.
        live_state = self._safe_live_state_summary(portfolio_slug=portfolio["slug"])
        if (
            self.runtime.is_live_portfolio(portfolio)
            and live_source_requested
            and live_state is not None
            and live_state.get("capital_usage") is not None
        ):
            capital_usage = dict(live_state["capital_usage"])
            if normalized_source in {"", "auto"}:
                return capital_usage
            capital_source = str(
                capital_usage.get("snapshot_source")
                or capital_usage.get("source")
                or ""
            ).strip().lower()
            if capital_source == normalized_source:
                return capital_usage
        return self.reads.latest_capital(portfolio_slug=portfolio_slug, source=source)

    def capital_history(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def recent_execution_results(
        self,
        *,
        limit: int = 25,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.recent_execution_results(limit=limit, portfolio_slug=portfolio_slug, account_id=account_id)

    def recent_execution_fills(
        self,
        *,
        limit: int = 50,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.recent_execution_fills(limit=limit, portfolio_slug=portfolio_slug, account_id=account_id)

    def recent_audit_events(self, *, limit: int = 50, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_audit_events(limit=limit, portfolio_slug=portfolio_slug)

    def report_decision_history(
        self,
        *,
        limit: int = 25,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.report_decision_history(limit=limit, portfolio_slug=portfolio_slug, account_id=account_id)

    def report_capital_history(
        self,
        *,
        limit: int = 25,
        source: str | None = None,
        portfolio_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.reads.report_capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)

    def portfolio_capital(self, portfolio_slug: str, *, source: str | None = "auto") -> dict[str, Any]:
        return self.reads.latest_capital(portfolio_slug=portfolio_slug, source=source)

    def desk_overview(self, *, desk_slug: str | None = None) -> dict[str, Any]:
        resolved_desk_slug = str(desk_slug or self.desk.slug)
        if resolved_desk_slug != self.desk.slug:
            raise ValueError(f"Unknown desk '{desk_slug}'.")
        cache_key = self._desk_overview_cache_key(desk_slug=resolved_desk_slug)
        cached, is_stale = self._cached_desk_overview(cache_key=cache_key)
        if cached is not None and not is_stale:
            return cached
        if cached is not None and is_stale:
            self._refresh_desk_overview_cache_async(
                cache_key=cache_key,
                desk_slug=resolved_desk_slug,
            )
            return cached
        with self._shared_desk_overview_compute_lock:
            cached, is_stale = self._cached_desk_overview(cache_key=cache_key)
            if cached is not None and not is_stale:
                return cached
            if cached is not None and is_stale:
                self._refresh_desk_overview_cache_async(
                    cache_key=cache_key,
                    desk_slug=resolved_desk_slug,
                )
                return cached
            payload = self._compute_desk_overview_payload(desk_slug=resolved_desk_slug)
            self._store_desk_overview_cache(cache_key=cache_key, payload=payload)
            return payload

    def latest_artifact(self, artifact_type: str) -> dict[str, Any] | None:
        return self.reads.latest_artifact(artifact_type)

    def latest_model_comparison(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        return self.reads.latest_model_comparison(portfolio_slug=portfolio_slug)

    def latest_risk_attribution(
        self,
        *,
        source: str = "auto",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        payload = self.reads.latest_risk_attribution(source=source, portfolio_slug=portfolio_slug)
        if payload is None:
            return None
        return self._enrich_risk_attribution_payload(payload)

    def latest_risk_budget(
        self,
        *,
        source: str = "auto",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        return self.reads.latest_risk_budget(source=source, portfolio_slug=portfolio_slug)

    def mt5_status(self, *, account_id: str | None = None) -> dict[str, Any]:
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        live_state = self.mt5.live_state(
            portfolio_slug=self.portfolio["slug"],
            detail_level="summary",
            account_id=resolved_account_id,
        )
        if live_state.get("terminal_status") is not None:
            payload = dict(live_state["terminal_status"])
            payload["account_id"] = resolved_account_id
            return payload
        return self.mt5.mt5_status(account_id=resolved_account_id)

    def mt5_account(self, *, account_id: str | None = None) -> dict[str, Any]:
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        live_state = self.mt5.live_state(
            portfolio_slug=self.portfolio["slug"],
            detail_level="summary",
            account_id=resolved_account_id,
        )
        if live_state.get("account") is not None:
            payload = dict(live_state["account"])
            payload["account_id"] = resolved_account_id
            return payload
        return self.mt5.mt5_account(account_id=resolved_account_id)

    def mt5_accounts(self) -> dict[str, Any]:
        return self.mt5.mt5_accounts()

    def mt5_positions(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.mt5.mt5_positions(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )

    def mt5_orders(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.mt5.mt5_orders(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )

    @staticmethod
    def _coerce_utc_datetime(value: Any) -> datetime | None:
        parsed = coerce_datetime(value)
        if parsed is None:
            return None
        return parsed.astimezone(timezone.utc)

    def _parse_utc_datetime(
        self,
        value: Any | None,
        *,
        field_name: str,
    ) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        parsed = self._coerce_utc_datetime(value)
        if parsed is None:
            raise ValueError(f"{field_name} must be a valid ISO-8601 datetime.")
        return parsed

    def _mt5_history_bounds(
        self,
        *,
        date_from: Any | None,
        date_to: Any | None,
    ) -> tuple[datetime, datetime]:
        end_at = self._parse_utc_datetime(date_to, field_name="date_to") or datetime.now(timezone.utc)
        start_at = self._parse_utc_datetime(date_from, field_name="date_from") or (end_at - timedelta(days=30))
        if start_at > end_at:
            raise ValueError("date_from must be less than or equal to date_to.")
        return start_at, end_at

    @staticmethod
    def _normalize_transaction_type(value: str | None = None) -> str:
        normalized = str(value or "all").strip().lower()
        if not normalized:
            return "all"
        if normalized not in {"all", "order", "deal", "manual", "desk"}:
            raise ValueError("type must be one of: all, order, deal, manual, desk.")
        return normalized

    @staticmethod
    def _normalize_sort(value: str | None = None) -> str:
        normalized = str(value or "time_desc").strip().lower()
        if normalized in {"desc", "time_desc"}:
            return "time_desc"
        if normalized in {"asc", "time_asc"}:
            return "time_asc"
        raise ValueError("sort must be one of: time_desc, time_asc.")

    def _mt5_order_history_rows(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        start_at, end_at = self._mt5_history_bounds(date_from=date_from, date_to=date_to)
        symbol_filter = str(symbol or "").strip().upper()
        with self.runtime._mt5_gateway(account_id=account_id) as live:
            rows = [
                item.to_dict()
                for item in live.order_history(
                    date_from=start_at,
                    date_to=end_at,
                    symbols=[symbol_filter] if symbol_filter else None,
                )
            ]
        filtered: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if symbol_filter and str(item.get("symbol") or "").strip().upper() != symbol_filter:
                continue
            event_time = self._coerce_utc_datetime(item.get("time_done_utc") or item.get("time_setup_utc"))
            if event_time is not None and (event_time < start_at or event_time > end_at):
                continue
            item["portfolio_slug"] = portfolio["slug"]
            filtered.append(item)
        filtered.sort(
            key=lambda item: (
                self._coerce_utc_datetime(item.get("time_done_utc") or item.get("time_setup_utc")) or datetime.min.replace(tzinfo=timezone.utc),
                int(item.get("ticket") or 0),
            ),
            reverse=True,
        )
        return filtered

    def _mt5_deal_history_rows(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        start_at, end_at = self._mt5_history_bounds(date_from=date_from, date_to=date_to)
        symbol_filter = str(symbol or "").strip().upper()
        with self.runtime._mt5_gateway(account_id=account_id) as live:
            rows = [
                item.to_dict()
                for item in live.deal_history(
                    date_from=start_at,
                    date_to=end_at,
                    symbols=[symbol_filter] if symbol_filter else None,
                )
            ]
        filtered: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if symbol_filter and str(item.get("symbol") or "").strip().upper() != symbol_filter:
                continue
            event_time = self._coerce_utc_datetime(item.get("time_utc"))
            if event_time is not None and (event_time < start_at or event_time > end_at):
                continue
            item["portfolio_slug"] = portfolio["slug"]
            filtered.append(item)
        filtered.sort(
            key=lambda item: (
                self._coerce_utc_datetime(item.get("time_utc")) or datetime.min.replace(tzinfo=timezone.utc),
                int(item.get("ticket") or 0),
            ),
            reverse=True,
        )
        return filtered

    def mt5_live_state(
        self,
        *,
        portfolio_slug: str | None = None,
        detail_level: str = "full",
        force_refresh: bool = False,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        return self.mt5.live_state(
            portfolio_slug=portfolio_slug,
            detail_level=detail_level,
            force_refresh=force_refresh,
            account_id=account_id,
        )

    def mt5_live_events(
        self,
        *,
        portfolio_slug: str | None = None,
        after: int = 0,
        limit: int = 100,
        wait_seconds: float = 15.0,
        detail_level: str = "full",
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.mt5.live_events(
            portfolio_slug=portfolio_slug,
            after=after,
            limit=limit,
            wait_seconds=wait_seconds,
            detail_level=detail_level,
            account_id=account_id,
        )

    def mt5_analytics_series(
        self,
        *,
        portfolio_slug: str | None = None,
        window_minutes: int = 240,
        max_points: int = 300,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        return self.mt5.analytics_series(
            portfolio_slug=portfolio_slug,
            window_minutes=window_minutes,
            max_points=max_points,
            account_id=account_id,
        )

    def mt5_history_orders(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 100,
        account_id: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._mt5_order_history_rows(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
        )
        return rows[: max(int(limit), 1)]

    def mt5_history_deals(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 100,
        account_id: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._mt5_deal_history_rows(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
        )
        return rows[: max(int(limit), 1)]

    def mt5_transaction_history(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
        symbol: str | None = None,
        type: str | None = None,
        sort: str = "time_desc",
        page: int = 1,
        page_size: int = 50,
        page_size_cap: int | None = 200,
    ) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        type_filter = self._normalize_transaction_type(type)
        sort_order = self._normalize_sort(sort)
        normalized_page = max(int(page), 1)
        normalized_page_size = max(int(page_size), 1)
        if page_size_cap is not None:
            normalized_page_size = min(normalized_page_size, max(int(page_size_cap), 1))
        start_at, end_at = self._mt5_history_bounds(date_from=date_from, date_to=date_to)
        symbol_filter = str(symbol or "").strip().upper() or None

        needs_orders = type_filter in {"all", "order", "manual", "desk"}
        needs_deals = type_filter in {"all", "deal", "manual", "desk"}
        orders = (
            self._mt5_order_history_rows(
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
                date_from=start_at,
                date_to=end_at,
                symbol=symbol_filter,
            )
            if needs_orders
            else []
        )
        deals = (
            self._mt5_deal_history_rows(
                portfolio_slug=portfolio["slug"],
                account_id=resolved_account_id,
                date_from=start_at,
                date_to=end_at,
                symbol=symbol_filter,
            )
            if needs_deals
            else []
        )

        transactions: list[dict[str, Any]] = []
        for row in orders:
            event_time = row.get("time_done_utc") or row.get("time_setup_utc")
            item = {
                "id": f"order:{row.get('ticket') or 'na'}:{event_time or 'na'}",
                "kind": "order",
                "ticket": row.get("ticket"),
                "order_ticket": None,
                "position_id": row.get("position_id"),
                "symbol": str(row.get("symbol") or ""),
                "side": row.get("side"),
                "transaction_type": row.get("order_type"),
                "state": row.get("state"),
                "volume": self._float_or_none(row.get("volume_initial") if row.get("volume_initial") is not None else row.get("volume_current")),
                "price": self._float_or_none(row.get("price_current") if row.get("price_current") is not None else row.get("price_open")),
                "profit": None,
                "commission": None,
                "swap": None,
                "fee": None,
                "comment": row.get("comment"),
                "is_manual": bool(row.get("is_manual")),
                "time_utc": event_time,
                "raw": dict(row.get("raw") or {}),
            }
            transactions.append(item)

        for row in deals:
            item = {
                "id": f"deal:{row.get('ticket') or 'na'}:{row.get('time_utc') or 'na'}",
                "kind": "deal",
                "ticket": row.get("ticket"),
                "order_ticket": row.get("order_ticket"),
                "position_id": row.get("position_id"),
                "symbol": str(row.get("symbol") or ""),
                "side": row.get("side"),
                "transaction_type": row.get("entry"),
                "state": None,
                "volume": self._float_or_none(row.get("volume")),
                "price": self._float_or_none(row.get("price")),
                "profit": self._float_or_none(row.get("profit")),
                "commission": self._float_or_none(row.get("commission")),
                "swap": self._float_or_none(row.get("swap")),
                "fee": self._float_or_none(row.get("fee")),
                "comment": row.get("comment"),
                "is_manual": bool(row.get("is_manual")),
                "time_utc": row.get("time_utc"),
                "raw": dict(row.get("raw") or {}),
            }
            transactions.append(item)

        if type_filter == "order":
            transactions = [item for item in transactions if item["kind"] == "order"]
        elif type_filter == "deal":
            transactions = [item for item in transactions if item["kind"] == "deal"]
        elif type_filter == "manual":
            transactions = [item for item in transactions if bool(item.get("is_manual"))]
        elif type_filter == "desk":
            transactions = [item for item in transactions if not bool(item.get("is_manual"))]

        transactions = [item for item in transactions if str(item.get("symbol") or "").strip()]
        reverse = sort_order == "time_desc"
        transactions.sort(
            key=lambda item: (
                self._coerce_utc_datetime(item.get("time_utc")) or datetime.min.replace(tzinfo=timezone.utc),
                str(item.get("kind") or ""),
                int(item.get("ticket") or 0),
            ),
            reverse=reverse,
        )

        total = len(transactions)
        start_index = (normalized_page - 1) * normalized_page_size
        end_index = start_index + normalized_page_size
        page_items = transactions[start_index:end_index]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "portfolio_slug": portfolio["slug"],
            "account_id": resolved_account_id,
            "filters": {
                "date_from": start_at.isoformat(),
                "date_to": end_at.isoformat(),
                "symbol": symbol_filter,
                "type": type_filter,
            },
            "sort": sort_order,
            "page": normalized_page,
            "page_size": normalized_page_size,
            "total": total,
            "has_next": end_index < total,
            "items": page_items,
        }

    def mt5_transaction_history_csv(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
        symbol: str | None = None,
        type: str | None = None,
        sort: str = "time_desc",
        max_rows: int = 5000,
    ) -> tuple[str, str]:
        rows_limit = min(max(int(max_rows), 1), 10000)
        payload = self.mt5_transaction_history(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
            type=type,
            sort=sort,
            page=1,
            page_size=rows_limit,
            page_size_cap=rows_limit,
        )
        items = list(payload.get("items") or [])
        buffer = StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "time_utc",
                "kind",
                "symbol",
                "ticket",
                "order_ticket",
                "position_id",
                "side",
                "transaction_type",
                "state",
                "volume",
                "price",
                "profit",
                "commission",
                "swap",
                "fee",
                "comment",
                "is_manual",
                "portfolio_slug",
                "account_id",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "time_utc": item.get("time_utc"),
                    "kind": item.get("kind"),
                    "symbol": item.get("symbol"),
                    "ticket": item.get("ticket"),
                    "order_ticket": item.get("order_ticket"),
                    "position_id": item.get("position_id"),
                    "side": item.get("side"),
                    "transaction_type": item.get("transaction_type"),
                    "state": item.get("state"),
                    "volume": item.get("volume"),
                    "price": item.get("price"),
                    "profit": item.get("profit"),
                    "commission": item.get("commission"),
                    "swap": item.get("swap"),
                    "fee": item.get("fee"),
                    "comment": item.get("comment"),
                    "is_manual": bool(item.get("is_manual")),
                    "portfolio_slug": payload.get("portfolio_slug"),
                    "account_id": payload.get("account_id"),
                }
            )
        filename = (
            f"mt5-transactions-"
            f"{str(payload.get('portfolio_slug') or 'portfolio')}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
        )
        return filename, buffer.getvalue()

    def list_instruments(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.market.list_instruments(portfolio_slug=portfolio_slug)

    def live_holdings(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        live_state = self._safe_live_state_summary(portfolio_slug=portfolio["slug"])
        live_holdings = list((live_state or {}).get("holdings") or [])
        if live_holdings:
            return live_holdings
        return self.market.live_holdings(portfolio_slug=portfolio["slug"])

    def live_exposure(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        live_state = self._safe_live_state_summary(portfolio_slug=portfolio["slug"])
        if (live_state or {}).get("exposure") is not None:
            return dict(live_state["exposure"])
        return self.market.live_exposure(portfolio_slug=portfolio["slug"])

    def _normalize_risk_data_quality(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        raw = dict(payload.get("data_quality") or {})
        estimation_window_days = int(
            raw.get("estimation_window_days")
            or payload.get("days")
            or self.runtime.risk_defaults.get("estimation_window_days")
            or 0
        )
        minimum_valid_days = int(
            raw.get("minimum_valid_days")
            or self.runtime.risk_defaults.get("minimum_valid_days")
            or 0
        )
        available_observations = int(raw.get("available_observations") or payload.get("sample_size") or 0)
        if not raw and available_observations <= 0 and estimation_window_days <= 0 and minimum_valid_days <= 0:
            return None
        default_epsilon = self._float_or_none(self.runtime.risk_defaults.get("no_exposure_epsilon_eur"))
        if default_epsilon is None or default_epsilon < 0.0:
            default_epsilon = 1.0
        configured_epsilons = {
            str(symbol).upper(): max(float(value), 0.0)
            for symbol, value in dict(self.runtime.risk_defaults.get("no_exposure_epsilon_by_symbol") or {}).items()
            if symbol not in {None, ""}
            and self._float_or_none(value) is not None
        }
        exposure_map = {
            str(symbol).upper(): abs(float(value))
            for symbol, value in dict(payload.get("exposure_by_symbol") or {}).items()
            if symbol not in {None, ""} and self._float_or_none(value) is not None
        }
        if not exposure_map:
            gross_from_holdings = 0.0
            holdings_found = False
            for item in list(payload.get("holdings") or []):
                if not isinstance(item, Mapping):
                    continue
                symbol = item.get("symbol")
                signed = self._float_or_none(item.get("signed_exposure_base_ccy"))
                absolute = self._float_or_none(item.get("exposure_base_ccy"))
                exposure = signed if signed is not None else absolute
                if symbol in {None, ""} or exposure is None:
                    continue
                normalized_symbol = str(symbol).upper()
                exposure_map[normalized_symbol] = exposure_map.get(normalized_symbol, 0.0) + abs(float(exposure))
                gross_from_holdings += abs(float(exposure))
                holdings_found = True
            if holdings_found and "gross_exposure_base_ccy" not in raw:
                raw["gross_exposure_base_ccy"] = gross_from_holdings
        epsilon_by_symbol = {
            str(symbol).upper(): max(float(value), 0.0)
            for symbol, value in dict(raw.get("no_exposure_epsilon_by_symbol") or {}).items()
            if symbol not in {None, ""} and self._float_or_none(value) is not None
        }
        if not epsilon_by_symbol and exposure_map:
            epsilon_by_symbol = {
                symbol: configured_epsilons.get(symbol, float(default_epsilon))
                for symbol in exposure_map
            }
        gross_exposure = self._float_or_none(raw.get("gross_exposure_base_ccy"))
        if gross_exposure is None and exposure_map:
            gross_exposure = float(sum(abs(value) for value in exposure_map.values()))
        gross_exposure_epsilon = self._float_or_none(raw.get("gross_exposure_epsilon_base_ccy"))
        if gross_exposure_epsilon is None and epsilon_by_symbol:
            gross_exposure_epsilon = float(sum(max(value, 0.0) for value in epsilon_by_symbol.values()))
        status = raw.get("status")
        if not status:
            has_no_exposure = (
                gross_exposure is not None
                and gross_exposure_epsilon is not None
                and gross_exposure <= max(gross_exposure_epsilon, 0.0)
            )
            if has_no_exposure:
                status = "no_exposure"
            elif available_observations <= 0:
                status = "incomplete"
            elif minimum_valid_days > 0 and available_observations < minimum_valid_days:
                status = "thin_history"
            else:
                status = "healthy"
        return {
            "status": str(status),
            "estimation_window_days": int(estimation_window_days),
            "minimum_valid_days": int(minimum_valid_days),
            "available_observations": int(available_observations),
            "oldest_observation": raw.get("oldest_observation"),
            "latest_observation": raw.get("latest_observation") or payload.get("latest_observation") or payload.get("time_utc"),
            "horizon_observations": dict(raw.get("horizon_observations") or {}),
            "symbol_count": int(raw.get("symbol_count") or 0),
            "gross_exposure_base_ccy": None if gross_exposure is None else float(gross_exposure),
            "gross_exposure_epsilon_base_ccy": (
                None if gross_exposure_epsilon is None else float(gross_exposure_epsilon)
            ),
            "no_exposure_epsilon_by_symbol": epsilon_by_symbol,
        }

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            if value in {None, "", "null"}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _build_risk_concentration(
        cls,
        *,
        items: Mapping[str, Any],
        metric_field: str,
        key_field: str = "symbol",
        basis: str,
        top_limit: int = 5,
    ) -> dict[str, Any] | None:
        rows: list[dict[str, Any]] = []
        for raw_key, raw_item in dict(items or {}).items():
            item = dict(raw_item or {})
            key = str(item.get(key_field) or raw_key or "").strip()
            if not key:
                continue
            metric_value = cls._float_or_none(item.get(metric_field))
            if metric_value is None:
                continue
            rows.append(
                {
                    "key": key,
                    "label": str(item.get(key_field) or key),
                    "abs_metric": abs(metric_value),
                    "component_var": cls._float_or_none(item.get("component_var")),
                    "component_es": cls._float_or_none(item.get("component_es")),
                }
            )
        if not rows:
            return None

        total_abs = float(sum(float(row["abs_metric"]) for row in rows))
        if total_abs <= 0.0:
            return {
                "basis": basis,
                "count": len(rows),
                "hhi": None,
                "normalized_hhi": None,
                "effective_count": None,
                "top1_share": None,
                "top3_share": None,
                "top5_share": None,
                "dominant_key": None,
                "dominant_label": None,
                "contributors": [],
            }

        ranked = sorted(rows, key=lambda item: float(item["abs_metric"]), reverse=True)
        shares = [float(item["abs_metric"]) / total_abs for item in ranked]
        hhi = float(sum(share * share for share in shares))
        effective_count = None if hhi <= 0.0 else (1.0 / hhi)
        normalized_hhi = None
        if len(shares) > 1:
            floor = 1.0 / float(len(shares))
            denom = 1.0 - floor
            if denom > 0.0:
                normalized_hhi = max(0.0, min((hhi - floor) / denom, 1.0))
        top1_share = shares[0] if shares else None
        top3_share = float(sum(shares[:3])) if shares else None
        top5_share = float(sum(shares[:5])) if shares else None
        dominant = ranked[0] if ranked else None

        contributors = []
        for index, item in enumerate(ranked[: max(int(top_limit), 1)]):
            contributors.append(
                {
                    "key": item["key"],
                    "label": item["label"],
                    "share": shares[index],
                    "component_var": item["component_var"],
                    "component_es": item["component_es"],
                }
            )

        return {
            "basis": basis,
            "count": len(ranked),
            "hhi": hhi,
            "normalized_hhi": normalized_hhi,
            "effective_count": effective_count,
            "top1_share": top1_share,
            "top3_share": top3_share,
            "top5_share": top5_share,
            "dominant_key": None if dominant is None else dominant["key"],
            "dominant_label": None if dominant is None else dominant["label"],
            "contributors": contributors,
        }

    @classmethod
    def _diversification_ratio(
        cls,
        *,
        positions: Mapping[str, Any],
        total_field_value: Any,
        standalone_field: str,
    ) -> float | None:
        total_risk = cls._float_or_none(total_field_value)
        if total_risk is None or abs(total_risk) <= 1e-12:
            return None
        standalone_sum = 0.0
        count = 0
        for raw_item in dict(positions or {}).values():
            item = dict(raw_item or {})
            standalone = cls._float_or_none(item.get(standalone_field))
            if standalone is None:
                continue
            standalone_sum += abs(standalone)
            count += 1
        if count <= 0:
            return None
        return float(standalone_sum / abs(total_risk))

    @staticmethod
    def _resolve_model_name(
        models: Mapping[str, Any],
        *candidates: Any,
    ) -> str | None:
        available = {str(key).lower(): str(key) for key in dict(models or {}).keys()}
        for candidate in candidates:
            normalized = str(candidate or "").strip().lower()
            if not normalized:
                continue
            if normalized in available:
                return available[normalized]
        if not available:
            return None
        return next(iter(available.values()))

    def _enrich_risk_attribution_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(payload or {})
        model_payloads = dict(enriched.get("models") or {})
        enriched_models: dict[str, Any] = {}
        for model_name, raw_model_payload in model_payloads.items():
            model_payload = dict(raw_model_payload or {})
            positions = dict(model_payload.get("positions") or {})
            model_payload["diversification_ratio_var"] = self._diversification_ratio(
                positions=positions,
                total_field_value=model_payload.get("total_var"),
                standalone_field="standalone_var",
            )
            model_payload["diversification_ratio_es"] = self._diversification_ratio(
                positions=positions,
                total_field_value=model_payload.get("total_es"),
                standalone_field="standalone_es",
            )
            model_payload["concentration_var"] = self._build_risk_concentration(
                items=positions,
                metric_field="component_var",
                key_field="symbol",
                basis="abs_component_var_share",
            )
            model_payload["concentration_es"] = self._build_risk_concentration(
                items=positions,
                metric_field="component_es",
                key_field="symbol",
                basis="abs_component_es_share",
            )
            enriched_models[str(model_name)] = model_payload
        enriched["models"] = enriched_models
        return enriched

    def _build_risk_summary_concentration(
        self,
        *,
        reference_model: Any,
        preferred_model: Any,
        attribution: Mapping[str, Any] | None = None,
        risk_budget: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        attribution_payload = dict(attribution or {})
        if attribution_payload:
            enriched = self._enrich_risk_attribution_payload(attribution_payload)
            models = dict(enriched.get("models") or {})
            selected_model = self._resolve_model_name(models, reference_model, preferred_model)
            if selected_model is not None:
                model_payload = dict(models.get(selected_model) or {})
                return {
                    "model": selected_model,
                    "diversification_ratio_var": model_payload.get("diversification_ratio_var"),
                    "diversification_ratio_es": model_payload.get("diversification_ratio_es"),
                    "var": model_payload.get("concentration_var"),
                    "es": model_payload.get("concentration_es"),
                }

        budget_models = dict(dict(risk_budget or {}).get("models") or {})
        selected_model = self._resolve_model_name(budget_models, reference_model, preferred_model)
        if selected_model is None:
            return None
        model_payload = dict(budget_models.get(selected_model) or {})
        positions = dict(model_payload.get("positions") or {})
        concentration_var = self._build_risk_concentration(
            items=positions,
            metric_field="component_var",
            key_field="symbol",
            basis="abs_component_var_share",
        )
        concentration_es = self._build_risk_concentration(
            items=positions,
            metric_field="component_es",
            key_field="symbol",
            basis="abs_component_es_share",
        )
        if concentration_var is None and concentration_es is None:
            return None
        return {
            "model": selected_model,
            "diversification_ratio_var": None,
            "diversification_ratio_es": None,
            "var": concentration_var,
            "es": concentration_es,
        }

    def risk_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        live_state = self._safe_live_state_summary(portfolio_slug=portfolio["slug"])
        if live_state is not None and live_state.get("risk_summary") is not None:
            summary = dict(live_state["risk_summary"])
            if summary.get("concentration") is None:
                concentration = self._build_risk_summary_concentration(
                    reference_model=summary.get("reference_model"),
                    preferred_model=summary.get("preferred_model"),
                    attribution=live_state.get("attribution"),
                    risk_budget=live_state.get("risk_budget"),
                )
                if concentration is not None:
                    summary["concentration"] = concentration
            return summary

        snapshot = None
        for source in self.reads._preferred_snapshot_sources(
            portfolio_slug=portfolio["slug"],
            source="auto",
        ):
            snapshot = self.reads.latest_snapshot(source=source, portfolio_slug=portfolio["slug"])
            if snapshot is not None:
                break
        if snapshot is None:
            return None
        payload = dict(snapshot.get("payload") or {})
        summary = {
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
            "data_quality": self._normalize_risk_data_quality(payload),
            "model_diagnostics": dict(payload.get("model_diagnostics") or {}),
            "risk_nowcast": dict(payload.get("risk_nowcast") or {}),
            "microstructure": dict(payload.get("microstructure") or {}),
            "tick_quality": dict(payload.get("tick_quality") or {}),
            "pnl_explain": dict(payload.get("pnl_explain") or {}),
        }
        concentration = self._build_risk_summary_concentration(
            reference_model=summary.get("reference_model"),
            preferred_model=summary.get("preferred_model"),
            attribution=payload.get("attribution"),
            risk_budget=payload.get("risk_budget"),
        )
        if concentration is not None:
            summary["concentration"] = concentration
        return summary

    def risk_contributions(self, *, portfolio_slug: str | None = None, source: str | None = None) -> dict[str, Any] | None:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        if source:
            payload = self.reads.latest_risk_attribution(source=source, portfolio_slug=portfolio["slug"])
            if payload is None:
                return None
            return self._enrich_risk_attribution_payload(payload)
        for candidate in self.reads._preferred_snapshot_sources(
            portfolio_slug=portfolio["slug"],
            source="auto",
        ):
            payload = self.reads.latest_risk_attribution(source=candidate, portfolio_slug=portfolio["slug"])
            if payload is not None:
                return self._enrich_risk_attribution_payload(payload)
        return None

    def market_data_status(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        payload = self.market.market_data_status(portfolio_slug=portfolio_slug)
        live_state = self.mt5.cached_live_state(portfolio_slug=portfolio_slug, detail_level="summary")
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

        # Keep this endpoint lightweight: do not force full live-state enrichment when cache is cold.
        # A terminal ping is enough to expose bridge liveness metadata for status cards.
        try:
            terminal = self.mt5.mt5_status()
        except Exception:
            terminal = None
        if terminal is not None:
            connected = bool(terminal.get("connected"))
            payload = {
                **payload,
                "live_bridge_status": "ok" if connected else "degraded",
                "live_bridge_connected": connected,
                "live_bridge_stale": None,
                "live_bridge_generated_at": terminal.get("timestamp_utc"),
                "live_bridge_sequence": None,
            }
        return payload

    def sync_market_data(
        self,
        *,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        days: int | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.market.sync_market_data(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
            days=days,
            timeframes=timeframes,
        )

    def market_data_sync_runs(
        self,
        *,
        portfolio_slug: str | None = None,
        status: str | list[str] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        raw_statuses: list[str] | None
        if status is None:
            raw_statuses = None
        elif isinstance(status, str):
            raw_statuses = [status]
        else:
            raw_statuses = [str(item) for item in status]
        statuses = (
            [item.strip() for item in raw_statuses if item.strip()]
            if raw_statuses is not None
            else None
        )
        if raw_statuses is not None and not statuses:
            return []
        return self.market.market_data_sync_runs(
            portfolio_slug=portfolio_slug,
            statuses=statuses,
            limit=limit,
        )

    def _normalize_operator_payload(
        self,
        *,
        action: str,
        request_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(request_payload or {})
        portfolio = self.runtime._resolve_portfolio_context(payload.get("portfolio_slug"))
        normalized: dict[str, Any] = {
            **payload,
            "portfolio_slug": portfolio["slug"],
        }
        normalized["account_id"] = self.runtime.resolve_mt5_account_id(payload.get("account_id"))
        if action in {"sync", "snapshot", "backtest"}:
            normalized["days"] = int(normalized.get("days") or self.runtime._default_days())
        if action in {"snapshot", "backtest"}:
            normalized["timeframe"] = str(normalized.get("timeframe") or self.runtime._default_timeframe())
        if action == "sync":
            raw_timeframes = normalized.get("timeframes")
            if raw_timeframes:
                normalized["timeframes"] = [str(item).upper() for item in list(raw_timeframes)]
            else:
                normalized["timeframes"] = list(self.runtime.market_data.startup_sync_timeframes())
        return normalized

    @staticmethod
    def _operator_artifact_refs(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {}
        refs: dict[str, Any] = {}
        for key, value in dict(payload).items():
            if value is None:
                continue
            if key.endswith("_id") or key.endswith("_path") or key in {"compare_csv", "validation_json", "report_markdown"}:
                refs[key] = value
        return refs

    @staticmethod
    def _operator_error_payload(exc: Exception) -> dict[str, str]:
        message = str(exc) or exc.__class__.__name__
        if isinstance(exc, FileNotFoundError):
            return {
                "error_code": "missing_artifact",
                "error_message": message,
                "hint": "Run a fresh sync/backtest before requesting this artifact again.",
            }
        if isinstance(exc, ValueError):
            return {
                "error_code": "invalid_request",
                "error_message": message,
                "hint": "Check the requested portfolio, timeframe, and history inputs.",
            }
        if isinstance(exc, RuntimeError):
            if "mt5_live_unavailable" in message.lower():
                return {
                    "error_code": "mt5_live_unavailable",
                    "error_message": message,
                    "hint": (
                        "Strict live mode requires a readable MT5 live book. "
                        "Verify VAR_PROJECT_MT5_AGENT_BASE_URL / VAR_PROJECT_MT5_AGENT_API_KEY on api/worker/celery-worker."
                    ),
                }
            return {
                "error_code": "runtime_error",
                "error_message": message,
                "hint": "Inspect the operator run details and the latest backend logs for the failing stage.",
            }
        return {
            "error_code": "unexpected_error",
            "error_message": message,
            "hint": "Retry the action once. If it fails again, inspect the request_id/run_id in backend logs.",
        }

    @staticmethod
    def _operator_requires_live_mt5(*, action: str, portfolio: Mapping[str, Any]) -> bool:
        mode = str(portfolio.get("mode") or "").lower()
        if mode != "live_mt5":
            return False
        return str(action or "").lower() in {"sync", "snapshot", "backtest", "report"}

    @staticmethod
    def _operator_sync_result_looks_offline(sync_result: Mapping[str, Any] | None) -> bool:
        payload = dict(sync_result or {})
        status = str(payload.get("status") or "").lower()
        configured = payload.get("configured")
        return status in {"offline_fixture", "not_configured"} or configured is False

    @staticmethod
    def _operator_mt5_not_configured_payload(*, action: str, portfolio_slug: str | None) -> dict[str, str]:
        slug = str(portfolio_slug or "unknown")
        return {
            "error_code": "mt5_live_unavailable",
            "error_message": (
                f"Operator action '{action}' requires strict live MT5 for portfolio '{slug}', "
                "but the live MT5 book is unavailable in this worker environment."
            ),
            "hint": (
                "Configure VAR_PROJECT_MT5_AGENT_BASE_URL (and VAR_PROJECT_MT5_AGENT_API_KEY if required) "
                "on api/worker/celery-worker, then retry."
            ),
        }

    def _refresh_market_data_for_operator(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        portfolio = self.runtime._resolve_portfolio_context(payload.get("portfolio_slug"))
        if not self.runtime.market_data.should_use_mt5_market_data(portfolio):
            return self.market_data_status(portfolio_slug=portfolio["slug"])
        requested_timeframes = payload.get("timeframes")
        if not requested_timeframes and payload.get("timeframe"):
            requested_timeframes = [payload["timeframe"]]
        return self.runtime.market_data.sync_market_data_if_stale(
            portfolio_slug=portfolio["slug"],
            account_id=payload.get("account_id"),
            max_age_seconds=90.0,
            days=int(payload.get("days") or self.runtime._default_days()),
            timeframes=requested_timeframes,
        )

    @staticmethod
    def _operator_env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)

    def _operator_running_ttl_seconds(self, action: str) -> int:
        defaults = {
            "sync": 120,
            "snapshot": 240,
            "backtest": 420,
            "report": 360,
        }
        action_key = str(action or "").lower()
        env_name = f"VAR_PROJECT_OPERATOR_STALE_RUNNING_{action_key.upper()}"
        fallback = defaults.get(action_key, 300)
        return max(30, self._operator_env_int(env_name, fallback))

    def _operator_queued_ttl_seconds(self, action: str) -> int:
        defaults = {
            "sync": 60,
            "snapshot": 90,
            "backtest": 120,
            "report": 120,
        }
        action_key = str(action or "").lower()
        env_name = f"VAR_PROJECT_OPERATOR_STALE_QUEUED_{action_key.upper()}"
        fallback = defaults.get(action_key, 90)
        return max(20, self._operator_env_int(env_name, fallback))

    def _operator_cache_ttl_seconds(self, action: str) -> int:
        defaults = {
            "sync": 20,
            "snapshot": 45,
            "backtest": 90,
            "report": 120,
        }
        action_key = str(action or "").lower()
        env_name = f"VAR_PROJECT_OPERATOR_CACHE_TTL_{action_key.upper()}"
        fallback = defaults.get(action_key, 45)
        return max(0, self._operator_env_int(env_name, fallback))

    def _operator_poll_after_ms(self, action: str) -> int:
        defaults = {
            "sync": 900,
            "snapshot": 1100,
            "backtest": 1500,
            "report": 1500,
        }
        action_key = str(action or "").lower()
        env_name = f"VAR_PROJECT_OPERATOR_POLL_MS_{action_key.upper()}"
        fallback = defaults.get(action_key, 1500)
        return max(400, self._operator_env_int(env_name, fallback))

    def _operator_sla(self, action: str) -> dict[str, int]:
        action_key = str(action or "").lower()
        queued_timeout_seconds = int(self._operator_queued_ttl_seconds(action_key))
        running_timeout_seconds = int(self._operator_running_ttl_seconds(action_key))
        poll_after_ms = int(self._operator_poll_after_ms(action_key))
        return {
            "queued_timeout_seconds": queued_timeout_seconds,
            "running_timeout_seconds": running_timeout_seconds,
            "sla_seconds": running_timeout_seconds,
            "poll_after_ms": poll_after_ms,
        }

    @staticmethod
    def _operator_elapsed_seconds(run: Mapping[str, Any]) -> float | None:
        status = str(run.get("status") or "").lower()
        started = coerce_datetime(run.get("started_at"))
        if started is None and status in {"queued", "running"}:
            started = coerce_datetime(run.get("created_at")) or coerce_datetime(run.get("updated_at"))
        if started is None:
            return None
        finished = coerce_datetime(run.get("finished_at"))
        end = finished or utcnow()
        return max(0.0, (end - started).total_seconds())

    def _operator_is_stale(self, run: Mapping[str, Any]) -> bool:
        status = str(run.get("status") or "").lower()
        action = str(run.get("action") or "").lower()
        now = utcnow()
        if status == "queued":
            queued_anchor = coerce_datetime(run.get("created_at")) or coerce_datetime(run.get("updated_at"))
            if queued_anchor is None:
                return False
            return (now - queued_anchor).total_seconds() > float(self._operator_queued_ttl_seconds(action))
        if status == "running":
            running_anchor = (
                coerce_datetime(run.get("started_at"))
                or coerce_datetime(run.get("updated_at"))
                or coerce_datetime(run.get("created_at"))
            )
            if running_anchor is None:
                return False
            return (now - running_anchor).total_seconds() > float(self._operator_running_ttl_seconds(action))
        return False

    def _operator_stale_context(self, run: Mapping[str, Any]) -> dict[str, Any]:
        status = str(run.get("status") or "").lower()
        action = str(run.get("action") or "").lower()
        now = utcnow()
        if status == "queued":
            anchor = coerce_datetime(run.get("created_at")) or coerce_datetime(run.get("updated_at"))
            ttl_seconds = int(self._operator_queued_ttl_seconds(action))
            status_reason = "abandoned"
            error_code = "abandoned_stale_run"
        else:
            anchor = (
                coerce_datetime(run.get("started_at"))
                or coerce_datetime(run.get("updated_at"))
                or coerce_datetime(run.get("created_at"))
            )
            ttl_seconds = int(self._operator_running_ttl_seconds(action))
            status_reason = "timeout"
            error_code = "timeout_stale_run"
        age_seconds = None if anchor is None else max(0.0, (now - anchor).total_seconds())
        return {
            "status": status or "running",
            "status_reason": status_reason,
            "error_code": error_code,
            "ttl_seconds": ttl_seconds,
            "age_seconds": age_seconds,
        }

    @staticmethod
    def _legacy_operator_status_reason(error_code: str | None) -> str | None:
        normalized_error_code = str(error_code or "").strip().lower()
        if normalized_error_code == "timeout_stale_run":
            return "timeout"
        if normalized_error_code == "abandoned_stale_run":
            return "abandoned"
        if normalized_error_code == "operator_interrupted":
            return "interrupted"
        return None

    def _decorate_operator_run(self, run: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(run)
        request_payload = dict(payload.get("request_payload") or {})
        try:
            payload["account_id"] = self.runtime.resolve_mt5_account_id(request_payload.get("account_id"))
        except ValueError:
            payload["account_id"] = self.runtime.resolve_mt5_account_id(None)
        elapsed_seconds = self._operator_elapsed_seconds(payload)
        status = str(payload.get("status") or "").lower()
        action = str(payload.get("action") or "").lower()
        status_reason = str(payload.get("status_reason") or "").strip().lower()
        if not status_reason:
            legacy_reason = self._legacy_operator_status_reason(payload.get("error_code"))
            if legacy_reason is not None:
                payload["status_reason"] = legacy_reason
        sla = self._operator_sla(action)
        payload["elapsed_seconds"] = None if elapsed_seconds is None else round(float(elapsed_seconds), 3)
        payload["is_stale"] = bool(self._operator_is_stale(payload))
        payload["queued_timeout_seconds"] = int(sla["queued_timeout_seconds"])
        payload["running_timeout_seconds"] = int(sla["running_timeout_seconds"])
        payload["sla_seconds"] = int(sla["sla_seconds"])
        payload["interruptible"] = status in {"queued", "running"}
        payload["poll_after_ms"] = (
            None
            if status in {"succeeded", "failed"}
            else int(sla["poll_after_ms"])
        )
        return payload

    def _latest_matching_operator_run(
        self,
        *,
        portfolio_slug: str,
        action: str,
        request_payload: Mapping[str, Any],
        statuses: list[str],
        limit: int = 20,
    ) -> dict[str, Any] | None:
        normalized_payload = jsonable(dict(request_payload))
        candidates = self.storage.list_operator_runs(
            portfolio_slug=portfolio_slug,
            action=action,
            statuses=statuses,
            limit=limit,
        )
        for run in candidates:
            if jsonable(dict(run.get("request_payload") or {})) == normalized_payload:
                return run
        return None

    def _fail_stale_operator_run(self, run: Mapping[str, Any]) -> dict[str, Any] | None:
        run_id = int(run["id"])
        action = str(run.get("action") or "operator")
        stale_context = self._operator_stale_context(run)
        status = str(stale_context.get("status") or "running")
        status_reason = str(stale_context.get("status_reason") or "timeout")
        error_code = str(stale_context.get("error_code") or "timeout_stale_run")
        ttl_seconds = int(stale_context.get("ttl_seconds") or 0)
        age_seconds = stale_context.get("age_seconds")
        age_fragment = (
            "unknown age"
            if age_seconds is None
            else f"age {round(float(age_seconds), 3)}s"
        )
        stale_hint = (
            f"Previous {action} run was auto-closed as '{status_reason}' from '{status}'. "
            f"Observed {age_fragment} for TTL {ttl_seconds}s."
        )
        stale_message = (
            f"Operator run {run_id} became stale while {status} "
            f"({age_fragment}, ttl={ttl_seconds}s, reason={status_reason}). "
            "The run was closed automatically to avoid an infinite pending state."
        )
        return self.storage.update_operator_run(
            run_id,
            status="failed",
            stage="failed",
            status_reason=status_reason,
            error_code=error_code,
            error_message=stale_message,
            hint=stale_hint,
            finished_at=utcnow(),
        )

    def reap_stale_operator_runs(
        self,
        *,
        portfolio_slug: str | None = None,
        action: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        stale_updates: list[dict[str, Any]] = []
        candidates = self.storage.list_operator_runs(
            portfolio_slug=portfolio_slug,
            action=action,
            statuses=["queued", "running"],
            limit=limit,
        )
        for run in candidates:
            if not self._operator_is_stale(run):
                continue
            updated = self._fail_stale_operator_run(run)
            if updated is not None:
                stale_updates.append(self._decorate_operator_run(updated))
        return stale_updates

    def operator_run(self, run_id: int) -> dict[str, Any] | None:
        run = self.storage.operator_run_by_id(run_id)
        if run is None:
            return None
        if self._operator_is_stale(run):
            failed = self._fail_stale_operator_run(run)
            if failed is not None:
                return self._decorate_operator_run(failed)
            run = self.storage.operator_run_by_id(run_id) or run
        return self._decorate_operator_run(run)

    def interrupt_operator_run(
        self,
        run_id: int,
        *,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        self.runtime.require_storage_ready()
        run = self.storage.operator_run_by_id(run_id)
        if run is None:
            return None
        status = str(run.get("status") or "").lower()
        if status in {"succeeded", "failed"}:
            return self._decorate_operator_run(run)

        normalized_reason = str(reason or "").strip()
        if not normalized_reason:
            normalized_reason = "Interrupted by operator."
        message = (
            f"Operator run {int(run_id)} was interrupted while in status '{status}'. "
            f"{normalized_reason}"
        )
        updated = self.storage.interrupt_operator_run(
            int(run_id),
            error_code="operator_interrupted",
            error_message=message,
            status_reason="interrupted",
            hint="The run was interrupted by the operator. Re-enqueue the action if needed.",
            stage="failed",
            finished_at=utcnow(),
        )
        if updated is not None:
            return self._decorate_operator_run(updated)

        refreshed = self.storage.operator_run_by_id(int(run_id))
        return None if refreshed is None else self._decorate_operator_run(refreshed)

    def _operator_interrupted_run_state(self, run_id: int) -> dict[str, Any] | None:
        latest_state = self.storage.operator_run_by_id(int(run_id))
        if latest_state is None:
            return None
        if str(latest_state.get("status") or "").lower() != "failed":
            return None
        if str(latest_state.get("error_code") or "") != "operator_interrupted":
            return None
        return latest_state

    def operator_runs(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 25,
        action: str | None = None,
        statuses: list[str] | None = None,
        status_reasons: list[str] | None = None,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        candidate_statuses = [str(item).lower() for item in (statuses or []) if str(item).strip()]
        if (not candidate_statuses) or any(item in {"queued", "running"} for item in candidate_statuses):
            self.reap_stale_operator_runs(portfolio_slug=portfolio_slug, action=action, limit=max(limit, 50))
        runs = self.storage.list_operator_runs(
            portfolio_slug=portfolio_slug,
            limit=limit,
            action=action,
            statuses=statuses,
            status_reasons=status_reasons,
        )
        decorated = [self._decorate_operator_run(run) for run in runs]
        if account_id in {None, "", "null"}:
            return decorated
        resolved_account_id = self.runtime.resolve_mt5_account_id(account_id)
        default_account_id = self.runtime.resolve_mt5_account_id(None)
        filtered: list[dict[str, Any]] = []
        for run in decorated:
            payload = dict(run.get("request_payload") or {})
            run_account_id = self.runtime.resolve_mt5_account_id(payload.get("account_id") or default_account_id)
            if run_account_id == resolved_account_id:
                filtered.append(run)
        return filtered

    def enqueue_operator_action(
        self,
        *,
        action: str,
        request_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        normalized_payload = self._normalize_operator_payload(action=action, request_payload=request_payload)
        portfolio = self.runtime._resolve_portfolio_context(normalized_payload.get("portfolio_slug"))
        self.reap_stale_operator_runs(portfolio_slug=portfolio["slug"], action=action, limit=50)

        existing = self.storage.latest_active_operator_run(
            portfolio_slug=portfolio["slug"],
            action=action,
            request_payload=normalized_payload,
        )
        if existing is not None:
            return self._decorate_operator_run(
                {
                    **existing,
                    "reused": True,
                    "reused_run_id": existing.get("id"),
                }
            )

        cache_ttl = self._operator_cache_ttl_seconds(action)
        if cache_ttl > 0:
            cached = self._latest_matching_operator_run(
                portfolio_slug=portfolio["slug"],
                action=action,
                request_payload=normalized_payload,
                statuses=["succeeded"],
                limit=25,
            )
            if cached is not None:
                finished_at = coerce_datetime(cached.get("finished_at")) or coerce_datetime(cached.get("updated_at"))
                if finished_at is not None:
                    cache_age_seconds = (utcnow() - finished_at).total_seconds()
                    if cache_age_seconds <= float(cache_ttl):
                        return self._decorate_operator_run(
                            {
                                **cached,
                                "reused": True,
                                "reused_run_id": cached.get("id"),
                            }
                        )

        request_id = uuid4().hex
        run_id = self.storage.create_operator_run(
            portfolio_id=self.runtime._resolve_portfolio_id(portfolio["slug"]),
            portfolio_slug=portfolio["slug"],
            action=action,
            request_id=request_id,
            status="queued",
            stage="accepted",
            request_payload=normalized_payload,
        )
        created = self.storage.operator_run_by_id(run_id)
        if created is None:
            raise RuntimeError("Operator run could not be persisted.")
        return self._decorate_operator_run(
            {
                **created,
                "reused": False,
            }
        )

    def process_operator_run(self, run_id: int) -> dict[str, Any]:
        self.runtime.require_storage_ready()
        run = self.storage.operator_run_by_id(run_id)
        if run is None:
            raise ValueError(f"Unknown operator run '{run_id}'.")
        if str(run.get("status")) in {"succeeded", "failed"}:
            return self._decorate_operator_run(run)
        if self._operator_is_stale(run):
            failed = self._fail_stale_operator_run(run)
            if failed is not None:
                return self._decorate_operator_run(failed)
            refreshed = self.storage.operator_run_by_id(run_id)
            if refreshed is not None:
                return self._decorate_operator_run(refreshed)
        run_status = str(run.get("status") or "").lower()
        if run_status == "running":
            return self._decorate_operator_run(run)

        claimed = self.storage.claim_operator_run(
            run_id,
            stage="starting",
            started_at=utcnow(),
        )
        if claimed is None:
            refreshed = self.storage.operator_run_by_id(run_id)
            if refreshed is not None:
                refreshed_status = str(refreshed.get("status") or "").lower()
                if refreshed_status in {"running", "succeeded", "failed"}:
                    return self._decorate_operator_run(refreshed)
                if self._operator_is_stale(refreshed):
                    failed = self._fail_stale_operator_run(refreshed)
                    if failed is not None:
                        return self._decorate_operator_run(failed)
            if run_status != "queued":
                return self._decorate_operator_run(run)
        else:
            run = claimed

        payload = dict(run.get("request_payload") or {})
        portfolio_slug = payload.get("portfolio_slug")
        action = str(run.get("action") or "").lower()
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        requires_live_mt5 = self._operator_requires_live_mt5(action=action, portfolio=portfolio)
        artifact_refs: dict[str, Any] = {}
        result_payload: dict[str, Any] = {}
        if requires_live_mt5 and not self.runtime.market_data.mt5_configured():
            error_payload = self._operator_mt5_not_configured_payload(
                action=action,
                portfolio_slug=portfolio_slug,
            )
            updated = self.storage.update_operator_run(
                run_id,
                status="failed",
                stage="failed",
                status_reason="mt5_unavailable",
                error_code=error_payload["error_code"],
                error_message=error_payload["error_message"],
                hint=error_payload["hint"],
                finished_at=utcnow(),
            )
            if updated is None:
                raise RuntimeError(f"Operator run '{run_id}' disappeared during MT5 preflight failure.")
            return self._decorate_operator_run(updated)
        try:
            if action == "sync":
                self.storage.update_operator_run(run_id, stage="syncing_market_data")
                interrupted = self._operator_interrupted_run_state(run_id)
                if interrupted is not None:
                    return self._decorate_operator_run(interrupted)
                sync_result = self.market.sync_market_data(
                    portfolio_slug=portfolio_slug,
                    account_id=payload.get("account_id"),
                    days=payload.get("days"),
                    timeframes=payload.get("timeframes"),
                )
                if requires_live_mt5 and self._operator_sync_result_looks_offline(sync_result):
                    error_payload = self._operator_mt5_not_configured_payload(
                        action=action,
                        portfolio_slug=portfolio_slug,
                    )
                    updated = self.storage.update_operator_run(
                        run_id,
                        status="failed",
                        stage="failed",
                        status_reason="mt5_unavailable",
                        result={"sync": sync_result},
                        error_code=error_payload["error_code"],
                        error_message=error_payload["error_message"],
                        hint=error_payload["hint"],
                        finished_at=utcnow(),
                    )
                    if updated is None:
                        raise RuntimeError(f"Operator run '{run_id}' disappeared during MT5 sync failure.")
                    return self._decorate_operator_run(updated)
                result_payload = {"sync": sync_result}
            elif action == "snapshot":
                self.storage.update_operator_run(run_id, stage="syncing_market_data")
                interrupted = self._operator_interrupted_run_state(run_id)
                if interrupted is not None:
                    return self._decorate_operator_run(interrupted)
                sync_result = self._refresh_market_data_for_operator(payload)
                if requires_live_mt5 and self._operator_sync_result_looks_offline(sync_result):
                    error_payload = self._operator_mt5_not_configured_payload(
                        action=action,
                        portfolio_slug=portfolio_slug,
                    )
                    updated = self.storage.update_operator_run(
                        run_id,
                        status="failed",
                        stage="failed",
                        status_reason="mt5_unavailable",
                        result={"sync": sync_result},
                        error_code=error_payload["error_code"],
                        error_message=error_payload["error_message"],
                        hint=error_payload["hint"],
                        finished_at=utcnow(),
                    )
                    if updated is None:
                        raise RuntimeError(f"Operator run '{run_id}' disappeared during MT5 sync failure.")
                    return self._decorate_operator_run(updated)
                self.storage.update_operator_run(run_id, stage="running_snapshot")
                interrupted = self._operator_interrupted_run_state(run_id)
                if interrupted is not None:
                    return self._decorate_operator_run(interrupted)
                snapshot_result = self.analytics.run_snapshot(**payload)
                artifact_refs.update(self._operator_artifact_refs(snapshot_result))
                result_payload = {"sync": sync_result, "snapshot": snapshot_result}
            elif action == "backtest":
                self.storage.update_operator_run(run_id, stage="syncing_market_data")
                interrupted = self._operator_interrupted_run_state(run_id)
                if interrupted is not None:
                    return self._decorate_operator_run(interrupted)
                sync_result = self._refresh_market_data_for_operator(payload)
                if requires_live_mt5 and self._operator_sync_result_looks_offline(sync_result):
                    error_payload = self._operator_mt5_not_configured_payload(
                        action=action,
                        portfolio_slug=portfolio_slug,
                    )
                    updated = self.storage.update_operator_run(
                        run_id,
                        status="failed",
                        stage="failed",
                        status_reason="mt5_unavailable",
                        result={"sync": sync_result},
                        error_code=error_payload["error_code"],
                        error_message=error_payload["error_message"],
                        hint=error_payload["hint"],
                        finished_at=utcnow(),
                    )
                    if updated is None:
                        raise RuntimeError(f"Operator run '{run_id}' disappeared during MT5 sync failure.")
                    return self._decorate_operator_run(updated)
                self.storage.update_operator_run(run_id, stage="running_backtest")
                interrupted = self._operator_interrupted_run_state(run_id)
                if interrupted is not None:
                    return self._decorate_operator_run(interrupted)
                backtest_result = self.analytics.run_backtest(**payload)
                artifact_refs.update(self._operator_artifact_refs(backtest_result))
                result_payload = {"sync": sync_result, "backtest": backtest_result}
            elif action == "report":
                compare_path = payload.get("compare_path")
                if compare_path is None and self.latest_backtest(portfolio_slug=portfolio_slug) is None:
                    self.storage.update_operator_run(run_id, stage="warming_backtest")
                    interrupted = self._operator_interrupted_run_state(run_id)
                    if interrupted is not None:
                        return self._decorate_operator_run(interrupted)
                    sync_result = self._refresh_market_data_for_operator(
                        {
                            "portfolio_slug": portfolio_slug,
                            "days": self.runtime._default_days(),
                            "timeframe": self.runtime._default_timeframe(),
                        }
                    )
                    if requires_live_mt5 and self._operator_sync_result_looks_offline(sync_result):
                        error_payload = self._operator_mt5_not_configured_payload(
                            action=action,
                            portfolio_slug=portfolio_slug,
                        )
                        updated = self.storage.update_operator_run(
                            run_id,
                            status="failed",
                            stage="failed",
                            status_reason="mt5_unavailable",
                            result={"sync": sync_result},
                            error_code=error_payload["error_code"],
                            error_message=error_payload["error_message"],
                            hint=error_payload["hint"],
                            finished_at=utcnow(),
                        )
                        if updated is None:
                            raise RuntimeError(f"Operator run '{run_id}' disappeared during MT5 sync failure.")
                        return self._decorate_operator_run(updated)
                    interrupted = self._operator_interrupted_run_state(run_id)
                    if interrupted is not None:
                        return self._decorate_operator_run(interrupted)
                    backtest_result = self.analytics.run_backtest(
                        portfolio_slug=portfolio_slug,
                        days=self.runtime._default_days(),
                        timeframe=self.runtime._default_timeframe(),
                    )
                    artifact_refs.update(self._operator_artifact_refs(backtest_result))
                    result_payload["sync"] = sync_result
                    result_payload["backtest"] = backtest_result
                    compare_path = backtest_result.get("compare_csv")
                self.storage.update_operator_run(run_id, stage="running_report")
                interrupted = self._operator_interrupted_run_state(run_id)
                if interrupted is not None:
                    return self._decorate_operator_run(interrupted)
                report_result = self.analytics.run_report(compare_path=compare_path, portfolio_slug=portfolio_slug)
                artifact_refs.update(self._operator_artifact_refs(report_result))
                result_payload["report"] = report_result
            else:
                raise ValueError(f"Unsupported operator action '{action}'.")

            interrupted = self._operator_interrupted_run_state(run_id)
            if interrupted is not None:
                return self._decorate_operator_run(interrupted)

            updated = self.storage.update_operator_run(
                run_id,
                status="succeeded",
                stage="completed",
                artifact_refs=artifact_refs,
                result=result_payload,
                finished_at=utcnow(),
            )
            if updated is None:
                raise RuntimeError(f"Operator run '{run_id}' disappeared during completion.")
            return self._decorate_operator_run(updated)
        except Exception as exc:
            interrupted = self._operator_interrupted_run_state(run_id)
            if interrupted is not None:
                return self._decorate_operator_run(interrupted)
            error_payload = self._operator_error_payload(exc)
            updated = self.storage.update_operator_run(
                run_id,
                status="failed",
                stage="failed",
                status_reason="execution_error",
                artifact_refs=artifact_refs,
                result=result_payload if result_payload else None,
                error_code=error_payload["error_code"],
                error_message=error_payload["error_message"],
                hint=error_payload["hint"],
                finished_at=utcnow(),
            )
            if updated is None:
                raise
            return self._decorate_operator_run(updated)

    def reconciliation_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug, force_refresh=True)
        if live_state.get("reconciliation") is not None:
            return self.market.reconciliation_summary_from_live_state(
                live_state,
                portfolio_slug=portfolio_slug,
            )
        return self.market.reconciliation_summary(portfolio_slug=portfolio_slug)

    def reconciliation_history(
        self,
        *,
        portfolio_slug: str | None = None,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.market.reconciliation_history(
            portfolio_slug=portfolio_slug,
            severity=severity,
            limit=limit,
        )

    @staticmethod
    def _normalize_trade_exposure_change(
        *,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        strict: bool = True,
    ) -> float:
        raw = exposure_change if exposure_change is not None else delta_position_eur
        try:
            parsed = float(raw if raw is not None else 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(TRADE_EXPOSURE_RULE_MESSAGE) from exc
        if not math.isfinite(parsed):
            raise ValueError(TRADE_EXPOSURE_RULE_MESSAGE)
        if not strict:
            return parsed

        magnitude = abs(parsed)
        if magnitude < TRADE_EXPOSURE_MIN_EUR:
            raise ValueError(TRADE_EXPOSURE_RULE_MESSAGE)

        normalized_steps = magnitude / TRADE_EXPOSURE_STEP_EUR
        if abs(normalized_steps - round(normalized_steps)) > TRADE_EXPOSURE_EPSILON:
            raise ValueError(TRADE_EXPOSURE_RULE_MESSAGE)
        return parsed

    def preview_execution(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        strict_trade_exposure_validation: bool = True,
    ) -> dict[str, Any]:
        normalized_exposure_change = self._normalize_trade_exposure_change(
            exposure_change=exposure_change,
            delta_position_eur=delta_position_eur,
            strict=strict_trade_exposure_validation,
        )
        return self.mt5.preview_execution(
            symbol=symbol,
            exposure_change=normalized_exposure_change,
            delta_position_eur=None,
            note=note,
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )

    def submit_execution(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        strict_trade_exposure_validation: bool = True,
    ) -> dict[str, Any]:
        normalized_exposure_change = self._normalize_trade_exposure_change(
            exposure_change=exposure_change,
            delta_position_eur=delta_position_eur,
            strict=strict_trade_exposure_validation,
        )
        return self.mt5.submit_execution(
            symbol=symbol,
            exposure_change=normalized_exposure_change,
            delta_position_eur=None,
            note=note,
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )

    def evaluate_trade_decision(
        self,
        *,
        symbol: str,
        exposure_change: float | None = None,
        delta_position_eur: float | None = None,
        note: str | None = None,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
        strict_trade_exposure_validation: bool = True,
    ) -> dict[str, Any]:
        normalized_exposure_change = self._normalize_trade_exposure_change(
            exposure_change=exposure_change,
            delta_position_eur=delta_position_eur,
            strict=strict_trade_exposure_validation,
        )
        return self.trading.evaluate_trade_decision(
            symbol=symbol,
            exposure_change=normalized_exposure_change,
            delta_position_eur=None,
            note=note,
            portfolio_slug=portfolio_slug,
            account_id=account_id,
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
        account_id: str | None = None,
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
            account_id=account_id,
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
        account_id: str | None = None,
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
            account_id=account_id,
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

    def run_report(
        self,
        *,
        compare_path: str | None = None,
        portfolio_slug: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        return self.analytics.run_report(
            compare_path=compare_path,
            portfolio_slug=portfolio_slug,
            account_id=account_id,
        )

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

    def latest_report_content(
        self,
        *,
        portfolio_slug: str | None = None,
        report_id: int | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        report = self.reads.latest_report_content(
            portfolio_slug=portfolio_slug,
            report_id=report_id,
            account_id=account_id,
        )
        if report is not None:
            return report
        if report_id is not None:
            return None
        compare_path = self.runtime._resolve_compare_path(None, portfolio_slug=portfolio_slug)
        if compare_path is None or not compare_path.exists():
            return None
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        live_snapshot = self.storage.latest_snapshot(source="mt5_live_bridge", portfolio_slug=portfolio["slug"])
        if live_snapshot is not None:
            self.runtime.governance.refresh_live_report_if_needed(
                portfolio=portfolio,
                portfolio_id=self.runtime._resolve_portfolio_id(portfolio["slug"]),
                snapshot_payload=dict(live_snapshot.get("payload") or {}),
                snapshot_id=int(live_snapshot["id"]),
                source="mt5_live_bridge",
                metadata=dict((live_snapshot.get("payload") or {}).get("metadata") or {}),
            )
            report = self.reads.latest_report_content(portfolio_slug=portfolio["slug"], account_id=account_id)
            if report is not None:
                return report
        self.analytics.run_report(compare_path=str(compare_path), portfolio_slug=portfolio_slug, account_id=account_id)
        return self.reads.latest_report_content(portfolio_slug=portfolio_slug, account_id=account_id)
