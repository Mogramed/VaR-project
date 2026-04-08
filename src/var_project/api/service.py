from __future__ import annotations

import os
from pathlib import Path
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

    def health_dependencies(self) -> dict[str, Any]:
        return self.reads.health_dependencies()

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
            live_state = self.mt5.cached_live_state(portfolio_slug=slug, detail_level="summary")
            if live_state is None:
                try:
                    live_state = self.mt5.live_state(portfolio_slug=slug, detail_level="summary")
                except Exception:
                    live_state = None
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
        severity_rank = {"breach": 0, "critical": 0, "warn": 1, "warning": 1, "info": 2, "ok": 3}
        payload.sort(
            key=lambda item: (
                int(severity_rank.get(str(item.get("severity", "")).lower(), 4)),
                str(item.get("code") or ""),
                str(item.get("message") or ""),
            )
        )
        return payload[: max(int(limit), 1)]

    def recent_decisions(self, *, limit: int = 25, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.reads.recent_decisions(limit=limit, portfolio_slug=portfolio_slug)

    def latest_capital(
        self,
        *,
        portfolio_slug: str | None = None,
        source: str | None = "auto",
    ) -> dict[str, Any]:
        normalized_source = str(source or "auto").strip().lower()
        live_source_requested = normalized_source in {"", "auto", "mt5_live_bridge", "mt5_live"}
        try:
            live_state = self.mt5.live_state(portfolio_slug=portfolio_slug, detail_level="summary")
        except Exception:
            live_state = None
        if live_source_requested and live_state is not None and live_state.get("capital_usage") is not None:
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

    def portfolio_capital(self, portfolio_slug: str, *, source: str | None = "auto") -> dict[str, Any]:
        return self.reads.latest_capital(portfolio_slug=portfolio_slug, source=source)

    def desk_overview(self, *, desk_slug: str | None = None) -> dict[str, Any]:
        if desk_slug is not None and desk_slug != self.desk.slug:
            raise ValueError(f"Unknown desk '{desk_slug}'.")
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
            snapshots.append(self.reads.latest_capital(portfolio_slug=portfolio["slug"]))
            if live_alert_count:
                alert_counts[portfolio["slug"]] = int(live_alert_count)
        return build_desk_snapshot(self.desk.to_dict(), snapshots, portfolio_map, alerts_by_portfolio=alert_counts).to_dict()

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
        return self.reads.latest_risk_attribution(source=source, portfolio_slug=portfolio_slug)

    def latest_risk_budget(
        self,
        *,
        source: str = "auto",
        portfolio_slug: str | None = None,
    ) -> dict[str, Any] | None:
        return self.reads.latest_risk_budget(source=source, portfolio_slug=portfolio_slug)

    def mt5_status(self) -> dict[str, Any]:
        live_state = self.mt5.live_state(portfolio_slug=self.portfolio["slug"], detail_level="summary")
        if live_state.get("terminal_status") is not None:
            return dict(live_state["terminal_status"])
        return self.mt5.mt5_status()

    def mt5_account(self) -> dict[str, Any]:
        live_state = self.mt5.live_state(portfolio_slug=self.portfolio["slug"], detail_level="summary")
        if live_state.get("account") is not None:
            return dict(live_state["account"])
        return self.mt5.mt5_account()

    def mt5_positions(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.mt5.mt5_positions(portfolio_slug=portfolio_slug)

    def mt5_orders(self, *, portfolio_slug: str | None = None) -> list[dict[str, Any]]:
        return self.mt5.mt5_orders(portfolio_slug=portfolio_slug)

    def mt5_live_state(
        self,
        *,
        portfolio_slug: str | None = None,
        detail_level: str = "full",
    ) -> dict[str, Any]:
        return self.mt5.live_state(
            portfolio_slug=portfolio_slug,
            detail_level=detail_level,
        )

    def mt5_live_events(
        self,
        *,
        portfolio_slug: str | None = None,
        after: int = 0,
        limit: int = 100,
        wait_seconds: float = 15.0,
        detail_level: str = "full",
    ) -> list[dict[str, Any]]:
        return self.mt5.live_events(
            portfolio_slug=portfolio_slug,
            after=after,
            limit=limit,
            wait_seconds=wait_seconds,
            detail_level=detail_level,
        )

    def mt5_analytics_series(
        self,
        *,
        portfolio_slug: str | None = None,
        window_minutes: int = 240,
        max_points: int = 300,
    ) -> dict[str, Any]:
        return self.mt5.analytics_series(
            portfolio_slug=portfolio_slug,
            window_minutes=window_minutes,
            max_points=max_points,
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
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug, detail_level="summary")
        live_holdings = list(live_state.get("holdings") or [])
        if live_holdings:
            return live_holdings
        return self.market.live_holdings(portfolio_slug=portfolio_slug)

    def live_exposure(self, *, portfolio_slug: str | None = None) -> dict[str, Any]:
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug, detail_level="summary")
        if live_state.get("exposure") is not None:
            return dict(live_state["exposure"])
        return self.market.live_exposure(portfolio_slug=portfolio_slug)

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
        status = raw.get("status")
        if not status:
            if available_observations <= 0:
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
        }

    def risk_summary(self, *, portfolio_slug: str | None = None) -> dict[str, Any] | None:
        portfolio = self.runtime._resolve_portfolio_context(portfolio_slug)
        try:
            live_state = self.mt5.live_state(portfolio_slug=portfolio["slug"], detail_level="summary")
        except Exception:
            live_state = None
        if live_state is not None and live_state.get("risk_summary") is not None:
            return dict(live_state["risk_summary"])

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
            "data_quality": self._normalize_risk_data_quality(payload),
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
        for candidate in self.reads._preferred_snapshot_sources(
            portfolio_slug=portfolio["slug"],
            source="auto",
        ):
            payload = self.reads.latest_risk_attribution(source=candidate, portfolio_slug=portfolio["slug"])
            if payload is not None:
                return payload
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
        days: int | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.market.sync_market_data(portfolio_slug=portfolio_slug, days=days, timeframes=timeframes)

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
        if mode not in {"live_mt5", "hybrid"}:
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

    def _decorate_operator_run(self, run: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(run)
        elapsed_seconds = self._operator_elapsed_seconds(payload)
        status = str(payload.get("status") or "").lower()
        action = str(payload.get("action") or "").lower()
        payload["elapsed_seconds"] = None if elapsed_seconds is None else round(float(elapsed_seconds), 3)
        payload["is_stale"] = bool(self._operator_is_stale(payload))
        payload["poll_after_ms"] = (
            None
            if status in {"succeeded", "failed"}
            else int(self._operator_poll_after_ms(action))
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
        status = str(run.get("status") or "running")
        stale_hint = (
            f"Previous {action} run timed out in '{status}'. A fresh run has been enqueued."
        )
        stale_message = (
            f"Operator run {run_id} became stale while {status}. "
            "The run was closed automatically to avoid an infinite pending state."
        )
        return self.storage.update_operator_run(
            run_id,
            status="failed",
            stage="failed",
            error_code="timeout_stale_run",
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

    def operator_runs(
        self,
        *,
        portfolio_slug: str | None = None,
        limit: int = 25,
        action: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_statuses = [str(item).lower() for item in (statuses or []) if str(item).strip()]
        if (not candidate_statuses) or any(item in {"queued", "running"} for item in candidate_statuses):
            self.reap_stale_operator_runs(portfolio_slug=portfolio_slug, action=action, limit=max(limit, 50))
        runs = self.storage.list_operator_runs(
            portfolio_slug=portfolio_slug,
            limit=limit,
            action=action,
            statuses=statuses,
        )
        return [self._decorate_operator_run(run) for run in runs]

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
                sync_result = self.market.sync_market_data(
                    portfolio_slug=portfolio_slug,
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
                snapshot_result = self.analytics.run_snapshot(**payload)
                artifact_refs.update(self._operator_artifact_refs(snapshot_result))
                result_payload = {"sync": sync_result, "snapshot": snapshot_result}
            elif action == "backtest":
                self.storage.update_operator_run(run_id, stage="syncing_market_data")
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
                backtest_result = self.analytics.run_backtest(**payload)
                artifact_refs.update(self._operator_artifact_refs(backtest_result))
                result_payload = {"sync": sync_result, "backtest": backtest_result}
            elif action == "report":
                compare_path = payload.get("compare_path")
                if compare_path is None and self.latest_backtest(portfolio_slug=portfolio_slug) is None:
                    self.storage.update_operator_run(run_id, stage="warming_backtest")
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
                            result={"sync": sync_result},
                            error_code=error_payload["error_code"],
                            error_message=error_payload["error_message"],
                            hint=error_payload["hint"],
                            finished_at=utcnow(),
                        )
                        if updated is None:
                            raise RuntimeError(f"Operator run '{run_id}' disappeared during MT5 sync failure.")
                        return self._decorate_operator_run(updated)
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
                report_result = self.analytics.run_report(compare_path=compare_path, portfolio_slug=portfolio_slug)
                artifact_refs.update(self._operator_artifact_refs(report_result))
                result_payload["report"] = report_result
            else:
                raise ValueError(f"Unsupported operator action '{action}'.")

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
            error_payload = self._operator_error_payload(exc)
            updated = self.storage.update_operator_run(
                run_id,
                status="failed",
                stage="failed",
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
        live_state = self.mt5.live_state(portfolio_slug=portfolio_slug)
        if live_state.get("reconciliation") is not None:
            return self.market.reconciliation_summary_from_live_state(
                live_state,
                portfolio_slug=portfolio_slug,
            )
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

    def latest_report_content(
        self,
        *,
        portfolio_slug: str | None = None,
        report_id: int | None = None,
    ) -> dict[str, Any] | None:
        report = self.reads.latest_report_content(portfolio_slug=portfolio_slug, report_id=report_id)
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
            report = self.reads.latest_report_content(portfolio_slug=portfolio["slug"])
            if report is not None:
                return report
        self.analytics.run_report(compare_path=str(compare_path), portfolio_slug=portfolio_slug)
        return self.reads.latest_report_content(portfolio_slug=portfolio_slug)
