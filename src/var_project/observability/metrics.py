from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
from threading import Lock
import time
from typing import Any, Mapping

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from var_project.storage.serialization import coerce_datetime


LOGGER = logging.getLogger("var_project.observability.metrics")
_METRICS_REFRESH_LOCK = Lock()

_KNOWN_OPERATOR_ACTIONS = ("sync", "snapshot", "backtest", "report", "other")
_KNOWN_OPERATOR_STATUSES = ("queued", "running", "succeeded", "failed", "other")
_KNOWN_OPERATOR_STALE_REASONS = ("timeout", "abandoned", "interrupted", "other")
_KNOWN_SYNC_STATUSES = ("ok", "running", "incomplete", "failed", "other")
_KNOWN_RECON_SEVERITIES = ("ok", "warn", "critical")
_SYNC_RUNS_OBSERVATION_WINDOW = timedelta(minutes=30)


API_HTTP_REQUESTS_TOTAL = Counter(
    "var_api_http_requests_total",
    "Total number of API HTTP requests.",
    ("method", "route", "status_code"),
)

API_HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "var_api_http_request_duration_seconds",
    "API HTTP request latency in seconds.",
    ("method", "route", "status_code"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

OBSERVABILITY_COLLECTION_ERRORS_TOTAL = Counter(
    "var_observability_collection_errors_total",
    "Number of metric collection errors.",
    ("scope",),
)

OBSERVABILITY_COLLECTION_DURATION_SECONDS = Histogram(
    "var_observability_collection_duration_seconds",
    "Duration of observability metric collection/scrape refresh.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

OBSERVABILITY_LAST_COLLECTION_UNIX_SECONDS = Gauge(
    "var_observability_last_collection_unix_seconds",
    "Unix timestamp of the latest successful observability metric collection.",
)

OPERATOR_RUNS_WINDOW_TOTAL = Gauge(
    "var_operator_runs_window_total",
    "Operator runs count over the current observation window.",
    ("portfolio_slug", "action", "status"),
)

OPERATOR_STALE_RUNS_WINDOW_TOTAL = Gauge(
    "var_operator_stale_runs_window_total",
    "Operator stale/interrupt closures observed in the current window.",
    ("portfolio_slug", "reason"),
)

OPERATOR_FAILURE_RATIO = Gauge(
    "var_operator_failure_ratio",
    "Operator failure ratio over terminal runs in the current window (0-1).",
    ("portfolio_slug", "action"),
)

MARKET_DATA_SYNC_RUNS_WINDOW_TOTAL = Gauge(
    "var_market_data_sync_runs_window_total",
    "Market-data sync runs count over the current observation window.",
    ("portfolio_slug", "status"),
)

MARKET_DATA_SYNC_LAST_STATUS = Gauge(
    "var_market_data_sync_last_status",
    "Latest market-data sync status as one-hot gauge (status label set to 1).",
    ("portfolio_slug", "status"),
)

RECONCILIATION_MISMATCHES_TOTAL = Gauge(
    "var_reconciliation_mismatches_total",
    "Reconciliation mismatch counts by severity.",
    ("portfolio_slug", "severity"),
)

RECONCILIATION_UNMATCHED_EXECUTIONS_TOTAL = Gauge(
    "var_reconciliation_unmatched_executions_total",
    "Current reconciliation unmatched desk execution count.",
    ("portfolio_slug",),
)

RECONCILIATION_WINDOW_EXPIRED_EXECUTIONS_TOTAL = Gauge(
    "var_reconciliation_window_expired_executions_total",
    "Current count of desk executions outside live reconciliation history window.",
    ("portfolio_slug",),
)

RECONCILIATION_ACTIVE_INCIDENTS_TOTAL = Gauge(
    "var_reconciliation_active_incidents_total",
    "Current count of active reconciliation incidents.",
    ("portfolio_slug",),
)

MT5_BRIDGE_STATE = Gauge(
    "var_mt5_bridge_state",
    "MT5 bridge state flags (connected/degraded/stale/fallback_snapshot_used) as 0|1.",
    ("portfolio_slug", "state"),
)

MT5_BRIDGE_CONSECUTIVE_FAILURES = Gauge(
    "var_mt5_bridge_consecutive_failures",
    "Number of consecutive MT5 bridge polling failures.",
    ("portfolio_slug",),
)


def _bounded_label(value: Any, *, fallback: str = "unknown", max_len: int = 96) -> str:
    if value in {None, "", "null"}:
        return fallback
    text = str(value).strip().lower()
    if not text:
        return fallback
    if len(text) > int(max_len):
        return text[: int(max_len)]
    return text


def observe_http_request(*, method: str, route: str, status_code: int, duration_seconds: float) -> None:
    method_label = _bounded_label(method, fallback="unknown", max_len=16)
    route_label = _bounded_label(route, fallback="unmatched", max_len=120)
    status_label = _bounded_label(status_code, fallback="0", max_len=8)
    safe_duration = max(float(duration_seconds), 0.0)
    API_HTTP_REQUESTS_TOTAL.labels(
        method=method_label,
        route=route_label,
        status_code=status_label,
    ).inc()
    API_HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method_label,
        route=route_label,
        status_code=status_label,
    ).observe(safe_duration)


def _clear_runtime_gauges() -> None:
    OPERATOR_RUNS_WINDOW_TOTAL.clear()
    OPERATOR_STALE_RUNS_WINDOW_TOTAL.clear()
    OPERATOR_FAILURE_RATIO.clear()
    MARKET_DATA_SYNC_RUNS_WINDOW_TOTAL.clear()
    MARKET_DATA_SYNC_LAST_STATUS.clear()
    RECONCILIATION_MISMATCHES_TOTAL.clear()
    RECONCILIATION_UNMATCHED_EXECUTIONS_TOTAL.clear()
    RECONCILIATION_WINDOW_EXPIRED_EXECUTIONS_TOTAL.clear()
    RECONCILIATION_ACTIVE_INCIDENTS_TOTAL.clear()
    MT5_BRIDGE_STATE.clear()
    MT5_BRIDGE_CONSECUTIVE_FAILURES.clear()


def _normalize_stale_reason(run: Mapping[str, Any]) -> str | None:
    status_reason = _bounded_label(run.get("status_reason"), fallback="")
    error_code = _bounded_label(run.get("error_code"), fallback="")
    if status_reason in {"timeout", "abandoned", "interrupted"}:
        return status_reason
    if error_code == "timeout_stale_run":
        return "timeout"
    if error_code == "abandoned_stale_run":
        return "abandoned"
    if error_code == "operator_interrupted":
        return "interrupted"
    if status_reason or error_code:
        return "other"
    return None


def _set_operator_metrics(*, portfolio_slug: str, runs: list[Mapping[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    stale_counts: dict[str, int] = defaultdict(int)
    terminal_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"failed": 0, "succeeded": 0})

    for run in runs:
        action = _bounded_label(run.get("action"), fallback="other")
        if action not in _KNOWN_OPERATOR_ACTIONS:
            action = "other"
        status = _bounded_label(run.get("status"), fallback="other")
        if status not in _KNOWN_OPERATOR_STATUSES:
            status = "other"
        counts[(action, status)] += 1

        if status in {"failed", "succeeded"}:
            terminal_totals[action][status] += 1

        stale_reason = _normalize_stale_reason(run)
        if stale_reason is not None:
            if stale_reason not in _KNOWN_OPERATOR_STALE_REASONS:
                stale_reason = "other"
            stale_counts[stale_reason] += 1

    for action in _KNOWN_OPERATOR_ACTIONS:
        for status in _KNOWN_OPERATOR_STATUSES:
            OPERATOR_RUNS_WINDOW_TOTAL.labels(
                portfolio_slug=portfolio_slug,
                action=action,
                status=status,
            ).set(float(counts.get((action, status), 0)))

        terminals = terminal_totals.get(action, {"failed": 0, "succeeded": 0})
        failed = int(terminals.get("failed", 0))
        succeeded = int(terminals.get("succeeded", 0))
        denominator = failed + succeeded
        ratio = float(failed / denominator) if denominator > 0 else 0.0
        OPERATOR_FAILURE_RATIO.labels(
            portfolio_slug=portfolio_slug,
            action=action,
        ).set(ratio)

    for reason in _KNOWN_OPERATOR_STALE_REASONS:
        OPERATOR_STALE_RUNS_WINDOW_TOTAL.labels(
            portfolio_slug=portfolio_slug,
            reason=reason,
        ).set(float(stale_counts.get(reason, 0)))


def _market_sync_run_timestamp(run: Mapping[str, Any]) -> datetime | None:
    for field in ("synced_at", "updated_at", "created_at"):
        candidate = coerce_datetime(run.get(field))
        if candidate is not None:
            return candidate
    return None


def _set_market_sync_metrics(
    *,
    portfolio_slug: str,
    runs: list[Mapping[str, Any]],
    now: datetime,
) -> None:
    cutoff = now - _SYNC_RUNS_OBSERVATION_WINDOW
    counts: dict[str, int] = defaultdict(int)
    latest_status = "other"
    latest_timestamp: datetime | None = None
    for idx, run in enumerate(runs):
        status = _bounded_label(run.get("status"), fallback="other")
        if status not in _KNOWN_SYNC_STATUSES:
            status = "other"
        timestamp = _market_sync_run_timestamp(run)
        if timestamp is not None:
            if timestamp >= cutoff:
                counts[status] += 1
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_status = status
        elif idx == 0 and latest_timestamp is None:
            # Keep alerting deterministic even if legacy rows miss timestamps.
            latest_status = status

    for status in _KNOWN_SYNC_STATUSES:
        MARKET_DATA_SYNC_RUNS_WINDOW_TOTAL.labels(
            portfolio_slug=portfolio_slug,
            status=status,
        ).set(float(counts.get(status, 0)))
        MARKET_DATA_SYNC_LAST_STATUS.labels(
            portfolio_slug=portfolio_slug,
            status=status,
        ).set(1.0 if status == latest_status else 0.0)


def _set_reconciliation_metrics(*, portfolio_slug: str, summary: Mapping[str, Any] | None) -> None:
    if summary is None:
        for severity in _KNOWN_RECON_SEVERITIES:
            RECONCILIATION_MISMATCHES_TOTAL.labels(portfolio_slug=portfolio_slug, severity=severity).set(0.0)
        RECONCILIATION_UNMATCHED_EXECUTIONS_TOTAL.labels(portfolio_slug=portfolio_slug).set(0.0)
        RECONCILIATION_WINDOW_EXPIRED_EXECUTIONS_TOTAL.labels(portfolio_slug=portfolio_slug).set(0.0)
        RECONCILIATION_ACTIVE_INCIDENTS_TOTAL.labels(portfolio_slug=portfolio_slug).set(0.0)
        return

    severity_counts = dict(summary.get("severity_counts") or {})
    for severity in _KNOWN_RECON_SEVERITIES:
        RECONCILIATION_MISMATCHES_TOTAL.labels(
            portfolio_slug=portfolio_slug,
            severity=severity,
        ).set(float(severity_counts.get(severity, 0)))

    RECONCILIATION_UNMATCHED_EXECUTIONS_TOTAL.labels(portfolio_slug=portfolio_slug).set(
        float(int(summary.get("unmatched_execution_count") or 0))
    )
    RECONCILIATION_WINDOW_EXPIRED_EXECUTIONS_TOTAL.labels(portfolio_slug=portfolio_slug).set(
        float(int(summary.get("history_window_expired_execution_count") or 0))
    )
    RECONCILIATION_ACTIVE_INCIDENTS_TOTAL.labels(portfolio_slug=portfolio_slug).set(
        float(int(summary.get("active_incident_count") or 0))
    )


def _set_mt5_bridge_metrics(*, portfolio_slug: str, live_state: Mapping[str, Any] | None) -> None:
    connected = bool(live_state and live_state.get("connected"))
    degraded = bool(live_state and live_state.get("degraded"))
    stale = bool(live_state and live_state.get("stale"))
    fallback_snapshot_used = bool(live_state and live_state.get("fallback_snapshot_used"))
    consecutive_failures = (
        int(live_state.get("bridge_consecutive_failures") or 0)
        if isinstance(live_state, Mapping)
        else 0
    )

    MT5_BRIDGE_STATE.labels(portfolio_slug=portfolio_slug, state="connected").set(1.0 if connected else 0.0)
    MT5_BRIDGE_STATE.labels(portfolio_slug=portfolio_slug, state="degraded").set(1.0 if degraded else 0.0)
    MT5_BRIDGE_STATE.labels(portfolio_slug=portfolio_slug, state="stale").set(1.0 if stale else 0.0)
    MT5_BRIDGE_STATE.labels(portfolio_slug=portfolio_slug, state="fallback_snapshot_used").set(
        1.0 if fallback_snapshot_used else 0.0
    )
    MT5_BRIDGE_CONSECUTIVE_FAILURES.labels(portfolio_slug=portfolio_slug).set(float(consecutive_failures))


def refresh_runtime_metrics(service: Any) -> None:
    started_at = time.perf_counter()
    collection_time_utc = datetime.now(timezone.utc)
    with _METRICS_REFRESH_LOCK:
        _clear_runtime_gauges()
        portfolios = [dict(item) for item in list(getattr(service, "portfolios", []) or [])]
        if not portfolios:
            OBSERVABILITY_LAST_COLLECTION_UNIX_SECONDS.set(time.time())
            OBSERVABILITY_COLLECTION_DURATION_SECONDS.observe(max(time.perf_counter() - started_at, 0.0))
            return

        for portfolio in portfolios:
            portfolio_slug_raw = str(portfolio.get("slug") or "").strip()
            portfolio_slug_query = None if not portfolio_slug_raw else portfolio_slug_raw
            portfolio_slug = _bounded_label(portfolio_slug_raw, fallback="unknown")

            operator_runs: list[Mapping[str, Any]]
            try:
                operator_runs = list(
                    service.storage.list_operator_runs(
                        portfolio_slug=portfolio_slug_query,
                        limit=200,
                    )
                )
            except Exception as exc:
                OBSERVABILITY_COLLECTION_ERRORS_TOTAL.labels(scope="operator_runs").inc()
                LOGGER.warning("operator-run metrics refresh failed for %s: %s", portfolio_slug, exc)
                operator_runs = []
            _set_operator_metrics(portfolio_slug=portfolio_slug, runs=operator_runs)

            sync_runs: list[Mapping[str, Any]]
            try:
                sync_runs = list(
                    service.storage.list_market_data_sync_runs(
                        portfolio_slug=portfolio_slug_query,
                        limit=200,
                    )
                )
            except Exception as exc:
                OBSERVABILITY_COLLECTION_ERRORS_TOTAL.labels(scope="market_data_sync").inc()
                LOGGER.warning("market-data sync metrics refresh failed for %s: %s", portfolio_slug, exc)
                sync_runs = []
            _set_market_sync_metrics(
                portfolio_slug=portfolio_slug,
                runs=sync_runs,
                now=collection_time_utc,
            )

            reconciliation_summary: Mapping[str, Any] | None
            try:
                events = service.storage.recent_audit_events(
                    limit=1,
                    portfolio_slug=portfolio_slug_query,
                    object_type="reconciliation_snapshot",
                )
                reconciliation_summary = None if not events else dict(events[0])
            except Exception as exc:
                OBSERVABILITY_COLLECTION_ERRORS_TOTAL.labels(scope="reconciliation").inc()
                LOGGER.warning("reconciliation metrics refresh failed for %s: %s", portfolio_slug, exc)
                reconciliation_summary = None
            _set_reconciliation_metrics(portfolio_slug=portfolio_slug, summary=reconciliation_summary)

            live_state: Mapping[str, Any] | None
            try:
                live_state = service.mt5.cached_live_state(portfolio_slug=portfolio_slug_query, detail_level="summary")
                if live_state is None:
                    live_state = service.mt5.live_state(
                        portfolio_slug=portfolio_slug_query,
                        detail_level="summary",
                        force_refresh=False,
                    )
                if live_state is not None:
                    live_state = dict(live_state)
            except Exception as exc:
                OBSERVABILITY_COLLECTION_ERRORS_TOTAL.labels(scope="mt5_live_state").inc()
                LOGGER.warning("mt5 live-state metrics refresh failed for %s: %s", portfolio_slug, exc)
                live_state = None
            _set_mt5_bridge_metrics(portfolio_slug=portfolio_slug, live_state=live_state)

        OBSERVABILITY_LAST_COLLECTION_UNIX_SECONDS.set(time.time())
        OBSERVABILITY_COLLECTION_DURATION_SECONDS.observe(max(time.perf_counter() - started_at, 0.0))


def build_metrics_payload(*, service: Any) -> tuple[bytes, str]:
    refresh_runtime_metrics(service)
    return generate_latest(), CONTENT_TYPE_LATEST
