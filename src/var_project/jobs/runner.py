from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.exc import OperationalError, ProgrammingError

from var_project.core.exceptions import MT5ConnectionError
from var_project.core.settings import (
    get_data_defaults,
    get_mt5_config,
    get_portfolio_contexts,
    get_risk_defaults,
    load_settings,
)
from var_project.storage import AppStorage
from var_project.jobs.operator_queue import dispatch_operator_run


@dataclass(frozen=True)
class ScheduledJob:
    enabled: bool = True
    interval_seconds: int = 3600
    timeframe: str | None = None
    days: int | None = None
    min_coverage: float | None = None
    alpha: float | None = None
    window: int | None = None
    n_sims: int | None = None
    dist: str | None = None
    df_t: int | None = None
    seed: int | None = None
    compare_path: str | None = None


@dataclass(frozen=True)
class WorkerSettings:
    loop_sleep_seconds: int = 30
    snapshot: ScheduledJob = ScheduledJob(enabled=True, interval_seconds=900)
    backtest: ScheduledJob = ScheduledJob(enabled=True, interval_seconds=3600)
    live_refresh: ScheduledJob = ScheduledJob(enabled=False, interval_seconds=15)
    report: ScheduledJob = ScheduledJob(enabled=True, interval_seconds=3600)


def _scheduled_job(
    payload: dict[str, Any],
    *,
    enabled_default: bool,
    interval_default: int,
    timeframe_default: str | None = None,
    days_default: int | None = None,
    min_coverage_default: float | None = None,
    alpha_default: float | None = None,
    window_default: int | None = None,
    n_sims_default: int | None = None,
    dist_default: str | None = None,
    df_t_default: int | None = None,
    seed_default: int | None = None,
) -> ScheduledJob:
    return ScheduledJob(
        enabled=bool(payload.get("enabled", enabled_default)),
        interval_seconds=int(payload.get("interval_seconds", interval_default)),
        timeframe=payload.get("timeframe", timeframe_default),
        days=payload.get("days", days_default),
        min_coverage=payload.get("min_coverage", min_coverage_default),
        alpha=payload.get("alpha", alpha_default),
        window=payload.get("window", window_default),
        n_sims=payload.get("n_sims", n_sims_default),
        dist=payload.get("dist", dist_default),
        df_t=payload.get("df_t", df_t_default),
        seed=payload.get("seed", seed_default),
        compare_path=payload.get("compare_path"),
    )


def load_worker_settings(root: Path) -> WorkerSettings:
    raw_cfg = load_settings(root)
    data_defaults = get_data_defaults(raw_cfg)
    risk_defaults = get_risk_defaults(raw_cfg)
    mt5_config = get_mt5_config(raw_cfg)
    portfolios = get_portfolio_contexts(raw_cfg)
    jobs_cfg = dict(raw_cfg.get("jobs") or {})

    default_timeframe = (data_defaults["timeframes"][0] if data_defaults["timeframes"] else "H1")
    default_days = (data_defaults["history_days_list"][-1] if data_defaults["history_days_list"] else 365)
    default_alpha = risk_defaults["alpha"]
    default_window = risk_defaults["window"]
    mc_defaults = dict(risk_defaults["mc"])

    snapshot = _scheduled_job(
        dict(jobs_cfg.get("snapshot") or {}),
        enabled_default=True,
        interval_default=900,
        timeframe_default=default_timeframe,
        days_default=default_days,
        min_coverage_default=data_defaults["min_coverage"],
        alpha_default=default_alpha,
        window_default=default_window,
        n_sims_default=mc_defaults["n_sims"],
        dist_default=mc_defaults["dist"],
        df_t_default=mc_defaults["df_t"],
        seed_default=mc_defaults["seed"],
    )
    backtest = _scheduled_job(
        dict(jobs_cfg.get("backtest") or {}),
        enabled_default=True,
        interval_default=3600,
        timeframe_default=default_timeframe,
        days_default=default_days,
        min_coverage_default=data_defaults["min_coverage"],
        alpha_default=default_alpha,
        window_default=default_window,
        n_sims_default=mc_defaults["n_sims"],
        dist_default=mc_defaults["dist"],
        df_t_default=mc_defaults["df_t"],
        seed_default=mc_defaults["seed"],
    )
    live_refresh_default_enabled = bool(
        mt5_config.live_enabled
        and any(str(portfolio.get("mode") or "").lower() in {"live_mt5", "hybrid"} for portfolio in portfolios)
        and any(
            [
                mt5_config.agent_base_url,
                mt5_config.path,
                mt5_config.login,
                mt5_config.server,
            ]
        )
    )
    live_refresh = _scheduled_job(
        dict(jobs_cfg.get("live_refresh") or {}),
        enabled_default=live_refresh_default_enabled,
        interval_default=max(5, min(15, int(max(float(mt5_config.live_history_poll_seconds), 1.0)))),
    )
    report = _scheduled_job(
        dict(jobs_cfg.get("report") or {}),
        enabled_default=True,
        interval_default=3600,
    )

    return WorkerSettings(
        loop_sleep_seconds=int(jobs_cfg.get("loop_sleep_seconds", 30)),
        snapshot=snapshot,
        backtest=backtest,
        live_refresh=live_refresh,
        report=report,
    )


def evaluate_worker_health(root: Path) -> dict[str, Any]:
    raw_cfg = load_settings(root)
    portfolios = get_portfolio_contexts(raw_cfg)
    mt5_config = get_mt5_config(raw_cfg)
    live_portfolios = [
        str(portfolio.get("slug") or "")
        for portfolio in portfolios
        if str(portfolio.get("mode") or "").lower() in {"live_mt5", "hybrid"}
    ]
    mt5_configured = bool(
        mt5_config.agent_base_url
        or mt5_config.path
        or mt5_config.login
        or mt5_config.server
    )
    if live_portfolios and not mt5_configured:
        return {
            "status": "degraded",
            "code": "mt5_not_configured",
            "message": (
                "Worker is running for live/hybrid portfolios but MT5 is not configured in this process."
            ),
            "hint": (
                "Set VAR_PROJECT_MT5_AGENT_BASE_URL (and VAR_PROJECT_MT5_AGENT_API_KEY if needed) "
                "for worker and celery-worker."
            ),
            "live_portfolios": live_portfolios,
        }
    return {
        "status": "ok",
        "code": "ok",
        "message": "Worker dependencies look healthy.",
        "live_portfolios": live_portfolios,
    }


class JobRunner:
    def __init__(
        self,
        root: Path,
        settings: WorkerSettings | None = None,
        *,
        bootstrap_storage: bool = False,
        mt5_connector_factory: Any | None = None,
    ):
        self.root = root.resolve()
        self.settings = settings or load_worker_settings(self.root)
        self.bootstrap_storage = bool(bootstrap_storage)
        self.mt5_connector_factory = mt5_connector_factory
        self.last_run: dict[str, float] = {}
        self.log = logging.getLogger("var_project.jobs")
        self.worker_health = evaluate_worker_health(self.root)
        if str(self.worker_health.get("status") or "").lower() == "degraded":
            self.log.warning(
                "%s [%s]",
                self.worker_health.get("message"),
                self.worker_health.get("code"),
            )

    def run_pending(self, *, force_all: bool = False) -> dict[str, Any]:
        results: dict[str, Any] = {}
        operator_results = self._drain_operator_runs()
        if operator_results:
            results["operator_runs"] = operator_results
        for job_name in ("snapshot", "backtest", "live_refresh", "report"):
            job_cfg = getattr(self.settings, job_name)
            if not job_cfg.enabled:
                continue
            if not force_all and not self._is_due(job_name, job_cfg.interval_seconds):
                continue
            try:
                results[job_name] = self._run_job(job_name, job_cfg)
                self.last_run[job_name] = time.time()
            except Exception as exc:
                self.log.exception("job %s failed: %s", job_name, exc)
                results[job_name] = {"status": "error", "error": str(exc)}
        return results

    def _drain_operator_runs(self) -> list[dict[str, Any]]:
        from var_project.api.service import DeskApiService

        service = DeskApiService(
            self.root,
            mt5_connector_factory=self.mt5_connector_factory,
            bootstrap_storage=self.bootstrap_storage,
        )
        try:
            stale_updates = service.reap_stale_operator_runs(limit=50)
            runs = service.operator_runs(limit=10, statuses=["queued"])
        except (OperationalError, ProgrammingError) as exc:
            self.log.warning("operator run storage not ready: %s", exc)
            return []
        processed: list[dict[str, Any]] = []
        for stale in stale_updates:
            processed.append(
                {
                    "id": stale.get("id"),
                    "action": stale.get("action"),
                    "status": stale.get("status"),
                    "stage": stale.get("stage"),
                    "reason": stale.get("status_reason") or stale.get("error_code") or "stale",
                }
            )
        for run in runs:
            run_id = int(run["id"])
            if str(run.get("queue_task_id") or "").strip():
                processed.append(
                    {
                        "id": run_id,
                        "action": run.get("action"),
                        "status": "queued",
                        "stage": run.get("stage"),
                        "dispatch_mode": "already_dispatched",
                        "task_id": run.get("queue_task_id"),
                    }
                )
                continue
            try:
                dispatch = dispatch_operator_run(
                    run_id=run_id,
                    action=str(run.get("action") or ""),
                    repo_root=self.root,
                )
                if dispatch is None:
                    updated = service.process_operator_run(run_id)
                    processed.append(
                        {
                            "id": updated.get("id"),
                            "action": updated.get("action"),
                            "status": updated.get("status"),
                            "stage": updated.get("stage"),
                            "dispatch_mode": "fallback",
                        }
                    )
                    continue
                if dispatch.get("task_id"):
                    service.storage.update_operator_run(
                        run_id,
                        queue_task_id=str(dispatch["task_id"]),
                    )
                processed.append(
                    {
                        "id": run_id,
                        "action": run.get("action"),
                        "status": "queued",
                        "stage": run.get("stage"),
                        "dispatch_mode": dispatch.get("mode"),
                        "task_id": dispatch.get("task_id"),
                    }
                )
            except Exception as exc:
                self.log.exception("operator run %s failed: %s", run.get("id"), exc)
                processed.append(
                    {
                        "id": run.get("id"),
                        "action": run.get("action"),
                        "status": "error",
                        "error": str(exc),
                    }
                )
        return processed

    def run_forever(self, *, once: bool = False) -> None:
        if once:
            self.run_pending(force_all=True)
            return

        while True:
            self.run_pending(force_all=False)
            time.sleep(max(1, int(self.settings.loop_sleep_seconds)))

    def _is_due(self, job_name: str, interval_seconds: int) -> bool:
        last = self.last_run.get(job_name)
        if last is None:
            return True
        return (time.time() - last) >= max(1, int(interval_seconds))

    @staticmethod
    def _extract_mt5_live_unavailable_detail(exc: Exception) -> str | None:
        seen: set[int] = set()
        pending: list[BaseException] = [exc]
        while pending:
            current = pending.pop()
            identifier = id(current)
            if identifier in seen:
                continue
            seen.add(identifier)
            detail = str(current or "").strip()
            if isinstance(current, MT5ConnectionError):
                return detail or "MT5 connection is unavailable."
            if detail.lower().startswith("mt5_live_unavailable"):
                return detail
            cause = getattr(current, "__cause__", None)
            if cause is not None:
                pending.append(cause)
            context = getattr(current, "__context__", None)
            if context is not None:
                pending.append(context)
        return None

    def _build_mt5_unavailable_job_result(
        self,
        *,
        job_name: str,
        portfolio_slug: str | None,
        exc: Exception,
    ) -> dict[str, Any] | None:
        detail = self._extract_mt5_live_unavailable_detail(exc)
        if detail is None:
            return None
        return {
            "status": "skipped",
            "reason": "mt5_live_unavailable",
            "job": str(job_name),
            "portfolio_slug": None if portfolio_slug is None else str(portfolio_slug),
            "detail": detail,
        }

    def _run_job(self, job_name: str, job_cfg: ScheduledJob) -> dict[str, Any]:
        from var_project.api.service import DeskApiService

        service = DeskApiService(
            self.root,
            mt5_connector_factory=self.mt5_connector_factory,
            bootstrap_storage=self.bootstrap_storage,
        )
        if job_name == "snapshot":
            portfolio = service.runtime._resolve_portfolio_context(None)
            portfolio_slug = str(portfolio.get("slug") or "")
            try:
                if (
                    str(portfolio.get("mode") or "").lower() in {"live_mt5", "hybrid"}
                    and service.runtime.market_data.should_use_mt5_market_data(portfolio)
                ):
                    service.runtime.market_data.sync_market_data_if_stale(
                        portfolio_slug=portfolio["slug"],
                        max_age_seconds=300.0,
                        days=job_cfg.days,
                        timeframes=[job_cfg.timeframe] if job_cfg.timeframe else None,
                    )
                result = service.run_snapshot(
                    timeframe=job_cfg.timeframe,
                    days=job_cfg.days,
                    min_coverage=job_cfg.min_coverage,
                    alpha=job_cfg.alpha,
                    window=job_cfg.window,
                    n_sims=job_cfg.n_sims,
                    dist=job_cfg.dist,
                    df_t=job_cfg.df_t,
                    seed=job_cfg.seed,
                )
            except Exception as exc:
                degraded = self._build_mt5_unavailable_job_result(
                    job_name=job_name,
                    portfolio_slug=portfolio_slug or None,
                    exc=exc,
                )
                if degraded is None:
                    raise
                self.log.warning(
                    "snapshot job skipped for %s: %s",
                    portfolio_slug or "unknown",
                    degraded["detail"],
                )
                return degraded
            self.log.info("snapshot job ok: %s", result["artifact_path"])
            return result

        if job_name == "backtest":
            portfolio = service.runtime._resolve_portfolio_context(None)
            portfolio_slug = str(portfolio.get("slug") or "")
            try:
                if (
                    str(portfolio.get("mode") or "").lower() in {"live_mt5", "hybrid"}
                    and service.runtime.market_data.should_use_mt5_market_data(portfolio)
                ):
                    service.runtime.market_data.sync_market_data_if_stale(
                        portfolio_slug=portfolio["slug"],
                        max_age_seconds=300.0,
                        days=job_cfg.days,
                        timeframes=[job_cfg.timeframe] if job_cfg.timeframe else None,
                    )
                result = service.run_backtest(
                    timeframe=job_cfg.timeframe,
                    days=job_cfg.days,
                    min_coverage=job_cfg.min_coverage,
                    alpha=job_cfg.alpha,
                    window=job_cfg.window,
                    n_sims=job_cfg.n_sims,
                    dist=job_cfg.dist,
                    df_t=job_cfg.df_t,
                    seed=job_cfg.seed,
                )
            except Exception as exc:
                degraded = self._build_mt5_unavailable_job_result(
                    job_name=job_name,
                    portfolio_slug=portfolio_slug or None,
                    exc=exc,
                )
                if degraded is None:
                    raise
                self.log.warning(
                    "backtest job skipped for %s: %s",
                    portfolio_slug or "unknown",
                    degraded["detail"],
                )
                return degraded
            self.log.info("backtest job ok: %s", result["compare_csv"])
            return result

        if job_name == "live_refresh":
            if not service.mt5_config.live_enabled:
                result = {"status": "skipped", "reason": "MT5 live bridge is disabled.", "refreshed_portfolios": []}
                self.log.info("live refresh job skipped: %s", result["reason"])
                return result

            storage_ready = bool(service.runtime.storage_ready)
            refreshed_portfolios: list[dict[str, Any]] = []
            skipped_portfolios: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            auto_report_count = 0

            for portfolio in service.portfolios:
                slug = str(portfolio["slug"])
                mode = str(portfolio.get("mode") or "").lower()
                if mode not in {"live_mt5", "hybrid"}:
                    skipped_portfolios.append({"portfolio_slug": slug, "reason": f"Portfolio mode '{mode or 'unknown'}' is not live."})
                    continue

                market_sync_error = None
                try:
                    service.runtime.market_data.sync_market_data_if_stale(
                        portfolio_slug=slug,
                        max_age_seconds=max(900.0, float(service.runtime.mt5_config.live_history_poll_seconds) * 10.0),
                        days=service.runtime.market_data.history_backfill_days(),
                        timeframes=service.runtime.market_data.startup_sync_timeframes(),
                    )
                except Exception as exc:
                    market_sync_error = str(exc)
                    self.log.warning("market data warm-up failed for %s: %s", slug, exc)

                before_snapshot = (
                    service.storage.latest_snapshot(source="mt5_live_bridge", portfolio_slug=slug)
                    if storage_ready
                    else None
                )
                before_report = (
                    service.storage.latest_artifact("daily_report", portfolio_slug=slug)
                    if storage_ready
                    else None
                )
                try:
                    live_state = service.mt5_live_state(portfolio_slug=slug)
                except Exception as exc:
                    errors.append({"portfolio_slug": slug, "error": str(exc)})
                    continue

                after_snapshot = (
                    service.storage.latest_snapshot(source="mt5_live_bridge", portfolio_slug=slug)
                    if storage_ready
                    else None
                )
                after_report = (
                    service.storage.latest_artifact("daily_report", portfolio_slug=slug)
                    if storage_ready
                    else None
                )

                fallback_report_result: dict[str, Any] | None = None
                if storage_ready and after_report is None:
                    try:
                        latest_backtest = service.latest_backtest(portfolio_slug=slug)
                        compare_path = None
                        if latest_backtest is not None:
                            artifact_id = latest_backtest.get("artifact_id")
                            if artifact_id:
                                artifact = service.storage.artifact_by_id(int(artifact_id))
                                compare_path = None if artifact is None else artifact.get("path")
                        if compare_path is None:
                            backtest_result = service.run_backtest(
                                portfolio_slug=slug,
                                timeframe=service.runtime._default_timeframe(),
                                days=service.runtime._default_days(),
                            )
                            compare_path = backtest_result.get("compare_csv")
                        fallback_report_result = service.run_report(compare_path=compare_path, portfolio_slug=slug)
                        after_report = service.storage.latest_artifact("daily_report", portfolio_slug=slug)
                    except Exception as exc:
                        self.log.warning("auto report fallback failed for %s: %s", slug, exc)

                report_changed = False
                report_auto_generated = False
                report_path = None
                if after_report is not None:
                    report_details = dict(after_report.get("details") or {})
                    report_changed = (
                        before_report is None
                        or before_report.get("updated_at") != after_report.get("updated_at")
                        or before_report.get("id") != after_report.get("id")
                    )
                    report_auto_generated = bool(report_details.get("auto_generated") or fallback_report_result is not None)
                    report_path = (
                        fallback_report_result.get("report_markdown")
                        if fallback_report_result is not None
                        else after_report.get("path")
                    )
                    if report_changed and report_auto_generated:
                        auto_report_count += 1

                refreshed_portfolios.append(
                    {
                        "portfolio_slug": slug,
                        "status": live_state.get("status"),
                        "connected": bool(live_state.get("connected")),
                        "stale": bool(live_state.get("stale")),
                        "sequence": int(live_state.get("sequence") or 0),
                        "snapshot_id": None if after_snapshot is None else after_snapshot.get("id"),
                        "snapshot_changed": (
                            before_snapshot is None
                            or after_snapshot is None
                            or before_snapshot.get("id") != after_snapshot.get("id")
                        ),
                        "market_sync_error": market_sync_error,
                        "report_markdown": report_path,
                        "report_changed": report_changed,
                        "report_auto_generated": report_auto_generated,
                    }
                )

            if refreshed_portfolios and not errors:
                status = "ok"
            elif refreshed_portfolios and errors:
                status = "partial"
            elif errors:
                status = "error"
            else:
                status = "skipped"

            result = {
                "status": status,
                "refreshed_portfolios": refreshed_portfolios,
                "skipped_portfolios": skipped_portfolios,
                "errors": errors,
                "auto_report_count": auto_report_count,
            }
            self.log.info(
                "live refresh job %s: %s refreshed, %s auto reports",
                status,
                len(refreshed_portfolios),
                auto_report_count,
            )
            return result

        result = service.run_report(compare_path=job_cfg.compare_path)
        self.log.info("report job ok: %s", result["report_markdown"])
        return result


def _coerce_age(timestamp: str | None, *, now: datetime) -> float | None:
    if not timestamp:
        return None
    try:
        created_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (now - created_at.astimezone(timezone.utc)).total_seconds())


def _build_operator_runs_metrics(*, recent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {
        "queued": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "other": 0,
    }
    stale_reason_counts: dict[str, int] = {
        "timeout": 0,
        "abandoned": 0,
    }
    stale_recent: list[dict[str, Any]] = []
    for run in recent_runs:
        status = str(run.get("status") or "").strip().lower()
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["other"] += 1

        status_reason = str(run.get("status_reason") or "").strip().lower()
        error_code = str(run.get("error_code") or "").strip().lower()
        stale_reason: str | None = None
        if status_reason in {"timeout", "abandoned"}:
            stale_reason = status_reason
        elif error_code == "timeout_stale_run":
            stale_reason = "timeout"
        elif error_code == "abandoned_stale_run":
            stale_reason = "abandoned"

        if stale_reason is None:
            continue

        stale_reason_counts[stale_reason] += 1
        stale_recent.append(
            {
                "id": run.get("id"),
                "action": run.get("action"),
                "status": run.get("status"),
                "status_reason": stale_reason,
                "error_code": run.get("error_code"),
                "updated_at": run.get("updated_at"),
                "finished_at": run.get("finished_at"),
            }
        )

    return {
        "window_size": len(recent_runs),
        "status_counts": status_counts,
        "stale_closed_total": int(sum(stale_reason_counts.values())),
        "stale_reason_counts": stale_reason_counts,
        "recent_stale": stale_recent[:10],
    }


def build_worker_status(
    root: Path,
    *,
    storage: AppStorage | None = None,
    strict_schema_revision: bool = False,
) -> dict[str, Any]:
    repo_root = root.resolve()
    raw_config = load_settings(repo_root)
    settings = load_worker_settings(repo_root)
    worker_health = evaluate_worker_health(repo_root)
    active_storage = storage or AppStorage.from_root(repo_root, raw_config)
    database_ready = active_storage.schema_ready(strict_revision=bool(strict_schema_revision))
    now = datetime.now(timezone.utc)

    latest_snapshot = active_storage.latest_snapshot() if database_ready else None
    latest_backtest = active_storage.latest_backtest_run() if database_ready else None
    latest_live_refresh = active_storage.latest_snapshot(source="mt5_live_bridge") if database_ready else None
    latest_report = active_storage.latest_artifact("daily_report") if database_ready else None
    recent_operator_runs = active_storage.list_operator_runs(limit=25) if database_ready else []
    operator_runs_metrics = _build_operator_runs_metrics(recent_runs=recent_operator_runs)

    jobs: dict[str, dict[str, Any]] = {}
    for job_name, latest in {
        "snapshot": latest_snapshot,
        "backtest": latest_backtest,
        "live_refresh": latest_live_refresh,
        "report": latest_report,
    }.items():
        job_cfg = getattr(settings, job_name)
        latest_payload = latest or {}
        if job_name == "report":
            last_run_at = latest_payload.get("updated_at") or latest_payload.get("created_at")
            artifact_path = latest_payload.get("path")
        else:
            last_run_at = latest_payload.get("created_at")
            artifact_path = None
            artifact_id = latest_payload.get("artifact_id")
            if artifact_id and database_ready:
                artifact = active_storage.artifact_by_id(int(artifact_id))
                artifact_path = None if artifact is None else artifact.get("path")

        age_seconds = _coerce_age(last_run_at, now=now)
        due = bool(job_cfg.enabled) and (age_seconds is None or age_seconds >= max(1, int(job_cfg.interval_seconds)))
        healthy = not job_cfg.enabled or (
            age_seconds is not None
            and age_seconds <= max(int(job_cfg.interval_seconds) * 2, int(job_cfg.interval_seconds) + settings.loop_sleep_seconds)
        )
        state = "disabled" if not job_cfg.enabled else "pending" if age_seconds is None else "due" if due else "ok"
        jobs[job_name] = {
            "enabled": bool(job_cfg.enabled),
            "interval_seconds": int(job_cfg.interval_seconds),
            "state": state,
            "due": due,
            "healthy": healthy,
            "last_run_at": last_run_at,
            "last_run_age_seconds": age_seconds,
            "artifact_path": artifact_path,
        }

    return {
        "generated_at": now.isoformat(),
        "loop_sleep_seconds": int(settings.loop_sleep_seconds),
        "database_ready": database_ready,
        "health": worker_health,
        "operator_runs": operator_runs_metrics,
        "jobs": jobs,
    }
