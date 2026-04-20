from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from test_mt5_execution_api import FakeMT5Connector

from var_project.api.service import DeskApiService
from var_project.core.exceptions import MT5ConnectionError
from var_project.jobs import JobRunner, build_worker_status


def _write_settings(
    root: Path,
    *,
    portfolio_mode: str | None = None,
    live_refresh_enabled: bool | None = None,
    report_enabled: bool = True,
) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    portfolio = {
        "name": "FX_EUR_20k",
        "configured_exposure": {"EURUSD": 10_000, "USDJPY": 10_000},
    }
    if portfolio_mode is not None:
        portfolio["mode"] = portfolio_mode
    settings = {
        "base_currency": "EUR",
        "symbols": ["EURUSD", "USDJPY"],
        "portfolio": portfolio,
        "data": {
            "timeframes": ["H1"],
            "history_days_list": [60],
            "storage_format": "csv",
            "min_coverage": 0.90,
        },
        "risk": {
            "alpha": 0.95,
            "window": 20,
            "ewma": {"lambda": 0.94},
            "fhs": {"lambda": 0.94},
            "mc": {"n_sims": 250, "dist": "normal", "df_t": 6, "seed": 7},
        },
        "storage": {
            "database_path": "data/app/test_jobs.db",
            "analytics_dir": "reports/backtests",
            "reports_dir": "reports/daily",
            "snapshots_dir": "data/snapshots",
        },
        "jobs": {
            "loop_sleep_seconds": 1,
            "snapshot": {"enabled": True, "interval_seconds": 5},
            "backtest": {"enabled": True, "interval_seconds": 5},
            "report": {"enabled": report_enabled, "interval_seconds": 5},
        },
    }
    if live_refresh_enabled is not None:
        settings["jobs"]["live_refresh"] = {"enabled": live_refresh_enabled, "interval_seconds": 5}
    (config_dir / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    (config_dir / "risk_limits.yaml").write_text("{}", encoding="utf-8")


def _write_processed_returns(root: Path, symbol: str, timeframe: str = "H1", days: int = 60) -> None:
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    bars_per_day = 24
    n_days = 45
    n_bars = n_days * bars_per_day
    times = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    x = np.arange(n_bars)

    if symbol == "EURUSD":
        log_returns = 0.0002 + 0.0007 * np.sin(x / 13.0) + 0.0002 * np.cos(x / 7.0)
    else:
        log_returns = -0.0001 + 0.0008 * np.cos(x / 11.0) - 0.00015 * np.sin(x / 5.0)

    frame = pd.DataFrame({"time": times, "log_return": log_returns})
    frame.to_csv(processed_dir / f"{symbol}_{timeframe}_{days}d_returns.csv", index=False)


def test_job_runner_executes_snapshot_backtest_and_report(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    runner = JobRunner(root, bootstrap_storage=True)
    results = runner.run_pending(force_all=True)

    assert set(results) == {"snapshot", "backtest", "report"}
    assert Path(results["snapshot"]["artifact_path"]).exists()
    assert Path(results["backtest"]["compare_csv"]).exists()
    assert Path(results["backtest"]["validation_json"]).exists()
    assert Path(results["report"]["report_markdown"]).exists()


def test_job_runner_live_refresh_auto_generates_report(tmp_path: Path):
    root = tmp_path
    _write_settings(
        root,
        portfolio_mode="live_mt5",
        live_refresh_enabled=True,
        report_enabled=False,
    )
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FakeMT5Connector.reset()

    runner = JobRunner(
        root,
        bootstrap_storage=True,
        mt5_connector_factory=FakeMT5Connector,
    )
    results = runner.run_pending(force_all=True)

    assert set(results) == {"snapshot", "backtest", "live_refresh"}
    assert results["live_refresh"]["status"] == "ok"
    assert results["live_refresh"]["auto_report_count"] == 1
    assert results["live_refresh"]["errors"] == []
    assert results["live_refresh"]["refreshed_portfolios"]
    refreshed = results["live_refresh"]["refreshed_portfolios"][0]
    assert refreshed["portfolio_slug"] == "fx_eur_20k"
    assert refreshed["report_auto_generated"] is True
    assert refreshed["report_changed"] is True
    assert refreshed["report_markdown"] is not None
    assert Path(refreshed["report_markdown"]).exists()

    status = build_worker_status(root)
    assert status["database_ready"] is True
    assert "live_refresh" in status["jobs"]
    assert status["jobs"]["live_refresh"]["enabled"] is True
    assert status["jobs"]["live_refresh"]["state"] in {"pending", "due", "ok"}


def test_build_worker_status_reports_stale_operator_run_metrics(tmp_path: Path):
    root = tmp_path
    _write_settings(root, report_enabled=False)
    service = DeskApiService(root, bootstrap_storage=True)
    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")

    service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="sync",
        request_id="worker-status-stale-timeout",
        status="failed",
        stage="failed",
        status_reason="timeout",
        error_code="timeout_stale_run",
    )
    service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="snapshot",
        request_id="worker-status-stale-abandoned",
        status="failed",
        stage="failed",
        status_reason="abandoned",
        error_code="abandoned_stale_run",
    )
    service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="backtest",
        request_id="worker-status-running",
        status="running",
        stage="running_backtest",
    )

    status = build_worker_status(root, storage=service.storage)
    operator_runs = dict(status.get("operator_runs") or {})
    stale_reason_counts = dict(operator_runs.get("stale_reason_counts") or {})
    status_counts = dict(operator_runs.get("status_counts") or {})

    assert operator_runs["window_size"] >= 3
    assert operator_runs["stale_closed_total"] >= 2
    assert stale_reason_counts.get("timeout") == 1
    assert stale_reason_counts.get("abandoned") == 1
    assert status_counts.get("running", 0) >= 1
    assert len(list(operator_runs.get("recent_stale") or [])) >= 2


def test_job_runner_skips_snapshot_and_backtest_when_mt5_is_unreachable(tmp_path: Path, monkeypatch):
    root = tmp_path
    _write_settings(
        root,
        portfolio_mode="hybrid",
        report_enabled=False,
    )

    class _UnavailableMarketData:
        @staticmethod
        def should_use_mt5_market_data(_portfolio):
            return True

        @staticmethod
        def sync_market_data_if_stale(**_kwargs):
            raise MT5ConnectionError("[Errno 101] Network is unreachable")

    class _UnavailableRuntime:
        def __init__(self):
            self.market_data = _UnavailableMarketData()

        @staticmethod
        def _resolve_portfolio_context(_slug):
            return {"slug": "fx_eur_20k", "mode": "hybrid"}

    class _UnavailableDeskApiService:
        def __init__(self, *_args, **_kwargs):
            self.runtime = _UnavailableRuntime()

        @staticmethod
        def reap_stale_operator_runs(limit=50):
            return []

        @staticmethod
        def operator_runs(limit=10, statuses=None):
            return []

    import var_project.api.service as service_module

    monkeypatch.setattr(service_module, "DeskApiService", _UnavailableDeskApiService)

    runner = JobRunner(root, bootstrap_storage=True)
    results = runner.run_pending(force_all=True)

    assert set(results) == {"snapshot", "backtest"}
    assert results["snapshot"]["status"] == "skipped"
    assert results["snapshot"]["reason"] == "mt5_live_unavailable"
    assert "Network is unreachable" in results["snapshot"]["detail"]
    assert results["backtest"]["status"] == "skipped"
    assert results["backtest"]["reason"] == "mt5_live_unavailable"
    assert "Network is unreachable" in results["backtest"]["detail"]
