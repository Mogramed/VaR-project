from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from var_project.jobs import JobRunner


def _write_settings(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "base_currency": "EUR",
        "symbols": ["EURUSD", "USDJPY"],
        "portfolio": {
            "name": "FX_EUR_20k",
            "positions_eur": {"EURUSD": 10_000, "USDJPY": 10_000},
        },
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
            "report": {"enabled": True, "interval_seconds": 5},
        },
    }
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
