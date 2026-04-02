from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def write_settings(
    root: Path,
    *,
    portfolios: list[dict[str, object]] | None = None,
    desk_name: str = "FX Risk Desk",
    database_path: str = "data/app/test_platform.db",
) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    default_portfolio = {
        "name": "FX_EUR_20k",
        "configured_exposure": {"EURUSD": 10_000, "USDJPY": 10_000},
    }
    settings = {
        "base_currency": "EUR",
        "symbols": ["EURUSD", "USDJPY"],
        "portfolio": default_portfolio,
        "data": {
            "timeframes": ["H1"],
            "history_days_list": [60],
            "market_history_days": 60,
            "market_retention_days": {"M1": 30, "H1": 60, "D1": 60},
            "tick_retention_days": 7,
            "tick_archive_dir": "data/market_ticks",
            "tick_archive_format": "parquet",
            "storage_format": "csv",
            "timezone": "Europe/Paris",
            "min_coverage": 0.90,
        },
        "risk": {
            "alpha": 0.95,
            "alphas": [0.95, 0.99],
            "horizons": [1, 5, 10],
            "estimation_window_days": 40,
            "minimum_valid_days": 20,
            "validation_window_days": 40,
            "window": 20,
            "ewma": {"lambda": 0.94},
            "fhs": {"lambda": 0.94},
            "mc": {"n_sims": 250, "dist": "normal", "df_t": 6, "seed": 7},
            "garch": {"enabled": True, "p": 1, "q": 1, "dist": "t", "mean": "constant"},
        },
        "storage": {
            "database_path": database_path,
            "analytics_dir": "reports/backtests",
            "reports_dir": "reports/daily",
            "snapshots_dir": "data/snapshots",
        },
        "desk": {"name": desk_name},
    }
    if portfolios:
        settings["portfolios"] = portfolios
    (config_dir / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    risk_limits = {
        "model_limits_eur": {
            "hist": {"var": 300.0, "es": 360.0},
            "param": {"var": 320.0, "es": 380.0},
            "mc": {"var": 320.0, "es": 380.0},
            "ewma": {"var": 320.0, "es": 380.0},
            "garch": {"var": 320.0, "es": 380.0},
            "fhs": {"var": 320.0, "es": 380.0},
        },
        "risk_budget": {
            "utilisation_warn": 0.85,
            "utilisation_breach": 1.00,
            "target_buffer": 0.95,
            "position_tolerance": 0.05,
            "preferred_model": "best_validation",
            "symbol_weights": {"EURUSD": 0.5, "USDJPY": 0.5},
        },
        "capital_management": {
            "reserve_ratio": 0.10,
            "rebalance_min_gap": 10.0,
            "preferred_model": "best_validation",
        },
        "risk_decision": {
            "decision_mode": "advisory",
            "reference_model": "best_validation",
            "warn_threshold": 0.85,
            "breach_threshold": 1.00,
            "min_fill_ratio": 0.25,
            "allow_risk_reducing_override": True,
        },
    }
    (config_dir / "risk_limits.yaml").write_text(yaml.safe_dump(risk_limits, sort_keys=False), encoding="utf-8")


def write_processed_returns(root: Path, symbol: str, timeframe: str = "H1", days: int = 60) -> None:
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
