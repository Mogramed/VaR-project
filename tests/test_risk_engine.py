from __future__ import annotations

import numpy as np
import pandas as pd

from var_project.engine.risk_engine import RiskEngine, RiskModelConfig
from var_project.risk.stress import StressScenario


def _sample_returns(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "date": idx,
            "EURUSD": rng.normal(0.0001, 0.004, size=rows),
            "USDJPY": rng.normal(-0.00005, 0.005, size=rows),
        }
    )


def test_snapshot_from_returns_exposes_all_models():
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    snapshot = engine.snapshot_from_returns(_sample_returns(), RiskModelConfig(alpha=0.95))

    assert set(snapshot.models) == {"hist", "param", "mc", "ewma", "garch", "fhs"}
    for result in snapshot.models.values():
        assert result.var >= 0.0
        assert result.es >= result.var


def test_risk_attribution_exposes_component_and_incremental_metrics():
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    returns = _sample_returns()
    snapshot = engine.snapshot_from_returns(returns, RiskModelConfig(alpha=0.95))
    attribution = engine.attribute_from_returns(returns, RiskModelConfig(alpha=0.95), base_snapshot=snapshot)

    assert set(attribution.models) == {"hist", "param", "mc", "ewma", "garch", "fhs"}
    hist = attribution.models["hist"]
    assert hist.total_var == snapshot.models["hist"].var
    assert set(hist.positions) == {"EURUSD", "USDJPY"}
    assert set(hist.asset_classes) == {"fx"}
    eurusd = hist.positions["EURUSD"]
    assert eurusd.asset_class == "fx"
    assert eurusd.standalone_var >= 0.0
    assert eurusd.incremental_var >= 0.0
    fx_bucket = hist.asset_classes["fx"]
    assert fx_bucket.symbol_count == 2
    assert set(fx_bucket.symbols) == {"EURUSD", "USDJPY"}
    assert fx_bucket.component_es >= 0.0


def test_backtest_returns_standardized_columns():
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    backtest = engine.backtest(_sample_returns(), window=60, config=RiskModelConfig(alpha=0.95))

    assert len(backtest) == 120
    expected = {
        "date",
        "pnl",
        "alpha",
        "window",
        "var_hist",
        "var_param",
        "var_mc",
        "var_ewma",
        "var_garch",
        "var_fhs",
        "es_hist",
        "es_param",
        "es_mc",
        "es_ewma",
        "es_garch",
        "es_fhs",
        "exc_hist",
        "exc_param",
        "exc_mc",
        "exc_ewma",
        "exc_garch",
        "exc_fhs",
        "ret_EURUSD",
        "ret_USDJPY",
        "pnl_EURUSD",
        "pnl_USDJPY",
    }
    assert expected.issubset(backtest.columns)


def test_stress_report_is_keyed_by_scenario():
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    frame = engine.build_portfolio_frame(_sample_returns())
    stressed = engine.stress(
        frame["pnl"],
        alpha=0.95,
        scenarios=[StressScenario(name="shock", vol_multiplier=1.2, shock_pnl=-10.0)],
    )

    assert "shock" in stressed
    assert stressed["shock"].var >= 0.0


def test_risk_surface_exposes_multi_horizon_metrics_and_quality():
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    surface = engine.build_risk_surface(
        _sample_returns(320),
        RiskModelConfig(alpha=0.95),
        alphas=[0.95, 0.99],
        horizons=[1, 5, 10],
        estimation_window_days=250,
        minimum_valid_days=120,
        reference_model="hist",
    )

    payload = surface.to_dict()
    assert payload["headline"]
    assert payload["data_quality"]["status"] in {"healthy", "thin_history", "stale", "incomplete"}
    assert any(point["alpha"] == 0.99 and point["horizon_days"] == 10 for point in payload["points"])


def test_backtest_adds_surface_columns_for_alpha_and_horizon():
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    backtest = engine.backtest(
        _sample_returns(220),
        window=80,
        config=RiskModelConfig(alpha=0.95),
        alphas=[0.95, 0.99],
        horizons=[1, 5],
    )

    assert "pnl_h5" in backtest.columns
    assert "var_hist_a99_h5" in backtest.columns
    assert "es_hist_a95_h5" in backtest.columns
    assert "exc_hist_a99_h5" in backtest.columns
