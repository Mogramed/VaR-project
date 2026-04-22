from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

import var_project.engine.risk_engine as risk_engine_module
from var_project.engine.risk_engine import RiskEngine, RiskModelConfig, RiskSurfacePoint
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
    assert payload["data_quality"]["status"] in {"healthy", "thin_history", "stale", "incomplete", "no_exposure"}
    assert any(point["alpha"] == 0.99 and point["horizon_days"] == 10 for point in payload["points"])


def test_risk_surface_diagnostics_flag_non_zero_equalities():
    diagnostics = RiskEngine._diagnostics_from_points(
        [
            RiskSurfacePoint(
                model="hist",
                alpha=0.99,
                horizon_days=1,
                var=125.0,
                es=165.0,
                observation_count=220,
                status="healthy",
            ),
            RiskSurfacePoint(
                model="param",
                alpha=0.99,
                horizon_days=1,
                var=125.0,
                es=165.0,
                observation_count=220,
                status="healthy",
            ),
        ],
        config=RiskModelConfig(alpha=0.99),
        exposure_by_symbol={"EURUSD": 10_000.0},
        sample_size=220,
        data_quality_status="healthy",
        no_exposure=False,
    )

    coherence = dict(diagnostics.get("coherence_checks") or {})
    assert coherence.get("suspicious_equalities_count") == 1
    assert diagnostics.get("coherence_alert_active") is True
    assert dict(diagnostics.get("input_trace") or {}).get("models", {}).get("hist") is not None


def test_risk_surface_marks_no_exposure_when_book_is_under_epsilon():
    engine = RiskEngine(
        {"EURUSD": 0.4, "USDJPY": -0.3},
        no_exposure_epsilon_by_symbol={"EURUSD": 1.0, "USDJPY": 1.0},
        default_no_exposure_epsilon=1.0,
    )
    surface = engine.build_risk_surface(
        _sample_returns(220),
        RiskModelConfig(alpha=0.95),
        alphas=[0.95],
        horizons=[1, 5],
        estimation_window_days=120,
        minimum_valid_days=60,
        reference_model="hist",
    )

    payload = surface.to_dict()
    assert payload["data_quality"]["status"] == "no_exposure"
    assert payload["data_quality"]["gross_exposure_base_ccy"] == pytest.approx(0.7)
    assert payload["data_quality"]["gross_exposure_epsilon_base_ccy"] == pytest.approx(2.0)
    assert payload["data_quality"]["no_exposure_epsilon_by_symbol"] == {"EURUSD": 1.0, "USDJPY": 1.0}
    assert payload["points"]
    assert all(point["status"] == "no_exposure" for point in payload["points"])


def test_risk_surface_keeps_non_no_exposure_when_history_misses_a_symbol():
    engine = RiskEngine(
        {"EURUSD": 1_200.0, "USDJPY": 800.0},
        no_exposure_epsilon_by_symbol={"EURUSD": 1.0, "USDJPY": 1.0},
        default_no_exposure_epsilon=1.0,
    )
    returns = _sample_returns(220)[["date", "EURUSD"]]
    surface = engine.build_risk_surface(
        returns,
        RiskModelConfig(alpha=0.95),
        alphas=[0.95],
        horizons=[1, 5],
        estimation_window_days=120,
        minimum_valid_days=60,
        reference_model="hist",
    )

    payload = surface.to_dict()
    assert payload["data_quality"]["status"] != "no_exposure"
    assert payload["data_quality"]["gross_exposure_base_ccy"] == pytest.approx(2_000.0)
    assert payload["points"]
    assert all(point["status"] != "no_exposure" for point in payload["points"])


def test_backtest_adds_surface_columns_for_alpha_and_horizon():
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    backtest = engine.backtest(
        _sample_returns(220),
        window=80,
        config=RiskModelConfig(alpha=0.95),
        alphas=[0.95, 0.975, 0.99],
        horizons=[1, 5],
    )

    assert "pnl_h5" in backtest.columns
    assert "var_hist_a99_h5" in backtest.columns
    assert "var_hist_a97p5_h5" in backtest.columns
    assert "es_hist_a95_h5" in backtest.columns
    assert "exc_hist_a99_h5" in backtest.columns


def test_evaluate_models_partial_failure_keeps_other_models(monkeypatch):
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})

    def _boom(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("forced param failure")

    monkeypatch.setattr(risk_engine_module, "normal_parametric_var_es", _boom)

    snapshot = engine.snapshot_from_returns(_sample_returns(), RiskModelConfig(alpha=0.95))

    assert "hist" in snapshot.models
    assert "param" not in snapshot.models


def test_evaluate_models_raises_with_diagnostics_when_all_models_fail(monkeypatch):
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})

    def _boom(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("forced failure")

    monkeypatch.setattr(risk_engine_module, "historical_var_es", _boom)
    monkeypatch.setattr(risk_engine_module, "normal_parametric_var_es", _boom)
    monkeypatch.setattr(risk_engine_module, "mc_var_es", _boom)
    monkeypatch.setattr(risk_engine_module, "ewma_var_es", _boom)
    monkeypatch.setattr(risk_engine_module, "garch_var_es", _boom)
    monkeypatch.setattr(risk_engine_module, "fhs_var_es", _boom)

    with pytest.raises(RuntimeError, match="No risk model could be evaluated"):
        engine.snapshot_from_returns(_sample_returns(), RiskModelConfig(alpha=0.95))


def test_evaluate_models_skips_fhs_when_history_is_too_short(caplog):
    engine = RiskEngine({"EURUSD": 10_000.0, "USDJPY": 8_000.0})
    frame = engine.build_portfolio_frame(_sample_returns(rows=15))

    with caplog.at_level(logging.WARNING):
        snapshot = engine.evaluate_models(
            frame["pnl"],
            frame[engine.portfolio_symbols(frame)],
            RiskModelConfig(alpha=0.95),
            enabled_models=["hist", "fhs"],
        )

    assert "hist" in snapshot.models
    assert "fhs" not in snapshot.models
    assert not any("fenetre trop courte pour FHS" in record.message for record in caplog.records)
