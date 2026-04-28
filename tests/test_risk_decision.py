from __future__ import annotations

import numpy as np
import pandas as pd

from var_project.engine.risk_engine import GarchConfig, MonteCarloConfig, RiskEngine, RiskModelConfig
from var_project.risk.decision_alpha import (
    backtest_decision_alpha_trajectory,
    compute_decision_alpha,
    forecast_decision_alpha,
    portfolio_decision_alpha_forecast,
    replay_decision_alpha,
)
from var_project.risk.decisioning import TradeProposal, evaluate_trade_proposal


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


def _risk_config() -> RiskModelConfig:
    return RiskModelConfig(
        alpha=0.95,
        mc=MonteCarloConfig(n_sims=250, seed=7),
        garch=GarchConfig(enabled=False),
    )


def _snapshot_var_es(returns: pd.DataFrame, positions: dict[str, float]) -> tuple[float, float]:
    engine = RiskEngine(positions)
    snapshot = engine.snapshot_from_returns(returns, _risk_config())
    return float(snapshot.models["hist"].var), float(snapshot.models["hist"].es)


def test_trade_decision_accepts_when_limits_are_loose():
    returns = _sample_returns()
    proposal = TradeProposal(symbol="EURUSD", delta_position_eur=1_000.0)
    limits_cfg = {
        "model_limits_eur": {"hist": {"var": 10_000.0, "es": 12_000.0}},
        "risk_budget": {"utilisation_warn": 0.85, "utilisation_breach": 1.0, "preferred_model": "hist"},
        "risk_decision": {"reference_model": "hist", "warn_threshold": 0.85, "min_fill_ratio": 0.25, "allow_risk_reducing_override": True},
    }

    result = evaluate_trade_proposal(
        returns,
        positions_eur={"EURUSD": 10_000.0, "USDJPY": 8_000.0},
        proposal=proposal,
        config=_risk_config(),
        limits_cfg=limits_cfg,
        reference_model="hist",
    )

    assert result.decision == "ACCEPT"
    assert result.approved_delta_position_eur == proposal.delta_position_eur


def test_trade_decision_reduces_when_partial_fill_is_admissible():
    returns = _sample_returns()
    base_positions = {"EURUSD": 10_000.0, "USDJPY": 8_000.0}
    full_positions = {"EURUSD": 15_000.0, "USDJPY": 8_000.0}
    pre_var, pre_es = _snapshot_var_es(returns, base_positions)
    full_var, full_es = _snapshot_var_es(returns, full_positions)
    limits_cfg = {
        "model_limits_eur": {"hist": {"var": (pre_var + full_var) / 2.0, "es": (pre_es + full_es) / 2.0}},
        "risk_budget": {"utilisation_warn": 1.0, "utilisation_breach": 1.0, "preferred_model": "hist"},
        "risk_decision": {"reference_model": "hist", "warn_threshold": 1.0, "min_fill_ratio": 0.25, "allow_risk_reducing_override": True},
    }

    result = evaluate_trade_proposal(
        returns,
        positions_eur=base_positions,
        proposal=TradeProposal(symbol="EURUSD", delta_position_eur=5_000.0),
        config=_risk_config(),
        limits_cfg=limits_cfg,
        reference_model="hist",
    )

    assert result.decision == "REDUCE"
    assert 0.0 < result.approved_delta_position_eur < 5_000.0
    assert result.suggested_delta_position_eur == result.approved_delta_position_eur


def test_trade_decision_rejects_when_fill_ratio_too_small():
    returns = _sample_returns()
    base_positions = {"EURUSD": 10_000.0, "USDJPY": 8_000.0}
    tiny_positions = {"EURUSD": 11_000.0, "USDJPY": 8_000.0}
    pre_var, pre_es = _snapshot_var_es(returns, base_positions)
    small_var, small_es = _snapshot_var_es(returns, tiny_positions)
    limits_cfg = {
        "model_limits_eur": {"hist": {"var": (pre_var + small_var) / 2.0, "es": (pre_es + small_es) / 2.0}},
        "risk_budget": {"utilisation_warn": 1.0, "utilisation_breach": 1.0, "preferred_model": "hist"},
        "risk_decision": {"reference_model": "hist", "warn_threshold": 1.0, "min_fill_ratio": 0.95, "allow_risk_reducing_override": True},
    }

    result = evaluate_trade_proposal(
        returns,
        positions_eur=base_positions,
        proposal=TradeProposal(symbol="EURUSD", delta_position_eur=5_000.0),
        config=_risk_config(),
        limits_cfg=limits_cfg,
        reference_model="hist",
    )

    assert result.decision == "REJECT"
    assert result.approved_delta_position_eur == 0.0


def test_trade_decision_accepts_risk_reducing_trade_even_when_desk_is_tense():
    returns = _sample_returns()
    positions = {"EURUSD": 10_000.0, "USDJPY": 8_000.0}
    pre_var, pre_es = _snapshot_var_es(returns, positions)
    limits_cfg = {
        "model_limits_eur": {"hist": {"var": pre_var * 0.95, "es": pre_es * 0.95}},
        "risk_budget": {"utilisation_warn": 0.85, "utilisation_breach": 1.0, "preferred_model": "hist"},
        "risk_decision": {"reference_model": "hist", "warn_threshold": 0.85, "min_fill_ratio": 0.25, "allow_risk_reducing_override": True},
    }

    result = evaluate_trade_proposal(
        returns,
        positions_eur=positions,
        proposal=TradeProposal(symbol="EURUSD", delta_position_eur=-3_000.0),
        config=_risk_config(),
        limits_cfg=limits_cfg,
        reference_model="hist",
    )

    assert result.decision == "ACCEPT"
    assert result.post_trade.var <= result.pre_trade.var


def test_trade_decision_rejects_invalid_symbol_and_zero_delta():
    returns = _sample_returns()
    limits_cfg = {
        "model_limits_eur": {"hist": {"var": 300.0, "es": 360.0}},
        "risk_budget": {"utilisation_warn": 0.85, "utilisation_breach": 1.0, "preferred_model": "hist"},
        "risk_decision": {"reference_model": "hist", "warn_threshold": 0.85, "min_fill_ratio": 0.25, "allow_risk_reducing_override": True},
    }
    invalid_symbol = evaluate_trade_proposal(
        returns,
        positions_eur={"EURUSD": 10_000.0, "USDJPY": 8_000.0},
        proposal=TradeProposal(symbol="GBPUSD", delta_position_eur=1_000.0),
        config=_risk_config(),
        limits_cfg=limits_cfg,
        reference_model="hist",
    )
    zero_delta = evaluate_trade_proposal(
        returns,
        positions_eur={"EURUSD": 10_000.0, "USDJPY": 8_000.0},
        proposal=TradeProposal(symbol="EURUSD", delta_position_eur=0.0),
        config=_risk_config(),
        limits_cfg=limits_cfg,
        reference_model="hist",
    )

    assert invalid_symbol.decision == "REJECT"
    assert zero_delta.decision == "REJECT"


def _decision_payload(decision: str = "ACCEPT") -> dict[str, object]:
    return {
        "symbol": "EURUSD",
        "decision": decision,
        "approved_exposure_change": 1_000.0,
        "requested_exposure_change": 1_000.0,
        "pre_trade": {
            "var": 120.0,
            "headroom_var": 180.0,
            "headline_risk": [{"status": "warning"}],
            "data_quality": {"status": "good", "available_observations": 200, "minimum_valid_days": 120},
        },
        "post_trade": {
            "var": 110.0,
            "headroom_var": 190.0,
            "headline_risk": [{"status": "ok"}],
            "data_quality": {"status": "good", "available_observations": 200, "minimum_valid_days": 120},
        },
    }


def test_decision_alpha_scores_are_bounded_and_explainable():
    returns = _sample_returns()
    intelligence = compute_decision_alpha(
        symbol="EURUSD",
        risk_decision=_decision_payload("ACCEPT"),
        bundle={"sample": returns},
        validation_summary={
            "best_model": "hist",
            "model_results": {
                "hist": {
                    "p_uc": 0.64,
                    "p_ind": 0.58,
                    "p_cc": 0.61,
                    "actual_rate": 0.045,
                    "expected_rate": 0.05,
                    "score": 72.0,
                }
            },
        },
    )

    assert intelligence["signal"] in {"BUY", "SELL", "HOLD"}
    assert -100.0 <= float(intelligence["score"]) <= 100.0
    assert 0.0 <= float(intelligence["confidence"]) <= 1.0
    assert 0.0 <= float(intelligence["size_multiplier"]) <= 1.0
    assert isinstance(intelligence["top_drivers"], list)
    assert intelligence["top_drivers"]
    assert set(intelligence["features"]).issuperset({"momentum_short_term", "volatility_recent"})


def test_decision_alpha_reject_guardrail_forces_hold():
    intelligence = compute_decision_alpha(
        symbol="EURUSD",
        risk_decision=_decision_payload("REJECT"),
    )

    assert intelligence["guardrail_applied"] is True
    assert intelligence["signal"] == "HOLD"
    assert float(intelligence["size_multiplier"]) == 0.0


def test_decision_alpha_fallback_caps_confidence_without_trained_model():
    intelligence = compute_decision_alpha(
        symbol="EURUSD",
        risk_decision=_decision_payload("ACCEPT"),
        bundle=None,
        validation_summary=None,
        model_state=None,
    )

    assert intelligence["model_version"] == "decision_alpha_v1"
    assert float(intelligence["confidence"]) <= 0.72
    assert 0.0 <= float(intelligence["size_multiplier"]) <= 1.0
    assert isinstance(intelligence["top_drivers"], list)
    feature_availability = dict(intelligence.get("feature_availability") or {})
    assert feature_availability.get("momentum_short_term") is False
    assert feature_availability.get("volatility_recent") is False
    assert feature_availability.get("spread_cost_norm") is False
    assert feature_availability.get("slippage_points") is False


def test_decision_alpha_replay_metrics_from_realized_pnl():
    replay = replay_decision_alpha(
        limit=25,
        decision_rows=[
            {
                "id": 1,
                "symbol": "EURUSD",
                "created_at": "2026-03-29T10:00:00+00:00",
                "decision_intelligence": {"score": 42.0, "signal": "BUY"},
            },
            {
                "id": 2,
                "symbol": "USDJPY",
                "created_at": "2026-03-30T10:00:00+00:00",
                "decision_intelligence": {"score": -38.0, "signal": "SELL"},
            },
        ],
        execution_rows=[
            {"id": 101, "decision_id": 1, "status": "EXECUTED"},
            {"id": 102, "decision_id": 2, "status": "EXECUTED"},
        ],
        fill_rows=[
            {"execution_result_id": 101, "profit": 14.0, "commission": -1.0, "swap": 0.0, "fee": 0.0},
            {"execution_result_id": 102, "profit": -9.0, "commission": -1.0, "swap": 0.0, "fee": 0.0},
        ],
    )

    assert replay["sample_size"] == 2
    assert replay["comparables"] == 2
    assert float(replay["cum_pnl"]) == 3.0
    assert abs(float(replay["hit_rate"]) - 1.0) <= 1e-9


def test_decision_alpha_forecast_emits_three_scenarios():
    prices = [1.08, 1.081, 1.0825, 1.0815, 1.0835, 1.084]
    market_bars = [
        {"time_utc": f"2026-03-{index + 1:02d}T12:00:00+00:00", "close": price}
        for index, price in enumerate(prices)
    ]

    forecast = forecast_decision_alpha(
        symbol="EURUSD",
        horizon_days=4,
        market_bars=market_bars,
    )

    assert forecast["horizon_days"] == 4
    assert len(forecast["scenarios"]) == 3
    scenario_names = {item["name"] for item in forecast["scenarios"]}
    assert scenario_names == {"bear", "base", "bull"}
    assert all(len(item["path"]) == 5 for item in forecast["scenarios"])
    probability_sum = sum(float(item["probability"]) for item in forecast["scenarios"])
    assert abs(probability_sum - 1.0) < 1e-6


def test_decision_alpha_forecast_uses_recent_h1_window_for_long_horizon():
    now = pd.Timestamp("2026-04-28T10:00:00+00:00")
    periods = 24 * 1000
    timeline = pd.date_range(now - pd.Timedelta(days=1000), periods=periods, freq="h", tz="UTC")
    prices = np.linspace(1.09, 1.18, periods)
    bars = [
        {"time_utc": timeline[index].isoformat(), "close": float(price)}
        for index, price in enumerate(prices)
    ]

    class FakeStorage:
        def __init__(self, rows):
            self.rows = list(rows)

        def market_bars(self, *, symbol, timeframe, since=None, limit=None):
            filtered = [row for row in self.rows if pd.to_datetime(row["time_utc"], utc=True) >= pd.to_datetime(since, utc=True)]
            if limit is not None:
                filtered = filtered[: int(limit)]
            return filtered

    forecast = forecast_decision_alpha(
        symbol="EURUSD",
        horizon_days=150,
        storage=FakeStorage(bars),
    )

    latest_price = float(prices[-1])
    current_price = float(forecast["current_price"])
    assert abs(current_price - latest_price) / latest_price < 0.01
    base = next(item for item in list(forecast["scenarios"]) if item["name"] == "base")
    assert abs(float(base["path"][0]["price"]) - current_price) < 1e-9


def test_decision_alpha_backtest_trajectory_outputs_predicted_vs_actual():
    prices = [1.08 + 0.0004 * np.sin(index / 3.0) + index * 0.00005 for index in range(140)]
    timeline = pd.date_range("2026-01-01", periods=len(prices), freq="h", tz="UTC")
    market_bars = [
        {"time_utc": timeline[index].isoformat(), "close": price}
        for index, price in enumerate(prices)
    ]

    trajectory = backtest_decision_alpha_trajectory(
        symbol="EURUSD",
        lookback_days=90,
        market_bars=market_bars,
    )

    assert trajectory["symbol"] == "EURUSD"
    assert trajectory["lookback_days"] == 90
    assert trajectory["sample_size"] >= 1
    assert 0.0 <= float(trajectory["hit_rate"]) <= 1.0
    assert isinstance(trajectory["predicted_vs_actual"], list)
    point = trajectory["predicted_vs_actual"][0]
    assert "predicted_price" in point
    assert "actual_price" in point
    assert "predicted_score" in point


def test_decision_alpha_backtest_trajectory_stays_in_realistic_fx_band():
    prices = [1.18 + 0.003 * np.sin(index / 24.0) + 0.0005 * np.sin(index / 7.0) for index in range(24 * 120)]
    timeline = pd.date_range("2026-01-01", periods=len(prices), freq="h", tz="UTC")
    market_bars = [
        {"time_utc": timeline[index].isoformat(), "close": price}
        for index, price in enumerate(prices)
    ]

    trajectory = backtest_decision_alpha_trajectory(
        symbol="EURUSD",
        lookback_days=90,
        market_bars=market_bars,
    )
    points = list(trajectory.get("predicted_vs_actual") or [])
    assert points

    last_predicted = float(points[-1]["predicted_price"])
    actual_values = [float(item["actual_price"]) for item in points]
    lower_bound = min(actual_values) * 0.85
    upper_bound = max(actual_values) * 1.15
    assert lower_bound <= last_predicted <= upper_bound

    max_relative_error = max(
        abs(float(item["predicted_price"]) - float(item["actual_price"])) / max(float(item["actual_price"]), 1e-9)
        for item in points
    )
    assert max_relative_error < 0.25


def test_portfolio_decision_alpha_forecast_builds_multi_symbol_pnl_scenarios():
    forecast = portfolio_decision_alpha_forecast(
        symbols=["EURUSD", "USDJPY"],
        exposures={"EURUSD": 50_000.0, "USDJPY": 40_000.0},
        horizon_days=150,
        storage=None,
    )

    assert forecast["horizon_days"] == 150
    assert forecast["symbol_count"] == 2
    assert len(forecast["symbols"]) == 2
    assert len(forecast["pnl_scenarios"]) == 3
    scenario_names = {item["name"] for item in forecast["pnl_scenarios"]}
    assert scenario_names == {"bear", "base", "bull"}
    assert all(len(item["path"]) == 151 for item in forecast["pnl_scenarios"])
