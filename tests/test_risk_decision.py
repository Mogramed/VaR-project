from __future__ import annotations

import numpy as np
import pandas as pd

from var_project.engine.risk_engine import GarchConfig, MonteCarloConfig, RiskEngine, RiskModelConfig
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
