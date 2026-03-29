from __future__ import annotations

from var_project.risk.budgeting import build_risk_budget_snapshot


def test_build_risk_budget_snapshot_uses_limits_and_positions():
    attribution = {
        "alpha": 0.95,
        "sample_size": 120,
        "models": {
            "hist": {
                "model": "hist",
                "total_var": 180.0,
                "total_es": 230.0,
                "positions": {
                    "EURUSD": {
                        "symbol": "EURUSD",
                        "position_eur": 10000.0,
                        "component_var": 120.0,
                        "component_es": 150.0,
                    },
                    "USDJPY": {
                        "symbol": "USDJPY",
                        "position_eur": 8000.0,
                        "component_var": 60.0,
                        "component_es": 80.0,
                    },
                },
            }
        },
    }
    limits_cfg = {
        "model_limits_eur": {"hist": {"var": 200.0, "es": 260.0}},
        "risk_budget": {
            "utilisation_warn": 0.85,
            "utilisation_breach": 1.0,
            "target_buffer": 0.95,
            "position_tolerance": 0.05,
            "symbol_weights": {"EURUSD": 0.5, "USDJPY": 0.5},
        },
    }

    budget = build_risk_budget_snapshot(
        attribution,
        limits_cfg,
        positions_eur={"EURUSD": 10000.0, "USDJPY": 8000.0},
        preferred_model="hist",
    )

    assert budget.preferred_model == "hist"
    assert budget.models["hist"].total_var_budget == 200.0
    assert budget.models["hist"].status == "WARN"
    assert budget.models["hist"].positions["EURUSD"].status == "BREACH"
    assert budget.models["hist"].positions["EURUSD"].action == "REDUCE"
    assert budget.models["hist"].positions["USDJPY"].recommended_position_eur is not None
