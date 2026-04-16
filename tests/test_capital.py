from __future__ import annotations

from var_project.risk.capital import build_capital_usage_snapshot


def _risk_budget_payload() -> dict[str, object]:
    return {
        "alpha": 0.95,
        "sample_size": 120,
        "preferred_model": "hist",
        "models": {
            "hist": {
                "model": "hist",
                "total_var": 520.0,
                "total_es": 560.0,
                "total_var_budget": 300.0,
                "total_es_budget": 380.0,
                "positions": {
                    "EURUSD": {
                        "symbol": "EURUSD",
                        "current_exposure": 10_000.0,
                        "weight": 0.5,
                        "component_var": 320.0,
                        "component_es": 350.0,
                        "action": "REDUCE",
                        "status": "BREACH",
                    },
                    "USDJPY": {
                        "symbol": "USDJPY",
                        "current_exposure": 9_000.0,
                        "weight": 0.5,
                        "component_var": 200.0,
                        "component_es": 210.0,
                        "action": "REDUCE",
                        "status": "BREACH",
                    },
                },
            }
        },
    }


def test_capital_snapshot_auto_budget_covers_consumption_and_reserve():
    limits_cfg = {
        "capital_management": {"reserve_ratio": 0.10, "preferred_model": "hist"},
        "risk_budget": {"utilisation_warn": 0.85, "utilisation_breach": 1.0},
    }
    snapshot = build_capital_usage_snapshot(
        _risk_budget_payload(),
        limits_cfg,
        portfolio_slug="fx_eur_20k",
        base_currency="EUR",
    )

    assert snapshot.total_capital_consumed_eur == 560.0
    assert snapshot.total_capital_budget_eur >= snapshot.total_capital_consumed_eur
    assert snapshot.total_capital_remaining_eur >= -1e-6


def test_capital_snapshot_honours_explicit_total_budget_override():
    limits_cfg = {
        "capital_management": {"reserve_ratio": 0.10, "preferred_model": "hist"},
        "risk_budget": {"utilisation_warn": 0.85, "utilisation_breach": 1.0},
    }
    snapshot = build_capital_usage_snapshot(
        _risk_budget_payload(),
        limits_cfg,
        portfolio_slug="fx_eur_20k",
        base_currency="EUR",
        overrides={"total_budget_eur": 420.0, "reserve_ratio": 0.05},
    )

    assert snapshot.total_capital_budget_eur == 420.0
    assert snapshot.total_capital_consumed_eur == 560.0
