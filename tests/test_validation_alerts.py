from __future__ import annotations

import pandas as pd

from var_project.alerts.engine import (
    alerts_from_capital_snapshot,
    alerts_from_live_snapshot,
    alerts_from_risk_budget,
    alerts_from_risk_decision,
    alerts_from_validation_summary,
)
from var_project.validation.model_validation import validate_compare_frame


def _compare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "pnl": [-12, 4, -15, 3, -11, 5, -13, 2, -10, 1],
            "var_hist": [10] * 10,
            "es_hist": [12] * 10,
            "exc_hist": [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            "var_param": [14] * 10,
            "es_param": [16] * 10,
            "exc_param": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )


def test_validation_summary_picks_best_model():
    summary = validate_compare_frame(_compare_frame(), alpha=0.95)

    assert set(summary.model_results) == {"hist", "param"}
    assert summary.best_model == "param"


def test_validation_alerts_flag_rejected_models():
    summary = validate_compare_frame(_compare_frame(), alpha=0.95)
    alerts = alerts_from_validation_summary(summary)

    assert any(alert.code == "KUPIEC_REJECTED" for alert in alerts)


def test_live_snapshot_alerts_flag_limit_breaches():
    snapshot = {
        "var": {"hist": 250.0, "ewma": 180.0},
        "es": {"hist": 270.0, "ewma": 190.0},
        "live_loss_proxy": 260.0,
        "limits": {"zone_hist": "RED", "zone_ewma": "AMBER"},
    }
    limits_cfg = {
        "model_limits_eur": {
            "hist": {"var": 200.0, "es": 260.0},
            "ewma": {"var": 220.0, "es": 280.0},
        }
    }

    alerts = alerts_from_live_snapshot(snapshot, limits_cfg)
    codes = {alert.code for alert in alerts}

    assert "MODEL_VAR_LIMIT" in codes
    assert "MODEL_ES_LIMIT" in codes
    assert "LIVE_ZONE_HIST_RED" in codes


def test_risk_budget_alerts_flag_model_and_position_pressure():
    budget = {
        "preferred_model": "hist",
        "models": {
            "hist": {
                "model": "hist",
                "status": "BREACH",
                "utilization_var": 1.04,
                "utilization_es": 0.98,
                "headroom_var": -8.0,
                "headroom_es": 4.0,
                "positions": {
                    "EURUSD": {
                        "symbol": "EURUSD",
                        "position_eur": 10000.0,
                        "utilization_var": 1.12,
                        "utilization_es": 1.01,
                        "recommended_position_eur": 9000.0,
                        "action": "REDUCE",
                        "status": "BREACH",
                    }
                },
            }
        },
    }

    alerts = alerts_from_risk_budget(budget)
    codes = {alert.code for alert in alerts}

    assert "MODEL_RISK_BUDGET_BREACH" in codes
    assert "POSITION_RISK_BUDGET_BREACH" in codes


def test_risk_decision_alerts_follow_decision_severity():
    reduce_alerts = alerts_from_risk_decision(
        {
            "symbol": "EURUSD",
            "decision": "REDUCE",
            "requested_delta_position_eur": 5000.0,
            "approved_delta_position_eur": 2500.0,
            "resulting_position_eur": 12500.0,
            "model_used": "hist",
            "reasons": ["Too much risk"],
            "pre_trade": {"var": 1.0, "es": 2.0, "headroom_var": 1.0, "headroom_es": 1.0, "gross_notional": 1.0, "position_eur": 1.0, "status": "WARN"},
            "post_trade": {"var": 1.0, "es": 2.0, "headroom_var": 1.0, "headroom_es": 1.0, "gross_notional": 1.0, "position_eur": 1.0, "status": "WARN"},
        }
    )
    reject_alerts = alerts_from_risk_decision(
        {
            "symbol": "EURUSD",
            "decision": "REJECT",
            "requested_delta_position_eur": 5000.0,
            "approved_delta_position_eur": 0.0,
            "resulting_position_eur": 10000.0,
            "model_used": "hist",
            "reasons": ["Rejected"],
            "pre_trade": {"var": 1.0, "es": 2.0, "headroom_var": 1.0, "headroom_es": 1.0, "gross_notional": 1.0, "position_eur": 1.0, "status": "BREACH"},
            "post_trade": {"var": 1.0, "es": 2.0, "headroom_var": 1.0, "headroom_es": 1.0, "gross_notional": 1.0, "position_eur": 1.0, "status": "BREACH"},
        }
    )

    assert reduce_alerts[0].severity == "WARN"
    assert reduce_alerts[0].code == "TRADE_DECISION_REDUCE"
    assert reject_alerts[0].severity == "BREACH"
    assert reject_alerts[0].code == "TRADE_DECISION_REJECT"


def test_capital_alerts_follow_capital_status():
    alerts = alerts_from_capital_snapshot(
        {
            "portfolio_slug": "fx_alpha",
            "reference_model": "hist",
            "status": "BREACH",
            "total_capital_budget_eur": 300.0,
            "total_capital_consumed_eur": 330.0,
            "total_capital_remaining_eur": -30.0,
            "allocations": {
                "EURUSD": {
                    "symbol": "EURUSD",
                    "status": "BREACH",
                    "utilization": 1.1,
                    "remaining_capital_eur": -10.0,
                    "action": "REDUCE",
                }
            },
        }
    )

    assert alerts
    assert alerts[0].severity == "BREACH"
