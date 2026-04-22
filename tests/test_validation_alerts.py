from __future__ import annotations

import pandas as pd

from var_project.alerts.engine import (
    alerts_from_capital_snapshot,
    alerts_from_live_operator_state,
    alerts_from_live_snapshot,
    alerts_from_risk_budget,
    alerts_from_risk_decision,
    alerts_from_validation_summary,
)
from var_project.validation.model_validation import (
    BacktestModelValidation,
    ValidationSummary,
    build_champion_challenger_summary,
    validate_compare_frame,
    validate_compare_surface,
)


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


def _surface_frame_with_observations(*, n: int, alpha_token: str, horizon_days: int) -> pd.DataFrame:
    observations = max(int(n), 1)
    exc = [0] * observations
    exc[min(observations - 1, observations // 4)] = 1
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=observations, freq="D", tz="UTC"),
            "pnl": [-10.0 if i % 2 == 0 else 4.0 for i in range(observations)],
            f"var_hist_a{alpha_token}_h{int(horizon_days)}": [10.0] * observations,
            f"es_hist_a{alpha_token}_h{int(horizon_days)}": [12.0] * observations,
            f"exc_hist_a{alpha_token}_h{int(horizon_days)}": exc,
        }
    )


def test_validation_summary_picks_best_model():
    summary = validate_compare_frame(_compare_frame(), alpha=0.95)

    assert set(summary.model_results) == {"hist", "param"}
    assert summary.best_model == "param"


def test_validation_summary_exposes_es_tail_diagnostics():
    summary = validate_compare_frame(_compare_frame(), alpha=0.95)
    hist = summary.model_results["hist"]

    assert hist.es_tail_observations == 4
    assert hist.es_shortfall_ratio is not None
    assert hist.es_shortfall_ratio > 1.0
    assert hist.es_breach_rate is not None
    assert 0.0 <= hist.es_breach_rate <= 1.0


def test_validation_summary_ignores_surface_suffixes():
    frame = _compare_frame().assign(
        var_hist_a95_h1=[10] * 10,
        es_hist_a95_h1=[12] * 10,
        exc_hist_a95_h1=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        var_param_a99_h5=[16] * 10,
        es_param_a99_h5=[18] * 10,
        exc_param_a99_h5=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    )

    summary = validate_compare_frame(frame, alpha=0.95)

    assert set(summary.model_results) == {"hist", "param"}


def test_validation_surface_supports_fractional_alpha_token_format():
    frame = _compare_frame().assign(
        var_hist_a97p5_h10=[17] * 10,
        es_hist_a97p5_h10=[20] * 10,
        exc_hist_a97p5_h10=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    )

    surface = validate_compare_surface(frame, alphas=[0.975], horizons=[10])

    assert surface["points"]
    assert "a97p5_h10" in surface["current_metrics"]


def test_validation_surface_reads_legacy_fractional_alpha_columns():
    frame = _compare_frame().assign(
        var_hist_a98_h10=[17] * 10,
        es_hist_a98_h10=[20] * 10,
        exc_hist_a98_h10=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    )

    surface = validate_compare_surface(frame, alphas=[0.975], horizons=[10])

    assert surface["points"]
    assert "a97p5_h10" in surface["current_metrics"]


def test_validation_surface_exposes_statistical_governance_summary():
    frame = _compare_frame().assign(
        var_hist_a95_h1=[10] * 10,
        es_hist_a95_h1=[12] * 10,
        exc_hist_a95_h1=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        var_param_a95_h1=[14] * 10,
        es_param_a95_h1=[16] * 10,
        exc_param_a95_h1=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    )

    surface = validate_compare_surface(frame, alphas=[0.95], horizons=[1])

    assert surface["points"]
    governance = surface.get("governance_summary")
    assert isinstance(governance, dict)
    assert governance["pvalue_threshold"] == 0.05
    assert governance["total_points"] == len(surface["points"])
    assert governance["confidence_level"] in {"high", "medium", "low", "unknown"}
    assert "confidence_score" in governance
    assert "confidence_reason" in governance
    assert "min_observations_by_horizon_days" in governance
    status_counts = governance["status_counts"]
    assert status_counts["PASS"] + status_counts["WARN"] + status_counts["FAIL"] == len(surface["points"])
    assert governance["coverage_fail_count"] >= 0
    assert governance["independence_fail_count"] >= 0
    assert governance["conditional_fail_count"] >= 0

    first_point = surface["points"][0]
    assert "coverage_pass" in first_point
    assert "independence_pass" in first_point
    assert "conditional_pass" in first_point
    assert first_point["statistical_status"] in {"PASS", "WARN", "FAIL"}


def test_validation_surface_sample_guardrails_below_threshold():
    frame = _surface_frame_with_observations(n=119, alpha_token="90", horizon_days=10)

    surface = validate_compare_surface(frame, alphas=[0.90], horizons=[10])
    governance = surface["governance_summary"]

    assert governance["total_points"] == 1
    assert governance["insufficient_sample_count"] == 1
    assert governance["effective_points"] == 0
    assert governance["confidence_level"] == "low"
    assert float(governance["confidence_score"]) < 60.0
    assert "Insufficient observations" in str(governance["confidence_reason"])
    assert surface["points"][0]["statistical_status"] == "WARN"


def test_validation_surface_sample_guardrails_at_threshold():
    frame = _surface_frame_with_observations(n=120, alpha_token="90", horizon_days=10)

    surface = validate_compare_surface(frame, alphas=[0.90], horizons=[10])
    governance = surface["governance_summary"]
    horizon_payload = surface["horizon_governance"]["horizons"]["h10"]

    assert governance["insufficient_sample_count"] == 0
    assert governance["effective_points"] == governance["total_points"]
    assert governance["confidence_level"] == "high"
    assert float(governance["confidence_score"]) >= 80.0
    assert horizon_payload["horizon_observation_floor"] == 120
    assert horizon_payload["confidence_level"] == "high"


def test_validation_surface_sample_guardrails_above_threshold():
    frame = _surface_frame_with_observations(n=180, alpha_token="90", horizon_days=10)

    surface = validate_compare_surface(frame, alphas=[0.90], horizons=[10])
    governance = surface["governance_summary"]

    assert governance["insufficient_sample_count"] == 0
    assert governance["effective_points"] == governance["total_points"]
    assert governance["confidence_level"] == "high"
    assert float(governance["confidence_score"]) >= 99.0


def test_validation_surface_sample_guardrails_apply_extrapolated_floor_for_long_horizon():
    frame = _surface_frame_with_observations(n=139, alpha_token="90", horizon_days=15)

    surface = validate_compare_surface(frame, alphas=[0.90], horizons=[15])
    governance = surface["governance_summary"]
    horizon_payload = surface["horizon_governance"]["horizons"]["h15"]

    assert horizon_payload["horizon_observation_floor"] == 140
    assert governance["insufficient_sample_count"] == 1
    assert horizon_payload["insufficient_sample_count"] == 1
    assert surface["points"][0]["statistical_status"] == "WARN"


def test_validation_alerts_long_horizon_fallback_counts_preserve_sample_thin_signal():
    summary = ValidationSummary(
        alpha=0.90,
        expected_rate=0.10,
        model_results={},
        best_model=None,
        surface={
            "governance_summary": {
                "total_points": 1,
                "status_counts": {"PASS": 0, "WARN": 1, "FAIL": 0},
            },
            "points": [
                {
                    "model": "hist",
                    "alpha": 0.90,
                    "horizon_days": 15,
                    "n": 130,
                    "expected_rate": 0.10,
                }
            ],
            "horizon_governance": {
                "horizon_order": [15],
                "overall_verdict": "WARN",
                "horizons": {
                    "h15": {
                        "horizon_days": 15,
                        "total_points": 1,
                        "status_counts": {"PASS": 0, "WARN": 1, "FAIL": 0},
                        "pass_rate": 0.0,
                        "verdict": "WARN",
                        "champion_model": "hist",
                    }
                },
            },
        },
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_SURFACE_SAMPLE_THIN" in codes
    assert "VALIDATION_HORIZON_SAMPLE_THIN" in codes
    assert "VALIDATION_HORIZON_WARN" not in codes


def test_validation_surface_exposes_horizon_governance_rollup():
    frame = _compare_frame().assign(
        var_hist_a95_h5=[11] * 10,
        es_hist_a95_h5=[13] * 10,
        exc_hist_a95_h5=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        var_param_a95_h5=[14] * 10,
        es_param_a95_h5=[16] * 10,
        exc_param_a95_h5=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    )

    surface = validate_compare_surface(frame, alphas=[0.95], horizons=[1, 5])

    horizon_governance = surface.get("horizon_governance")
    assert isinstance(horizon_governance, dict)
    assert horizon_governance.get("horizon_order") == [1, 5]
    assert horizon_governance.get("overall_verdict") in {"PASS", "WARN", "FAIL"}

    horizons_payload = horizon_governance.get("horizons")
    assert isinstance(horizons_payload, dict)
    assert "h1" in horizons_payload
    assert "h5" in horizons_payload

    h1 = horizons_payload["h1"]
    h5 = horizons_payload["h5"]
    assert h1["horizon_days"] == 1
    assert h5["horizon_days"] == 5
    assert h1["verdict"] in {"PASS", "WARN", "FAIL"}
    assert h5["verdict"] in {"PASS", "WARN", "FAIL"}
    assert h1["champion_model"] in {"hist", "param"}
    assert h5["champion_model"] in {"hist", "param"}


def test_validation_alerts_flag_rejected_models():
    summary = validate_compare_frame(_compare_frame(), alpha=0.95)
    alerts = alerts_from_validation_summary(summary)

    assert any(alert.code == "KUPIEC_REJECTED" for alert in alerts)


def test_validation_alerts_include_surface_governance_failures():
    summary = ValidationSummary(
        alpha=0.95,
        expected_rate=0.05,
        model_results={},
        best_model=None,
        surface={
            "governance_summary": {
                "pvalue_threshold": 0.05,
                "total_points": 12,
                "status_counts": {"PASS": 7, "WARN": 3, "FAIL": 2},
                "coverage_fail_count": 2,
                "independence_fail_count": 1,
                "conditional_fail_count": 2,
                "pass_rate": 7 / 12,
            }
        },
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_GOVERNANCE_FAIL" in codes
    assert "VALIDATION_SURFACE_COVERAGE_FAIL" in codes
    assert "VALIDATION_SURFACE_CONDITIONAL_FAIL" in codes
    assert "VALIDATION_SURFACE_INDEPENDENCE_FAIL" in codes


def test_validation_alerts_include_surface_governance_warning_without_fail():
    summary = ValidationSummary(
        alpha=0.95,
        expected_rate=0.05,
        model_results={},
        best_model=None,
        surface={
            "governance_summary": {
                "pvalue_threshold": 0.05,
                "total_points": 10,
                "status_counts": {"PASS": 8, "WARN": 2, "FAIL": 0},
                "coverage_fail_count": 0,
                "independence_fail_count": 2,
                "conditional_fail_count": 0,
                "pass_rate": 0.8,
            }
        },
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_GOVERNANCE_WARN" in codes
    assert "VALIDATION_GOVERNANCE_FAIL" not in codes
    assert "VALIDATION_SURFACE_INDEPENDENCE_FAIL" in codes
    assert "VALIDATION_SURFACE_COVERAGE_FAIL" not in codes
    assert "VALIDATION_SURFACE_CONDITIONAL_FAIL" not in codes


def test_validation_alerts_include_horizon_governance_codes():
    summary = ValidationSummary(
        alpha=0.95,
        expected_rate=0.05,
        model_results={},
        best_model=None,
        surface={
            "governance_summary": {
                "pvalue_threshold": 0.05,
                "total_points": 6,
                "status_counts": {"PASS": 4, "WARN": 1, "FAIL": 1},
                "coverage_fail_count": 1,
                "independence_fail_count": 0,
                "conditional_fail_count": 1,
                "pass_rate": 4 / 6,
            },
            "horizon_governance": {
                "horizon_order": [1, 5],
                "overall_verdict": "FAIL",
                "horizons": {
                    "h1": {
                        "horizon_days": 1,
                        "total_points": 3,
                        "status_counts": {"PASS": 2, "WARN": 0, "FAIL": 1},
                        "coverage_fail_count": 1,
                        "independence_fail_count": 0,
                        "conditional_fail_count": 1,
                        "pass_rate": 2 / 3,
                        "verdict": "FAIL",
                        "champion_model": "hist",
                    },
                    "h5": {
                        "horizon_days": 5,
                        "total_points": 3,
                        "status_counts": {"PASS": 2, "WARN": 1, "FAIL": 0},
                        "coverage_fail_count": 0,
                        "independence_fail_count": 1,
                        "conditional_fail_count": 0,
                        "pass_rate": 2 / 3,
                        "verdict": "WARN",
                        "champion_model": "param",
                    },
                },
            },
        },
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_HORIZON_FAIL" in codes
    assert "VALIDATION_HORIZON_WARN" in codes
    fail_alert = next(alert for alert in alerts if alert.code == "VALIDATION_HORIZON_FAIL")
    warn_alert = next(alert for alert in alerts if alert.code == "VALIDATION_HORIZON_WARN")
    assert fail_alert.context.get("horizon_days") == 1
    assert warn_alert.context.get("horizon_days") == 5


def test_validation_alerts_collapse_thin_sample_surface_to_dedicated_warning():
    summary = ValidationSummary(
        alpha=0.95,
        expected_rate=0.05,
        model_results={},
        best_model=None,
        surface={
            "governance_summary": {
                "pvalue_threshold": 0.05,
                "total_points": 48,
                "effective_points": 0,
                "insufficient_sample_count": 48,
                "status_counts": {"PASS": 0, "WARN": 48, "FAIL": 0},
                "coverage_fail_count": 0,
                "independence_fail_count": 0,
                "conditional_fail_count": 0,
                "pass_rate": 0.0,
            },
            "horizon_governance": {
                "horizon_order": [1, 10],
                "overall_verdict": "WARN",
                "horizons": {
                    "h1": {
                        "horizon_days": 1,
                        "total_points": 24,
                        "effective_points": 0,
                        "insufficient_sample_count": 24,
                        "status_counts": {"PASS": 0, "WARN": 24, "FAIL": 0},
                        "pass_rate": 0.0,
                        "verdict": "WARN",
                        "champion_model": "fhs",
                    },
                    "h10": {
                        "horizon_days": 10,
                        "total_points": 24,
                        "effective_points": 0,
                        "insufficient_sample_count": 24,
                        "status_counts": {"PASS": 0, "WARN": 24, "FAIL": 0},
                        "pass_rate": 0.0,
                        "verdict": "WARN",
                        "champion_model": "fhs",
                    },
                },
            },
        },
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_SURFACE_SAMPLE_THIN" in codes
    assert "VALIDATION_GOVERNANCE_WARN" not in codes
    assert "VALIDATION_HORIZON_WARN" not in codes
    assert "VALIDATION_HORIZON_SAMPLE_THIN" in codes


def test_validation_alerts_downgrade_legacy_fail_surface_when_effective_points_are_zero():
    summary = ValidationSummary(
        alpha=0.95,
        expected_rate=0.05,
        model_results={},
        best_model=None,
        surface={
            "governance_summary": {
                "pvalue_threshold": 0.05,
                "total_points": 48,
                "effective_points": 0,
                "insufficient_sample_count": 48,
                "status_counts": {"PASS": 11, "WARN": 0, "FAIL": 37},
                "coverage_fail_count": 37,
                "independence_fail_count": 24,
                "conditional_fail_count": 32,
                "pass_rate": 11 / 48,
            },
            "horizon_governance": {
                "horizon_order": [1, 5, 10],
                "overall_verdict": "FAIL",
                "horizons": {
                    "h1": {
                        "horizon_days": 1,
                        "total_points": 18,
                        "effective_points": 0,
                        "insufficient_sample_count": 18,
                        "status_counts": {"PASS": 8, "WARN": 0, "FAIL": 10},
                        "pass_rate": 8 / 18,
                        "verdict": "FAIL",
                        "champion_model": "ewma",
                    },
                    "h5": {
                        "horizon_days": 5,
                        "total_points": 15,
                        "effective_points": 0,
                        "insufficient_sample_count": 15,
                        "status_counts": {"PASS": 3, "WARN": 0, "FAIL": 12},
                        "pass_rate": 3 / 15,
                        "verdict": "FAIL",
                        "champion_model": "ewma",
                    },
                    "h10": {
                        "horizon_days": 10,
                        "total_points": 15,
                        "effective_points": 0,
                        "insufficient_sample_count": 15,
                        "status_counts": {"PASS": 0, "WARN": 0, "FAIL": 15},
                        "pass_rate": 0.0,
                        "verdict": "FAIL",
                        "champion_model": "hist",
                    },
                },
            },
        },
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_SURFACE_SAMPLE_THIN" in codes
    assert "VALIDATION_HORIZON_SAMPLE_THIN" in codes
    assert "VALIDATION_GOVERNANCE_FAIL" not in codes
    assert "VALIDATION_SURFACE_COVERAGE_FAIL" not in codes
    assert "VALIDATION_SURFACE_CONDITIONAL_FAIL" not in codes
    assert "VALIDATION_SURFACE_INDEPENDENCE_FAIL" not in codes
    assert "VALIDATION_HORIZON_FAIL" not in codes


def test_validation_alerts_include_es_shortfall_and_breach_rate_signals():
    summary = ValidationSummary(
        alpha=0.99,
        expected_rate=0.01,
        model_results={
            "hist": BacktestModelValidation(
                model="hist",
                n=250,
                exceptions=7,
                expected_rate=0.01,
                actual_rate=7 / 250,
                lr_uc=0.0,
                p_uc=0.12,
                lr_ind=0.0,
                p_ind=0.22,
                lr_cc=0.0,
                p_cc=0.11,
                exceptions_last_250=7,
                traffic_light="YELLOW",
                score=74.0,
                es_tail_observations=12,
                es_shortfall_ratio=1.32,
                es_breach_rate=0.58,
            ),
            "param": BacktestModelValidation(
                model="param",
                n=250,
                exceptions=4,
                expected_rate=0.01,
                actual_rate=4 / 250,
                lr_uc=0.0,
                p_uc=0.55,
                lr_ind=0.0,
                p_ind=0.65,
                lr_cc=0.0,
                p_cc=0.62,
                exceptions_last_250=4,
                traffic_light="GREEN",
                score=88.0,
                es_tail_observations=9,
                es_shortfall_ratio=1.14,
                es_breach_rate=0.41,
            ),
        },
        best_model="param",
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_ES_SHORTFALL_BREACH" in codes
    assert "VALIDATION_ES_BREACH_RATE_BREACH" in codes
    assert "VALIDATION_ES_SHORTFALL_WARN" in codes
    assert "VALIDATION_ES_BREACH_RATE_WARN" in codes


def test_validation_alerts_skip_es_signals_when_tail_sample_is_too_small():
    summary = ValidationSummary(
        alpha=0.99,
        expected_rate=0.01,
        model_results={
            "hist": BacktestModelValidation(
                model="hist",
                n=250,
                exceptions=2,
                expected_rate=0.01,
                actual_rate=2 / 250,
                lr_uc=0.0,
                p_uc=0.5,
                lr_ind=0.0,
                p_ind=0.5,
                lr_cc=0.0,
                p_cc=0.5,
                exceptions_last_250=2,
                traffic_light="GREEN",
                score=92.0,
                es_tail_observations=2,
                es_shortfall_ratio=1.40,
                es_breach_rate=0.75,
            ),
        },
        best_model="hist",
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_ES_SHORTFALL_BREACH" not in codes
    assert "VALIDATION_ES_BREACH_RATE_BREACH" not in codes


def test_validation_alerts_skip_es_signals_when_total_sample_is_too_small():
    summary = ValidationSummary(
        alpha=0.95,
        expected_rate=0.05,
        model_results={
            "hist": BacktestModelValidation(
                model="hist",
                n=24,
                exceptions=6,
                expected_rate=0.05,
                actual_rate=6 / 24,
                lr_uc=0.0,
                p_uc=0.4,
                lr_ind=0.0,
                p_ind=0.4,
                lr_cc=0.0,
                p_cc=0.4,
                exceptions_last_250=6,
                traffic_light=None,
                score=65.0,
                es_tail_observations=8,
                es_shortfall_ratio=1.35,
                es_breach_rate=0.62,
            ),
        },
        best_model="hist",
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_ES_SHORTFALL_BREACH" not in codes
    assert "VALIDATION_ES_SHORTFALL_WARN" not in codes
    assert "VALIDATION_ES_BREACH_RATE_BREACH" not in codes
    assert "VALIDATION_ES_BREACH_RATE_WARN" not in codes


def test_validation_alerts_include_es_acerbi_signals():
    summary = ValidationSummary(
        alpha=0.99,
        expected_rate=0.01,
        model_results={
            "hist": BacktestModelValidation(
                model="hist",
                n=500,
                exceptions=12,
                expected_rate=0.01,
                actual_rate=12 / 500,
                lr_uc=0.0,
                p_uc=0.32,
                lr_ind=0.0,
                p_ind=0.28,
                lr_cc=0.0,
                p_cc=0.30,
                exceptions_last_250=6,
                traffic_light="YELLOW",
                score=70.0,
                es_tail_observations=12,
                es_shortfall_ratio=1.08,
                es_breach_rate=0.29,
                es_acerbi_stat=2.95,
                es_acerbi_p_value=0.0032,
                es_acerbi_observations=320,
            ),
            "param": BacktestModelValidation(
                model="param",
                n=500,
                exceptions=8,
                expected_rate=0.01,
                actual_rate=8 / 500,
                lr_uc=0.0,
                p_uc=0.61,
                lr_ind=0.0,
                p_ind=0.54,
                lr_cc=0.0,
                p_cc=0.58,
                exceptions_last_250=4,
                traffic_light="GREEN",
                score=88.0,
                es_tail_observations=8,
                es_shortfall_ratio=1.04,
                es_breach_rate=0.25,
                es_acerbi_stat=1.98,
                es_acerbi_p_value=0.047,
                es_acerbi_observations=320,
            ),
        },
        best_model="param",
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_ES_ACERBI_BREACH" in codes
    assert "VALIDATION_ES_ACERBI_WARN" in codes


def test_validation_alerts_skip_es_acerbi_when_observations_are_too_small():
    summary = ValidationSummary(
        alpha=0.99,
        expected_rate=0.01,
        model_results={
            "hist": BacktestModelValidation(
                model="hist",
                n=250,
                exceptions=4,
                expected_rate=0.01,
                actual_rate=4 / 250,
                lr_uc=0.0,
                p_uc=0.5,
                lr_ind=0.0,
                p_ind=0.5,
                lr_cc=0.0,
                p_cc=0.5,
                exceptions_last_250=4,
                traffic_light="GREEN",
                score=90.0,
                es_tail_observations=6,
                es_shortfall_ratio=1.02,
                es_breach_rate=0.20,
                es_acerbi_stat=3.10,
                es_acerbi_p_value=0.002,
                es_acerbi_observations=20,
            ),
        },
        best_model="hist",
    )

    alerts = alerts_from_validation_summary(summary)
    codes = {alert.code for alert in alerts}

    assert "VALIDATION_ES_ACERBI_BREACH" not in codes
    assert "VALIDATION_ES_ACERBI_WARN" not in codes


def test_champion_ranking_downgrades_acerbi_failed_model():
    summary = ValidationSummary(
        alpha=0.99,
        expected_rate=0.01,
        model_results={
            "hist": BacktestModelValidation(
                model="hist",
                n=500,
                exceptions=7,
                expected_rate=0.01,
                actual_rate=7 / 500,
                lr_uc=0.0,
                p_uc=0.75,
                lr_ind=0.0,
                p_ind=0.72,
                lr_cc=0.0,
                p_cc=0.71,
                exceptions_last_250=4,
                traffic_light="GREEN",
                score=95.0,
                es_tail_observations=8,
                es_shortfall_ratio=1.01,
                es_breach_rate=0.11,
                es_acerbi_stat=3.02,
                es_acerbi_p_value=0.0025,
                es_acerbi_observations=320,
            ),
            "param": BacktestModelValidation(
                model="param",
                n=500,
                exceptions=9,
                expected_rate=0.01,
                actual_rate=9 / 500,
                lr_uc=0.0,
                p_uc=0.55,
                lr_ind=0.0,
                p_ind=0.61,
                lr_cc=0.0,
                p_cc=0.57,
                exceptions_last_250=5,
                traffic_light="GREEN",
                score=90.0,
                es_tail_observations=9,
                es_shortfall_ratio=1.03,
                es_breach_rate=0.15,
                es_acerbi_stat=0.41,
                es_acerbi_p_value=0.68,
                es_acerbi_observations=320,
            ),
        },
        best_model="hist",
        champion_model_live="hist",
    )

    comparison = build_champion_challenger_summary(summary)

    assert comparison.champion_model == "param"
    assert comparison.challenger_model == "hist"
    assert comparison.ranking[0].model == "param"
    assert comparison.ranking[0].es_acerbi_status == "PASS"
    assert comparison.ranking[1].es_acerbi_status == "FAIL"


def test_champion_ranking_keeps_score_priority_when_acerbi_bucket_is_same():
    summary = ValidationSummary(
        alpha=0.99,
        expected_rate=0.01,
        model_results={
            "hist": BacktestModelValidation(
                model="hist",
                n=500,
                exceptions=6,
                expected_rate=0.01,
                actual_rate=6 / 500,
                lr_uc=0.0,
                p_uc=0.80,
                lr_ind=0.0,
                p_ind=0.74,
                lr_cc=0.0,
                p_cc=0.76,
                exceptions_last_250=3,
                traffic_light="GREEN",
                score=96.0,
                es_tail_observations=7,
                es_shortfall_ratio=1.01,
                es_breach_rate=0.10,
                es_acerbi_stat=0.48,
                es_acerbi_p_value=0.63,
                es_acerbi_observations=320,
            ),
            "param": BacktestModelValidation(
                model="param",
                n=500,
                exceptions=8,
                expected_rate=0.01,
                actual_rate=8 / 500,
                lr_uc=0.0,
                p_uc=0.61,
                lr_ind=0.0,
                p_ind=0.66,
                lr_cc=0.0,
                p_cc=0.62,
                exceptions_last_250=4,
                traffic_light="GREEN",
                score=90.0,
                es_tail_observations=8,
                es_shortfall_ratio=1.03,
                es_breach_rate=0.13,
                es_acerbi_stat=0.72,
                es_acerbi_p_value=0.47,
                es_acerbi_observations=320,
            ),
        },
        best_model="hist",
        champion_model_live="param",
    )

    comparison = build_champion_challenger_summary(summary)

    assert comparison.champion_model == "hist"
    assert comparison.challenger_model == "param"
    assert comparison.ranking[0].model == "hist"
    assert comparison.ranking[0].es_acerbi_status == "PASS"
    assert comparison.ranking[1].es_acerbi_status == "PASS"


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


def test_live_snapshot_alerts_flag_suspicious_model_equalities():
    snapshot = {
        "var": {"hist": 210.0, "param": 210.0},
        "es": {"hist": 260.0, "param": 260.0},
        "model_diagnostics": {
            "coherence_checks": {
                "suspicious_equalities": [
                    {
                        "models": ["hist", "param"],
                        "alpha": 0.99,
                        "horizon_days": 1,
                    }
                ]
            }
        },
    }

    alerts = alerts_from_live_snapshot(snapshot, limits_cfg={})
    codes = {alert.code for alert in alerts}

    assert "VAR_MODEL_EQUALITY_SUSPECT" in codes


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


def test_live_operator_alerts_suppress_reconciliation_noise_when_bridge_is_down():
    alerts = alerts_from_live_operator_state(
        {
            "connected": False,
            "status": "degraded",
            "generated_at": "2026-04-03T12:00:00+00:00",
            "last_error": "bridge offline",
            "reconciliation": {
                "live_base_ready": False,
                "manual_event_count": 3,
                "unmatched_execution_count": 2,
                "status_counts": {
                    "desk_vs_broker_drift": 1,
                    "pending_broker": 2,
                },
            },
        }
    )

    codes = {alert.code for alert in alerts}

    assert "MT5_LIVE_DISCONNECTED" in codes
    assert "MT5_LIVE_ERROR" in codes
    assert "DESK_BROKER_DRIFT" not in codes
    assert "PENDING_BROKER_ACTIVITY" not in codes
    assert "EXECUTION_UNMATCHED" not in codes


def test_live_operator_alerts_surface_incomplete_reconciliation_base_explicitly():
    alerts = alerts_from_live_operator_state(
        {
            "connected": True,
            "status": "ok",
            "generated_at": "2026-04-04T12:00:00+00:00",
            "reconciliation": {
                "bridge_connected": True,
                "live_base_ready": False,
                "live_evidence_present": False,
                "history_window_minutes": 180,
                "history_window_expired_execution_count": 2,
                "suppressed_status_counts": {
                    "live_base_incomplete": 1,
                    "pending_broker": 2,
                },
                "status_counts": {
                    "live_base_incomplete": 1,
                    "pending_broker": 2,
                },
                "diagnostic_code": "MT5_RECONCILIATION_INCOMPLETE",
                "diagnostic_message": "Broker book is empty for the current reconciliation window.",
            },
        }
    )

    codes = {alert.code for alert in alerts}

    assert "MT5_RECONCILIATION_INCOMPLETE" in codes
    assert "MT5_RECONCILIATION_WINDOW_EXPIRED" not in codes
    assert "DESK_BROKER_DRIFT" not in codes
    assert "PENDING_BROKER_ACTIVITY" not in codes
    assert "EXECUTION_UNMATCHED" not in codes


def test_live_operator_alerts_skip_window_expired_when_only_historical_and_non_actionable():
    alerts = alerts_from_live_operator_state(
        {
            "connected": True,
            "status": "ok",
            "generated_at": "2026-04-04T12:00:00+00:00",
            "reconciliation": {
                "bridge_connected": True,
                "live_base_ready": True,
                "history_window_minutes": 180,
                "history_window_expired_execution_count": 3,
                "manual_event_count": 0,
                "unmatched_execution_count": 0,
                "status_counts": {},
                "suppressed_status_counts": {},
            },
        }
    )

    codes = {alert.code for alert in alerts}
    assert "MT5_RECONCILIATION_WINDOW_EXPIRED" not in codes


def test_live_operator_alerts_skip_window_expired_when_only_history_window_expired_status():
    alerts = alerts_from_live_operator_state(
        {
            "connected": True,
            "status": "ok",
            "generated_at": "2026-04-04T12:00:00+00:00",
            "reconciliation": {
                "bridge_connected": True,
                "live_base_ready": True,
                "history_window_minutes": 180,
                "history_window_expired_execution_count": 3,
                "manual_event_count": 0,
                "unmatched_execution_count": 0,
                "status_counts": {"history_window_expired": 3},
                "suppressed_status_counts": {},
            },
        }
    )

    codes = {alert.code for alert in alerts}
    assert "MT5_RECONCILIATION_WINDOW_EXPIRED" not in codes
