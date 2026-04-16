from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping

from var_project.risk.decisioning import RiskDecisionResult
from var_project.risk.budgeting import RiskBudgetSnapshot
from var_project.validation.model_validation import ValidationSummary

ES_ALERT_MIN_TAIL_OBSERVATIONS = 3
ES_SHORTFALL_WARN_THRESHOLD = 1.10
ES_SHORTFALL_BREACH_THRESHOLD = 1.25
ES_BREACH_RATE_WARN_THRESHOLD = 0.35
ES_BREACH_RATE_BREACH_THRESHOLD = 0.50
ES_ACERBI_MIN_OBSERVATIONS = 60
ES_ACERBI_WARN_PVALUE = 0.05
ES_ACERBI_BREACH_PVALUE = 0.01


@dataclass(frozen=True)
class AlertEvent:
    source: str
    severity: str
    code: str
    message: str
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _alerts_from_validation_surface(summary: ValidationSummary) -> List[AlertEvent]:
    surface = dict(summary.surface or {})
    governance = dict(surface.get("governance_summary") or {})
    horizon_governance = dict(surface.get("horizon_governance") or {})
    if not governance:
        return []

    total_points = _int_or_zero(governance.get("total_points"))
    if total_points <= 0:
        return []

    status_counts = {
        str(key).upper(): _int_or_zero(value)
        for key, value in dict(governance.get("status_counts") or {}).items()
    }
    pass_count = _int_or_zero(status_counts.get("PASS"))
    warn_count = _int_or_zero(status_counts.get("WARN"))
    fail_count = _int_or_zero(status_counts.get("FAIL"))

    coverage_fail_count = _int_or_zero(governance.get("coverage_fail_count"))
    independence_fail_count = _int_or_zero(governance.get("independence_fail_count"))
    conditional_fail_count = _int_or_zero(governance.get("conditional_fail_count"))
    pvalue_threshold = _float_or_none(governance.get("pvalue_threshold"))
    pass_rate = _float_or_none(governance.get("pass_rate"))
    if pass_rate is None and total_points > 0:
        pass_rate = float(pass_count / total_points)

    alerts: List[AlertEvent] = []
    if fail_count > 0:
        alerts.append(
            AlertEvent(
                source="validation_surface",
                severity="BREACH",
                code="VALIDATION_GOVERNANCE_FAIL",
                message=(
                    f"Validation governance surface has {fail_count} failing points "
                    f"out of {total_points}."
                ),
                context={
                    "total_points": total_points,
                    "pass_count": pass_count,
                    "warn_count": warn_count,
                    "fail_count": fail_count,
                    "pass_rate": pass_rate,
                    "pvalue_threshold": pvalue_threshold,
                },
            )
        )
    elif warn_count > 0:
        alerts.append(
            AlertEvent(
                source="validation_surface",
                severity="WARN",
                code="VALIDATION_GOVERNANCE_WARN",
                message=(
                    f"Validation governance surface has {warn_count} warning points "
                    f"out of {total_points}."
                ),
                context={
                    "total_points": total_points,
                    "pass_count": pass_count,
                    "warn_count": warn_count,
                    "fail_count": fail_count,
                    "pass_rate": pass_rate,
                    "pvalue_threshold": pvalue_threshold,
                },
            )
        )

    if coverage_fail_count > 0:
        alerts.append(
            AlertEvent(
                source="validation_surface",
                severity="BREACH",
                code="VALIDATION_SURFACE_COVERAGE_FAIL",
                message=(
                    f"Unconditional coverage fails on {coverage_fail_count} validation-surface points."
                ),
                context={
                    "coverage_fail_count": coverage_fail_count,
                    "total_points": total_points,
                    "pvalue_threshold": pvalue_threshold,
                },
            )
        )

    if conditional_fail_count > 0:
        alerts.append(
            AlertEvent(
                source="validation_surface",
                severity="BREACH",
                code="VALIDATION_SURFACE_CONDITIONAL_FAIL",
                message=(
                    f"Conditional coverage fails on {conditional_fail_count} validation-surface points."
                ),
                context={
                    "conditional_fail_count": conditional_fail_count,
                    "total_points": total_points,
                    "pvalue_threshold": pvalue_threshold,
                },
            )
        )

    if independence_fail_count > 0:
        alerts.append(
            AlertEvent(
                source="validation_surface",
                severity="WARN",
                code="VALIDATION_SURFACE_INDEPENDENCE_FAIL",
                message=(
                    f"Independence test fails on {independence_fail_count} validation-surface points."
                ),
                context={
                    "independence_fail_count": independence_fail_count,
                    "total_points": total_points,
                    "pvalue_threshold": pvalue_threshold,
                },
            )
        )

    horizon_payload = {
        str(key): dict(value)
        for key, value in dict(horizon_governance.get("horizons") or {}).items()
        if isinstance(value, Mapping)
    }
    horizon_order_raw = horizon_governance.get("horizon_order") or []
    horizon_order: list[int] = []
    if isinstance(horizon_order_raw, list):
        for item in horizon_order_raw:
            try:
                horizon_days = int(item)
            except (TypeError, ValueError):
                continue
            if horizon_days > 0:
                horizon_order.append(horizon_days)
    if not horizon_order:
        inferred_horizons: list[int] = []
        for key in horizon_payload.keys():
            if not key.startswith("h"):
                continue
            try:
                inferred_horizons.append(int(key[1:]))
            except (TypeError, ValueError):
                continue
        horizon_order = sorted({int(item) for item in inferred_horizons if int(item) > 0})

    for horizon_days in horizon_order:
        payload = dict(horizon_payload.get(f"h{int(horizon_days)}") or {})
        status_counts = {
            str(key).upper(): _int_or_zero(value)
            for key, value in dict(payload.get("status_counts") or {}).items()
        }
        pass_count = _int_or_zero(status_counts.get("PASS"))
        warn_count = _int_or_zero(status_counts.get("WARN"))
        fail_count = _int_or_zero(status_counts.get("FAIL"))
        horizon_total_points = _int_or_zero(payload.get("total_points"))
        if horizon_total_points <= 0:
            horizon_total_points = pass_count + warn_count + fail_count
        horizon_pass_rate = _float_or_none(payload.get("pass_rate"))
        if horizon_pass_rate is None and horizon_total_points > 0:
            horizon_pass_rate = float(pass_count / horizon_total_points)

        horizon_verdict = str(payload.get("verdict") or "").upper()
        champion_model = str(payload.get("champion_model") or "").lower() or None
        context = {
            "horizon_days": int(horizon_days),
            "total_points": horizon_total_points,
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
            "pass_rate": horizon_pass_rate,
            "verdict": horizon_verdict or None,
            "champion_model": champion_model,
            "pvalue_threshold": pvalue_threshold,
        }
        if horizon_verdict == "FAIL":
            alerts.append(
                AlertEvent(
                    source="validation_surface",
                    severity="BREACH",
                    code="VALIDATION_HORIZON_FAIL",
                    message=(
                        f"Validation horizon {int(horizon_days)}d is FAIL with "
                        f"{fail_count} failing points out of {horizon_total_points}."
                    ),
                    context=context,
                )
            )
        elif horizon_verdict == "WARN":
            alerts.append(
                AlertEvent(
                    source="validation_surface",
                    severity="WARN",
                    code="VALIDATION_HORIZON_WARN",
                    message=(
                        f"Validation horizon {int(horizon_days)}d is WARN "
                        f"({warn_count} warning points out of {horizon_total_points})."
                    ),
                    context=context,
                )
            )

    return alerts


def alerts_from_validation_summary(summary: ValidationSummary) -> List[AlertEvent]:
    alerts: List[AlertEvent] = []
    for model_name, result in summary.model_results.items():
        if result.p_uc < 0.05:
            alerts.append(
                AlertEvent(
                    source="backtest",
                    severity="WARN",
                    code="KUPIEC_REJECTED",
                    message=f"{model_name} fails unconditional coverage at 5%.",
                    context={"model": model_name, "p_uc": result.p_uc, "exceptions": result.exceptions},
                )
            )
        if not (result.p_cc != result.p_cc) and result.p_cc < 0.05:
            alerts.append(
                AlertEvent(
                    source="backtest",
                    severity="WARN",
                    code="CC_REJECTED",
                    message=f"{model_name} fails conditional coverage at 5%.",
                    context={"model": model_name, "p_cc": result.p_cc},
                )
            )
        if result.traffic_light == "RED":
            alerts.append(
                AlertEvent(
                    source="backtest",
                    severity="BREACH",
                    code="TRAFFIC_LIGHT_RED",
                    message=f"{model_name} is in Basel traffic-light RED zone.",
                    context={"model": model_name, "exceptions_last_250": result.exceptions_last_250},
                )
            )
        elif result.traffic_light == "YELLOW":
            alerts.append(
                AlertEvent(
                    source="backtest",
                    severity="WARN",
                    code="TRAFFIC_LIGHT_YELLOW",
                    message=f"{model_name} is in Basel traffic-light YELLOW zone.",
                    context={"model": model_name, "exceptions_last_250": result.exceptions_last_250},
                )
            )
        es_tail_observations = _int_or_zero(result.es_tail_observations)
        es_shortfall_ratio = _float_or_none(result.es_shortfall_ratio)
        es_breach_rate = _float_or_none(result.es_breach_rate)
        es_acerbi_observations = _int_or_zero(getattr(result, "es_acerbi_observations", 0))
        es_acerbi_stat = _float_or_none(getattr(result, "es_acerbi_stat", None))
        es_acerbi_p_value = _float_or_none(getattr(result, "es_acerbi_p_value", None))
        if es_tail_observations >= ES_ALERT_MIN_TAIL_OBSERVATIONS:
            if es_shortfall_ratio is not None and es_shortfall_ratio >= ES_SHORTFALL_BREACH_THRESHOLD:
                alerts.append(
                    AlertEvent(
                        source="validation_es",
                        severity="BREACH",
                        code="VALIDATION_ES_SHORTFALL_BREACH",
                        message=(
                            f"{model_name} ES shortfall ratio is elevated "
                            f"({es_shortfall_ratio:.3f}) with {es_tail_observations} tail observations."
                        ),
                        context={
                            "model": model_name,
                            "es_tail_observations": es_tail_observations,
                            "es_shortfall_ratio": es_shortfall_ratio,
                            "warn_threshold": ES_SHORTFALL_WARN_THRESHOLD,
                            "breach_threshold": ES_SHORTFALL_BREACH_THRESHOLD,
                        },
                    )
                )
            elif es_shortfall_ratio is not None and es_shortfall_ratio >= ES_SHORTFALL_WARN_THRESHOLD:
                alerts.append(
                    AlertEvent(
                        source="validation_es",
                        severity="WARN",
                        code="VALIDATION_ES_SHORTFALL_WARN",
                        message=(
                            f"{model_name} ES shortfall ratio is above warning "
                            f"threshold ({es_shortfall_ratio:.3f})."
                        ),
                        context={
                            "model": model_name,
                            "es_tail_observations": es_tail_observations,
                            "es_shortfall_ratio": es_shortfall_ratio,
                            "warn_threshold": ES_SHORTFALL_WARN_THRESHOLD,
                            "breach_threshold": ES_SHORTFALL_BREACH_THRESHOLD,
                        },
                    )
                )

            if es_breach_rate is not None and es_breach_rate >= ES_BREACH_RATE_BREACH_THRESHOLD:
                alerts.append(
                    AlertEvent(
                        source="validation_es",
                        severity="BREACH",
                        code="VALIDATION_ES_BREACH_RATE_BREACH",
                        message=(
                            f"{model_name} ES breach rate is elevated "
                            f"({es_breach_rate:.2%}) on tail observations."
                        ),
                        context={
                            "model": model_name,
                            "es_tail_observations": es_tail_observations,
                            "es_breach_rate": es_breach_rate,
                            "warn_threshold": ES_BREACH_RATE_WARN_THRESHOLD,
                            "breach_threshold": ES_BREACH_RATE_BREACH_THRESHOLD,
                        },
                    )
                )
            elif es_breach_rate is not None and es_breach_rate >= ES_BREACH_RATE_WARN_THRESHOLD:
                alerts.append(
                    AlertEvent(
                        source="validation_es",
                        severity="WARN",
                        code="VALIDATION_ES_BREACH_RATE_WARN",
                        message=(
                            f"{model_name} ES breach rate is above warning "
                            f"threshold ({es_breach_rate:.2%})."
                        ),
                        context={
                            "model": model_name,
                            "es_tail_observations": es_tail_observations,
                            "es_breach_rate": es_breach_rate,
                            "warn_threshold": ES_BREACH_RATE_WARN_THRESHOLD,
                            "breach_threshold": ES_BREACH_RATE_BREACH_THRESHOLD,
                        },
                    )
                )
        if (
            es_acerbi_observations >= ES_ACERBI_MIN_OBSERVATIONS
            and es_acerbi_p_value is not None
        ):
            if es_acerbi_p_value <= ES_ACERBI_BREACH_PVALUE:
                stat_label = "n/a" if es_acerbi_stat is None else f"{es_acerbi_stat:.3f}"
                alerts.append(
                    AlertEvent(
                        source="validation_es",
                        severity="BREACH",
                        code="VALIDATION_ES_ACERBI_BREACH",
                        message=(
                            f"{model_name} ES Acerbi backtest rejects calibration "
                            f"(p={es_acerbi_p_value:.4f}, z={stat_label})."
                        ),
                        context={
                            "model": model_name,
                            "es_acerbi_observations": es_acerbi_observations,
                            "es_acerbi_stat": es_acerbi_stat,
                            "es_acerbi_p_value": es_acerbi_p_value,
                            "warn_threshold": ES_ACERBI_WARN_PVALUE,
                            "breach_threshold": ES_ACERBI_BREACH_PVALUE,
                        },
                    )
                )
            elif es_acerbi_p_value <= ES_ACERBI_WARN_PVALUE:
                alerts.append(
                    AlertEvent(
                        source="validation_es",
                        severity="WARN",
                        code="VALIDATION_ES_ACERBI_WARN",
                        message=(
                            f"{model_name} ES Acerbi backtest is close to rejection "
                            f"(p={es_acerbi_p_value:.4f})."
                        ),
                        context={
                            "model": model_name,
                            "es_acerbi_observations": es_acerbi_observations,
                            "es_acerbi_stat": es_acerbi_stat,
                            "es_acerbi_p_value": es_acerbi_p_value,
                            "warn_threshold": ES_ACERBI_WARN_PVALUE,
                            "breach_threshold": ES_ACERBI_BREACH_PVALUE,
                        },
                    )
                )
    alerts.extend(_alerts_from_validation_surface(summary))
    return alerts


def alerts_from_live_snapshot(snapshot: Mapping[str, Any], limits_cfg: Mapping[str, Any]) -> List[AlertEvent]:
    alerts: List[AlertEvent] = []
    vars_map = dict(snapshot.get("var") or {})
    es_map = dict(snapshot.get("es") or {})
    live_loss = snapshot.get("live_loss_proxy")

    model_limits = limits_cfg.get("model_limits_eur", {}) if limits_cfg else {}
    for model, rules in model_limits.items():
        var_limit = rules.get("var")
        es_limit = rules.get("es")
        current_var = vars_map.get(model)
        current_es = es_map.get(model)

        if var_limit is not None and current_var is not None and float(current_var) >= float(var_limit):
            alerts.append(
                AlertEvent(
                    source="live",
                    severity="BREACH",
                    code="MODEL_VAR_LIMIT",
                    message=f"{model} VaR exceeds configured limit.",
                    context={"model": model, "var": current_var, "limit": var_limit},
                )
            )
        if es_limit is not None and current_es is not None and float(current_es) >= float(es_limit):
            alerts.append(
                AlertEvent(
                    source="live",
                    severity="BREACH",
                    code="MODEL_ES_LIMIT",
                    message=f"{model} ES exceeds configured limit.",
                    context={"model": model, "es": current_es, "limit": es_limit},
                )
            )

    zone_hist = ((snapshot.get("limits") or {}).get("zone_hist"))
    zone_ewma = ((snapshot.get("limits") or {}).get("zone_ewma"))
    if zone_hist == "RED":
        alerts.append(
            AlertEvent(
                source="live",
                severity="BREACH",
                code="LIVE_ZONE_HIST_RED",
                message="Historical live zone is RED.",
                context={"zone": zone_hist, "live_loss": live_loss, "var_hist": vars_map.get("hist")},
            )
        )
    elif zone_hist == "AMBER":
        alerts.append(
            AlertEvent(
                source="live",
                severity="WARN",
                code="LIVE_ZONE_HIST_AMBER",
                message="Historical live zone is AMBER.",
                context={"zone": zone_hist, "live_loss": live_loss, "var_hist": vars_map.get("hist")},
            )
        )

    if zone_ewma == "RED":
        alerts.append(
            AlertEvent(
                source="live",
                severity="BREACH",
                code="LIVE_ZONE_EWMA_RED",
                message="EWMA live zone is RED.",
                context={"zone": zone_ewma, "live_loss": live_loss, "var_ewma": vars_map.get("ewma")},
            )
        )
    elif zone_ewma == "AMBER":
        alerts.append(
            AlertEvent(
                source="live",
                severity="WARN",
                code="LIVE_ZONE_EWMA_AMBER",
                message="EWMA live zone is AMBER.",
                context={"zone": zone_ewma, "live_loss": live_loss, "var_ewma": vars_map.get("ewma")},
            )
        )

    return alerts


def alerts_from_risk_budget(snapshot: Mapping[str, Any]) -> List[AlertEvent]:
    budget = RiskBudgetSnapshot.from_dict(dict(snapshot))
    model = budget.models.get(budget.preferred_model)
    if model is None:
        return []

    alerts: List[AlertEvent] = []
    if model.status == "BREACH":
        alerts.append(
            AlertEvent(
                source="risk_budget",
                severity="BREACH",
                code="MODEL_RISK_BUDGET_BREACH",
                message=f"{model.model} consumes more than its configured risk budget.",
                context={
                    "model": model.model,
                    "utilization_var": model.utilization_var,
                    "utilization_es": model.utilization_es,
                    "headroom_var": model.headroom_var,
                    "headroom_es": model.headroom_es,
                },
            )
        )
    elif model.status == "WARN":
        alerts.append(
            AlertEvent(
                source="risk_budget",
                severity="WARN",
                code="MODEL_RISK_BUDGET_WARN",
                message=f"{model.model} is close to its configured risk budget.",
                context={
                    "model": model.model,
                    "utilization_var": model.utilization_var,
                    "utilization_es": model.utilization_es,
                    "headroom_var": model.headroom_var,
                    "headroom_es": model.headroom_es,
                },
            )
        )

    ranked_positions = sorted(
        model.positions.values(),
        key=lambda item: (
            0 if item.status == "BREACH" else 1 if item.status == "WARN" else 2,
            -(item.utilization_var or 0.0),
            -(item.utilization_es or 0.0),
        ),
    )
    for item in ranked_positions[:3]:
        if item.status not in {"BREACH", "WARN"}:
            continue
        severity = "BREACH" if item.status == "BREACH" else "WARN"
        alerts.append(
            AlertEvent(
                source="risk_budget",
                severity=severity,
                code=f"POSITION_RISK_BUDGET_{item.status}",
                message=f"{item.symbol} uses {item.status.lower()}-level risk budget on {model.model}.",
                context={
                    "model": model.model,
                    "symbol": item.symbol,
                    "current_exposure": item.current_exposure,
                    "utilization_var": item.utilization_var,
                    "utilization_es": item.utilization_es,
                    "recommended_exposure": item.recommended_exposure,
                    "action": item.action,
                },
            )
        )
    return alerts


def alerts_from_risk_decision(result: Mapping[str, Any]) -> List[AlertEvent]:
    decision = RiskDecisionResult.from_dict(dict(result))
    if decision.decision == "ACCEPT":
        return []

    severity = "WARN" if decision.decision == "REDUCE" else "BREACH"
    code = "TRADE_DECISION_REDUCE" if decision.decision == "REDUCE" else "TRADE_DECISION_REJECT"
    return [
        AlertEvent(
            source="decision",
            severity=severity,
            code=code,
            message=f"{decision.decision.title()} trade recommendation on {decision.symbol}.",
            context={
                "symbol": decision.symbol,
                "decision": decision.decision,
                "model_used": decision.model_used,
                "requested_exposure_change": decision.requested_exposure_change,
                "approved_exposure_change": decision.approved_exposure_change,
                "resulting_exposure": decision.resulting_exposure,
                "reasons": list(decision.reasons),
            },
        )
    ]


def alerts_from_capital_snapshot(snapshot: Mapping[str, Any]) -> List[AlertEvent]:
    capital = dict(snapshot)
    alerts: List[AlertEvent] = []
    if str(capital.get("status", "OK")) == "BREACH":
        alerts.append(
            AlertEvent(
                source="capital",
                severity="BREACH",
                code="CAPITAL_BUDGET_BREACH",
                message=f"{capital.get('portfolio_slug')} exceeds its capital budget on {capital.get('reference_model')}.",
                context={
                    "portfolio_slug": capital.get("portfolio_slug"),
                    "reference_model": capital.get("reference_model"),
                    "budget": capital.get("total_capital_budget_eur"),
                    "consumed": capital.get("total_capital_consumed_eur"),
                    "remaining": capital.get("total_capital_remaining_eur"),
                },
            )
        )
    elif str(capital.get("status", "OK")) == "WARN":
        alerts.append(
            AlertEvent(
                source="capital",
                severity="WARN",
                code="CAPITAL_BUDGET_WARN",
                message=f"{capital.get('portfolio_slug')} is close to its capital budget on {capital.get('reference_model')}.",
                context={
                    "portfolio_slug": capital.get("portfolio_slug"),
                    "reference_model": capital.get("reference_model"),
                    "budget": capital.get("total_capital_budget_eur"),
                    "consumed": capital.get("total_capital_consumed_eur"),
                    "remaining": capital.get("total_capital_remaining_eur"),
                },
            )
        )

    hot_allocations = sorted(
        [dict(item) for item in dict(capital.get("allocations") or {}).values()],
        key=lambda item: (
            0 if str(item.get("status", "OK")) == "BREACH" else 1 if str(item.get("status", "OK")) == "WARN" else 2,
            -(float(item.get("utilization")) if item.get("utilization") is not None else 0.0),
        ),
    )
    for allocation in hot_allocations[:3]:
        if str(allocation.get("status", "OK")) not in {"BREACH", "WARN"}:
            continue
        alerts.append(
            AlertEvent(
                source="capital",
                severity="BREACH" if str(allocation.get("status", "OK")) == "BREACH" else "WARN",
                code=f"CAPITAL_ALLOCATION_{str(allocation.get('status', 'OK'))}",
                message=f"{allocation.get('symbol')} is at {str(allocation.get('status', 'OK')).lower()} capital utilization.",
                context={
                    "portfolio_slug": capital.get("portfolio_slug"),
                    "symbol": allocation.get("symbol"),
                    "utilization": allocation.get("utilization"),
                    "remaining_capital_eur": allocation.get("remaining_capital_eur"),
                    "action": allocation.get("action"),
                },
            )
        )
    return alerts


def alerts_from_execution_result(result: Mapping[str, Any]) -> List[AlertEvent]:
    payload = dict(result)
    status = str(payload.get("status", "UNKNOWN")).upper()
    symbol = str(payload.get("symbol", "")).upper()
    guard = dict(payload.get("guard") or {})
    reasons = list(guard.get("reasons") or [])

    if status == "EXECUTED":
        return []

    severity = "WARN" if status in {"REDUCED", "BLOCKED"} else "BREACH"
    code = f"EXECUTION_{status}"
    message = (
        f"MT5 execution blocked on {symbol}."
        if status in {"REJECTED", "BLOCKED"}
        else f"MT5 execution failed on {symbol}."
    )
    return [
        AlertEvent(
            source="execution",
            severity=severity,
            code=code,
            message=message,
            context={
                "symbol": symbol,
                "status": status,
                "requested_exposure_change": payload.get(
                    "requested_exposure_change", payload.get("requested_delta_position_eur")
                ),
                "approved_exposure_change": payload.get(
                    "approved_exposure_change", payload.get("approved_delta_position_eur")
                ),
                "executed_exposure_change": payload.get(
                    "executed_exposure_change", payload.get("executed_delta_position_eur")
                ),
                "reasons": reasons,
                "mt5_result": dict(payload.get("mt5_result") or {}),
            },
        )
    ]


def alerts_from_live_operator_state(state: Mapping[str, Any]) -> List[AlertEvent]:
    payload = dict(state)
    reconciliation = dict(payload.get("reconciliation") or {})
    live_base_ready = bool(reconciliation.get("live_base_ready", payload.get("connected")))
    bridge_connected = bool(reconciliation.get("bridge_connected", payload.get("connected")))
    market_closed = bool(reconciliation.get("market_closed", False))
    history_window_expired_execution_count = int(reconciliation.get("history_window_expired_execution_count") or 0)
    suppressed_status_counts = {
        str(key): int(value)
        for key, value in dict(reconciliation.get("suppressed_status_counts") or {}).items()
        if int(value or 0) > 0
    }
    status_counts = {
        str(key): int(value)
        for key, value in dict(reconciliation.get("status_counts") or {}).items()
        if int(value or 0) > 0
    }
    non_actionable_statuses = {
        "ok",
        "match",
        "matched",
        "history_window_expired",
        "live_base_incomplete",
    }
    active_status_signal_count = sum(
        count for status, count in status_counts.items() if status not in non_actionable_statuses
    )
    suppressed_status_signal_count = sum(
        count for status, count in suppressed_status_counts.items() if status not in non_actionable_statuses
    )
    manual_event_count = int(reconciliation.get("manual_event_count") or 0)
    unmatched_execution_count = int(reconciliation.get("unmatched_execution_count") or 0)
    actionable_reconciliation_signal_count = (
        active_status_signal_count
        + suppressed_status_signal_count
        + manual_event_count
        + unmatched_execution_count
    )
    alerts: List[AlertEvent] = []

    if payload.get("connected") is False:
        alerts.append(
            AlertEvent(
                source="mt5_live",
                severity="BREACH",
                code="MT5_LIVE_DISCONNECTED",
                message="The MT5 live bridge is disconnected.",
                context={
                    "status": payload.get("status"),
                    "generated_at": payload.get("generated_at"),
                    "last_success_at": payload.get("last_success_at"),
                },
            )
        )
    elif payload.get("stale"):
        alerts.append(
            AlertEvent(
                source="mt5_live",
                severity="WARN",
                code="MT5_LIVE_STALE",
                message="The MT5 live bridge is connected but stale.",
                context={
                    "status": payload.get("status"),
                    "generated_at": payload.get("generated_at"),
                    "last_success_at": payload.get("last_success_at"),
                },
            )
        )
    elif payload.get("degraded"):
        alerts.append(
            AlertEvent(
                source="mt5_live",
                severity="WARN",
                code="MT5_LIVE_DEGRADED",
                message="The MT5 live bridge is running in degraded mode.",
                context={
                    "status": payload.get("status"),
                    "generated_at": payload.get("generated_at"),
                    "last_success_at": payload.get("last_success_at"),
                },
            )
        )

    if payload.get("last_error"):
        alerts.append(
            AlertEvent(
                source="mt5_live",
                severity="WARN",
                code="MT5_LIVE_ERROR",
                message="The live bridge reported a recent MT5 error.",
                context={
                    "last_error": payload.get("last_error"),
                    "generated_at": payload.get("generated_at"),
                },
            )
        )

    if bridge_connected and not live_base_ready and (
        bool(suppressed_status_counts) or history_window_expired_execution_count > 0
    ):
        alerts.append(
            AlertEvent(
                source="reconciliation",
                severity="INFO" if market_closed else "WARN",
                code=str(reconciliation.get("diagnostic_code") or "MT5_RECONCILIATION_INCOMPLETE"),
                message=str(
                    reconciliation.get("diagnostic_message")
                    or (
                        "The MT5 bridge is connected, but the broker live book/history is empty for the current "
                        "reconciliation window. Derived reconciliation alerts are withheld until live broker evidence is available."
                    )
                ),
                context={
                    "portfolio_slug": reconciliation.get("portfolio_slug") or payload.get("portfolio_slug"),
                    "evidence_state": "empty_live_book" if not reconciliation.get("live_evidence_present") else "insufficient_live_evidence",
                    "history_window_minutes": reconciliation.get("history_window_minutes"),
                    "history_window_expired_execution_count": history_window_expired_execution_count,
                    "live_evidence_counts": dict(reconciliation.get("live_evidence_counts") or {}),
                    "suppressed_status_counts": suppressed_status_counts,
                },
            )
        )

    window_expired_actionable = (
        history_window_expired_execution_count > 0
        and live_base_ready
        and actionable_reconciliation_signal_count > 0
    )
    if bridge_connected and window_expired_actionable:
        alerts.append(
            AlertEvent(
                source="reconciliation",
                severity="INFO" if market_closed else "WARN",
                code="MT5_RECONCILIATION_WINDOW_EXPIRED",
                message=(
                    "Some stored desk executions are older than the live MT5 history window and cannot be confirmed "
                    "from the current broker feed alone."
                    if not market_closed
                    else "Market is closed: reconciliation uses the last valid broker reference while older desk executions remain outside the current broker history window."
                ),
                context={
                    "portfolio_slug": reconciliation.get("portfolio_slug") or payload.get("portfolio_slug"),
                    "history_window_minutes": reconciliation.get("history_window_minutes"),
                    "history_window_expired_execution_count": history_window_expired_execution_count,
                    "market_reference_timestamp": reconciliation.get("market_reference_timestamp"),
                },
            )
        )

    if live_base_ready and manual_event_count > 0:
        alerts.append(
            AlertEvent(
                source="reconciliation",
                severity="WARN",
                code="MT5_MANUAL_EVENTS",
                message="Manual MT5 activity was detected outside the desk workflow.",
                context={
                    "manual_event_count": manual_event_count,
                    "portfolio_slug": reconciliation.get("portfolio_slug") or payload.get("portfolio_slug"),
                },
            )
        )

    if live_base_ready and unmatched_execution_count > 0:
        alerts.append(
            AlertEvent(
                source="reconciliation",
                severity="WARN",
                code="EXECUTION_UNMATCHED",
                message="Some desk execution attempts are not fully matched to broker activity.",
                context={
                    "unmatched_execution_count": unmatched_execution_count,
                    "portfolio_slug": reconciliation.get("portfolio_slug") or payload.get("portfolio_slug"),
                },
            )
        )

    status_alerts = {
        "partial_fill": ("WARN", "PARTIAL_FILL_ACTIVE", "Partial MT5 fills still require operator follow-up."),
        "pending_broker": ("WARN", "PENDING_BROKER_ACTIVITY", "Some submissions are still pending broker confirmation."),
        "rejected_by_broker": ("BREACH", "BROKER_REJECTION_ACTIVE", "A broker rejection is present in the live reconciliation feed."),
        "manual_trade_detected": ("WARN", "MANUAL_TRADE_DETECTED", "A manual MT5 trade was detected in the blotter."),
        "orphan_live_position": ("BREACH", "ORPHAN_LIVE_POSITION", "A live MT5 position has no matching desk execution lineage."),
        "orphan_live_order": ("WARN", "ORPHAN_LIVE_ORDER", "A live MT5 order has no matching desk execution lineage."),
        "desk_vs_broker_drift": ("BREACH", "DESK_BROKER_DRIFT", "Desk vs broker exposure drift is currently detected."),
        "overfill_or_volume_drift": ("BREACH", "OVERFILL_OR_VOLUME_DRIFT", "Broker fills exceed the approved desk volume."),
    }
    if live_base_ready:
        for status, (severity, code, message) in status_alerts.items():
            count = status_counts.get(status, 0)
            if count <= 0:
                continue
            alerts.append(
                AlertEvent(
                    source="reconciliation",
                    severity=severity,
                    code=code,
                    message=message,
                    context={
                        "count": count,
                        "status": status,
                        "portfolio_slug": reconciliation.get("portfolio_slug") or payload.get("portfolio_slug"),
                    },
                )
            )

    return alerts
