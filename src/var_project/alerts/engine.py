from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping

from var_project.risk.decisioning import RiskDecisionResult
from var_project.risk.budgeting import RiskBudgetSnapshot
from var_project.validation.model_validation import ValidationSummary


@dataclass(frozen=True)
class AlertEvent:
    source: str
    severity: str
    code: str
    message: str
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
                    "position_eur": item.position_eur,
                    "utilization_var": item.utilization_var,
                    "utilization_es": item.utilization_es,
                    "recommended_exposure": item.recommended_exposure,
                    "recommended_position_eur": item.recommended_position_eur,
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
                "requested_delta_position_eur": decision.requested_delta_position_eur,
                "approved_delta_position_eur": decision.approved_delta_position_eur,
                "resulting_position_eur": decision.resulting_position_eur,
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
                "requested_delta_position_eur": payload.get("requested_delta_position_eur"),
                "approved_delta_position_eur": payload.get("approved_delta_position_eur"),
                "executed_delta_position_eur": payload.get("executed_delta_position_eur"),
                "reasons": reasons,
                "mt5_result": dict(payload.get("mt5_result") or {}),
            },
        )
    ]


def alerts_from_live_operator_state(state: Mapping[str, Any]) -> List[AlertEvent]:
    payload = dict(state)
    reconciliation = dict(payload.get("reconciliation") or {})
    status_counts = {
        str(key): int(value)
        for key, value in dict(reconciliation.get("status_counts") or {}).items()
        if int(value or 0) > 0
    }
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

    manual_event_count = int(reconciliation.get("manual_event_count") or 0)
    if manual_event_count > 0:
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

    unmatched_execution_count = int(reconciliation.get("unmatched_execution_count") or 0)
    if unmatched_execution_count > 0:
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
