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
                    "position_eur": item.position_eur,
                    "utilization_var": item.utilization_var,
                    "utilization_es": item.utilization_es,
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
