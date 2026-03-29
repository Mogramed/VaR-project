from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from var_project.risk.budgeting import RiskBudgetSnapshot


def _status_from_utilization(utilization: float | None, *, warn: float, breach: float) -> str:
    value = 0.0 if utilization is None else float(utilization)
    if value >= breach:
        return "BREACH"
    if value >= warn:
        return "WARN"
    return "OK"


def _resolve_reference_model(
    budget: RiskBudgetSnapshot,
    limits_cfg: Mapping[str, Any] | None,
    explicit: str | None = None,
) -> str:
    capital_cfg = dict((limits_cfg or {}).get("capital_management") or {})
    candidates = [explicit, capital_cfg.get("preferred_model"), budget.preferred_model, "hist"]
    for candidate in candidates:
        name = str(candidate or "").strip().lower()
        if not name or name in {"best_validation", "best", "best_model", "auto"}:
            continue
        if name in budget.models:
            return name
    return budget.preferred_model if budget.preferred_model in budget.models else next(iter(budget.models), "hist")


@dataclass(frozen=True)
class CapitalBudget:
    reference_model: str
    total_budget_eur: float
    reserve_ratio: float
    reserved_capital_eur: float
    model_budgets: dict[str, float]
    symbol_budgets: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_model": self.reference_model,
            "total_budget_eur": self.total_budget_eur,
            "reserve_ratio": self.reserve_ratio,
            "reserved_capital_eur": self.reserved_capital_eur,
            "model_budgets": dict(self.model_budgets),
            "symbol_budgets": dict(self.symbol_budgets),
        }


@dataclass(frozen=True)
class CapitalAllocation:
    symbol: str
    weight: float
    target_capital_eur: float
    consumed_capital_eur: float
    reserved_capital_eur: float
    remaining_capital_eur: float
    utilization: float | None
    action: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "weight": self.weight,
            "target_capital_eur": self.target_capital_eur,
            "consumed_capital_eur": self.consumed_capital_eur,
            "reserved_capital_eur": self.reserved_capital_eur,
            "remaining_capital_eur": self.remaining_capital_eur,
            "utilization": self.utilization,
            "action": self.action,
            "status": self.status,
        }


@dataclass(frozen=True)
class ModelCapitalBudget:
    model: str
    budget_eur: float
    consumed_eur: float
    remaining_eur: float
    utilization: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "budget_eur": self.budget_eur,
            "consumed_eur": self.consumed_eur,
            "remaining_eur": self.remaining_eur,
            "utilization": self.utilization,
            "status": self.status,
        }


@dataclass(frozen=True)
class ReallocationRecommendation:
    symbol_from: str
    symbol_to: str
    amount_eur: float
    reason: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_from": self.symbol_from,
            "symbol_to": self.symbol_to,
            "amount_eur": self.amount_eur,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class CapitalUsageSnapshot:
    portfolio_slug: str
    base_currency: str
    reference_model: str
    snapshot_source: str
    snapshot_timestamp: str | None
    total_capital_budget_eur: float
    total_capital_consumed_eur: float
    total_capital_reserved_eur: float
    total_capital_remaining_eur: float
    headroom_ratio: float | None
    status: str
    budget: CapitalBudget
    models: dict[str, ModelCapitalBudget]
    allocations: dict[str, CapitalAllocation]
    recommendations: list[ReallocationRecommendation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_slug": self.portfolio_slug,
            "base_currency": self.base_currency,
            "reference_model": self.reference_model,
            "snapshot_source": self.snapshot_source,
            "snapshot_timestamp": self.snapshot_timestamp,
            "total_capital_budget_eur": self.total_capital_budget_eur,
            "total_capital_consumed_eur": self.total_capital_consumed_eur,
            "total_capital_reserved_eur": self.total_capital_reserved_eur,
            "total_capital_remaining_eur": self.total_capital_remaining_eur,
            "headroom_ratio": self.headroom_ratio,
            "status": self.status,
            "budget": self.budget.to_dict(),
            "models": {name: item.to_dict() for name, item in self.models.items()},
            "allocations": {name: item.to_dict() for name, item in self.allocations.items()},
            "recommendations": [item.to_dict() for item in self.recommendations],
        }


def build_capital_usage_snapshot(
    risk_budget_snapshot: Mapping[str, Any] | RiskBudgetSnapshot,
    limits_cfg: Mapping[str, Any] | None,
    *,
    portfolio_slug: str,
    base_currency: str,
    reference_model: str | None = None,
    snapshot_source: str = "historical",
    snapshot_timestamp: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CapitalUsageSnapshot:
    budget = risk_budget_snapshot if isinstance(risk_budget_snapshot, RiskBudgetSnapshot) else RiskBudgetSnapshot.from_dict(dict(risk_budget_snapshot))
    limits_cfg = dict(limits_cfg or {})
    overrides = dict(overrides or {})
    capital_cfg = dict(limits_cfg.get("capital_management") or {})
    risk_budget_cfg = dict(limits_cfg.get("risk_budget") or {})

    warn = float(risk_budget_cfg.get("utilisation_warn", 0.85))
    breach = float(risk_budget_cfg.get("utilisation_breach", 1.0))
    reserve_ratio = float(overrides.get("reserve_ratio", capital_cfg.get("reserve_ratio", 0.10)))
    rebalance_min_gap = float(capital_cfg.get("rebalance_min_gap", 10.0))

    resolved_model = _resolve_reference_model(budget, limits_cfg, explicit=reference_model or overrides.get("reference_model"))
    ref_budget = budget.models.get(resolved_model)
    if ref_budget is None:
        raise ValueError(f"Reference model '{resolved_model}' not available in risk budget snapshot.")

    model_budgets_cfg = dict(capital_cfg.get("model_budgets_eur") or {})
    symbol_budget_cfg = dict(overrides.get("symbol_weights") or capital_cfg.get("symbol_weights") or {})

    default_total_budget = max(float(ref_budget.total_var_budget), float(ref_budget.total_es_budget))
    total_budget = float(overrides.get("total_budget_eur", capital_cfg.get("total_budget_eur", default_total_budget)))
    total_reserved = float(total_budget * reserve_ratio)
    total_consumed = float(max(ref_budget.total_var, ref_budget.total_es))
    total_remaining = float(total_budget - total_consumed - total_reserved)
    headroom_ratio = None if total_budget <= 0.0 else float(total_remaining / total_budget)
    overall_status = _status_from_utilization(None if total_budget <= 0.0 else total_consumed / total_budget, warn=warn, breach=breach)

    models: dict[str, ModelCapitalBudget] = {}
    for model_name, model_budget in budget.models.items():
        configured_budget = model_budgets_cfg.get(model_name)
        model_total_budget = float(configured_budget if configured_budget is not None else max(model_budget.total_var_budget, model_budget.total_es_budget))
        model_consumed = float(max(model_budget.total_var, model_budget.total_es))
        model_remaining = float(model_total_budget - model_consumed)
        utilization = None if model_total_budget <= 0.0 else float(model_consumed / model_total_budget)
        models[model_name] = ModelCapitalBudget(
            model=model_name,
            budget_eur=model_total_budget,
            consumed_eur=model_consumed,
            remaining_eur=model_remaining,
            utilization=utilization,
            status=_status_from_utilization(utilization, warn=warn, breach=breach),
        )

    positions = ref_budget.positions
    raw_weights = {
        symbol: float(symbol_budget_cfg.get(symbol, item.weight if item.weight > 0.0 else 0.0))
        for symbol, item in positions.items()
    }
    weight_total = float(sum(max(value, 0.0) for value in raw_weights.values()))
    if weight_total <= 0.0:
        equal = 1.0 / float(len(positions) or 1)
        normalized_weights = {symbol: equal for symbol in positions}
    else:
        normalized_weights = {symbol: max(weight, 0.0) / weight_total for symbol, weight in raw_weights.items()}

    symbol_budgets = {symbol: float((total_budget - total_reserved) * weight) for symbol, weight in normalized_weights.items()}
    allocations: dict[str, CapitalAllocation] = {}
    overloaded: list[tuple[str, float, str]] = []
    underused: list[tuple[str, float]] = []
    for symbol, item in positions.items():
        target_capital = float(symbol_budgets.get(symbol, 0.0))
        consumed = float(max(abs(item.component_var), abs(item.component_es)))
        reserved = float(total_reserved * normalized_weights.get(symbol, 0.0))
        remaining = float(target_capital - consumed - reserved)
        utilization = None if target_capital <= 0.0 else float(consumed / target_capital)
        status = _status_from_utilization(utilization, warn=warn, breach=breach)
        action = "HOLD"
        if remaining < -rebalance_min_gap:
            action = "REDUCE"
            overloaded.append((symbol, abs(remaining), status))
        elif remaining > rebalance_min_gap:
            action = "ADD"
            underused.append((symbol, remaining))
        elif item.action == "HEDGE":
            action = "HEDGE"

        allocations[symbol] = CapitalAllocation(
            symbol=symbol,
            weight=float(normalized_weights.get(symbol, 0.0)),
            target_capital_eur=target_capital,
            consumed_capital_eur=consumed,
            reserved_capital_eur=reserved,
            remaining_capital_eur=remaining,
            utilization=utilization,
            action=action,
            status=status,
        )

    recommendations: list[ReallocationRecommendation] = []
    overloaded.sort(key=lambda item: item[1], reverse=True)
    underused.sort(key=lambda item: item[1], reverse=True)
    mutable_surplus = {symbol: amount for symbol, amount in underused}
    for symbol_from, deficit, status in overloaded:
        remaining_deficit = deficit
        for symbol_to, _ in underused:
            available = mutable_surplus.get(symbol_to, 0.0)
            if available <= rebalance_min_gap or remaining_deficit <= rebalance_min_gap:
                continue
            amount = float(min(available, remaining_deficit))
            recommendations.append(
                ReallocationRecommendation(
                    symbol_from=symbol_from,
                    symbol_to=symbol_to,
                    amount_eur=amount,
                    reason=f"Reallocate risk capital from {symbol_from} to {symbol_to} on {resolved_model}.",
                    priority=1 if status == "BREACH" else 2,
                )
            )
            mutable_surplus[symbol_to] = float(max(0.0, available - amount))
            remaining_deficit = float(max(0.0, remaining_deficit - amount))
        if remaining_deficit > rebalance_min_gap:
            recommendations.append(
                ReallocationRecommendation(
                    symbol_from=symbol_from,
                    symbol_to="CASH",
                    amount_eur=float(remaining_deficit),
                    reason=f"Reduce {symbol_from} risk capital to restore headroom on {resolved_model}.",
                    priority=1 if status == "BREACH" else 2,
                )
            )

    return CapitalUsageSnapshot(
        portfolio_slug=portfolio_slug,
        base_currency=base_currency,
        reference_model=resolved_model,
        snapshot_source=str(snapshot_source),
        snapshot_timestamp=snapshot_timestamp,
        total_capital_budget_eur=total_budget,
        total_capital_consumed_eur=total_consumed,
        total_capital_reserved_eur=total_reserved,
        total_capital_remaining_eur=total_remaining,
        headroom_ratio=headroom_ratio,
        status=overall_status,
        budget=CapitalBudget(
            reference_model=resolved_model,
            total_budget_eur=total_budget,
            reserve_ratio=reserve_ratio,
            reserved_capital_eur=total_reserved,
            model_budgets={name: item.budget_eur for name, item in models.items()},
            symbol_budgets=symbol_budgets,
        ),
        models=models,
        allocations=allocations,
        recommendations=recommendations,
    )
