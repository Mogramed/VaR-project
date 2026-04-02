from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from var_project.core.model_registry import ordered_model_names


@dataclass(frozen=True)
class PositionRiskBudget:
    symbol: str
    current_exposure: float
    weight: float
    target_var_budget: float
    target_es_budget: float
    component_var: float
    component_es: float
    utilized_var: float
    utilized_es: float
    utilization_var: float | None
    utilization_es: float | None
    headroom_var: float
    headroom_es: float
    max_exposure: float | None
    recommended_exposure: float | None
    action: str
    status: str

    @property
    def position_eur(self) -> float:
        return self.current_exposure

    @property
    def max_position_eur(self) -> float | None:
        return self.max_exposure

    @property
    def recommended_position_eur(self) -> float | None:
        return self.recommended_exposure

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PositionRiskBudget":
        return cls(
            symbol=str(payload.get("symbol", "")),
            current_exposure=float(payload.get("current_exposure", payload.get("exposure_base_ccy", payload.get("position_eur", 0.0)))),
            weight=float(payload.get("weight", 0.0)),
            target_var_budget=float(payload.get("target_var_budget", 0.0)),
            target_es_budget=float(payload.get("target_es_budget", 0.0)),
            component_var=float(payload.get("component_var", 0.0)),
            component_es=float(payload.get("component_es", 0.0)),
            utilized_var=float(payload.get("utilized_var", 0.0)),
            utilized_es=float(payload.get("utilized_es", 0.0)),
            utilization_var=None if payload.get("utilization_var") is None else float(payload.get("utilization_var")),
            utilization_es=None if payload.get("utilization_es") is None else float(payload.get("utilization_es")),
            headroom_var=float(payload.get("headroom_var", 0.0)),
            headroom_es=float(payload.get("headroom_es", 0.0)),
            max_exposure=None
            if payload.get("max_exposure", payload.get("max_position_eur")) is None
            else float(payload.get("max_exposure", payload.get("max_position_eur"))),
            recommended_exposure=None
            if payload.get("recommended_exposure", payload.get("recommended_position_eur")) is None
            else float(payload.get("recommended_exposure", payload.get("recommended_position_eur"))),
            action=str(payload.get("action", "HOLD")),
            status=str(payload.get("status", "OK")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_exposure": self.current_exposure,
            "exposure_base_ccy": self.current_exposure,
            "weight": self.weight,
            "target_var_budget": self.target_var_budget,
            "target_es_budget": self.target_es_budget,
            "component_var": self.component_var,
            "component_es": self.component_es,
            "utilized_var": self.utilized_var,
            "utilized_es": self.utilized_es,
            "utilization_var": self.utilization_var,
            "utilization_es": self.utilization_es,
            "headroom_var": self.headroom_var,
            "headroom_es": self.headroom_es,
            "max_exposure": self.max_exposure,
            "recommended_exposure": self.recommended_exposure,
            "action": self.action,
            "status": self.status,
        }


@dataclass(frozen=True)
class ModelRiskBudget:
    model: str
    total_var: float
    total_es: float
    total_var_budget: float
    total_es_budget: float
    utilization_var: float | None
    utilization_es: float | None
    headroom_var: float
    headroom_es: float
    scale_to_var_budget: float | None
    scale_to_es_budget: float | None
    recommended_scale: float | None
    current_gross_exposure: float
    recommended_gross_exposure: float | None
    status: str
    positions: dict[str, PositionRiskBudget]

    @property
    def current_gross_notional(self) -> float:
        return self.current_gross_exposure

    @property
    def recommended_gross_notional(self) -> float | None:
        return self.recommended_gross_exposure

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelRiskBudget":
        positions_payload = dict(payload.get("positions") or {})
        return cls(
            model=str(payload.get("model", "")),
            total_var=float(payload.get("total_var", 0.0)),
            total_es=float(payload.get("total_es", 0.0)),
            total_var_budget=float(payload.get("total_var_budget", 0.0)),
            total_es_budget=float(payload.get("total_es_budget", 0.0)),
            utilization_var=None if payload.get("utilization_var") is None else float(payload.get("utilization_var")),
            utilization_es=None if payload.get("utilization_es") is None else float(payload.get("utilization_es")),
            headroom_var=float(payload.get("headroom_var", 0.0)),
            headroom_es=float(payload.get("headroom_es", 0.0)),
            scale_to_var_budget=None if payload.get("scale_to_var_budget") is None else float(payload.get("scale_to_var_budget")),
            scale_to_es_budget=None if payload.get("scale_to_es_budget") is None else float(payload.get("scale_to_es_budget")),
            recommended_scale=None if payload.get("recommended_scale") is None else float(payload.get("recommended_scale")),
            current_gross_exposure=float(
                payload.get("current_gross_exposure", payload.get("current_gross_notional", 0.0))
            ),
            recommended_gross_exposure=None
            if payload.get("recommended_gross_exposure", payload.get("recommended_gross_notional")) is None
            else float(payload.get("recommended_gross_exposure", payload.get("recommended_gross_notional"))),
            status=str(payload.get("status", "OK")),
            positions={symbol: PositionRiskBudget.from_dict(item) for symbol, item in positions_payload.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total_var": self.total_var,
            "total_es": self.total_es,
            "total_var_budget": self.total_var_budget,
            "total_es_budget": self.total_es_budget,
            "utilization_var": self.utilization_var,
            "utilization_es": self.utilization_es,
            "headroom_var": self.headroom_var,
            "headroom_es": self.headroom_es,
            "scale_to_var_budget": self.scale_to_var_budget,
            "scale_to_es_budget": self.scale_to_es_budget,
            "recommended_scale": self.recommended_scale,
            "current_gross_exposure": self.current_gross_exposure,
            "recommended_gross_exposure": self.recommended_gross_exposure,
            "status": self.status,
            "positions": {symbol: item.to_dict() for symbol, item in self.positions.items()},
        }


@dataclass(frozen=True)
class RiskBudgetSnapshot:
    alpha: float
    sample_size: int
    preferred_model: str
    models: dict[str, ModelRiskBudget]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskBudgetSnapshot":
        models_payload = dict(payload.get("models") or {})
        return cls(
            alpha=float(payload.get("alpha", 0.0)),
            sample_size=int(payload.get("sample_size", 0)),
            preferred_model=str(payload.get("preferred_model", "hist")),
            models={name: ModelRiskBudget.from_dict(item) for name, item in models_payload.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "sample_size": self.sample_size,
            "preferred_model": self.preferred_model,
            "models": {name: item.to_dict() for name, item in self.models.items()},
        }


def _normalize_budget_weights(
    symbols: list[str],
    exposure_by_symbol: Mapping[str, Any],
    configured: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    configured = dict(configured or {})
    positive_config = {
        symbol: float(configured[symbol])
        for symbol in symbols
        if configured.get(symbol) is not None and float(configured[symbol]) > 0.0
    }

    if len(positive_config) == len(symbols):
        raw = positive_config
    else:
        raw = {symbol: abs(float(exposure_by_symbol.get(symbol, 0.0))) for symbol in symbols}

    total = float(sum(raw.values()))
    if total <= 0.0 and symbols:
        equal = 1.0 / float(len(symbols))
        return {symbol: equal for symbol in symbols}
    if total <= 0.0:
        return {}
    return {symbol: float(raw.get(symbol, 0.0)) / total for symbol in symbols}


def _status_from_utilization(utilization_var: float | None, utilization_es: float | None, *, warn: float, breach: float) -> str:
    values = [value for value in (utilization_var, utilization_es) if value is not None]
    max_value = max(values) if values else 0.0
    if max_value >= breach:
        return "BREACH"
    if max_value >= warn:
        return "WARN"
    return "OK"


def _recommended_exposure(
    current_exposure: float,
    *,
    target_var_budget: float,
    target_es_budget: float,
    utilized_var: float,
    utilized_es: float,
    target_buffer: float,
) -> tuple[float | None, float | None]:
    current_abs = abs(float(current_exposure))
    if current_abs <= 1e-9:
        return None, None

    max_candidates: list[float] = []
    if utilized_var > 0.0 and target_var_budget > 0.0:
        max_candidates.append(current_abs * (target_var_budget / utilized_var))
    if utilized_es > 0.0 and target_es_budget > 0.0:
        max_candidates.append(current_abs * (target_es_budget / utilized_es))
    if not max_candidates:
        return None, None

    max_position_abs = min(max_candidates)
    recommended_abs = max_position_abs * float(target_buffer)
    sign = -1.0 if float(current_exposure) < 0.0 else 1.0
    return float(max_position_abs), float(sign * recommended_abs)


def _action_from_budget(
    current_exposure: float,
    *,
    component_var: float,
    recommended_exposure: float | None,
    tolerance: float,
    utilization_var: float | None,
    warn: float,
) -> str:
    current_abs = abs(float(current_exposure))
    if component_var < 0.0 and (utilization_var or 0.0) < warn:
        return "HEDGE"
    if recommended_exposure is None:
        return "HOLD"

    target_abs = abs(float(recommended_exposure))
    if current_abs > target_abs * (1.0 + tolerance):
        return "REDUCE"
    if current_abs < target_abs * (1.0 - tolerance):
        return "ADD"
    return "HOLD"


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value)


def _resolve_preferred_model(
    models: list[str],
    *,
    preferred_model: str | None,
    configured_model: str | None,
) -> str:
    candidates = [preferred_model, configured_model, "hist"]
    normalized = [str(item).strip().lower() for item in candidates if str(item).strip()]
    for candidate in normalized:
        if candidate in {"auto", "best", "best_model", "best_validation"}:
            continue
        if candidate in models:
            return candidate
    return models[0] if models else "hist"


def build_risk_budget_snapshot(
    attribution: Mapping[str, Any] | Any,
    limits_cfg: Mapping[str, Any] | None,
    *,
    exposure_by_symbol: Mapping[str, Any] | None = None,
    positions_eur: Mapping[str, Any] | None = None,
    preferred_model: str | None = None,
) -> RiskBudgetSnapshot:
    attribution_payload = _coerce_mapping(attribution)
    limits_cfg = dict(limits_cfg or {})
    budget_cfg = dict(limits_cfg.get("risk_budget") or {})
    model_limits = dict(limits_cfg.get("model_limits_eur") or {})
    models_payload = dict(attribution_payload.get("models") or {})
    model_names = ordered_model_names(models_payload.keys())

    warn = float(budget_cfg.get("utilisation_warn", 0.85))
    breach = float(budget_cfg.get("utilisation_breach", 1.0))
    target_buffer = float(budget_cfg.get("target_buffer", 0.95))
    tolerance = float(budget_cfg.get("position_tolerance", 0.05))
    preferred = _resolve_preferred_model(
        model_names,
        preferred_model=preferred_model,
        configured_model=budget_cfg.get("preferred_model"),
    )

    inferred_exposure: dict[str, float] = {}
    selected_exposure = exposure_by_symbol if exposure_by_symbol is not None else positions_eur
    if selected_exposure is not None:
        inferred_exposure = {str(symbol): float(value) for symbol, value in selected_exposure.items()}
    elif model_names:
        first_positions = dict((dict(models_payload.get(model_names[0]) or {})).get("positions") or {})
        inferred_exposure = {
            str(symbol): float(
                (dict(item or {})).get(
                    "exposure_base_ccy",
                    (dict(item or {})).get("position_eur", 0.0),
                )
            )
            for symbol, item in first_positions.items()
        }
    symbols = list(inferred_exposure.keys())
    weights = _normalize_budget_weights(symbols, inferred_exposure, configured=budget_cfg.get("symbol_weights"))
    current_gross_exposure = float(sum(abs(value) for value in inferred_exposure.values()))

    budget_models: dict[str, ModelRiskBudget] = {}
    for model_name in model_names:
        model_payload = dict(models_payload.get(model_name) or {})
        positions_payload = dict(model_payload.get("positions") or {})
        limit_payload = dict(model_limits.get(model_name) or {})

        total_var = float(model_payload.get("total_var", 0.0))
        total_es = float(model_payload.get("total_es", 0.0))
        total_var_budget = float(limit_payload.get("var", total_var))
        total_es_budget = float(limit_payload.get("es", total_es))
        utilization_var = None if total_var_budget <= 0.0 else float(total_var / total_var_budget)
        utilization_es = None if total_es_budget <= 0.0 else float(total_es / total_es_budget)
        headroom_var = float(total_var_budget - total_var)
        headroom_es = float(total_es_budget - total_es)
        scale_to_var_budget = None if total_var <= 0.0 or total_var_budget <= 0.0 else float(total_var_budget / total_var)
        scale_to_es_budget = None if total_es <= 0.0 or total_es_budget <= 0.0 else float(total_es_budget / total_es)
        scale_candidates = [value for value in (scale_to_var_budget, scale_to_es_budget) if value is not None]
        recommended_scale = None if not scale_candidates else float(min(scale_candidates) * target_buffer)
        recommended_gross_exposure = None if recommended_scale is None else float(current_gross_exposure * recommended_scale)

        position_budgets: dict[str, PositionRiskBudget] = {}
        for symbol in symbols:
            position_payload = dict(positions_payload.get(symbol) or {})
            current_exposure = float(
                position_payload.get(
                    "current_exposure",
                    position_payload.get(
                        "exposure_base_ccy",
                        position_payload.get("position_eur", inferred_exposure.get(symbol, 0.0)),
                    ),
                )
            )
            component_var = float(position_payload.get("component_var", 0.0))
            component_es = float(position_payload.get("component_es", 0.0))
            utilized_var = abs(component_var)
            utilized_es = abs(component_es)
            weight = float(weights.get(symbol, 0.0))
            target_var_budget = float(total_var_budget * weight)
            target_es_budget = float(total_es_budget * weight)
            pos_utilization_var = None if target_var_budget <= 0.0 else float(utilized_var / target_var_budget)
            pos_utilization_es = None if target_es_budget <= 0.0 else float(utilized_es / target_es_budget)
            max_exposure, recommended_exposure = _recommended_exposure(
                current_exposure,
                target_var_budget=target_var_budget,
                target_es_budget=target_es_budget,
                utilized_var=utilized_var,
                utilized_es=utilized_es,
                target_buffer=target_buffer,
            )
            status = _status_from_utilization(pos_utilization_var, pos_utilization_es, warn=warn, breach=breach)
            action = _action_from_budget(
                current_exposure,
                component_var=component_var,
                recommended_exposure=recommended_exposure,
                tolerance=tolerance,
                utilization_var=pos_utilization_var,
                warn=warn,
            )

            position_budgets[symbol] = PositionRiskBudget(
                symbol=symbol,
                current_exposure=current_exposure,
                weight=weight,
                target_var_budget=target_var_budget,
                target_es_budget=target_es_budget,
                component_var=component_var,
                component_es=component_es,
                utilized_var=utilized_var,
                utilized_es=utilized_es,
                utilization_var=pos_utilization_var,
                utilization_es=pos_utilization_es,
                headroom_var=float(target_var_budget - utilized_var),
                headroom_es=float(target_es_budget - utilized_es),
                max_exposure=max_exposure,
                recommended_exposure=recommended_exposure,
                action=action,
                status=status,
            )

        budget_models[model_name] = ModelRiskBudget(
            model=model_name,
            total_var=total_var,
            total_es=total_es,
            total_var_budget=total_var_budget,
            total_es_budget=total_es_budget,
            utilization_var=utilization_var,
            utilization_es=utilization_es,
            headroom_var=headroom_var,
            headroom_es=headroom_es,
            scale_to_var_budget=scale_to_var_budget,
            scale_to_es_budget=scale_to_es_budget,
            recommended_scale=recommended_scale,
            current_gross_exposure=current_gross_exposure,
            recommended_gross_exposure=recommended_gross_exposure,
            status=_status_from_utilization(utilization_var, utilization_es, warn=warn, breach=breach),
            positions=position_budgets,
        )

    return RiskBudgetSnapshot(
        alpha=float(attribution_payload.get("alpha", 0.0)),
        sample_size=int(attribution_payload.get("sample_size", 0)),
        preferred_model=preferred,
        models=budget_models,
    )
