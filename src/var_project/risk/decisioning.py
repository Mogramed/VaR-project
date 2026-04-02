from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from var_project.engine.risk_engine import RiskEngine, RiskModelConfig
from var_project.risk.budgeting import RiskBudgetSnapshot, build_risk_budget_snapshot


@dataclass(frozen=True, init=False)
class TradeProposal:
    symbol: str
    exposure_change: float
    note: str | None = None

    def __init__(
        self,
        symbol: str,
        exposure_change: float | None = None,
        note: str | None = None,
        *,
        delta_position_eur: float | None = None,
    ) -> None:
        selected_change = exposure_change if exposure_change is not None else delta_position_eur
        object.__setattr__(self, "symbol", str(symbol).upper())
        object.__setattr__(self, "exposure_change", float(0.0 if selected_change is None else selected_change))
        object.__setattr__(self, "note", note)

    @property
    def delta_position_eur(self) -> float:
        return self.exposure_change

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeProposal":
        return cls(
            symbol=str(payload.get("symbol", "")).upper(),
            exposure_change=float(payload.get("exposure_change", payload.get("delta_position_eur", 0.0))),
            note=None if payload.get("note") is None else str(payload.get("note")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exposure_change": self.exposure_change,
            "note": self.note,
        }


@dataclass(frozen=True)
class DecisionRiskState:
    var: float
    es: float
    budget_utilization_var: float | None
    budget_utilization_es: float | None
    headroom_var: float
    headroom_es: float
    gross_exposure: float
    symbol_exposure: float
    status: str
    headline_risk: list[dict[str, Any]] = field(default_factory=list)
    data_quality: dict[str, Any] | None = None

    @property
    def gross_notional(self) -> float:
        return self.gross_exposure

    @property
    def position_eur(self) -> float:
        return self.symbol_exposure

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionRiskState":
        return cls(
            var=float(payload.get("var", 0.0)),
            es=float(payload.get("es", 0.0)),
            budget_utilization_var=None
            if payload.get("budget_utilization_var") is None
            else float(payload.get("budget_utilization_var")),
            budget_utilization_es=None
            if payload.get("budget_utilization_es") is None
            else float(payload.get("budget_utilization_es")),
            headroom_var=float(payload.get("headroom_var", 0.0)),
            headroom_es=float(payload.get("headroom_es", 0.0)),
            gross_exposure=float(payload.get("gross_exposure", payload.get("gross_notional", 0.0))),
            symbol_exposure=float(payload.get("symbol_exposure", payload.get("position_eur", 0.0))),
            status=str(payload.get("status", "OK")),
            headline_risk=[dict(item) for item in payload.get("headline_risk") or []],
            data_quality=None if payload.get("data_quality") is None else dict(payload.get("data_quality") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "es": self.es,
            "budget_utilization_var": self.budget_utilization_var,
            "budget_utilization_es": self.budget_utilization_es,
            "headroom_var": self.headroom_var,
            "headroom_es": self.headroom_es,
            "gross_exposure": self.gross_exposure,
            "symbol_exposure": self.symbol_exposure,
            "status": self.status,
            "headline_risk": list(self.headline_risk),
            "data_quality": self.data_quality,
        }


@dataclass(frozen=True)
class RiskDecisionResult:
    symbol: str
    decision: str
    requested_exposure_change: float
    approved_exposure_change: float
    suggested_exposure_change: float | None
    resulting_exposure: float
    model_used: str
    reasons: list[str]
    note: str | None
    pre_trade: DecisionRiskState
    post_trade: DecisionRiskState

    @property
    def requested_delta_position_eur(self) -> float:
        return self.requested_exposure_change

    @property
    def approved_delta_position_eur(self) -> float:
        return self.approved_exposure_change

    @property
    def suggested_delta_position_eur(self) -> float | None:
        return self.suggested_exposure_change

    @property
    def resulting_position_eur(self) -> float:
        return self.resulting_exposure

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskDecisionResult":
        return cls(
            symbol=str(payload.get("symbol", "")).upper(),
            decision=str(payload.get("decision", "REJECT")).upper(),
            requested_exposure_change=float(
                payload.get("requested_exposure_change", payload.get("requested_delta_position_eur", 0.0))
            ),
            approved_exposure_change=float(
                payload.get("approved_exposure_change", payload.get("approved_delta_position_eur", 0.0))
            ),
            suggested_exposure_change=None
            if payload.get("suggested_exposure_change", payload.get("suggested_delta_position_eur")) is None
            else float(payload.get("suggested_exposure_change", payload.get("suggested_delta_position_eur"))),
            resulting_exposure=float(payload.get("resulting_exposure", payload.get("resulting_position_eur", 0.0))),
            model_used=str(payload.get("model_used", "hist")).lower(),
            reasons=[str(item) for item in payload.get("reasons") or []],
            note=None if payload.get("note") is None else str(payload.get("note")),
            pre_trade=DecisionRiskState.from_dict(dict(payload.get("pre_trade") or {})),
            post_trade=DecisionRiskState.from_dict(dict(payload.get("post_trade") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "requested_exposure_change": self.requested_exposure_change,
            "approved_exposure_change": self.approved_exposure_change,
            "suggested_exposure_change": self.suggested_exposure_change,
            "resulting_exposure": self.resulting_exposure,
            "model_used": self.model_used,
            "reasons": list(self.reasons),
            "note": self.note,
            "pre_trade": self.pre_trade.to_dict(),
            "post_trade": self.post_trade.to_dict(),
        }


@dataclass(frozen=True)
class _DecisionEvaluation:
    budget: RiskBudgetSnapshot
    state: DecisionRiskState


def _decision_settings(limits_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    limits_cfg = dict(limits_cfg or {})
    budget_cfg = dict(limits_cfg.get("risk_budget") or {})
    decision_cfg = dict(limits_cfg.get("risk_decision") or {})
    return {
        "decision_mode": str(decision_cfg.get("decision_mode", "advisory")),
        "warn_threshold": float(decision_cfg.get("warn_threshold", budget_cfg.get("utilisation_warn", 0.85))),
        "breach_threshold": float(decision_cfg.get("breach_threshold", budget_cfg.get("utilisation_breach", 1.0))),
        "min_fill_ratio": float(decision_cfg.get("min_fill_ratio", 0.25)),
        "allow_risk_reducing_override": bool(decision_cfg.get("allow_risk_reducing_override", True)),
    }


def _resolve_model(reference_model: str | None, budget: RiskBudgetSnapshot) -> str:
    normalized = str(reference_model or "").strip().lower()
    if normalized and normalized in budget.models:
        return normalized
    preferred = str(budget.preferred_model or "").strip().lower()
    if preferred in budget.models:
        return preferred
    return "hist" if "hist" in budget.models else next(iter(budget.models), "hist")


def _gross_exposure(exposure_by_symbol: Mapping[str, Any]) -> float:
    return float(sum(abs(float(value)) for value in exposure_by_symbol.values()))


def _build_state(
    budget: RiskBudgetSnapshot,
    *,
    model_name: str,
    exposure_by_symbol: Mapping[str, Any],
    symbol: str,
    headline_risk: list[dict[str, Any]] | None = None,
    data_quality: Mapping[str, Any] | None = None,
) -> DecisionRiskState:
    model_budget = budget.models[model_name]
    position_budget = model_budget.positions.get(symbol)
    symbol_exposure = 0.0 if position_budget is None else float(position_budget.current_exposure)
    return DecisionRiskState(
        var=float(model_budget.total_var),
        es=float(model_budget.total_es),
        budget_utilization_var=None if model_budget.utilization_var is None else float(model_budget.utilization_var),
        budget_utilization_es=None if model_budget.utilization_es is None else float(model_budget.utilization_es),
        headroom_var=float(model_budget.headroom_var),
        headroom_es=float(model_budget.headroom_es),
        gross_exposure=_gross_exposure(exposure_by_symbol),
        symbol_exposure=symbol_exposure,
        status=str(model_budget.status),
        headline_risk=[dict(item) for item in (headline_risk or [])],
        data_quality=None if data_quality is None else dict(data_quality),
    )


def _evaluate_positions(
    returns_wide: pd.DataFrame,
    *,
    exposure_by_symbol: Mapping[str, Any],
    config: RiskModelConfig,
    limits_cfg: Mapping[str, Any] | None,
    reference_model: str | None,
    symbol: str,
    headline_risk: list[dict[str, Any]] | None = None,
    data_quality: Mapping[str, Any] | None = None,
) -> _DecisionEvaluation:
    engine = RiskEngine({str(name): float(value) for name, value in exposure_by_symbol.items()})
    snapshot = engine.snapshot_from_returns(returns_wide, config)
    attribution = engine.attribute_from_returns(returns_wide, config, base_snapshot=snapshot)
    budget = build_risk_budget_snapshot(
        attribution,
        limits_cfg,
        exposure_by_symbol=exposure_by_symbol,
        preferred_model=reference_model,
    )
    model_name = _resolve_model(reference_model, budget)
    return _DecisionEvaluation(
        budget=budget,
        state=_build_state(
            budget,
            model_name=model_name,
            exposure_by_symbol=exposure_by_symbol,
            symbol=symbol,
            headline_risk=headline_risk,
            data_quality=data_quality,
        ),
    )


def _position_reduces_exposure(pre_position: float, post_position: float) -> bool:
    return abs(float(post_position)) < abs(float(pre_position)) - 1e-9


def _is_strict_improvement(before: DecisionRiskState, after: DecisionRiskState) -> bool:
    return (
        after.var < before.var - 1e-9
        or after.es < before.es - 1e-9
        or after.headroom_var > before.headroom_var + 1e-9
        or after.headroom_es > before.headroom_es + 1e-9
        or (
            before.budget_utilization_var is not None
            and after.budget_utilization_var is not None
            and after.budget_utilization_var < before.budget_utilization_var - 1e-9
        )
        or (
            before.budget_utilization_es is not None
            and after.budget_utilization_es is not None
            and after.budget_utilization_es < before.budget_utilization_es - 1e-9
        )
    )


def _is_admissible(state: DecisionRiskState, *, warn_threshold: float) -> bool:
    util_var_ok = state.budget_utilization_var is None or float(state.budget_utilization_var) <= float(warn_threshold)
    util_es_ok = state.budget_utilization_es is None or float(state.budget_utilization_es) <= float(warn_threshold)
    return util_var_ok and util_es_ok and state.headroom_var >= -1e-9 and state.headroom_es >= -1e-9


def _can_override_risk_reducing(
    before: DecisionRiskState,
    after: DecisionRiskState,
    *,
    pre_position: float,
    post_position: float,
    allow_override: bool,
) -> bool:
    if not allow_override:
        return False
    if not _position_reduces_exposure(pre_position, post_position):
        return False
    if after.var > before.var + 1e-9 or after.es > before.es + 1e-9:
        return False
    if after.headroom_var < before.headroom_var - 1e-9 or after.headroom_es < before.headroom_es - 1e-9:
        return False
    if (
        before.budget_utilization_var is not None
        and after.budget_utilization_var is not None
        and after.budget_utilization_var > before.budget_utilization_var + 1e-9
    ):
        return False
    if (
        before.budget_utilization_es is not None
        and after.budget_utilization_es is not None
        and after.budget_utilization_es > before.budget_utilization_es + 1e-9
    ):
        return False
    return _is_strict_improvement(before, after)


def _candidate_is_allowed(
    before: DecisionRiskState,
    after: DecisionRiskState,
    *,
    pre_position: float,
    post_position: float,
    settings: Mapping[str, Any],
) -> bool:
    if _is_admissible(after, warn_threshold=float(settings["warn_threshold"])):
        return True
    return _can_override_risk_reducing(
        before,
        after,
        pre_position=pre_position,
        post_position=post_position,
        allow_override=bool(settings["allow_risk_reducing_override"]),
    )


def evaluate_trade_proposal(
    returns_wide: pd.DataFrame,
    *,
    exposure_by_symbol: Mapping[str, Any] | None = None,
    positions_eur: Mapping[str, Any] | None = None,
    proposal: TradeProposal,
    config: RiskModelConfig,
    limits_cfg: Mapping[str, Any] | None = None,
    reference_model: str | None = None,
) -> RiskDecisionResult:
    symbol = str(proposal.symbol).upper()
    selected_exposure = exposure_by_symbol if exposure_by_symbol is not None else positions_eur or {}
    current_positions = {str(name): float(value) for name, value in selected_exposure.items()}
    settings = _decision_settings(limits_cfg)
    pre_eval = _evaluate_positions(
        returns_wide,
        exposure_by_symbol=current_positions,
        config=config,
        limits_cfg=limits_cfg,
        reference_model=reference_model,
        symbol=symbol if symbol in current_positions else next(iter(current_positions), symbol),
    )
    model_used = _resolve_model(reference_model, pre_eval.budget)

    current_position = float(current_positions.get(symbol, 0.0))
    if symbol not in current_positions:
        return RiskDecisionResult(
            symbol=symbol,
            decision="REJECT",
            requested_exposure_change=float(proposal.exposure_change),
            approved_exposure_change=0.0,
            suggested_exposure_change=None,
            resulting_exposure=current_position,
            model_used=model_used,
            reasons=[f"Unknown symbol '{symbol}' for the current portfolio."],
            note=proposal.note,
            pre_trade=pre_eval.state,
            post_trade=pre_eval.state,
        )

    requested_delta = float(proposal.exposure_change)
    if abs(requested_delta) <= 1e-9:
        return RiskDecisionResult(
            symbol=symbol,
            decision="REJECT",
            requested_exposure_change=requested_delta,
            approved_exposure_change=0.0,
            suggested_exposure_change=None,
            resulting_exposure=current_position,
            model_used=model_used,
            reasons=["exposure_change must be non-zero."],
            note=proposal.note,
            pre_trade=pre_eval.state,
            post_trade=pre_eval.state,
        )

    def candidate(delta: float) -> tuple[_DecisionEvaluation, float]:
        candidate_positions = dict(current_positions)
        candidate_positions[symbol] = float(current_position + delta)
        evaluation = _evaluate_positions(
            returns_wide,
            exposure_by_symbol=candidate_positions,
            config=config,
            limits_cfg=limits_cfg,
            reference_model=model_used,
            symbol=symbol,
        )
        return evaluation, float(candidate_positions[symbol])

    post_eval_full, full_position = candidate(requested_delta)
    if _candidate_is_allowed(
        pre_eval.state,
        post_eval_full.state,
        pre_position=current_position,
        post_position=full_position,
        settings=settings,
    ):
        reasons = [f"Post-trade risk remains acceptable under {model_used.upper()}."]
        if _can_override_risk_reducing(
            pre_eval.state,
            post_eval_full.state,
            pre_position=current_position,
            post_position=full_position,
            allow_override=bool(settings["allow_risk_reducing_override"]),
        ) and not _is_admissible(post_eval_full.state, warn_threshold=float(settings["warn_threshold"])):
            reasons = [f"Trade reduces risk on {symbol} and improves VaR/ES/headroom under {model_used.upper()}."]
        return RiskDecisionResult(
            symbol=symbol,
            decision="ACCEPT",
            requested_exposure_change=requested_delta,
            approved_exposure_change=requested_delta,
            suggested_exposure_change=None,
            resulting_exposure=full_position,
            model_used=model_used,
            reasons=reasons,
            note=proposal.note,
            pre_trade=pre_eval.state,
            post_trade=post_eval_full.state,
        )

    best_ratio = 0.0
    best_eval = pre_eval
    best_position = current_position
    low, high = 0.0, 1.0
    for _ in range(18):
        ratio = (low + high) / 2.0
        delta = requested_delta * ratio
        mid_eval, mid_position = candidate(delta)
        if _candidate_is_allowed(
            pre_eval.state,
            mid_eval.state,
            pre_position=current_position,
            post_position=mid_position,
            settings=settings,
        ):
            best_ratio = ratio
            best_eval = mid_eval
            best_position = mid_position
            low = ratio
        else:
            high = ratio

    if best_ratio >= float(settings["min_fill_ratio"]):
        approved_delta = requested_delta * best_ratio
        return RiskDecisionResult(
            symbol=symbol,
            decision="REDUCE",
            requested_exposure_change=requested_delta,
            approved_exposure_change=approved_delta,
            suggested_exposure_change=approved_delta,
            resulting_exposure=best_position,
            model_used=model_used,
            reasons=[
                f"Requested trade would exceed the advisory thresholds under {model_used.upper()}.",
                f"A reduced exposure change of {approved_delta:,.2f} remains admissible.",
            ],
            note=proposal.note,
            pre_trade=pre_eval.state,
            post_trade=best_eval.state,
        )

    return RiskDecisionResult(
        symbol=symbol,
        decision="REJECT",
        requested_exposure_change=requested_delta,
        approved_exposure_change=0.0,
        suggested_exposure_change=None,
        resulting_exposure=current_position,
        model_used=model_used,
        reasons=[
            f"Requested trade exceeds the advisory thresholds under {model_used.upper()}.",
            f"No admissible fill above the minimum ratio of {100.0 * float(settings['min_fill_ratio']):.0f}% was found.",
        ],
        note=proposal.note,
        pre_trade=pre_eval.state,
        post_trade=pre_eval.state,
    )
