from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised indirectly when sklearn is installed.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path for constrained environments.
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

if TYPE_CHECKING:
    from var_project.storage.app_storage import AppStorage


DECISION_ALPHA_VERSION = "decision_alpha_v1"
MAX_DECISION_ALPHA_FORECAST_HORIZON_DAYS = 180
MAX_BACKTEST_TRAINING_ROWS = 50_000
FEATURE_COLUMNS: tuple[str, ...] = (
    "momentum_short_term",
    "volatility_recent",
    "headroom_delta",
    "risk_delta",
    "validation_confidence",
    "exception_pressure",
    "spread_cost_norm",
    "slippage_points",
)

_FEATURE_LABELS: dict[str, str] = {
    "momentum_short_term": "momentum",
    "volatility_recent": "volatility",
    "headroom_delta": "headroom_delta",
    "risk_delta": "risk_delta",
    "validation_confidence": "validation_confidence",
    "exception_pressure": "exception_pressure",
    "spread_cost_norm": "spread_cost",
    "slippage_points": "slippage",
}

_HEURISTIC_WEIGHTS: dict[str, float] = {
    "momentum_short_term": 120.0,
    "volatility_recent": -0.8,
    "headroom_delta": 0.0009,
    "risk_delta": -0.0012,
    "validation_confidence": 1.6,
    "exception_pressure": -1.8,
    "spread_cost_norm": -2.3,
    "slippage_points": -0.04,
}

# Keep backtest trajectory predictions realistic for liquid FX symbols by
# avoiding open-loop drift accumulation on long hourly windows.
TRAJECTORY_EDGE_DAMPING = 4.0
TRAJECTORY_ANCHOR_WEIGHT = 0.2
TRAJECTORY_DIVERGENCE_REVERSION = 0.35
TRAJECTORY_RETURN_FLOOR = 0.0002
TRAJECTORY_RETURN_MIN_CAP = 0.0015
TRAJECTORY_RETURN_MAX_CAP = 0.01
FORECAST_H1_LOOKBACK_DAYS_MIN = 90
FORECAST_H1_LOOKBACK_DAYS_MAX = 300
FORECAST_D1_LOOKBACK_DAYS_MIN = 180
FORECAST_D1_LOOKBACK_DAYS_MAX = 1200


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, "", "null"}:
            return float(default)
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _has_finite_numeric(value: Any) -> bool:
    if value in {None, "", "null"}:
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(parsed))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, "", "null"}:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_datetime(value: Any) -> datetime | None:
    if value in {None, "", "null"}:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if parsed is None or pd.isna(parsed):
        return None
    return parsed.to_pydatetime().astimezone(timezone.utc)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), float(low)), float(high)))


def _sigmoid(value: float) -> float:
    bounded = _clip(value, -50.0, 50.0)
    return float(1.0 / (1.0 + math.exp(-bounded)))


def _extract_best_validation_row(validation_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not validation_summary:
        return {}
    payload = dict(validation_summary.get("summary") or validation_summary)
    best_model = str(validation_summary.get("best_model") or payload.get("best_model") or "").strip().lower()
    model_results = dict(payload.get("model_results") or {})
    if best_model and best_model in model_results:
        candidate = dict(model_results.get(best_model) or {})
        candidate.setdefault("model", best_model)
        return candidate
    if model_results:
        first_key = str(next(iter(model_results.keys())))
        candidate = dict(model_results.get(first_key) or {})
        candidate.setdefault("model", first_key)
        return candidate
    return {}


def _data_quality_confidence(data_quality: Mapping[str, Any] | None) -> float:
    if not data_quality:
        return 0.5
    status = str(data_quality.get("status") or "").strip().lower()
    available = _safe_float(data_quality.get("available_observations"), 0.0)
    minimum = max(_safe_float(data_quality.get("minimum_valid_days"), 1.0), 1.0)
    coverage = _clip(available / minimum, 0.0, 1.0)
    status_score = 0.45
    if status in {"good", "healthy", "ok"}:
        status_score = 0.8
    elif status in {"fair", "warning"}:
        status_score = 0.6
    elif status in {"poor", "bad", "degraded"}:
        status_score = 0.35
    return _clip(0.55 * status_score + 0.45 * coverage, 0.05, 0.95)


def _validation_confidence(
    validation_summary: Mapping[str, Any] | None,
    data_quality: Mapping[str, Any] | None,
) -> float:
    row = _extract_best_validation_row(validation_summary)
    candidate_scores: list[float] = []
    for key in ("p_uc", "p_ind", "p_cc"):
        value = row.get(key)
        if value is None:
            continue
        candidate_scores.append(_clip(_safe_float(value, 0.5), 0.0, 1.0))
    if row.get("score") is not None:
        candidate_scores.append(_clip(_safe_float(row.get("score"), 50.0) / 100.0, 0.0, 1.0))
    if row.get("actual_rate") is not None and row.get("expected_rate") is not None:
        actual = max(_safe_float(row.get("actual_rate"), 0.0), 0.0)
        expected = max(_safe_float(row.get("expected_rate"), 0.0), 1e-6)
        overrun = max(actual - expected, 0.0) / expected
        candidate_scores.append(_clip(1.0 - overrun, 0.0, 1.0))
    quality_score = _data_quality_confidence(data_quality)
    if not candidate_scores:
        return quality_score
    validation_score = float(sum(candidate_scores) / len(candidate_scores))
    return _clip(0.7 * validation_score + 0.3 * quality_score, 0.05, 0.95)


def _exception_pressure(
    *,
    validation_summary: Mapping[str, Any] | None,
    headline_risk: Sequence[Mapping[str, Any]] | None,
) -> float:
    pressure = 0.0
    row = _extract_best_validation_row(validation_summary)
    if row:
        actual = max(_safe_float(row.get("actual_rate"), 0.0), 0.0)
        expected = max(_safe_float(row.get("expected_rate"), 0.0), 1e-6)
        if expected > 0.0:
            pressure += max(actual - expected, 0.0) / expected
        exceptions = _safe_float(row.get("exceptions"), 0.0)
        sample = max(_safe_float(row.get("n"), 1.0), 1.0)
        pressure += max((exceptions / sample) - expected, 0.0)
    for item in headline_risk or []:
        status = str(item.get("status") or "").strip().lower()
        if status in {"breach", "critical"}:
            pressure += 0.3
        elif status in {"warning", "warn"}:
            pressure += 0.15
    return _clip(pressure, 0.0, 1.0)


def _extract_symbol_microstructure(
    symbol: str,
    microstructure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not microstructure:
        return {}
    symbol_key = str(symbol).upper()
    for item in list(microstructure.get("items") or []):
        if str(item.get("symbol") or "").upper() == symbol_key:
            return dict(item)
    return {}


def _feature_values(
    *,
    symbol: str,
    risk_decision: Mapping[str, Any],
    bundle: Mapping[str, Any] | None,
    validation_summary: Mapping[str, Any] | None,
    microstructure: Mapping[str, Any] | None,
    spread_cost: float | None,
    slippage_points: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    symbol_key = str(symbol).upper()
    pre_trade = dict(risk_decision.get("pre_trade") or {})
    post_trade = dict(risk_decision.get("post_trade") or {})
    pre_var_raw = pre_trade.get("var")
    post_var_raw = post_trade.get("var")
    pre_headroom_raw = pre_trade.get("headroom_var")
    post_headroom_raw = post_trade.get("headroom_var")
    pre_budget_util_raw = pre_trade.get("budget_utilization_var")
    post_budget_util_raw = post_trade.get("budget_utilization_var")
    approved_exposure_raw = risk_decision.get("approved_exposure_change")

    pre_var_available = _has_finite_numeric(pre_var_raw)
    post_var_available = _has_finite_numeric(post_var_raw)
    pre_headroom_available = _has_finite_numeric(pre_headroom_raw)
    post_headroom_available = _has_finite_numeric(post_headroom_raw)
    pre_budget_util_available = _has_finite_numeric(pre_budget_util_raw)
    post_budget_util_available = _has_finite_numeric(post_budget_util_raw)
    approved_exposure_available = _has_finite_numeric(approved_exposure_raw)

    pre_var = _safe_float(pre_var_raw, 0.0)
    post_var = _safe_float(post_var_raw, pre_var)
    pre_headroom = _safe_float(pre_headroom_raw, 0.0)
    post_headroom = _safe_float(post_headroom_raw, pre_headroom)
    approved_exposure = abs(_safe_float(approved_exposure_raw, 0.0))

    momentum_short_term = 0.0
    volatility_recent = 0.0
    momentum_available = False
    volatility_available = False
    sample = None if not bundle else bundle.get("sample")
    if isinstance(sample, pd.DataFrame) and symbol_key in sample.columns:
        series = pd.to_numeric(sample[symbol_key], errors="coerce").dropna().astype(float).tail(64)
        if not series.empty:
            momentum_short_term = _safe_float(series.tail(8).mean(), 0.0)
            momentum_available = True
            if len(series) >= 2:
                volatility_recent = _safe_float(series.tail(16).std(ddof=0), 0.0)
                volatility_available = True

    symbol_micro = _extract_symbol_microstructure(symbol_key, microstructure)
    if not volatility_available:
        micro_vol = symbol_micro.get("realized_vol_30m", microstructure.get("realized_vol_30m") if microstructure else None)
        if _has_finite_numeric(micro_vol):
            volatility_recent = _safe_float(micro_vol, 0.0)
            volatility_available = True

    resolved_spread_cost = spread_cost
    spread_cost_available = _has_finite_numeric(resolved_spread_cost)
    spread_cost_derived = False
    if not spread_cost_available:
        spread_bps = symbol_micro.get("spread_bps")
        if spread_bps is None and microstructure is not None:
            spread_bps = microstructure.get("avg_spread_bps")
        if spread_bps is not None and approved_exposure > 0.0:
            resolved_spread_cost = approved_exposure * _safe_float(spread_bps) / 10_000.0 * 0.5
            spread_cost_available = True
            spread_cost_derived = True
    spread_cost_norm = 0.0
    spread_cost_norm_available = bool(spread_cost_available and approved_exposure > 0.0)
    if spread_cost_norm_available:
        spread_cost_norm = abs(_safe_float(resolved_spread_cost, 0.0)) / max(approved_exposure, 1e-6)

    validation_row = _extract_best_validation_row(validation_summary)
    data_quality = pre_trade.get("data_quality")
    validation_conf_available = bool(validation_row or data_quality)
    validation_conf = _validation_confidence(validation_summary, pre_trade.get("data_quality"))
    headline_risk = list(pre_trade.get("headline_risk") or [])
    exception_pressure_available = bool(validation_row or headline_risk)
    exception_pressure = _exception_pressure(
        validation_summary=validation_summary,
        headline_risk=headline_risk,
    )
    slippage_available = _has_finite_numeric(slippage_points)
    slippage = abs(_safe_float(slippage_points, 0.0))
    headroom_pre_available = bool(
        pre_headroom_available
        and (pre_budget_util_available or abs(pre_headroom) > 1e-12)
    )
    headroom_post_available = bool(
        post_headroom_available
        and (post_budget_util_available or abs(post_headroom) > 1e-12)
    )
    headroom_delta_available = bool(headroom_pre_available and headroom_post_available)
    risk_delta_available = bool(pre_var_available and post_var_available)
    values = {
        "momentum_short_term": _safe_float(momentum_short_term, 0.0),
        "volatility_recent": abs(_safe_float(volatility_recent, 0.0)),
        "headroom_delta": _safe_float(post_headroom - pre_headroom, 0.0),
        "risk_delta": _safe_float(post_var - pre_var, 0.0),
        "validation_confidence": _clip(validation_conf, 0.0, 1.0),
        "exception_pressure": _clip(exception_pressure, 0.0, 1.0),
        "spread_cost_norm": max(_safe_float(spread_cost_norm, 0.0), 0.0),
        "slippage_points": slippage,
    }
    calculations = {
        "pre_var": pre_var,
        "post_var": post_var,
        "pre_headroom_var": pre_headroom,
        "post_headroom_var": post_headroom,
        "resolved_spread_cost": _safe_float(resolved_spread_cost, 0.0),
        "resolved_slippage_points": slippage,
        "approved_exposure_abs": approved_exposure,
        "momentum_window": 8.0,
        "volatility_window": 16.0,
        "feature_available_momentum_short_term": 1.0 if momentum_available else 0.0,
        "feature_available_volatility_recent": 1.0 if volatility_available else 0.0,
        "feature_available_headroom_delta": 1.0 if headroom_delta_available else 0.0,
        "feature_available_risk_delta": 1.0 if risk_delta_available else 0.0,
        "feature_available_validation_confidence": 1.0 if validation_conf_available else 0.0,
        "feature_available_exception_pressure": 1.0 if exception_pressure_available else 0.0,
        "feature_available_spread_cost_norm": 1.0 if spread_cost_norm_available else 0.0,
        "feature_available_slippage_points": 1.0 if slippage_available else 0.0,
        "calc_available_pre_var": 1.0 if pre_var_available else 0.0,
        "calc_available_post_var": 1.0 if post_var_available else 0.0,
        "calc_available_pre_headroom_var": 1.0 if headroom_pre_available else 0.0,
        "calc_available_post_headroom_var": 1.0 if headroom_post_available else 0.0,
        "calc_available_resolved_spread_cost": 1.0 if spread_cost_available else 0.0,
        "calc_available_resolved_slippage_points": 1.0 if slippage_available else 0.0,
        "calc_available_approved_exposure_abs": 1.0 if approved_exposure_available else 0.0,
        "calc_derived_resolved_spread_cost": 1.0 if spread_cost_derived else 0.0,
    }
    return values, calculations


@dataclass
class DecisionAlphaModelState:
    model: Any | None = None
    trained_at: str | None = None
    sample_count: int = 0
    class_balance: float = 0.5
    source_rows: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.source_rows is None:
            self.source_rows = {}


class DecisionAlphaRuntime:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state = DecisionAlphaModelState()

    @property
    def state(self) -> DecisionAlphaModelState:
        return self._state

    def warm_start(self, *, storage: AppStorage | None, portfolio_slug: str | None = None) -> DecisionAlphaModelState:
        state = train_decision_alpha_model(storage=storage, portfolio_slug=portfolio_slug)
        with self._lock:
            self._state = state
        return state


def _model_runtime_metadata(model_state: DecisionAlphaModelState | None) -> dict[str, Any]:
    state = model_state or DecisionAlphaModelState()
    source_rows: dict[str, int] = {}
    for key, value in dict(getattr(state, "source_rows", {}) or {}).items():
        source_rows[str(key)] = _safe_int(value, 0)
    return {
        "trained_model": bool(getattr(state, "model", None) is not None),
        "trained_at": getattr(state, "trained_at", None),
        "sample_count": _safe_int(getattr(state, "sample_count", 0), 0),
        "class_balance": _clip(_safe_float(getattr(state, "class_balance", 0.5), 0.5), 0.0, 1.0),
        "source_rows": source_rows,
    }


def _estimate_csv_data_rows(csv_path: Path) -> int | None:
    try:
        row_count = 0
        with csv_path.open("rb") as handle:
            for _ in handle:
                row_count += 1
    except Exception:
        return None
    return max(row_count - 1, 0)


def _build_backtest_training_frame(
    *,
    storage: AppStorage | None,
    portfolio_slug: str | None,
    max_rows: int = MAX_BACKTEST_TRAINING_ROWS,
) -> pd.DataFrame:
    if storage is None:
        return pd.DataFrame()
    backtest = storage.latest_backtest_run(portfolio_slug=portfolio_slug)
    if not backtest:
        return pd.DataFrame()
    artifact_id = _safe_int(backtest.get("artifact_id"), 0)
    artifact = storage.artifact_by_id(artifact_id) if artifact_id > 0 else None
    if not artifact:
        return pd.DataFrame()
    csv_path = Path(str(artifact.get("path") or "")).expanduser()
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return pd.DataFrame()
    available_columns = {str(column) for column in header.columns}
    selected_columns = [
        column
        for column in ("pnl", "net_total", "strategy_pnl", "daily_pnl", "close", "var_hist", "hist_var", "var")
        if column in available_columns
    ]
    if not selected_columns:
        return pd.DataFrame()

    read_kwargs: dict[str, Any] = {
        "usecols": selected_columns,
    }
    data_rows = _estimate_csv_data_rows(csv_path)
    bounded_rows = max(_safe_int(max_rows, MAX_BACKTEST_TRAINING_ROWS), 2_000)
    if data_rows is not None and data_rows > bounded_rows:
        read_kwargs["skiprows"] = range(1, data_rows - bounded_rows + 1)

    try:
        frame = pd.read_csv(csv_path, **read_kwargs)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    if len(frame) > bounded_rows:
        frame = frame.tail(bounded_rows).copy()

    pnl_column = next(
        (
            candidate
            for candidate in ("pnl", "net_total", "strategy_pnl", "daily_pnl")
            if candidate in frame.columns
        ),
        None,
    )
    if pnl_column is None and "close" in frame.columns:
        close_series = pd.to_numeric(frame["close"], errors="coerce")
        pnl_series = close_series.diff()
    elif pnl_column is not None:
        pnl_series = pd.to_numeric(frame[pnl_column], errors="coerce")
    else:
        return pd.DataFrame()

    var_column = next((candidate for candidate in ("var_hist", "hist_var", "var") if candidate in frame.columns), None)
    if var_column is None:
        var_ref = pnl_series.rolling(20, min_periods=5).std(ddof=0).abs().fillna(0.0)
    else:
        var_ref = pd.to_numeric(frame[var_column], errors="coerce").abs().fillna(0.0)

    training = pd.DataFrame()
    training["momentum_short_term"] = pnl_series.rolling(6, min_periods=3).mean().shift(1)
    training["volatility_recent"] = pnl_series.rolling(12, min_periods=4).std(ddof=0).shift(1)
    training["headroom_delta"] = (-var_ref).diff().shift(1)
    training["risk_delta"] = var_ref.diff().shift(1)
    training["validation_confidence"] = 0.58
    exceptions = (pnl_series < -var_ref.abs()).astype(float)
    training["exception_pressure"] = exceptions.rolling(20, min_periods=3).mean().shift(1)
    training["spread_cost_norm"] = 0.0
    training["slippage_points"] = 0.0
    training["target"] = (pnl_series.shift(-1) > 0.0).astype(float)
    training = training.replace([np.inf, -np.inf], np.nan).dropna()
    if training.empty:
        return pd.DataFrame()
    training["target"] = training["target"].astype(int)
    return training


def _fill_realized_pnl(fill: Mapping[str, Any]) -> float:
    pnl = 0.0
    for field in ("profit", "commission", "swap", "fee"):
        pnl += _safe_float(fill.get(field), 0.0)
    return float(pnl)


def _build_execution_training_frame(
    *,
    storage: AppStorage | None,
    portfolio_slug: str | None,
    limit: int,
) -> pd.DataFrame:
    if storage is None:
        return pd.DataFrame()
    executions = storage.recent_execution_results(limit=max(int(limit), 200), portfolio_slug=portfolio_slug)
    if not executions:
        return pd.DataFrame()
    fills = storage.recent_execution_fills(limit=max(int(limit) * 3, 600), portfolio_slug=portfolio_slug)

    pnl_by_execution: dict[int, float] = {}
    for fill in fills:
        execution_id = _safe_int(fill.get("execution_result_id"), 0)
        if execution_id <= 0:
            continue
        pnl_by_execution[execution_id] = pnl_by_execution.get(execution_id, 0.0) + _fill_realized_pnl(fill)

    rows: list[dict[str, float | int]] = []
    for record in executions:
        execution_id = _safe_int(record.get("id"), 0)
        if execution_id <= 0 or execution_id not in pnl_by_execution:
            continue
        status = str(record.get("status") or "").strip().upper()
        if status in {"PREVIEW", "FAILED", "BLOCKED", "REJECTED"}:
            continue
        risk_decision = dict(record.get("risk_decision") or {})
        intelligence = dict(risk_decision.get("decision_intelligence") or {})
        pre_trade = dict(risk_decision.get("pre_trade") or {})
        post_trade = dict(risk_decision.get("post_trade") or {})
        approved = _safe_float(record.get("approved_exposure_change"), 0.0)
        requested = _safe_float(record.get("requested_exposure_change"), 0.0)
        slippage = abs(_safe_float(record.get("slippage_points"), 0.0))
        spread_cost = _safe_float(
            dict(intelligence.get("calculations") or {}).get("resolved_spread_cost"),
            0.0,
        )
        spread_cost_norm = 0.0 if abs(approved) <= 1e-6 else abs(spread_cost) / max(abs(approved), 1e-6)
        align = 0.0
        if abs(requested) > 1e-9:
            align = approved / requested
        exception_pressure = 0.0
        reconciliation_status = str(record.get("reconciliation_status") or "").strip().lower()
        if reconciliation_status in {"partial_fill", "pending_broker", "rejected_by_broker"}:
            exception_pressure = 0.35
        if status == "PLACED":
            exception_pressure += 0.15
        rows.append(
            {
                "momentum_short_term": _clip(align, -2.0, 2.0),
                "volatility_recent": min(slippage / 20.0, 3.0),
                "headroom_delta": _safe_float(post_trade.get("headroom_var"), 0.0)
                - _safe_float(pre_trade.get("headroom_var"), 0.0),
                "risk_delta": _safe_float(post_trade.get("var"), 0.0) - _safe_float(pre_trade.get("var"), 0.0),
                "validation_confidence": _clip(_safe_float(intelligence.get("confidence"), 0.5), 0.0, 1.0),
                "exception_pressure": _clip(exception_pressure, 0.0, 1.0),
                "spread_cost_norm": max(spread_cost_norm, 0.0),
                "slippage_points": slippage,
                "target": 1 if pnl_by_execution.get(execution_id, 0.0) > 0.0 else 0,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def train_decision_alpha_model(
    *,
    storage: AppStorage | None,
    portfolio_slug: str | None = None,
    lookback_limit: int = 800,
) -> DecisionAlphaModelState:
    backtest_frame = _build_backtest_training_frame(storage=storage, portfolio_slug=portfolio_slug)
    execution_frame = _build_execution_training_frame(
        storage=storage,
        portfolio_slug=portfolio_slug,
        limit=lookback_limit,
    )
    source_rows = {
        "backtest": int(len(backtest_frame)),
        "execution": int(len(execution_frame)),
    }
    if backtest_frame.empty and execution_frame.empty:
        return DecisionAlphaModelState(
            model=None,
            trained_at=_utcnow_iso(),
            sample_count=0,
            class_balance=0.5,
            source_rows=source_rows,
        )

    dataset = pd.concat([backtest_frame, execution_frame], ignore_index=True, sort=False)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURE_COLUMNS) + ["target"])
    if dataset.empty:
        return DecisionAlphaModelState(
            model=None,
            trained_at=_utcnow_iso(),
            sample_count=0,
            class_balance=0.5,
            source_rows=source_rows,
        )
    dataset = dataset.tail(max(int(lookback_limit), 100)).copy()
    for feature in FEATURE_COLUMNS:
        dataset[feature] = pd.to_numeric(dataset[feature], errors="coerce").fillna(0.0).astype(float)
    dataset["target"] = pd.to_numeric(dataset["target"], errors="coerce").fillna(0).astype(int)
    class_balance = _clip(float(dataset["target"].mean()), 0.01, 0.99)
    if len(dataset) < 24 or dataset["target"].nunique() < 2 or not SKLEARN_AVAILABLE:
        return DecisionAlphaModelState(
            model=None,
            trained_at=_utcnow_iso(),
            sample_count=int(len(dataset)),
            class_balance=class_balance,
            source_rows=source_rows,
        )

    try:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        model.fit(dataset.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float), dataset["target"].to_numpy(dtype=int))
        return DecisionAlphaModelState(
            model=model,
            trained_at=_utcnow_iso(),
            sample_count=int(len(dataset)),
            class_balance=class_balance,
            source_rows=source_rows,
        )
    except Exception:
        return DecisionAlphaModelState(
            model=None,
            trained_at=_utcnow_iso(),
            sample_count=int(len(dataset)),
            class_balance=class_balance,
            source_rows=source_rows,
        )


def _predict_probability(features: Mapping[str, float], model_state: DecisionAlphaModelState | None) -> float:
    fallback_base = 0.5 if model_state is None else _clip(model_state.class_balance, 0.01, 0.99)
    if model_state is None or model_state.model is None:
        logit = math.log(fallback_base / (1.0 - fallback_base))
        for name, weight in _HEURISTIC_WEIGHTS.items():
            value = _safe_float(features.get(name), 0.0)
            logit += weight * value
        return _clip(_sigmoid(logit), 0.01, 0.99)

    try:
        vector = np.array([_safe_float(features.get(name), 0.0) for name in FEATURE_COLUMNS], dtype=float).reshape(1, -1)
        probabilities = model_state.model.predict_proba(vector)
        return _clip(_safe_float(probabilities[0][1], fallback_base), 0.01, 0.99)
    except Exception:
        return fallback_base


def _feature_contributions(
    *,
    features: Mapping[str, float],
    model_state: DecisionAlphaModelState | None,
) -> dict[str, float]:
    if model_state is not None and model_state.model is not None:
        try:
            scaler = model_state.model.named_steps.get("scaler")
            classifier = model_state.model.named_steps.get("classifier")
            if classifier is not None and hasattr(classifier, "coef_"):
                raw_vector = np.array([_safe_float(features.get(name), 0.0) for name in FEATURE_COLUMNS], dtype=float)
                if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    scaled = (raw_vector - scaler.mean_) / np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
                else:
                    scaled = raw_vector
                coef = np.array(classifier.coef_[0], dtype=float)
                return {
                    name: _safe_float(coef[index] * scaled[index], 0.0)
                    for index, name in enumerate(FEATURE_COLUMNS)
                }
        except Exception:
            pass
    return {name: _safe_float(_HEURISTIC_WEIGHTS.get(name, 0.0) * features.get(name, 0.0), 0.0) for name in FEATURE_COLUMNS}


def compute_decision_alpha(
    *,
    symbol: str,
    risk_decision: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
    validation_summary: Mapping[str, Any] | None = None,
    microstructure: Mapping[str, Any] | None = None,
    spread_cost: float | None = None,
    slippage_points: float | None = None,
    model_state: DecisionAlphaModelState | None = None,
) -> dict[str, Any]:
    normalized_symbol = str(symbol).upper()
    feature_values, calculations = _feature_values(
        symbol=normalized_symbol,
        risk_decision=risk_decision,
        bundle=bundle,
        validation_summary=validation_summary,
        microstructure=microstructure,
        spread_cost=spread_cost,
        slippage_points=slippage_points,
    )
    probability_up = _predict_probability(feature_values, model_state)
    score = _clip((probability_up - 0.5) * 200.0, -100.0, 100.0)
    if score >= 15.0:
        signal = "BUY"
    elif score <= -15.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = _clip(
        0.2
        + 0.55 * abs(score) / 100.0
        + 0.25 * _safe_float(feature_values.get("validation_confidence"), 0.5)
        - 0.2 * _safe_float(feature_values.get("exception_pressure"), 0.0),
        0.0,
        1.0,
    )
    # Keep fallback mode conservative when no trained model is available.
    if model_state is None or model_state.model is None:
        confidence = min(confidence, 0.72)
    spread_slippage_penalty = _clip(
        3.0 * _safe_float(feature_values.get("spread_cost_norm"), 0.0)
        + _safe_float(feature_values.get("slippage_points"), 0.0) / 120.0,
        0.0,
        0.8,
    )
    size_multiplier = _clip(confidence * (1.0 - spread_slippage_penalty), 0.0, 1.0)
    if signal == "HOLD":
        size_multiplier = 0.0

    risk_verdict = str(risk_decision.get("decision") or "").strip().upper()
    guardrail_applied = False
    if risk_verdict == "REJECT":
        guardrail_applied = True
        signal = "HOLD"
        size_multiplier = 0.0
        score = min(score, 0.0)
    elif risk_verdict == "REDUCE":
        size_multiplier = min(size_multiplier, 0.65)

    feature_availability = {
        name: _safe_float(calculations.get(f"feature_available_{name}"), 1.0) >= 0.5
        for name in FEATURE_COLUMNS
    }
    contributions = _feature_contributions(features=feature_values, model_state=model_state)
    for name, available in feature_availability.items():
        if not available:
            contributions[name] = 0.0
    top_driver_keys = [
        key
        for key in sorted(contributions.keys(), key=lambda name: abs(contributions[name]), reverse=True)
        if abs(contributions[key]) > 1e-9
    ][:3]
    top_drivers = [f"{_FEATURE_LABELS.get(name, name)}:{contributions[name]:+.2f}" for name in top_driver_keys]
    runtime_meta = _model_runtime_metadata(model_state)
    feature_availability_payload = {
        name: bool(_safe_float(calculations.get(f"feature_available_{name}"), 1.0) >= 0.5)
        for name in FEATURE_COLUMNS
    }
    calculation_availability_payload = {
        key[len("calc_available_") :]: bool(_safe_float(value, 0.0) >= 0.5)
        for key, value in calculations.items()
        if str(key).startswith("calc_available_")
    }

    return {
        "signal": signal,
        "score": _clip(score, -100.0, 100.0),
        "confidence": _clip(confidence, 0.0, 1.0),
        "size_multiplier": _clip(size_multiplier, 0.0, 1.0),
        "top_drivers": top_drivers,
        "model_version": DECISION_ALPHA_VERSION,
        "guardrail_applied": guardrail_applied,
        "model_runtime": runtime_meta,
        "features": {name: _safe_float(feature_values.get(name), 0.0) for name in FEATURE_COLUMNS},
        "feature_contributions": {name: _safe_float(contributions.get(name), 0.0) for name in FEATURE_COLUMNS},
        "feature_availability": feature_availability_payload,
        "calculation_availability": calculation_availability_payload,
        "calculations": {
            **{key: _safe_float(value, 0.0) for key, value in calculations.items()},
            "probability_up": probability_up,
            "trained_model_active": 1.0 if runtime_meta.get("trained_model") else 0.0,
            "trained_sample_count": _safe_int(runtime_meta.get("sample_count"), 0),
            "training_backtest_rows": _safe_int(dict(runtime_meta.get("source_rows") or {}).get("backtest"), 0),
            "training_execution_rows": _safe_int(dict(runtime_meta.get("source_rows") or {}).get("execution"), 0),
        },
    }


def _decision_timestamp(record: Mapping[str, Any]) -> str:
    for key in ("created_at", "time_utc"):
        raw = record.get(key)
        if raw in {None, "", "null"}:
            continue
        try:
            parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        except Exception:
            parsed = None
        if parsed is None or pd.isna(parsed):
            continue
        return parsed.to_pydatetime().astimezone(timezone.utc).isoformat()
    return _utcnow_iso()


def replay_decision_alpha(
    *,
    storage: AppStorage | None = None,
    portfolio_slug: str | None = None,
    limit: int = 200,
    lookback_days: int | None = None,
    decision_rows: Sequence[Mapping[str, Any]] | None = None,
    execution_rows: Sequence[Mapping[str, Any]] | None = None,
    fill_rows: Sequence[Mapping[str, Any]] | None = None,
    model_state: DecisionAlphaModelState | None = None,
) -> dict[str, Any]:
    normalized_limit = max(int(limit), 1)
    if decision_rows is None:
        if storage is None:
            decision_rows = []
        else:
            decision_rows = storage.recent_decisions(
                limit=max(normalized_limit * 5, 200),
                portfolio_slug=portfolio_slug,
            )
    if execution_rows is None:
        execution_rows = (
            []
            if storage is None
            else storage.recent_execution_results(
                limit=max(normalized_limit * 5, 200),
                portfolio_slug=portfolio_slug,
            )
        )
    if fill_rows is None:
        fill_rows = (
            []
            if storage is None
            else storage.recent_execution_fills(
                limit=max(normalized_limit * 20, 600),
                portfolio_slug=portfolio_slug,
            )
        )

    pnl_by_execution: dict[int, float] = {}
    for fill in fill_rows:
        execution_id = _safe_int(fill.get("execution_result_id"), 0)
        if execution_id <= 0:
            continue
        pnl_by_execution[execution_id] = pnl_by_execution.get(execution_id, 0.0) + _fill_realized_pnl(fill)

    pnl_by_decision: dict[int, float] = {}
    for execution in execution_rows:
        status = str(execution.get("status") or "").strip().upper()
        if status in {"PREVIEW", "BLOCKED", "REJECTED", "FAILED"}:
            continue
        decision_id = _safe_int(execution.get("decision_id"), 0)
        execution_id = _safe_int(execution.get("id"), 0)
        if decision_id <= 0 or execution_id <= 0:
            continue
        realized = pnl_by_execution.get(execution_id)
        if realized is None:
            continue
        pnl_by_decision[decision_id] = pnl_by_decision.get(decision_id, 0.0) + float(realized)

    ordered_decisions = sorted(
        [dict(item) for item in decision_rows],
        key=lambda item: _decision_timestamp(item),
    )
    lookback_cutoff: datetime | None = None
    if lookback_days not in {None, "", "null"}:
        lookback_cutoff = datetime.now(timezone.utc) - timedelta(days=max(_safe_int(lookback_days, 90), 1))
    replay_points: list[dict[str, Any]] = []

    for record in ordered_decisions:
        timestamp = _safe_datetime(_decision_timestamp(record))
        if lookback_cutoff is not None and timestamp is not None and timestamp < lookback_cutoff:
            continue
        decision_id = _safe_int(record.get("id"), 0)
        if decision_id <= 0 or decision_id not in pnl_by_decision:
            continue
        intelligence = dict(record.get("decision_intelligence") or {})
        if not intelligence:
            intelligence = compute_decision_alpha(
                symbol=str(record.get("symbol") or ""),
                risk_decision=record,
                model_state=model_state,
            )
        score = _safe_float(intelligence.get("score"), 0.0)
        realized_pnl = _safe_float(pnl_by_decision.get(decision_id), 0.0)
        comparable = abs(score) > 1e-9 and abs(realized_pnl) > 1e-9
        hit: bool | None = None
        if comparable:
            hit = (score > 0.0 and realized_pnl > 0.0) or (score < 0.0 and realized_pnl < 0.0)
        replay_points.append(
            {
                "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
                "decision_id": decision_id,
                "symbol": str(record.get("symbol") or "").upper(),
                "predicted_score": score,
                "predicted_signal": str(intelligence.get("signal") or "HOLD").upper(),
                "realized_pnl": realized_pnl,
                "hit": hit,
                "cum_pnl": 0.0,
            }
        )
    replay_points = replay_points[-normalized_limit:]
    cumulative_pnl = 0.0
    for point in replay_points:
        cumulative_pnl += _safe_float(point.get("realized_pnl"), 0.0)
        point["cum_pnl"] = cumulative_pnl
    comparable_count = sum(1 for item in replay_points if item.get("hit") is not None)
    hit_count = sum(1 for item in replay_points if item.get("hit") is True)
    hit_rate = 0.0 if comparable_count <= 0 else float(hit_count / comparable_count)
    runtime_meta = _model_runtime_metadata(model_state)
    return {
        "model_version": DECISION_ALPHA_VERSION,
        "generated_at": _utcnow_iso(),
        "portfolio_slug": portfolio_slug,
        "lookback_days": None if lookback_cutoff is None else max(_safe_int(lookback_days, 90), 1),
        "model_runtime": runtime_meta,
        "sample_size": len(replay_points),
        "hit_rate": _clip(hit_rate, 0.0, 1.0),
        "cum_pnl": _safe_float(replay_points[-1]["cum_pnl"], 0.0) if replay_points else 0.0,
        "predicted_vs_realized": replay_points,
        "comparables": comparable_count,
    }


def _build_price_frame(market_bars: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not market_bars:
        return pd.DataFrame(columns=["time_utc", "close"])
    frame = pd.DataFrame(list(market_bars))
    if frame.empty:
        return pd.DataFrame(columns=["time_utc", "close"])
    if "time_utc" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame(columns=["time_utc", "close"])
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["time_utc", "close"]).sort_values("time_utc").reset_index(drop=True)
    return frame


def _prediction_features_from_returns(
    returns: pd.Series,
    *,
    index: int,
) -> dict[str, float]:
    tail = returns.iloc[max(index - 96, 0):index]
    momentum = _safe_float(tail.tail(16).mean(), 0.0) if not tail.empty else 0.0
    volatility = abs(_safe_float(tail.tail(32).std(ddof=0), 0.0)) if not tail.empty else 0.0
    return {
        "momentum_short_term": momentum,
        "volatility_recent": volatility,
        "headroom_delta": 0.0,
        "risk_delta": 0.0,
        "validation_confidence": 0.58,
        "exception_pressure": 0.15,
        "spread_cost_norm": 0.0,
        "slippage_points": 0.0,
    }


def _trajectory_predicted_return(
    *,
    probability_up: float,
    momentum: float,
    volatility: float,
    predicted_path: float,
    previous_actual: float,
) -> float:
    # Non-linear edge damping to reduce long-horizon compounding bias.
    edge = _clip(probability_up - 0.5, -0.49, 0.49)
    damped_edge = math.tanh(edge * TRAJECTORY_EDGE_DAMPING) / TRAJECTORY_EDGE_DAMPING
    directional_component = damped_edge * max(2.0 * volatility, TRAJECTORY_RETURN_FLOOR)
    momentum_component = _clip(momentum * 0.25, -0.0012, 0.0012)
    divergence = 0.0 if previous_actual <= 1e-9 else (predicted_path / previous_actual) - 1.0
    reversion_component = _clip(
        -TRAJECTORY_DIVERGENCE_REVERSION * divergence,
        -0.002,
        0.002,
    )
    dynamic_cap = _clip(3.0 * volatility, TRAJECTORY_RETURN_MIN_CAP, TRAJECTORY_RETURN_MAX_CAP)
    return _clip(
        directional_component + momentum_component + reversion_component,
        -dynamic_cap,
        dynamic_cap,
    )


def backtest_decision_alpha_trajectory(
    *,
    symbol: str,
    lookback_days: int = 90,
    storage: AppStorage | None = None,
    portfolio_slug: str | None = None,
    market_bars: Sequence[Mapping[str, Any]] | None = None,
    model_state: DecisionAlphaModelState | None = None,
) -> dict[str, Any]:
    normalized_symbol = str(symbol).upper()
    lookback = max(_safe_int(lookback_days, 90), 7)
    if market_bars is None:
        if storage is None:
            market_bars = []
        else:
            since = datetime.now(timezone.utc) - timedelta(days=max(lookback + 21, 45))
            market_bars = storage.market_bars(
                symbol=normalized_symbol,
                timeframe="H1",
                since=since,
                limit=max(lookback * 36, 1_500),
            )
            if not market_bars:
                market_bars = storage.market_bars(
                    symbol=normalized_symbol,
                    timeframe="D1",
                    since=since,
                    limit=max(lookback + 45, 600),
                )
    frame = _build_price_frame(market_bars)
    if frame.empty:
        return {
            "model_version": DECISION_ALPHA_VERSION,
            "generated_at": _utcnow_iso(),
            "portfolio_slug": portfolio_slug,
            "symbol": normalized_symbol,
            "lookback_days": lookback,
            "sample_size": 0,
            "hit_rate": 0.0,
            "mean_abs_error": 0.0,
            "predicted_vs_actual": [],
            "model_runtime": _model_runtime_metadata(model_state),
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)
    working = frame[frame["time_utc"] >= cutoff].copy()
    if working.empty or len(working) < 5:
        working = frame.tail(min(len(frame), max(lookback * 8, 96))).copy()
    working = working.reset_index(drop=True)
    returns = working["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if len(working) < 2:
        return {
            "model_version": DECISION_ALPHA_VERSION,
            "generated_at": _utcnow_iso(),
            "portfolio_slug": portfolio_slug,
            "symbol": normalized_symbol,
            "lookback_days": lookback,
            "sample_size": 0,
            "hit_rate": 0.0,
            "mean_abs_error": 0.0,
            "predicted_vs_actual": [],
            "model_runtime": _model_runtime_metadata(model_state),
        }

    predicted_path = _safe_float(working["close"].iloc[0], 1.0)
    points: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    for index in range(1, len(working)):
        prev_actual = _safe_float(working["close"].iloc[index - 1], 1.0)
        features = _prediction_features_from_returns(returns, index=index)
        momentum = _safe_float(features.get("momentum_short_term"), 0.0)
        volatility = max(abs(_safe_float(features.get("volatility_recent"), 0.0)), 1e-6)
        probability_up = _predict_probability(features, model_state)
        predicted_score = _clip((probability_up - 0.5) * 200.0, -100.0, 100.0)
        predicted_return = _trajectory_predicted_return(
            probability_up=probability_up,
            momentum=momentum,
            volatility=volatility,
            predicted_path=predicted_path,
            previous_actual=prev_actual,
        )
        open_loop_path = predicted_path * (1.0 + predicted_return)
        anchored_path = prev_actual * (1.0 + predicted_return)
        predicted_path = max(
            TRAJECTORY_ANCHOR_WEIGHT * open_loop_path
            + (1.0 - TRAJECTORY_ANCHOR_WEIGHT) * anchored_path,
            1e-8,
        )

        actual = _safe_float(working["close"].iloc[index], prev_actual)
        realized_return = 0.0 if prev_actual <= 1e-9 else (actual / prev_actual) - 1.0
        comparable = abs(predicted_return) > 1e-10 and abs(realized_return) > 1e-10
        hit: bool | None = None
        if comparable:
            hit = (predicted_return > 0.0 and realized_return > 0.0) or (predicted_return < 0.0 and realized_return < 0.0)
        absolute_errors.append(abs(predicted_path - actual))
        timestamp = _safe_datetime(working.loc[index, "time_utc"]) or datetime.now(timezone.utc)
        points.append(
            {
                "timestamp": timestamp.isoformat(),
                "predicted_price": _safe_float(predicted_path, actual),
                "actual_price": _safe_float(actual, predicted_path),
                "predicted_return": _safe_float(predicted_return, 0.0),
                "realized_return": _safe_float(realized_return, 0.0),
                "predicted_score": _safe_float(predicted_score, 0.0),
                "hit": hit,
            }
        )

    comparable_count = sum(1 for item in points if item.get("hit") is not None)
    hit_count = sum(1 for item in points if item.get("hit") is True)
    hit_rate = 0.0 if comparable_count <= 0 else float(hit_count / comparable_count)
    mean_abs_error = 0.0 if not absolute_errors else float(sum(absolute_errors) / len(absolute_errors))
    return {
        "model_version": DECISION_ALPHA_VERSION,
        "generated_at": _utcnow_iso(),
        "portfolio_slug": portfolio_slug,
        "symbol": normalized_symbol,
        "lookback_days": lookback,
        "sample_size": len(points),
        "hit_rate": _clip(hit_rate, 0.0, 1.0),
        "mean_abs_error": _safe_float(mean_abs_error, 0.0),
        "predicted_vs_actual": points,
        "model_runtime": _model_runtime_metadata(model_state),
    }


def forecast_decision_alpha(
    *,
    symbol: str,
    horizon_days: int = 5,
    storage: AppStorage | None = None,
    portfolio_slug: str | None = None,
    market_bars: Sequence[Mapping[str, Any]] | None = None,
    model_state: DecisionAlphaModelState | None = None,
) -> dict[str, Any]:
    normalized_symbol = str(symbol).upper()
    horizon = _clip(_safe_int(horizon_days, 5), 1, MAX_DECISION_ALPHA_FORECAST_HORIZON_DAYS)
    if market_bars is None:
        if storage is None:
            market_bars = []
        else:
            # Keep H1 query inside a bounded recent window so the storage
            # read (ordered ASC + LIMIT) still includes the latest prices.
            h1_lookback_days = int(
                max(
                    FORECAST_H1_LOOKBACK_DAYS_MIN,
                    min(int(horizon) * 2, FORECAST_H1_LOOKBACK_DAYS_MAX),
                )
            )
            since = datetime.now(timezone.utc) - timedelta(days=h1_lookback_days)
            market_bars = storage.market_bars(
                symbol=normalized_symbol,
                timeframe="H1",
                since=since,
                limit=8_000,
            )
            if not market_bars:
                d1_lookback_days = int(
                    max(
                        FORECAST_D1_LOOKBACK_DAYS_MIN,
                        min(int(horizon) * 6, FORECAST_D1_LOOKBACK_DAYS_MAX),
                    )
                )
                since = datetime.now(timezone.utc) - timedelta(days=d1_lookback_days)
                market_bars = storage.market_bars(
                    symbol=normalized_symbol,
                    timeframe="D1",
                    since=since,
                    limit=2_000,
                )
    frame = _build_price_frame(market_bars)
    if frame.empty:
        current_price = 1.0
        returns = pd.Series(dtype=float)
    else:
        current_price = _safe_float(frame["close"].iloc[-1], 1.0)
        returns = frame["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna().astype(float)

    recent_returns = returns.tail(240)
    drift = _safe_float(recent_returns.mean(), 0.0)
    volatility = abs(_safe_float(recent_returns.std(ddof=0), 0.0))
    if volatility <= 1e-6:
        volatility = max(abs(drift) * 2.0, 0.002)
    momentum = _safe_float(returns.tail(24).mean(), drift)

    forecast_features = {
        "momentum_short_term": momentum,
        "volatility_recent": volatility,
        "headroom_delta": 0.0,
        "risk_delta": 0.0,
        "validation_confidence": 0.58,
        "exception_pressure": 0.15,
        "spread_cost_norm": 0.0,
        "slippage_points": 0.0,
    }
    probability_up = _predict_probability(forecast_features, model_state)
    score = _clip((probability_up - 0.5) * 200.0, -100.0, 100.0)

    bull_probability = _clip(0.2 + 0.6 * probability_up, 0.15, 0.75)
    bear_probability = _clip(0.2 + 0.6 * (1.0 - probability_up), 0.15, 0.75)
    base_probability = max(1.0 - bull_probability - bear_probability, 0.1)
    total_probability = bull_probability + bear_probability + base_probability
    bull_probability /= total_probability
    bear_probability /= total_probability
    base_probability /= total_probability

    today = datetime.now(timezone.utc).date()

    def _scenario_path(*, name: str, drift_adjustment: float, vol_adjustment: float, probability: float) -> dict[str, Any]:
        mu = drift + drift_adjustment
        sigma = max(volatility * vol_adjustment, 1e-5)
        direction_bias = 1.0 if name == "bull" else (-1.0 if name == "bear" else 0.0)
        points: list[dict[str, Any]] = []
        for day in range(int(horizon) + 1):
            time_component = (mu - 0.5 * sigma * sigma) * day
            volatility_component = direction_bias * sigma * math.sqrt(max(day, 1)) * 0.15
            projected = current_price * math.exp(time_component + volatility_component)
            points.append(
                {
                    "day": int(day),
                    "date": (today + timedelta(days=day)).isoformat(),
                    "price": _safe_float(projected, current_price),
                }
            )
        projected_return = 0.0 if current_price <= 1e-9 else (points[-1]["price"] / current_price) - 1.0
        return {
            "name": name,
            "probability": _clip(probability, 0.0, 1.0),
            "projected_return": projected_return,
            "path": points,
        }

    scenarios = [
        _scenario_path(name="bear", drift_adjustment=-0.6 * volatility, vol_adjustment=1.25, probability=bear_probability),
        _scenario_path(name="base", drift_adjustment=0.0, vol_adjustment=1.0, probability=base_probability),
        _scenario_path(name="bull", drift_adjustment=0.6 * volatility, vol_adjustment=0.85, probability=bull_probability),
    ]
    return {
        "model_version": DECISION_ALPHA_VERSION,
        "generated_at": _utcnow_iso(),
        "symbol": normalized_symbol,
        "horizon_days": int(horizon),
        "current_price": current_price,
        "score": score,
        "probability_up": probability_up,
        "model_runtime": _model_runtime_metadata(model_state),
        "features": forecast_features,
        "scenarios": scenarios,
    }


def portfolio_decision_alpha_forecast(
    *,
    symbols: Sequence[str],
    exposures: Mapping[str, Any] | None = None,
    horizon_days: int = 150,
    storage: AppStorage | None = None,
    portfolio_slug: str | None = None,
    model_state: DecisionAlphaModelState | None = None,
) -> dict[str, Any]:
    horizon = _safe_int(horizon_days, 150)
    horizon = max(7, min(horizon, MAX_DECISION_ALPHA_FORECAST_HORIZON_DAYS))
    normalized_symbols: list[str] = []
    seen: set[str] = set()
    for raw in list(symbols or []):
        key = str(raw or "").strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized_symbols.append(key)

    normalized_exposure = {
        str(key).upper(): _safe_float(value, 0.0)
        for key, value in dict(exposures or {}).items()
        if str(key).strip()
    }
    if not normalized_symbols:
        normalized_symbols = list(normalized_exposure.keys())

    if not normalized_symbols:
        return {
            "model_version": DECISION_ALPHA_VERSION,
            "generated_at": _utcnow_iso(),
            "portfolio_slug": portfolio_slug,
            "horizon_days": horizon,
            "symbol_count": 0,
            "current_notional_eur": 0.0,
            "symbols": [],
            "pnl_scenarios": [],
            "model_runtime": _model_runtime_metadata(model_state),
        }

    symbol_forecasts: list[dict[str, Any]] = []
    scenario_prob_totals: dict[str, float] = {}
    scenario_prob_weight: dict[str, float] = {}
    total_notional = sum(abs(normalized_exposure.get(symbol, 0.0)) for symbol in normalized_symbols)
    if total_notional <= 1e-9:
        total_notional = float(len(normalized_symbols))

    for symbol in normalized_symbols:
        exposure = normalized_exposure.get(symbol, 0.0)
        if abs(exposure) <= 1e-9:
            exposure = total_notional / max(len(normalized_symbols), 1)
        weight = abs(exposure) / max(total_notional, 1e-9)
        forecast = forecast_decision_alpha(
            symbol=symbol,
            horizon_days=horizon,
            storage=storage,
            portfolio_slug=portfolio_slug,
            model_state=model_state,
        )
        symbol_forecasts.append(
            {
                "symbol": symbol,
                "exposure_eur": _safe_float(exposure, 0.0),
                "weight": _clip(weight, 0.0, 1.0),
                "forecast": forecast,
            }
        )
        for scenario in list(forecast.get("scenarios") or []):
            name = str(scenario.get("name") or "").lower()
            probability = _clip(_safe_float(scenario.get("probability"), 0.0), 0.0, 1.0)
            scenario_prob_totals[name] = scenario_prob_totals.get(name, 0.0) + weight * probability
            scenario_prob_weight[name] = scenario_prob_weight.get(name, 0.0) + weight

    def _scenario_price(forecast_payload: Mapping[str, Any], *, name: str, day: int) -> float:
        scenarios = list(forecast_payload.get("scenarios") or [])
        for scenario in scenarios:
            if str(scenario.get("name") or "").lower() != name:
                continue
            path = list(scenario.get("path") or [])
            if not path:
                return _safe_float(forecast_payload.get("current_price"), 1.0)
            bounded_day = min(max(int(day), 0), len(path) - 1)
            return _safe_float(dict(path[bounded_day]).get("price"), _safe_float(forecast_payload.get("current_price"), 1.0))
        return _safe_float(forecast_payload.get("current_price"), 1.0)

    def _scenario_path(name: str) -> dict[str, Any]:
        points: list[dict[str, Any]] = []
        probability = 0.0
        if scenario_prob_weight.get(name, 0.0) > 1e-9:
            probability = scenario_prob_totals.get(name, 0.0) / scenario_prob_weight[name]
        start_date = datetime.now(timezone.utc).date()
        for day in range(horizon + 1):
            pnl = 0.0
            for item in symbol_forecasts:
                forecast_payload = dict(item.get("forecast") or {})
                current_price = max(_safe_float(forecast_payload.get("current_price"), 1.0), 1e-9)
                exposure_eur = _safe_float(item.get("exposure_eur"), 0.0)
                scenario_price = _scenario_price(forecast_payload, name=name, day=day)
                projected_return = (scenario_price / current_price) - 1.0
                pnl += exposure_eur * projected_return
            points.append(
                {
                    "day": int(day),
                    "date": (start_date + timedelta(days=day)).isoformat(),
                    "pnl": _safe_float(pnl, 0.0),
                }
            )
        projected_return = 0.0 if total_notional <= 1e-9 else _safe_float(points[-1]["pnl"], 0.0) / total_notional
        return {
            "name": name,
            "probability": _clip(probability, 0.0, 1.0),
            "projected_return": projected_return,
            "path": points,
        }

    pnl_scenarios = [_scenario_path("bear"), _scenario_path("base"), _scenario_path("bull")]
    probability_sum = sum(_safe_float(item.get("probability"), 0.0) for item in pnl_scenarios)
    if probability_sum > 1e-9:
        for item in pnl_scenarios:
            item["probability"] = _clip(_safe_float(item.get("probability"), 0.0) / probability_sum, 0.0, 1.0)

    return {
        "model_version": DECISION_ALPHA_VERSION,
        "generated_at": _utcnow_iso(),
        "portfolio_slug": portfolio_slug,
        "horizon_days": horizon,
        "symbol_count": len(symbol_forecasts),
        "current_notional_eur": _safe_float(total_notional, 0.0),
        "symbols": symbol_forecasts,
        "pnl_scenarios": pnl_scenarios,
        "model_runtime": _model_runtime_metadata(model_state),
    }
