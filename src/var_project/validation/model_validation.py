from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from var_project.core.alpha_codec import alpha_lookup_tokens, decode_alpha_token, encode_alpha_token
from var_project.core.model_registry import infer_model_names_from_columns

P_VALUE_SIGNIFICANCE_LEVEL = 0.05
VALIDATION_STATUS_PASS = "PASS"
VALIDATION_STATUS_WARN = "WARN"
VALIDATION_STATUS_FAIL = "FAIL"
ES_ACERBI_MIN_OBSERVATIONS = 60
ES_ACERBI_WARN_PVALUE = 0.05
ES_ACERBI_BREACH_PVALUE = 0.01


@dataclass(frozen=True)
class BacktestModelValidation:
    model: str
    n: int
    exceptions: int
    expected_rate: float
    actual_rate: float
    lr_uc: float
    p_uc: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float
    exceptions_last_250: int
    traffic_light: Optional[str]
    score: float
    es_tail_observations: int = 0
    es_shortfall_ratio: float | None = None
    es_breach_rate: float | None = None
    es_acerbi_stat: float | None = None
    es_acerbi_p_value: float | None = None
    es_acerbi_observations: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BacktestModelValidation":
        return cls(
            model=str(payload["model"]),
            n=int(payload["n"]),
            exceptions=int(payload["exceptions"]),
            expected_rate=float(payload["expected_rate"]),
            actual_rate=float(payload["actual_rate"]),
            lr_uc=float(payload["lr_uc"]),
            p_uc=float(payload["p_uc"]),
            lr_ind=float(payload["lr_ind"]),
            p_ind=float(payload["p_ind"]),
            lr_cc=float(payload["lr_cc"]),
            p_cc=float(payload["p_cc"]),
            exceptions_last_250=int(payload["exceptions_last_250"]),
            traffic_light=payload.get("traffic_light"),
            score=float(payload["score"]),
            es_tail_observations=int(payload.get("es_tail_observations") or 0),
            es_shortfall_ratio=(
                None
                if payload.get("es_shortfall_ratio") is None
                else float(payload.get("es_shortfall_ratio"))
            ),
            es_breach_rate=(
                None
                if payload.get("es_breach_rate") is None
                else float(payload.get("es_breach_rate"))
            ),
            es_acerbi_stat=(
                None
                if payload.get("es_acerbi_stat") is None and payload.get("es_acerbi_z") is None
                else float(payload.get("es_acerbi_stat", payload.get("es_acerbi_z")))
            ),
            es_acerbi_p_value=(
                None
                if payload.get("es_acerbi_p_value") is None and payload.get("es_acerbi_p") is None
                else float(payload.get("es_acerbi_p_value", payload.get("es_acerbi_p")))
            ),
            es_acerbi_observations=int(
                payload.get("es_acerbi_observations", payload.get("es_acerbi_n") or 0) or 0
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationSummary:
    alpha: float
    expected_rate: float
    model_results: Dict[str, BacktestModelValidation]
    best_model: Optional[str]
    champion_model_live: Optional[str] = None
    champion_model_reporting: Optional[str] = None
    surface: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ValidationSummary":
        models_payload = dict(payload.get("models") or payload.get("model_results") or {})
        return cls(
            alpha=float(payload["alpha"]),
            expected_rate=float(payload["expected_rate"]),
            model_results={name: BacktestModelValidation.from_dict(dict(model_payload)) for name, model_payload in models_payload.items()},
            best_model=payload.get("best_model"),
            champion_model_live=payload.get("champion_model_live"),
            champion_model_reporting=payload.get("champion_model_reporting"),
            surface=None if payload.get("surface") is None else dict(payload.get("surface") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "alpha": self.alpha,
            "expected_rate": self.expected_rate,
            "best_model": self.best_model,
            "models": {name: result.to_dict() for name, result in self.model_results.items()},
            "champion_model_live": self.champion_model_live,
            "champion_model_reporting": self.champion_model_reporting,
        }
        if self.surface is not None:
            payload["surface"] = dict(self.surface)
        return payload


@dataclass(frozen=True)
class ValidationSurfacePoint:
    model: str
    alpha: float
    horizon_days: int
    n: int
    exceptions: int
    expected_rate: float
    actual_rate: float
    p_uc: float
    p_ind: float
    p_cc: float
    traffic_light: Optional[str]
    score: float
    coverage_pass: bool
    independence_pass: bool | None = None
    conditional_pass: bool | None = None
    statistical_status: str = VALIDATION_STATUS_WARN

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankedModelComparison:
    rank: int
    model: str
    score: float
    actual_rate: float
    expected_rate: float
    exceptions: int
    p_uc: float
    p_ind: float
    p_cc: float
    traffic_light: Optional[str]
    current_var: float | None = None
    current_es: float | None = None
    es_acerbi_status: str = "N/A"
    es_acerbi_p_value: float | None = None
    es_acerbi_observations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChampionChallengerSummary:
    alpha: float
    champion_model: Optional[str]
    champion_model_reporting: Optional[str]
    challenger_model: Optional[str]
    score_gap: float | None
    rate_gap: float | None
    exception_gap: int | None
    current_var_gap: float | None
    current_es_gap: float | None
    ranking: list[RankedModelComparison]
    validation_surface: dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "champion_model": self.champion_model,
            "champion_model_reporting": self.champion_model_reporting,
            "challenger_model": self.challenger_model,
            "score_gap": self.score_gap,
            "rate_gap": self.rate_gap,
            "exception_gap": self.exception_gap,
            "current_var_gap": self.current_var_gap,
            "current_es_gap": self.current_es_gap,
            "ranking": [row.to_dict() for row in self.ranking],
            "validation_surface": self.validation_surface,
        }


def _clip_probability(value: float | None, fallback: float = 0.0) -> float:
    if value is None or np.isnan(value):
        return fallback
    return float(min(max(value, 0.0), 1.0))


def _infer_models(df: pd.DataFrame) -> list[str]:
    return infer_model_names_from_columns(df.columns)


def _score_model(
    actual_rate: float,
    expected_rate: float,
    p_uc: float,
    p_ind: float,
    p_cc: float,
) -> float:
    denom = max(expected_rate, 1e-9)
    accuracy = max(0.0, 1.0 - abs(actual_rate - expected_rate) / denom)
    score = (
        0.45 * accuracy
        + 0.20 * _clip_probability(p_uc)
        + 0.15 * _clip_probability(p_ind, fallback=0.5)
        + 0.20 * _clip_probability(p_cc, fallback=0.5)
    )
    return round(100.0 * score, 2)


def _rank_key(item: BacktestModelValidation) -> tuple[int, float, float, int, float]:
    acerbi_bucket, acerbi_p_value = _es_acerbi_rank_bucket(item)
    return (
        acerbi_bucket,
        -item.score,
        abs(item.actual_rate - item.expected_rate),
        item.exceptions,
        -acerbi_p_value,
    )


def _es_acerbi_rank_bucket(item: BacktestModelValidation) -> tuple[int, float]:
    observations = int(item.es_acerbi_observations or 0)
    p_value = pd.to_numeric(item.es_acerbi_p_value, errors="coerce")
    if observations < ES_ACERBI_MIN_OBSERVATIONS or pd.isna(p_value):
        return 1, 0.0
    clipped = _clip_probability(float(p_value))
    if clipped <= ES_ACERBI_BREACH_PVALUE:
        return 3, clipped
    if clipped <= ES_ACERBI_WARN_PVALUE:
        return 2, clipped
    return 0, clipped


def _es_acerbi_status(item: BacktestModelValidation) -> str:
    bucket, _ = _es_acerbi_rank_bucket(item)
    if bucket == 0:
        return VALIDATION_STATUS_PASS
    if bucket == 2:
        return VALIDATION_STATUS_WARN
    if bucket == 3:
        return VALIDATION_STATUS_FAIL
    return "N/A"


def _finite_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _es_backtest_diagnostics(
    *,
    losses: pd.Series,
    var_series: pd.Series,
    es_series: pd.Series,
) -> tuple[int, float | None, float | None]:
    valid = losses.notna() & var_series.notna() & es_series.notna()
    if not bool(valid.any()):
        return 0, None, None
    tail_mask = valid & (losses > var_series)
    tail_count = int(tail_mask.sum())
    if tail_count <= 0:
        return 0, None, None
    tail_losses = losses[tail_mask].astype(float)
    tail_es = es_series[tail_mask].astype(float)
    es_mean = float(tail_es.mean())
    ratio = None if es_mean <= 1e-12 else float(tail_losses.mean() / es_mean)
    breach_rate = float((tail_losses > tail_es).mean()) if tail_count > 0 else None
    return tail_count, ratio, breach_rate


def _es_acerbi_backtest(
    *,
    losses: pd.Series,
    var_series: pd.Series,
    es_series: pd.Series,
    alpha: float,
) -> tuple[int, float | None, float | None]:
    """
    Acerbi-Szekely style ES diagnostic (approximate z-test).

    We evaluate the centered series:
        Z_t = 1_{L_t > VaR_t} * (L_t / ES_t) - (1 - alpha)
    Under a correctly calibrated VaR/ES pair, E[Z_t] ~= 0.
    """
    valid = losses.notna() & var_series.notna() & es_series.notna()
    if not bool(valid.any()):
        return 0, None, None

    losses_values = losses[valid].astype(float).to_numpy()
    var_values = var_series[valid].astype(float).to_numpy()
    es_values = es_series[valid].astype(float).to_numpy()

    finite = (
        np.isfinite(losses_values)
        & np.isfinite(var_values)
        & np.isfinite(es_values)
        & (es_values > 1e-12)
    )
    if not bool(np.any(finite)):
        return 0, None, None

    losses_values = losses_values[finite]
    var_values = var_values[finite]
    es_values = es_values[finite]
    observation_count = int(len(losses_values))
    if observation_count < 3:
        return observation_count, None, None

    exceedance = (losses_values > var_values).astype(float)
    centered = exceedance * (losses_values / es_values) - (1.0 - float(alpha))

    mean_centered = float(np.mean(centered))
    std_centered = float(np.std(centered, ddof=1))
    if not np.isfinite(std_centered) or std_centered <= 1e-12:
        return observation_count, None, None

    standard_error = std_centered / float(np.sqrt(observation_count))
    if standard_error <= 1e-12:
        return observation_count, None, None

    stat = float(mean_centered / standard_error)
    p_value = float(2.0 * norm.sf(abs(stat)))
    if not np.isfinite(stat) or not np.isfinite(p_value):
        return observation_count, None, None

    return observation_count, stat, p_value


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def kupiec_uc(exc: pd.Series, alpha: float) -> dict:
    e = exc.dropna().astype(int).to_numpy()
    n = int(len(e))
    x = int(e.sum())

    p = 1.0 - float(alpha)
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)

    phat = x / n if n > 0 else 0.0
    phat = min(max(phat, eps), 1 - eps)

    ll_h0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_h1 = (n - x) * np.log(1 - phat) + x * np.log(phat)

    lr_uc = float(-2.0 * (ll_h0 - ll_h1))
    p_value = float(chi2.sf(lr_uc, df=1))

    return {"n": n, "x": x, "rate": x / n if n else 0.0, "LR_uc": lr_uc, "p_uc": p_value}


def christoffersen_ind(exc: pd.Series) -> dict:
    e = exc.dropna().astype(int).to_numpy()
    if len(e) < 2:
        return {"LR_ind": np.nan, "p_ind": np.nan, "n00": 0, "n01": 0, "n10": 0, "n11": 0}

    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(e)):
        prev, cur = e[i - 1], e[i]
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    eps = 1e-12
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    pi0 = min(max(pi0, eps), 1 - eps)
    pi1 = min(max(pi1, eps), 1 - eps)
    pi = min(max(pi, eps), 1 - eps)

    ll_indep = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_markov = (
        n00 * np.log(1 - pi0) + n01 * np.log(pi0) +
        n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    )

    lr_ind = float(-2.0 * (ll_indep - ll_markov))
    p_value = float(chi2.sf(lr_ind, df=1))

    return {"LR_ind": lr_ind, "p_ind": p_value, "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def conditional_coverage(exc: pd.Series, alpha: float) -> dict:
    uc = kupiec_uc(exc, alpha)
    ind = christoffersen_ind(exc)

    if np.isnan(ind["LR_ind"]):
        return {"LR_cc": np.nan, "p_cc": np.nan}

    lr_cc = float(uc["LR_uc"] + ind["LR_ind"])
    p_cc = float(chi2.sf(lr_cc, df=2))
    return {"LR_cc": lr_cc, "p_cc": p_cc}


def _traffic_light(last_250_exceptions: int, alpha: float) -> Optional[str]:
    if abs(alpha - 0.99) > 1e-9:
        return None
    if last_250_exceptions <= 4:
        return "GREEN"
    if last_250_exceptions <= 9:
        return "YELLOW"
    return "RED"


_SURFACE_COLUMN_RE = re.compile(
    r"^(?P<prefix>var|es|exc)_(?P<model>[a-z0-9]+)_a(?P<alpha>\d+(?:p\d+)?)_h(?P<horizon>\d+)$"
)


def _surface_dimensions(df: pd.DataFrame) -> tuple[list[str], list[float], list[int]]:
    models: set[str] = set()
    alphas: set[float] = set()
    horizons: set[int] = set()
    for column in df.columns:
        match = _SURFACE_COLUMN_RE.match(str(column))
        if not match:
            continue
        models.add(str(match.group("model")))
        try:
            alphas.add(decode_alpha_token(match.group("alpha")))
        except ValueError:
            continue
        horizons.add(int(match.group("horizon")))
    return (
        infer_model_names_from_columns([f"var_{model}" for model in models]),
        sorted(alphas),
        sorted(horizons),
    )


def _surface_exc_series(df: pd.DataFrame, *, model: str, alpha: float, horizon_days: int) -> pd.Series | None:
    for alpha_token in alpha_lookup_tokens(alpha):
        surface_col = f"exc_{model}_a{alpha_token}_h{int(horizon_days)}"
        if surface_col in df.columns:
            return pd.to_numeric(df[surface_col], errors="coerce")
    if int(horizon_days) == 1 and f"exc_{model}" in df.columns:
        return pd.to_numeric(df[f"exc_{model}"], errors="coerce")
    return None


def _surface_metric_map(
    df: pd.DataFrame,
    *,
    prefix: str,
    alpha: float,
    horizon_days: int,
) -> dict[str, float]:
    alpha_tokens = set(alpha_lookup_tokens(alpha))
    metrics: dict[str, float] = {}
    for column in df.columns:
        match = _SURFACE_COLUMN_RE.match(str(column))
        if not match or match.group("prefix") != prefix:
            continue
        if str(match.group("alpha")) not in alpha_tokens or int(match.group("horizon")) != int(horizon_days):
            continue
        metrics[str(match.group("model"))] = float(pd.to_numeric(df[column], errors="coerce").iloc[-1])
    if int(horizon_days) == 1 and not metrics:
        for model in infer_model_names_from_columns(df.columns):
            column = f"{prefix}_{model}"
            if column in df.columns:
                metrics[model] = float(pd.to_numeric(df[column], errors="coerce").iloc[-1])
    return metrics


def _pick_surface_champion(
    points: list[ValidationSurfacePoint],
    *,
    alpha: float,
    horizon_days: int,
) -> str | None:
    candidates = [
        point
        for point in points
        if abs(float(point.alpha) - float(alpha)) <= 1e-9 and int(point.horizon_days) == int(horizon_days)
    ]
    if not candidates:
        return None
    preferred_rank = {"hist": 0, "fhs": 1, "ewma": 2, "param": 3, "garch": 4, "mc": 5}
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item.score),
            abs(float(item.actual_rate) - float(item.expected_rate)),
            preferred_rank.get(item.model, 99),
            int(item.exceptions),
        ),
    )
    return ordered[0].model


def _pvalue_pass(value: float) -> bool | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return bool(float(numeric) >= P_VALUE_SIGNIFICANCE_LEVEL)


def _surface_statistical_status(
    *,
    coverage_pass: bool,
    independence_pass: bool | None,
    conditional_pass: bool | None,
) -> str:
    if not coverage_pass or conditional_pass is False:
        return VALIDATION_STATUS_FAIL
    if independence_pass is False:
        return VALIDATION_STATUS_WARN
    if coverage_pass and (conditional_pass in {None, True}) and (independence_pass in {None, True}):
        return VALIDATION_STATUS_PASS
    return VALIDATION_STATUS_WARN


def _surface_slice_validation_summary(points: list[ValidationSurfacePoint]) -> dict[str, Any] | None:
    if not points:
        return None
    ordered = sorted(
        points,
        key=lambda item: (
            -float(item.score),
            abs(float(item.actual_rate) - float(item.expected_rate)),
            int(item.exceptions),
            item.model,
        ),
    )
    status_counts = {
        VALIDATION_STATUS_PASS: 0,
        VALIDATION_STATUS_WARN: 0,
        VALIDATION_STATUS_FAIL: 0,
    }
    for point in points:
        status_counts[point.statistical_status] = status_counts.get(point.statistical_status, 0) + 1

    coverage_fail_count = int(sum(1 for point in points if point.coverage_pass is False))
    independence_fail_count = int(sum(1 for point in points if point.independence_pass is False))
    conditional_fail_count = int(sum(1 for point in points if point.conditional_pass is False))

    leader = ordered[0]
    total_points = int(len(points))
    pass_count = int(status_counts.get(VALIDATION_STATUS_PASS, 0))
    warn_count = int(status_counts.get(VALIDATION_STATUS_WARN, 0))
    fail_count = int(status_counts.get(VALIDATION_STATUS_FAIL, 0))
    non_fail_count = pass_count + warn_count
    if fail_count > 0:
        verdict = VALIDATION_STATUS_FAIL
    elif warn_count > 0:
        verdict = VALIDATION_STATUS_WARN
    else:
        verdict = VALIDATION_STATUS_PASS
    return {
        "model_count": total_points,
        "champion_model": str(leader.model),
        "champion_score": float(leader.score),
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "total_points": total_points,
        "pass_rate": (None if total_points <= 0 else float(pass_count / total_points)),
        "non_fail_rate": (None if total_points <= 0 else float(non_fail_count / total_points)),
        "verdict": verdict,
        "pvalue_threshold": float(P_VALUE_SIGNIFICANCE_LEVEL),
        "coverage_fail_count": coverage_fail_count,
        "independence_fail_count": independence_fail_count,
        "conditional_fail_count": conditional_fail_count,
    }


def _surface_governance_summary(points: list[ValidationSurfacePoint]) -> dict[str, Any]:
    status_counts = {
        VALIDATION_STATUS_PASS: 0,
        VALIDATION_STATUS_WARN: 0,
        VALIDATION_STATUS_FAIL: 0,
    }
    traffic_lights = {"GREEN": 0, "YELLOW": 0, "RED": 0, "UNAVAILABLE": 0}

    for point in points:
        status_counts[point.statistical_status] = status_counts.get(point.statistical_status, 0) + 1
        traffic = str(point.traffic_light or "UNAVAILABLE").upper()
        if traffic not in traffic_lights:
            traffic = "UNAVAILABLE"
        traffic_lights[traffic] = traffic_lights.get(traffic, 0) + 1

    total_points = int(len(points))
    coverage_fail_count = int(sum(1 for point in points if point.coverage_pass is False))
    independence_fail_count = int(sum(1 for point in points if point.independence_pass is False))
    conditional_fail_count = int(sum(1 for point in points if point.conditional_pass is False))

    pass_count = int(status_counts.get(VALIDATION_STATUS_PASS, 0))
    warn_count = int(status_counts.get(VALIDATION_STATUS_WARN, 0))
    fail_count = int(status_counts.get(VALIDATION_STATUS_FAIL, 0))
    non_fail_count = pass_count + int(status_counts.get(VALIDATION_STATUS_WARN, 0))
    if fail_count > 0:
        verdict = VALIDATION_STATUS_FAIL
    elif warn_count > 0:
        verdict = VALIDATION_STATUS_WARN
    elif pass_count > 0:
        verdict = VALIDATION_STATUS_PASS
    else:
        verdict = "N/A"
    return {
        "pvalue_threshold": float(P_VALUE_SIGNIFICANCE_LEVEL),
        "total_points": total_points,
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "coverage_fail_count": coverage_fail_count,
        "independence_fail_count": independence_fail_count,
        "conditional_fail_count": conditional_fail_count,
        "traffic_lights": traffic_lights,
        "verdict": verdict,
        "pass_rate": (None if total_points <= 0 else float(pass_count / total_points)),
        "non_fail_rate": (None if total_points <= 0 else float(non_fail_count / total_points)),
    }


def _surface_horizon_governance_summary(
    points: list[ValidationSurfacePoint],
    *,
    horizons: list[int],
    alphas: list[float],
) -> dict[str, Any]:
    horizon_payload: dict[str, Any] = {}
    ordered_horizons = [int(item) for item in horizons]
    ordered_alphas = sorted({float(item) for item in alphas}, reverse=True)

    for horizon_days in ordered_horizons:
        horizon_points = [point for point in points if int(point.horizon_days) == int(horizon_days)]
        status_counts = {
            VALIDATION_STATUS_PASS: 0,
            VALIDATION_STATUS_WARN: 0,
            VALIDATION_STATUS_FAIL: 0,
        }
        traffic_lights = {"GREEN": 0, "YELLOW": 0, "RED": 0, "UNAVAILABLE": 0}
        for point in horizon_points:
            status_counts[point.statistical_status] = status_counts.get(point.statistical_status, 0) + 1
            traffic = str(point.traffic_light or "UNAVAILABLE").upper()
            if traffic not in traffic_lights:
                traffic = "UNAVAILABLE"
            traffic_lights[traffic] = traffic_lights.get(traffic, 0) + 1

        total_points = int(len(horizon_points))
        pass_count = int(status_counts.get(VALIDATION_STATUS_PASS, 0))
        warn_count = int(status_counts.get(VALIDATION_STATUS_WARN, 0))
        fail_count = int(status_counts.get(VALIDATION_STATUS_FAIL, 0))
        if fail_count > 0:
            verdict = VALIDATION_STATUS_FAIL
        elif warn_count > 0:
            verdict = VALIDATION_STATUS_WARN
        elif pass_count > 0:
            verdict = VALIDATION_STATUS_PASS
        else:
            verdict = "N/A"

        champion_model = None
        for alpha in ordered_alphas:
            champion_model = _pick_surface_champion(points, alpha=float(alpha), horizon_days=int(horizon_days))
            if champion_model:
                break
        if champion_model is None and horizon_points:
            fallback = sorted(
                horizon_points,
                key=lambda item: (
                    -float(item.score),
                    abs(float(item.actual_rate) - float(item.expected_rate)),
                    int(item.exceptions),
                    str(item.model),
                ),
            )
            champion_model = fallback[0].model if fallback else None

        horizon_payload[f"h{int(horizon_days)}"] = {
            "horizon_days": int(horizon_days),
            "total_points": total_points,
            "status_counts": {key: int(value) for key, value in status_counts.items()},
            "coverage_fail_count": int(sum(1 for point in horizon_points if point.coverage_pass is False)),
            "independence_fail_count": int(sum(1 for point in horizon_points if point.independence_pass is False)),
            "conditional_fail_count": int(sum(1 for point in horizon_points if point.conditional_pass is False)),
            "traffic_lights": traffic_lights,
            "pass_rate": (None if total_points <= 0 else float(pass_count / total_points)),
            "non_fail_rate": (
                None
                if total_points <= 0
                else float((pass_count + warn_count) / total_points)
            ),
            "verdict": verdict,
            "champion_model": champion_model,
            "alpha_priority": [float(item) for item in ordered_alphas],
            "pvalue_threshold": float(P_VALUE_SIGNIFICANCE_LEVEL),
        }

    overall_verdict = VALIDATION_STATUS_PASS
    for payload in horizon_payload.values():
        verdict = str(payload.get("verdict") or "").upper()
        if verdict == VALIDATION_STATUS_FAIL:
            overall_verdict = VALIDATION_STATUS_FAIL
            break
        if verdict == VALIDATION_STATUS_WARN and overall_verdict != VALIDATION_STATUS_FAIL:
            overall_verdict = VALIDATION_STATUS_WARN
    if not horizon_payload:
        overall_verdict = "N/A"

    return {
        "horizon_order": ordered_horizons,
        "horizons": horizon_payload,
        "overall_verdict": overall_verdict,
        "pvalue_threshold": float(P_VALUE_SIGNIFICANCE_LEVEL),
    }


def validate_compare_surface(
    df: pd.DataFrame,
    *,
    alphas: list[float] | None = None,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    inferred_models, inferred_alphas, inferred_horizons = _surface_dimensions(df)
    selected_alphas = list(alphas or inferred_alphas or [0.95])
    selected_horizons = list(horizons or inferred_horizons or [1])
    models = inferred_models or _infer_models(df)

    points: list[ValidationSurfacePoint] = []
    summary_by_key: dict[str, dict[str, Any]] = {}

    for alpha in selected_alphas:
        expected_rate = 1.0 - float(alpha)
        for horizon_days in selected_horizons:
            for model in models:
                exc_series = _surface_exc_series(df, model=model, alpha=alpha, horizon_days=horizon_days)
                if exc_series is None:
                    continue
                exc = exc_series.dropna().astype(int)
                uc = kupiec_uc(exc, alpha)
                ind = christoffersen_ind(exc)
                cc = conditional_coverage(exc, alpha)
                exceptions_last_250 = int(exc.iloc[-250:].sum()) if len(exc) >= 250 else int(exc.sum())
                traffic_light = _traffic_light(exceptions_last_250, alpha)
                score = _score_model(
                    actual_rate=uc["rate"],
                    expected_rate=expected_rate,
                    p_uc=uc["p_uc"],
                    p_ind=ind["p_ind"],
                    p_cc=cc["p_cc"],
                )
                coverage_pass = bool(_pvalue_pass(uc["p_uc"]))
                independence_pass = _pvalue_pass(ind["p_ind"])
                conditional_pass = _pvalue_pass(cc["p_cc"])
                statistical_status = _surface_statistical_status(
                    coverage_pass=coverage_pass,
                    independence_pass=independence_pass,
                    conditional_pass=conditional_pass,
                )
                points.append(
                    ValidationSurfacePoint(
                        model=model,
                        alpha=float(alpha),
                        horizon_days=int(horizon_days),
                        n=int(uc["n"]),
                        exceptions=int(uc["x"]),
                        expected_rate=float(expected_rate),
                        actual_rate=float(uc["rate"]),
                        p_uc=float(uc["p_uc"]),
                        p_ind=float(ind["p_ind"]) if not np.isnan(ind["p_ind"]) else float("nan"),
                        p_cc=float(cc["p_cc"]) if not np.isnan(cc["p_cc"]) else float("nan"),
                        traffic_light=traffic_light,
                        score=float(score),
                        coverage_pass=coverage_pass,
                        independence_pass=independence_pass,
                        conditional_pass=conditional_pass,
                        statistical_status=statistical_status,
                    )
                )
            slice_points = [
                point
                for point in points
                if abs(float(point.alpha) - float(alpha)) <= 1e-9 and int(point.horizon_days) == int(horizon_days)
            ]
            summary_by_key[f"a{encode_alpha_token(alpha)}_h{int(horizon_days)}"] = {
                "var": _surface_metric_map(df, prefix="var", alpha=alpha, horizon_days=horizon_days),
                "es": _surface_metric_map(df, prefix="es", alpha=alpha, horizon_days=horizon_days),
                "validation": _surface_slice_validation_summary(slice_points),
            }

    champion_live = _pick_surface_champion(points, alpha=0.95, horizon_days=1)
    champion_reporting = (
        _pick_surface_champion(points, alpha=0.99, horizon_days=10)
        or _pick_surface_champion(points, alpha=0.99, horizon_days=5)
        or champion_live
    )
    return {
        "alphas": [float(item) for item in selected_alphas],
        "horizons": [int(item) for item in selected_horizons],
        "points": [point.to_dict() for point in points],
        "current_metrics": summary_by_key,
        "champion_model_live": champion_live,
        "champion_model_reporting": champion_reporting,
        "governance_summary": _surface_governance_summary(points),
        "horizon_governance": _surface_horizon_governance_summary(
            points,
            horizons=[int(item) for item in selected_horizons],
            alphas=[float(item) for item in selected_alphas],
        ),
    }


def validate_compare_frame(df: pd.DataFrame, alpha: float) -> ValidationSummary:
    models = _infer_models(df)
    expected_rate = 1.0 - float(alpha)
    results: Dict[str, BacktestModelValidation] = {}

    for model in models:
        col = f"exc_{model}"
        if col not in df.columns:
            continue

        exc = df[col].dropna().astype(int)
        uc = kupiec_uc(exc, alpha)
        ind = christoffersen_ind(exc)
        cc = conditional_coverage(exc, alpha)
        exceptions_last_250 = int(exc.iloc[-250:].sum()) if len(exc) >= 250 else int(exc.sum())
        traffic_light = _traffic_light(exceptions_last_250, alpha)
        score = _score_model(
            actual_rate=uc["rate"],
            expected_rate=expected_rate,
            p_uc=uc["p_uc"],
            p_ind=ind["p_ind"],
            p_cc=cc["p_cc"],
        )
        losses = -_numeric_series(df, "pnl")
        var_series = _numeric_series(df, f"var_{model}")
        es_series = _numeric_series(df, f"es_{model}")
        es_tail_observations, es_shortfall_ratio, es_breach_rate = _es_backtest_diagnostics(
            losses=losses,
            var_series=var_series,
            es_series=es_series,
        )
        es_acerbi_observations, es_acerbi_stat, es_acerbi_p_value = _es_acerbi_backtest(
            losses=losses,
            var_series=var_series,
            es_series=es_series,
            alpha=alpha,
        )

        results[model] = BacktestModelValidation(
            model=model,
            n=uc["n"],
            exceptions=uc["x"],
            expected_rate=expected_rate,
            actual_rate=uc["rate"],
            lr_uc=uc["LR_uc"],
            p_uc=uc["p_uc"],
            lr_ind=float(ind["LR_ind"]) if not np.isnan(ind["LR_ind"]) else float("nan"),
            p_ind=float(ind["p_ind"]) if not np.isnan(ind["p_ind"]) else float("nan"),
            lr_cc=float(cc["LR_cc"]) if not np.isnan(cc["LR_cc"]) else float("nan"),
            p_cc=float(cc["p_cc"]) if not np.isnan(cc["p_cc"]) else float("nan"),
            exceptions_last_250=exceptions_last_250,
            traffic_light=traffic_light,
            score=score,
            es_tail_observations=es_tail_observations,
            es_shortfall_ratio=es_shortfall_ratio,
            es_breach_rate=es_breach_rate,
            es_acerbi_stat=es_acerbi_stat,
            es_acerbi_p_value=es_acerbi_p_value,
            es_acerbi_observations=es_acerbi_observations,
        )

    ranked = sorted(results.values(), key=_rank_key)
    best_model = ranked[0].model if ranked else None
    surface = validate_compare_surface(df, alphas=[float(alpha)], horizons=[1])

    return ValidationSummary(
        alpha=float(alpha),
        expected_rate=expected_rate,
        model_results=results,
        best_model=best_model,
        champion_model_live=surface.get("champion_model_live"),
        champion_model_reporting=surface.get("champion_model_reporting"),
        surface=surface,
    )


def rank_validation_models(
    summary: ValidationSummary,
    *,
    current_var: Mapping[str, Any] | None = None,
    current_es: Mapping[str, Any] | None = None,
) -> list[RankedModelComparison]:
    vars_map = dict(current_var or {})
    es_map = dict(current_es or {})
    ranked = sorted(summary.model_results.values(), key=_rank_key)

    return [
        RankedModelComparison(
            rank=idx + 1,
            model=item.model,
            score=float(item.score),
            actual_rate=float(item.actual_rate),
            expected_rate=float(item.expected_rate),
            exceptions=int(item.exceptions),
            p_uc=float(item.p_uc),
            p_ind=float(item.p_ind),
            p_cc=float(item.p_cc),
            traffic_light=item.traffic_light,
            current_var=None if vars_map.get(item.model) is None else float(vars_map[item.model]),
            current_es=None if es_map.get(item.model) is None else float(es_map[item.model]),
            es_acerbi_status=_es_acerbi_status(item),
            es_acerbi_p_value=_finite_float(item.es_acerbi_p_value),
            es_acerbi_observations=int(item.es_acerbi_observations or 0),
        )
        for idx, item in enumerate(ranked)
    ]


def build_champion_challenger_summary(
    summary: ValidationSummary,
    *,
    current_var: Mapping[str, Any] | None = None,
    current_es: Mapping[str, Any] | None = None,
) -> ChampionChallengerSummary:
    ranking = rank_validation_models(summary, current_var=current_var, current_es=current_es)
    champion = ranking[0] if ranking else None
    challenger = ranking[1] if len(ranking) > 1 else None
    champion_model = None if champion is None else champion.model

    return ChampionChallengerSummary(
        alpha=float(summary.alpha),
        champion_model=champion_model or summary.champion_model_live,
        champion_model_reporting=summary.champion_model_reporting,
        challenger_model=None if challenger is None else challenger.model,
        score_gap=None if champion is None or challenger is None else round(float(champion.score - challenger.score), 2),
        rate_gap=None if champion is None or challenger is None else float(champion.actual_rate - challenger.actual_rate),
        exception_gap=None if champion is None or challenger is None else int(champion.exceptions - challenger.exceptions),
        current_var_gap=(
            None
            if champion is None or challenger is None or champion.current_var is None or challenger.current_var is None
            else float(champion.current_var - challenger.current_var)
        ),
        current_es_gap=(
            None
            if champion is None or challenger is None or champion.current_es is None or challenger.current_es is None
            else float(champion.current_es - challenger.current_es)
        ),
        ranking=ranking,
        validation_surface=None if summary.surface is None else dict(summary.surface),
    )
