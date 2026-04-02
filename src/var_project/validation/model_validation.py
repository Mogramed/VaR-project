from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2

from var_project.core.model_registry import infer_model_names_from_columns


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


def _rank_key(item: BacktestModelValidation) -> tuple[float, float, int]:
    return (-item.score, abs(item.actual_rate - item.expected_rate), item.exceptions)


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


_SURFACE_COLUMN_RE = re.compile(r"^(?P<prefix>var|es|exc)_(?P<model>[a-z0-9]+)_a(?P<alpha>\d+)_h(?P<horizon>\d+)$")


def _surface_dimensions(df: pd.DataFrame) -> tuple[list[str], list[float], list[int]]:
    models: set[str] = set()
    alphas: set[float] = set()
    horizons: set[int] = set()
    for column in df.columns:
        match = _SURFACE_COLUMN_RE.match(str(column))
        if not match:
            continue
        models.add(str(match.group("model")))
        alphas.add(int(match.group("alpha")) / 100.0)
        horizons.add(int(match.group("horizon")))
    return (
        infer_model_names_from_columns([f"var_{model}" for model in models]),
        sorted(alphas),
        sorted(horizons),
    )


def _surface_exc_series(df: pd.DataFrame, *, model: str, alpha: float, horizon_days: int) -> pd.Series | None:
    alpha_suffix = int(round(float(alpha) * 100))
    surface_col = f"exc_{model}_a{alpha_suffix}_h{int(horizon_days)}"
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
    alpha_suffix = int(round(float(alpha) * 100))
    metrics: dict[str, float] = {}
    for column in df.columns:
        match = _SURFACE_COLUMN_RE.match(str(column))
        if not match or match.group("prefix") != prefix:
            continue
        if int(match.group("alpha")) != alpha_suffix or int(match.group("horizon")) != int(horizon_days):
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
                    )
                )
            summary_by_key[f"a{int(round(alpha * 100))}_h{int(horizon_days)}"] = {
                "var": _surface_metric_map(df, prefix="var", alpha=alpha, horizon_days=horizon_days),
                "es": _surface_metric_map(df, prefix="es", alpha=alpha, horizon_days=horizon_days),
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

    return ChampionChallengerSummary(
        alpha=float(summary.alpha),
        champion_model=summary.champion_model_live or (None if champion is None else champion.model),
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
