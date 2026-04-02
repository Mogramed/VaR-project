from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from var_project.core.model_registry import ordered_model_names
from var_project.engine.monte_carlo import mc_var_es
from var_project.portfolio.holdings import (
    PortfolioHolding,
    aggregate_exposure_by_symbol,
    normalize_holdings,
)
from var_project.portfolio.pnl import daily_from_intraday_pnl, portfolio_pnl_from_returns
from var_project.risk.ewma import ewma_var_es
from var_project.risk.expected_shortfall import RiskTail, historical_var_es, normal_parametric_var_es
from var_project.risk.fhs import fhs_var_es
from var_project.risk.garch import garch_var_es
from var_project.risk.stress import StressScenario, stress_report


@dataclass(frozen=True)
class MonteCarloConfig:
    n_sims: int = 20_000
    dist: str = "normal"
    df_t: int = 6
    seed: int | None = 42


@dataclass(frozen=True)
class GarchConfig:
    enabled: bool = True
    p: int = 1
    q: int = 1
    dist: str = "normal"
    mean: str = "constant"


@dataclass(frozen=True)
class RiskModelConfig:
    alpha: float
    ewma_lambda: float = 0.94
    fhs_lambda: float = 0.94
    mc: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    garch: GarchConfig = field(default_factory=GarchConfig)


@dataclass(frozen=True)
class ModelRiskResult:
    model: str
    var: float
    es: float


@dataclass(frozen=True)
class RiskSnapshot:
    alpha: float
    sample_size: int
    models: Dict[str, ModelRiskResult]

    def vars_dict(self) -> Dict[str, float]:
        return {name: result.var for name, result in self.models.items()}

    def es_dict(self) -> Dict[str, float]:
        return {name: result.es for name, result in self.models.items()}


@dataclass(frozen=True)
class RiskSurfacePoint:
    model: str
    alpha: float
    horizon_days: int
    var: float
    es: float
    observation_count: int
    status: str
    latest_observation: str | None = None
    is_stressed: bool = False
    scenario_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "alpha": self.alpha,
            "horizon_days": self.horizon_days,
            "var": self.var,
            "es": self.es,
            "observation_count": self.observation_count,
            "status": self.status,
            "latest_observation": self.latest_observation,
            "is_stressed": self.is_stressed,
            "scenario_name": self.scenario_name,
        }


@dataclass(frozen=True)
class HeadlineRiskPoint:
    key: str
    label: str
    model: str
    alpha: float
    horizon_days: int
    var: float
    es: float
    status: str
    observation_count: int
    is_stressed: bool = False
    scenario_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "model": self.model,
            "alpha": self.alpha,
            "horizon_days": self.horizon_days,
            "var": self.var,
            "es": self.es,
            "status": self.status,
            "observation_count": self.observation_count,
            "is_stressed": self.is_stressed,
            "scenario_name": self.scenario_name,
        }


@dataclass(frozen=True)
class RiskDataQuality:
    status: str
    estimation_window_days: int
    minimum_valid_days: int
    available_observations: int
    oldest_observation: str | None = None
    latest_observation: str | None = None
    horizon_observations: Dict[str, int] = field(default_factory=dict)
    symbol_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "estimation_window_days": self.estimation_window_days,
            "minimum_valid_days": self.minimum_valid_days,
            "available_observations": self.available_observations,
            "oldest_observation": self.oldest_observation,
            "latest_observation": self.latest_observation,
            "horizon_observations": dict(self.horizon_observations),
            "symbol_count": self.symbol_count,
        }


@dataclass(frozen=True)
class RiskSurfaceSnapshot:
    reference_model: str
    alphas: list[float]
    horizons: list[int]
    points: list[RiskSurfacePoint]
    headline: list[HeadlineRiskPoint]
    data_quality: RiskDataQuality
    model_diagnostics: dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        by_model: dict[str, dict[str, dict[str, Any]]] = {}
        for point in self.points:
            model_payload = by_model.setdefault(point.model, {})
            alpha_key = f"a{int(round(point.alpha * 100))}"
            horizon_key = f"h{point.horizon_days}"
            model_payload.setdefault(alpha_key, {})[horizon_key] = {
                "var": point.var,
                "es": point.es,
                "status": point.status,
                "observation_count": point.observation_count,
                "latest_observation": point.latest_observation,
                "is_stressed": point.is_stressed,
                "scenario_name": point.scenario_name,
            }
        return {
            "reference_model": self.reference_model,
            "alphas": list(self.alphas),
            "horizons": list(self.horizons),
            "points": [point.to_dict() for point in self.points],
            "headline": [item.to_dict() for item in self.headline],
            "data_quality": self.data_quality.to_dict(),
            "model_diagnostics": dict(self.model_diagnostics),
            "by_model": by_model,
        }


@dataclass(frozen=True)
class PositionRiskAttribution:
    symbol: str
    asset_class: str
    exposure_base_ccy: float
    standalone_var: float
    standalone_es: float
    incremental_var: float
    incremental_es: float
    marginal_var: float
    marginal_es: float
    component_var: float
    component_es: float
    contribution_pct_var: float | None
    contribution_pct_es: float | None

    @property
    def position_eur(self) -> float:
        return self.exposure_base_ccy

    def to_dict(self) -> Dict[str, float | str | None]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "exposure_base_ccy": self.exposure_base_ccy,
            "standalone_var": self.standalone_var,
            "standalone_es": self.standalone_es,
            "incremental_var": self.incremental_var,
            "incremental_es": self.incremental_es,
            "marginal_var": self.marginal_var,
            "marginal_es": self.marginal_es,
            "component_var": self.component_var,
            "component_es": self.component_es,
            "contribution_pct_var": self.contribution_pct_var,
            "contribution_pct_es": self.contribution_pct_es,
        }


@dataclass(frozen=True)
class AssetClassRiskAttribution:
    asset_class: str
    symbols: list[str]
    symbol_count: int
    exposure_base_ccy: float
    standalone_var: float
    standalone_es: float
    incremental_var: float
    incremental_es: float
    marginal_var: float
    marginal_es: float
    component_var: float
    component_es: float
    contribution_pct_var: float | None
    contribution_pct_es: float | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_class": self.asset_class,
            "symbols": list(self.symbols),
            "symbol_count": int(self.symbol_count),
            "exposure_base_ccy": self.exposure_base_ccy,
            "standalone_var": self.standalone_var,
            "standalone_es": self.standalone_es,
            "incremental_var": self.incremental_var,
            "incremental_es": self.incremental_es,
            "marginal_var": self.marginal_var,
            "marginal_es": self.marginal_es,
            "component_var": self.component_var,
            "component_es": self.component_es,
            "contribution_pct_var": self.contribution_pct_var,
            "contribution_pct_es": self.contribution_pct_es,
        }


@dataclass(frozen=True)
class ModelRiskAttribution:
    model: str
    total_var: float
    total_es: float
    positions: Dict[str, PositionRiskAttribution]
    asset_classes: Dict[str, AssetClassRiskAttribution]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_var": self.total_var,
            "total_es": self.total_es,
            "positions": {symbol: payload.to_dict() for symbol, payload in self.positions.items()},
            "asset_classes": {asset_class: payload.to_dict() for asset_class, payload in self.asset_classes.items()},
        }


@dataclass(frozen=True)
class RiskAttributionSnapshot:
    alpha: float
    sample_size: int
    models: Dict[str, ModelRiskAttribution]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "sample_size": self.sample_size,
            "models": {model: payload.to_dict() for model, payload in self.models.items()},
        }


class RiskEngine:
    """
    Canonical engine for portfolio-level VaR analytics.

    It takes aligned daily returns, builds portfolio PnL and computes standard
    snapshot/backtest/stress outputs used by scripts today and later by the API/UI.
    """

    def __init__(
        self,
        holdings_or_exposure: Mapping[str, Any] | Iterable[Mapping[str, Any] | PortfolioHolding],
        *,
        base_currency: str = "EUR",
    ):
        self.base_currency = str(base_currency).upper()
        self.holdings = normalize_holdings(holdings_or_exposure, base_currency=self.base_currency)
        self.exposure_by_symbol = aggregate_exposure_by_symbol(self.holdings, base_currency=self.base_currency)
        self.asset_class_by_symbol: dict[str, str] = {}
        for holding in self.holdings:
            symbol = str(holding.symbol).upper()
            asset_class = str(holding.asset_class or "unknown").lower()
            if symbol not in self.asset_class_by_symbol:
                self.asset_class_by_symbol[symbol] = asset_class

    def portfolio_symbols(self, frame: pd.DataFrame) -> list[str]:
        return [c for c in frame.columns if c in self.exposure_by_symbol]

    @staticmethod
    def configured_model_names(config: RiskModelConfig) -> list[str]:
        names = ["hist", "param", "mc", "ewma", "fhs"]
        if config.garch.enabled:
            names.append("garch")
        return ordered_model_names(names)

    def normalize_returns_frame(self, returns_wide: pd.DataFrame) -> pd.DataFrame:
        if returns_wide.empty:
            raise ValueError("returns_wide is empty")

        df = returns_wide.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).set_index("time")
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[~df.index.isna()]

        symbols = self.portfolio_symbols(df)
        if not symbols:
            raise ValueError("No portfolio symbol columns found in returns_wide")

        return df[symbols].sort_index().astype(float)

    def compute_intraday_pnl(self, returns_wide: pd.DataFrame) -> pd.Series:
        returns = self.normalize_returns_frame(returns_wide)
        return portfolio_pnl_from_returns(returns, self.exposure_by_symbol)

    def compute_daily_pnl(self, returns_wide: pd.DataFrame) -> pd.Series:
        intraday = self.compute_intraday_pnl(returns_wide)
        return daily_from_intraday_pnl(intraday)

    def build_portfolio_frame(self, returns_wide: pd.DataFrame) -> pd.DataFrame:
        returns = self.normalize_returns_frame(returns_wide)
        out = returns.copy()

        for sym in self.portfolio_symbols(returns):
            out[f"pnl_{sym}"] = float(self.exposure_by_symbol[sym]) * out[sym]

        out["pnl"] = portfolio_pnl_from_returns(returns, self.exposure_by_symbol)
        out.index.name = "date"
        return out

    def evaluate_models(
        self,
        pnl: pd.Series,
        returns_wide: pd.DataFrame,
        config: RiskModelConfig,
        *,
        enabled_models: Sequence[str] | None = None,
    ) -> RiskSnapshot:
        returns = self.normalize_returns_frame(returns_wide)
        pnl_series = pd.Series(pnl, index=returns.index, name="pnl").astype(float)
        exposure_by_symbol = {
            sym: float(self.exposure_by_symbol[sym]) for sym in self.portfolio_symbols(returns)
        }
        weights = np.array([exposure_by_symbol[sym] for sym in returns.columns], dtype=float)
        requested_models = (
            self.configured_model_names(config)
            if enabled_models is None
            else ordered_model_names(enabled_models)
        )
        selected_models = set(requested_models)
        models: dict[str, ModelRiskResult] = {}

        if "hist" in selected_models:
            try:
                hist = historical_var_es(pnl_series, config.alpha)
                models["hist"] = ModelRiskResult("hist", hist.var, hist.es)
            except Exception:
                pass
        if "param" in selected_models:
            try:
                param = normal_parametric_var_es(pnl_series, config.alpha)
                models["param"] = ModelRiskResult("param", param.var, param.es)
            except Exception:
                pass
        if "mc" in selected_models:
            try:
                mc_var, mc_es = mc_var_es(
                    returns=returns,
                    positions=exposure_by_symbol,
                    alpha=config.alpha,
                    n_sims=config.mc.n_sims,
                    dist=config.mc.dist,
                    df_t=config.mc.df_t,
                    seed=config.mc.seed,
                )
                models["mc"] = ModelRiskResult("mc", float(mc_var), float(mc_es))
            except Exception:
                pass
        if "ewma" in selected_models:
            try:
                ewma_var, ewma_es = ewma_var_es(
                    returns=returns.to_numpy(dtype=float),
                    weights=weights,
                    alpha=config.alpha,
                    lam=config.ewma_lambda,
                )
                models["ewma"] = ModelRiskResult("ewma", float(ewma_var), float(ewma_es))
            except Exception:
                pass
        if "garch" in selected_models and config.garch.enabled:
            try:
                garch_tail = garch_var_es(
                    pnl_series.to_numpy(dtype=float),
                    alpha=config.alpha,
                    p=config.garch.p,
                    q=config.garch.q,
                    dist=config.garch.dist,
                    mean=config.garch.mean,
                )
                models["garch"] = ModelRiskResult("garch", float(garch_tail.var), float(garch_tail.es))
            except Exception:
                pass
        if "fhs" in selected_models:
            try:
                fhs_var, fhs_es = fhs_var_es(
                    pnl_train=pnl_series.to_numpy(dtype=float),
                    alpha=config.alpha,
                    lam=config.fhs_lambda,
                )
                models["fhs"] = ModelRiskResult("fhs", float(fhs_var), float(fhs_es))
            except Exception:
                pass

        ordered_models = {name: models[name] for name in ordered_model_names(models.keys())}
        if not ordered_models:
            raise RuntimeError("No risk model could be evaluated for the requested window.")
        return RiskSnapshot(alpha=config.alpha, sample_size=len(pnl_series), models=ordered_models)

    def snapshot_from_returns(
        self,
        returns_wide: pd.DataFrame,
        config: RiskModelConfig,
        *,
        enabled_models: Sequence[str] | None = None,
    ) -> RiskSnapshot:
        frame = self.build_portfolio_frame(returns_wide)
        return self.evaluate_models(
            frame["pnl"],
            frame[self.portfolio_symbols(frame)],
            config,
            enabled_models=enabled_models,
        )

    @staticmethod
    def _aggregate_returns_for_horizon(returns: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        if int(horizon_days) <= 1:
            return returns.copy()
        aggregated = returns.rolling(int(horizon_days)).sum().dropna(how="any")
        aggregated.index.name = returns.index.name
        return aggregated

    @staticmethod
    def _aggregate_pnl_for_horizon(pnl: pd.Series, horizon_days: int) -> pd.Series:
        if int(horizon_days) <= 1:
            return pnl.copy()
        aggregated = pnl.rolling(int(horizon_days)).sum().dropna()
        aggregated.name = pnl.name
        return aggregated

    @staticmethod
    def _history_status(
        observation_count: int,
        *,
        minimum_valid_days: int,
        latest_observation: str | None,
    ) -> str:
        if observation_count <= 0:
            return "incomplete"
        if latest_observation:
            latest_dt = pd.to_datetime(latest_observation, utc=True, errors="coerce")
            if latest_dt is not None and not pd.isna(latest_dt):
                age = datetime.now(timezone.utc) - latest_dt.to_pydatetime()
                if age.days >= 5:
                    return "stale"
        if observation_count < int(minimum_valid_days):
            return "thin_history"
        return "healthy"

    def _apply_stress_scenario_to_returns(
        self,
        returns: pd.DataFrame,
        scenario: StressScenario,
    ) -> pd.DataFrame:
        stressed = returns.mean() + float(scenario.vol_multiplier) * (returns - returns.mean())
        gross_exposure = float(sum(abs(value) for value in self.exposure_by_symbol.values()))
        if abs(float(scenario.shock_pnl)) <= 1e-9 or gross_exposure <= 1e-9:
            return stressed

        for symbol in stressed.columns:
            exposure = float(self.exposure_by_symbol.get(symbol, 0.0))
            abs_exposure = abs(exposure)
            if abs_exposure <= 1e-9:
                continue
            exposure_weight = abs_exposure / gross_exposure
            pnl_share = float(scenario.shock_pnl) * exposure_weight
            stressed[symbol] = stressed[symbol] + (pnl_share / exposure)
        return stressed

    @staticmethod
    def _reference_point(
        points: Sequence[RiskSurfacePoint],
        *,
        reference_model: str,
        alpha: float,
        horizon_days: int,
        is_stressed: bool = False,
    ) -> RiskSurfacePoint | None:
        rounded_alpha = round(float(alpha), 6)
        for point in points:
            if (
                point.model == reference_model
                and round(point.alpha, 6) == rounded_alpha
                and int(point.horizon_days) == int(horizon_days)
                and bool(point.is_stressed) == bool(is_stressed)
            ):
                return point
        fallback = [
            point
            for point in points
            if round(point.alpha, 6) == rounded_alpha
            and int(point.horizon_days) == int(horizon_days)
            and bool(point.is_stressed) == bool(is_stressed)
        ]
        return fallback[0] if fallback else None

    @staticmethod
    def _headline_from_points(
        points: Sequence[RiskSurfacePoint],
        *,
        reference_model: str,
    ) -> list[HeadlineRiskPoint]:
        requested = [
            ("live_1d_95", "Live 1D 95%", 0.95, 1, False),
            ("live_1d_99", "Live 1D 99%", 0.99, 1, False),
            ("watch_5d_99", "Watch 5D 99%", 0.99, 5, False),
            ("governance_10d_975", "Governance ES 10D 97.5%", 0.975, 10, False),
            ("governance_10d_99", "Governance 10D 99%", 0.99, 10, False),
            ("stressed_10d_975", "Stressed ES 10D 97.5%", 0.975, 10, True),
            ("stressed_10d_99", "Stressed ES 10D 99%", 0.99, 10, True),
        ]
        headline: list[HeadlineRiskPoint] = []
        for key, label, alpha, horizon_days, is_stressed in requested:
            point = RiskEngine._reference_point(
                points,
                reference_model=reference_model,
                alpha=alpha,
                horizon_days=horizon_days,
                is_stressed=is_stressed,
            )
            if point is None:
                continue
            headline.append(
                HeadlineRiskPoint(
                    key=key,
                    label=label,
                    model=point.model,
                    alpha=point.alpha,
                    horizon_days=point.horizon_days,
                    var=point.var,
                    es=point.es,
                    status=point.status,
                    observation_count=point.observation_count,
                    is_stressed=point.is_stressed,
                    scenario_name=point.scenario_name,
                )
            )
        return headline

    @staticmethod
    def _diagnostics_from_points(points: Sequence[RiskSurfacePoint]) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        grouped: dict[str, list[RiskSurfacePoint]] = {}
        for point in points:
            if point.is_stressed:
                continue
            grouped.setdefault(point.model, []).append(point)
        for model, model_points in grouped.items():
            vars_values = [float(point.var) for point in model_points]
            es_values = [float(point.es) for point in model_points]
            diagnostics[model] = {
                "role": "primary" if model in {"hist", "fhs"} else "challenger",
                "surface_points": len(model_points),
                "avg_var": float(np.mean(vars_values)) if vars_values else 0.0,
                "avg_es": float(np.mean(es_values)) if es_values else 0.0,
                "max_var": float(np.max(vars_values)) if vars_values else 0.0,
                "max_es": float(np.max(es_values)) if es_values else 0.0,
                "min_var": float(np.min(vars_values)) if vars_values else 0.0,
                "min_es": float(np.min(es_values)) if es_values else 0.0,
                "stability_ratio": (
                    None
                    if not vars_values or min(vars_values) <= 1e-9
                    else float(max(vars_values) / min(vars_values))
                ),
            }
        return diagnostics

    def build_risk_surface(
        self,
        returns_wide: pd.DataFrame,
        config: RiskModelConfig,
        *,
        alphas: Sequence[float],
        horizons: Sequence[int],
        estimation_window_days: int,
        minimum_valid_days: int,
        reference_model: str = "hist",
        enabled_models: Sequence[str] | None = None,
        is_stressed: bool = False,
        scenario_name: str | None = None,
    ) -> RiskSurfaceSnapshot:
        returns = self.normalize_returns_frame(returns_wide).tail(max(int(estimation_window_days), 1))
        alphas_sorted = sorted({round(float(item), 6) for item in alphas if float(item) > 0.0})
        horizons_sorted = sorted({int(item) for item in horizons if int(item) > 0})
        points: list[RiskSurfacePoint] = []
        horizon_observations: dict[str, int] = {}

        for horizon_days in horizons_sorted:
            aggregated_returns = self._aggregate_returns_for_horizon(returns, horizon_days)
            if aggregated_returns.empty:
                horizon_observations[f"h{horizon_days}"] = 0
                continue
            pnl = portfolio_pnl_from_returns(aggregated_returns, self.exposure_by_symbol)
            latest_observation = aggregated_returns.index[-1].isoformat()
            status = self._history_status(
                len(pnl),
                minimum_valid_days=int(minimum_valid_days),
                latest_observation=latest_observation,
            )
            horizon_observations[f"h{horizon_days}"] = int(len(pnl))

            for alpha in alphas_sorted:
                alpha_config = replace(config, alpha=float(alpha))
                snapshot = self.evaluate_models(
                    pnl,
                    aggregated_returns,
                    alpha_config,
                    enabled_models=enabled_models,
                )
                for model_name, result in snapshot.models.items():
                    points.append(
                        RiskSurfacePoint(
                            model=model_name,
                            alpha=float(alpha),
                            horizon_days=int(horizon_days),
                            var=float(result.var),
                            es=float(result.es),
                            observation_count=int(snapshot.sample_size),
                            status=status,
                            latest_observation=latest_observation,
                            is_stressed=is_stressed,
                            scenario_name=scenario_name,
                        )
                    )

        oldest_observation = None if returns.empty else returns.index[0].isoformat()
        latest_observation = None if returns.empty else returns.index[-1].isoformat()
        available_observations = int(len(returns))
        data_quality = RiskDataQuality(
            status=self._history_status(
                available_observations,
                minimum_valid_days=int(minimum_valid_days),
                latest_observation=latest_observation,
            ),
            estimation_window_days=int(estimation_window_days),
            minimum_valid_days=int(minimum_valid_days),
            available_observations=available_observations,
            oldest_observation=oldest_observation,
            latest_observation=latest_observation,
            horizon_observations=horizon_observations,
            symbol_count=len(list(returns.columns)),
        )
        order_index = {name: idx for idx, name in enumerate(ordered_model_names({point.model for point in points}))}
        ordered_points = sorted(
            points,
            key=lambda item: (
                1 if item.is_stressed else 0,
                order_index.get(item.model, 999),
                item.alpha,
                item.horizon_days,
            ),
        )
        return RiskSurfaceSnapshot(
            reference_model=reference_model,
            alphas=list(alphas_sorted),
            horizons=list(horizons_sorted),
            points=ordered_points,
            headline=self._headline_from_points(ordered_points, reference_model=reference_model),
            data_quality=data_quality,
            model_diagnostics=self._diagnostics_from_points(ordered_points),
        )

    def _asset_class_groups(self, symbols: Sequence[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for symbol in symbols:
            asset_class = str(self.asset_class_by_symbol.get(str(symbol).upper(), "unknown")).lower()
            groups.setdefault(asset_class, []).append(str(symbol).upper())
        return groups

    def _scaled_group_exposure(
        self,
        base_exposure: Mapping[str, float],
        *,
        group_symbols: Sequence[str],
        scale: float,
    ) -> dict[str, float]:
        scaled = {str(symbol).upper(): float(value) for symbol, value in dict(base_exposure).items()}
        for symbol in group_symbols:
            normalized = str(symbol).upper()
            scaled[normalized] = float(scaled.get(normalized, 0.0)) * float(scale)
        return scaled

    def attribute_from_returns(
        self,
        returns_wide: pd.DataFrame,
        config: RiskModelConfig,
        *,
        base_snapshot: RiskSnapshot | None = None,
        bump_relative: float = 0.01,
        bump_absolute: float = 100.0,
        enabled_models: Sequence[str] | None = None,
    ) -> RiskAttributionSnapshot:
        returns = self.normalize_returns_frame(returns_wide)
        base = base_snapshot or self.snapshot_from_returns(returns, config, enabled_models=enabled_models)
        symbols = self.portfolio_symbols(returns)
        base_exposure = {symbol: float(self.exposure_by_symbol.get(symbol, 0.0)) for symbol in symbols}
        model_names = ordered_model_names(base.models.keys())
        asset_class_groups = self._asset_class_groups(symbols)

        model_payloads: dict[str, dict[str, PositionRiskAttribution]] = {model_name: {} for model_name in model_names}
        asset_class_payloads: dict[str, dict[str, AssetClassRiskAttribution]] = {
            model_name: {} for model_name in model_names
        }

        for symbol in symbols:
            position = float(base_exposure[symbol])
            asset_class = str(self.asset_class_by_symbol.get(symbol, "unknown")).lower()

            standalone_exposure = {name: 0.0 for name in symbols}
            standalone_exposure[symbol] = position
            standalone_snapshot = RiskEngine(standalone_exposure, base_currency=self.base_currency).snapshot_from_returns(
                returns,
                config,
                enabled_models=model_names,
            )

            without_exposure = dict(base_exposure)
            without_exposure[symbol] = 0.0
            without_snapshot = RiskEngine(without_exposure, base_currency=self.base_currency).snapshot_from_returns(
                returns,
                config,
                enabled_models=model_names,
            )

            bump = self._position_bump(position, bump_relative=bump_relative, bump_absolute=bump_absolute)
            plus_exposure = dict(base_exposure)
            plus_exposure[symbol] = position + bump
            plus_snapshot = RiskEngine(plus_exposure, base_currency=self.base_currency).snapshot_from_returns(
                returns,
                config,
                enabled_models=model_names,
            )

            if abs(position) > 1e-9:
                minus_exposure = dict(base_exposure)
                minus_exposure[symbol] = position - bump
                minus_snapshot = RiskEngine(minus_exposure, base_currency=self.base_currency).snapshot_from_returns(
                    returns,
                    config,
                    enabled_models=model_names,
                )
            else:
                minus_snapshot = base

            for model_name, result in base.models.items():
                standalone = standalone_snapshot.models[model_name]
                reduced = without_snapshot.models[model_name]
                plus = plus_snapshot.models[model_name]
                if abs(position) > 1e-9:
                    minus = minus_snapshot.models[model_name]
                    marginal_var = (plus.var - minus.var) / (2.0 * bump)
                    marginal_es = (plus.es - minus.es) / (2.0 * bump)
                else:
                    marginal_var = (plus.var - result.var) / bump
                    marginal_es = (plus.es - result.es) / bump

                component_var = position * marginal_var
                component_es = position * marginal_es
                model_payloads[model_name][symbol] = PositionRiskAttribution(
                    symbol=symbol,
                    asset_class=asset_class,
                    exposure_base_ccy=position,
                    standalone_var=float(standalone.var),
                    standalone_es=float(standalone.es),
                    incremental_var=float(result.var - reduced.var),
                    incremental_es=float(result.es - reduced.es),
                    marginal_var=float(marginal_var),
                    marginal_es=float(marginal_es),
                    component_var=float(component_var),
                    component_es=float(component_es),
                    contribution_pct_var=None if result.var <= 0 else float(component_var / result.var),
                    contribution_pct_es=None if result.es <= 0 else float(component_es / result.es),
                )

        for asset_class, group_symbols in asset_class_groups.items():
            group_exposure = float(sum(float(base_exposure.get(symbol, 0.0)) for symbol in group_symbols))
            group_gross_exposure = float(sum(abs(float(base_exposure.get(symbol, 0.0))) for symbol in group_symbols))

            standalone_exposure = {name: 0.0 for name in symbols}
            for symbol in group_symbols:
                standalone_exposure[symbol] = float(base_exposure.get(symbol, 0.0))
            standalone_snapshot = RiskEngine(standalone_exposure, base_currency=self.base_currency).snapshot_from_returns(
                returns,
                config,
                enabled_models=model_names,
            )

            without_exposure = dict(base_exposure)
            for symbol in group_symbols:
                without_exposure[symbol] = 0.0
            without_snapshot = RiskEngine(without_exposure, base_currency=self.base_currency).snapshot_from_returns(
                returns,
                config,
                enabled_models=model_names,
            )

            if group_gross_exposure > 1e-9:
                bump = self._position_bump(group_gross_exposure, bump_relative=bump_relative, bump_absolute=bump_absolute)
                scale = 1.0 + (float(bump) / group_gross_exposure)
                plus_exposure = self._scaled_group_exposure(base_exposure, group_symbols=group_symbols, scale=scale)
                plus_snapshot = RiskEngine(plus_exposure, base_currency=self.base_currency).snapshot_from_returns(
                    returns,
                    config,
                    enabled_models=model_names,
                )
                minus_scale = max(1.0 - (float(bump) / group_gross_exposure), 0.0)
                minus_exposure = self._scaled_group_exposure(base_exposure, group_symbols=group_symbols, scale=minus_scale)
                minus_snapshot = RiskEngine(minus_exposure, base_currency=self.base_currency).snapshot_from_returns(
                    returns,
                    config,
                    enabled_models=model_names,
                )
            else:
                bump = self._position_bump(group_exposure, bump_relative=bump_relative, bump_absolute=bump_absolute)
                plus_exposure = dict(base_exposure)
                if group_symbols:
                    anchor = group_symbols[0]
                    plus_exposure[anchor] = float(base_exposure.get(anchor, 0.0)) + float(bump)
                plus_snapshot = RiskEngine(plus_exposure, base_currency=self.base_currency).snapshot_from_returns(
                    returns,
                    config,
                    enabled_models=model_names,
                )
                minus_snapshot = base

            for model_name, result in base.models.items():
                standalone = standalone_snapshot.models[model_name]
                reduced = without_snapshot.models[model_name]
                plus = plus_snapshot.models[model_name]
                if group_gross_exposure > 1e-9:
                    minus = minus_snapshot.models[model_name]
                    marginal_var = (plus.var - minus.var) / (2.0 * bump)
                    marginal_es = (plus.es - minus.es) / (2.0 * bump)
                else:
                    marginal_var = (plus.var - result.var) / bump
                    marginal_es = (plus.es - result.es) / bump

                component_var = float(
                    sum(
                        float(model_payloads[model_name][symbol].component_var)
                        for symbol in group_symbols
                        if symbol in model_payloads[model_name]
                    )
                )
                component_es = float(
                    sum(
                        float(model_payloads[model_name][symbol].component_es)
                        for symbol in group_symbols
                        if symbol in model_payloads[model_name]
                    )
                )
                asset_class_payloads[model_name][asset_class] = AssetClassRiskAttribution(
                    asset_class=asset_class,
                    symbols=list(group_symbols),
                    symbol_count=len(group_symbols),
                    exposure_base_ccy=group_exposure,
                    standalone_var=float(standalone.var),
                    standalone_es=float(standalone.es),
                    incremental_var=float(result.var - reduced.var),
                    incremental_es=float(result.es - reduced.es),
                    marginal_var=float(marginal_var),
                    marginal_es=float(marginal_es),
                    component_var=component_var,
                    component_es=component_es,
                    contribution_pct_var=None if result.var <= 0 else float(component_var / result.var),
                    contribution_pct_es=None if result.es <= 0 else float(component_es / result.es),
                )

        models = {
            model_name: ModelRiskAttribution(
                model=model_name,
                total_var=float(base.models[model_name].var),
                total_es=float(base.models[model_name].es),
                positions=model_payloads[model_name],
                asset_classes=asset_class_payloads[model_name],
            )
            for model_name in model_names
        }
        return RiskAttributionSnapshot(alpha=base.alpha, sample_size=base.sample_size, models=models)

    def stress(self, pnl: pd.Series, alpha: float, scenarios: Iterable[StressScenario]) -> Dict[str, RiskTail]:
        return stress_report(pd.Series(pnl).astype(float), alpha, list(scenarios))

    def default_stress_scenarios(self) -> list[StressScenario]:
        gross_exposure = float(sum(abs(value) for value in self.exposure_by_symbol.values()))
        directional_shock = -0.01 * gross_exposure if gross_exposure > 0 else -1_000.0
        correlated_shock = -0.015 * gross_exposure if gross_exposure > 0 else -1_500.0
        return [
            StressScenario(name="Volatility regime shift", vol_multiplier=1.5, shock_pnl=0.0),
            StressScenario(name="FX directional down shock", vol_multiplier=1.1, shock_pnl=directional_shock),
            StressScenario(name="Correlated multi-asset drawdown", vol_multiplier=2.0, shock_pnl=correlated_shock),
        ]

    def build_stress_surface(
        self,
        returns_wide: pd.DataFrame,
        config: RiskModelConfig,
        *,
        alphas: Sequence[float],
        horizons: Sequence[int],
        estimation_window_days: int,
        minimum_valid_days: int,
        scenarios: Sequence[StressScenario] | None = None,
    ) -> dict[str, Any]:
        returns = self.normalize_returns_frame(returns_wide).tail(max(int(estimation_window_days), 1))
        baseline_snapshot = self.snapshot_from_returns(returns, config, enabled_models=["hist"])
        baseline_surface = self.build_risk_surface(
            returns,
            config,
            alphas=alphas,
            horizons=horizons,
            estimation_window_days=estimation_window_days,
            minimum_valid_days=minimum_valid_days,
            reference_model="hist",
            enabled_models=["hist"],
        )
        baseline_attribution = self.attribute_from_returns(
            returns,
            config,
            base_snapshot=baseline_snapshot,
            enabled_models=["hist"],
        )
        scenario_defs = list(scenarios or self.default_stress_scenarios())
        scenario_surfaces: list[dict[str, Any]] = []
        stressed_points: list[RiskSurfacePoint] = []

        for scenario in scenario_defs:
            stressed_returns = self._apply_stress_scenario_to_returns(returns, scenario)
            scenario_snapshot = self.snapshot_from_returns(stressed_returns, config, enabled_models=["hist"])
            surface = self.build_risk_surface(
                stressed_returns,
                config,
                alphas=alphas,
                horizons=horizons,
                estimation_window_days=estimation_window_days,
                minimum_valid_days=minimum_valid_days,
                reference_model="hist",
                enabled_models=["hist"],
                is_stressed=True,
                scenario_name=scenario.name,
            )
            scenario_attribution = self.attribute_from_returns(
                stressed_returns,
                config,
                base_snapshot=scenario_snapshot,
                enabled_models=["hist"],
            )
            scenario_payload = surface.to_dict()
            headline = list(scenario_payload.get("headline") or [])
            primary = next(
                (
                    item
                    for item in headline
                    if item.get("key") in {
                        "live_1d_99",
                        "governance_10d_975",
                        "governance_10d_99",
                        "stressed_10d_975",
                        "stressed_10d_99",
                    }
                ),
                headline[0] if headline else None,
            )
            scenario_surfaces.append(
                {
                    "name": scenario.name,
                    "vol_multiplier": float(scenario.vol_multiplier),
                    "shock_pnl": float(scenario.shock_pnl),
                    "risk_surface": scenario_payload,
                    "headline_risk": headline,
                    "attribution": scenario_attribution.to_dict(),
                    "primary_metric": primary,
                }
            )
            stressed_points.extend(surface.points)

        baseline_pnl = portfolio_pnl_from_returns(returns, self.exposure_by_symbol)
        historical_extremes: list[dict[str, Any]] = []
        for horizon_days in sorted({int(item) for item in horizons if int(item) > 0}):
            aggregated_pnl = self._aggregate_pnl_for_horizon(baseline_pnl, horizon_days)
            if aggregated_pnl.empty:
                continue
            losses = -aggregated_pnl
            worst_idx = losses.idxmax()
            historical_extremes.append(
                {
                    "horizon_days": int(horizon_days),
                    "worst_loss": float(losses.max()),
                    "worst_end_date": worst_idx.isoformat() if hasattr(worst_idx, "isoformat") else str(worst_idx),
                    "tail_mean_loss": float(losses[losses >= np.quantile(losses, 0.95)].mean()),
                }
            )

        all_points = list(baseline_surface.points) + stressed_points
        headline = self._headline_from_points(all_points, reference_model="hist")
        baseline_point = self._reference_point(
            baseline_surface.points,
            reference_model="hist",
            alpha=float(config.alpha),
            horizon_days=1,
        )

        return {
            "reference_model": "hist",
            "baseline_var": None if baseline_point is None else float(baseline_point.var),
            "baseline_es": None if baseline_point is None else float(baseline_point.es),
            "risk_surface": baseline_surface.to_dict(),
            "headline_risk": [item.to_dict() for item in headline],
            "attribution": baseline_attribution.to_dict(),
            "scenarios": scenario_surfaces,
            "historical_extremes": historical_extremes,
        }

    def backtest(
        self,
        returns_wide: pd.DataFrame,
        window: int,
        config: RiskModelConfig,
        metadata: Mapping[str, Any] | None = None,
        *,
        alphas: Sequence[float] | None = None,
        horizons: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        frame = self.build_portfolio_frame(returns_wide)
        symbols = self.portfolio_symbols(frame)
        selected_alphas = sorted({round(float(item), 6) for item in (alphas or [config.alpha]) if float(item) > 0.0})
        selected_horizons = sorted({int(item) for item in (horizons or [1]) if int(item) > 0})
        viable_horizons = [horizon_days for horizon_days in selected_horizons if len(frame) >= window + horizon_days]
        if not viable_horizons:
            requested_horizon = max(selected_horizons, default=1)
            raise RuntimeError(
                f"Not enough observations ({len(frame)}) for window={window} and horizon={requested_horizon}"
            )
        selected_horizons = viable_horizons
        max_horizon = max(selected_horizons, default=1)

        out_rows: list[dict[str, Any]] = []
        for i in range(window, len(frame) - max_horizon + 1):
            window_df = frame.iloc[i - window : i]

            row: dict[str, Any] = {
                "date": frame.index[i],
                "pnl": float(frame.iloc[i]["pnl"]),
                "alpha": float(config.alpha),
                "window": int(window),
            }

            for horizon_days in selected_horizons:
                realized_pnl = float(frame["pnl"].iloc[i : i + horizon_days].sum())
                row[f"pnl_h{horizon_days}"] = realized_pnl
                aggregated_returns = self._aggregate_returns_for_horizon(window_df[symbols], horizon_days)
                if aggregated_returns.empty:
                    continue
                aggregated_pnl = portfolio_pnl_from_returns(aggregated_returns, self.exposure_by_symbol)

                for alpha in selected_alphas:
                    alpha_config = replace(config, alpha=float(alpha))
                    snapshot = self.evaluate_models(aggregated_pnl, aggregated_returns, alpha_config)
                    alpha_suffix = f"_a{int(round(alpha * 100))}_h{horizon_days}"

                    for model_name, result in snapshot.models.items():
                        row[f"var_{model_name}{alpha_suffix}"] = result.var
                        row[f"es_{model_name}{alpha_suffix}"] = result.es
                        row[f"exc_{model_name}{alpha_suffix}"] = int((-realized_pnl) > result.var)
                        if abs(alpha - float(config.alpha)) <= 1e-9 and int(horizon_days) == 1:
                            row[f"var_{model_name}"] = result.var
                            row[f"es_{model_name}"] = result.es
                            row[f"exc_{model_name}"] = int((-row["pnl"]) > result.var)

            for sym in symbols:
                row[f"ret_{sym}"] = float(frame.iloc[i][sym])
                row[f"pnl_{sym}"] = float(frame.iloc[i][f"pnl_{sym}"])

            if metadata:
                row.update(metadata)

            out_rows.append(row)

        return pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _position_bump(position: float, *, bump_relative: float, bump_absolute: float) -> float:
        if abs(position) <= 1e-9:
            return float(bump_absolute)
        candidate = max(abs(position) * float(bump_relative), float(bump_absolute))
        return float(min(candidate, abs(position) * 0.5))
