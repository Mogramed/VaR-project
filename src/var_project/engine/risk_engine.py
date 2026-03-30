from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

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
class PositionRiskAttribution:
    symbol: str
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
            "exposure_base_ccy": self.exposure_base_ccy,
            "position_eur": self.position_eur,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_var": self.total_var,
            "total_es": self.total_es,
            "positions": {symbol: payload.to_dict() for symbol, payload in self.positions.items()},
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
    ) -> RiskSnapshot:
        returns = self.normalize_returns_frame(returns_wide)
        pnl_series = pd.Series(pnl, index=returns.index, name="pnl").astype(float)
        exposure_by_symbol = {
            sym: float(self.exposure_by_symbol[sym]) for sym in self.portfolio_symbols(returns)
        }
        weights = np.array([exposure_by_symbol[sym] for sym in returns.columns], dtype=float)

        hist = historical_var_es(pnl_series, config.alpha)
        param = normal_parametric_var_es(pnl_series, config.alpha)
        mc_var, mc_es = mc_var_es(
            returns=returns,
            positions=exposure_by_symbol,
            alpha=config.alpha,
            n_sims=config.mc.n_sims,
            dist=config.mc.dist,
            df_t=config.mc.df_t,
            seed=config.mc.seed,
        )
        ewma_var, ewma_es = ewma_var_es(
            returns=returns.to_numpy(dtype=float),
            weights=weights,
            alpha=config.alpha,
            lam=config.ewma_lambda,
        )
        fhs_var, fhs_es = fhs_var_es(
            pnl_train=pnl_series.to_numpy(dtype=float),
            alpha=config.alpha,
            lam=config.fhs_lambda,
        )

        models = {
            "hist": ModelRiskResult("hist", hist.var, hist.es),
            "param": ModelRiskResult("param", param.var, param.es),
            "mc": ModelRiskResult("mc", float(mc_var), float(mc_es)),
            "ewma": ModelRiskResult("ewma", float(ewma_var), float(ewma_es)),
        }
        if config.garch.enabled:
            garch_tail = garch_var_es(
                pnl_series.to_numpy(dtype=float),
                alpha=config.alpha,
                p=config.garch.p,
                q=config.garch.q,
                dist=config.garch.dist,
                mean=config.garch.mean,
            )
            models["garch"] = ModelRiskResult("garch", float(garch_tail.var), float(garch_tail.es))
        models["fhs"] = ModelRiskResult("fhs", float(fhs_var), float(fhs_es))
        return RiskSnapshot(alpha=config.alpha, sample_size=len(pnl_series), models=models)

    def snapshot_from_returns(self, returns_wide: pd.DataFrame, config: RiskModelConfig) -> RiskSnapshot:
        frame = self.build_portfolio_frame(returns_wide)
        return self.evaluate_models(frame["pnl"], frame[self.portfolio_symbols(frame)], config)

    def attribute_from_returns(
        self,
        returns_wide: pd.DataFrame,
        config: RiskModelConfig,
        *,
        base_snapshot: RiskSnapshot | None = None,
        bump_relative: float = 0.01,
        bump_absolute: float = 100.0,
    ) -> RiskAttributionSnapshot:
        returns = self.normalize_returns_frame(returns_wide)
        base = base_snapshot or self.snapshot_from_returns(returns, config)
        symbols = self.portfolio_symbols(returns)
        base_exposure = {symbol: float(self.exposure_by_symbol.get(symbol, 0.0)) for symbol in symbols}

        model_payloads: dict[str, dict[str, PositionRiskAttribution]] = {
            model_name: {} for model_name in ordered_model_names(base.models.keys())
        }

        for symbol in symbols:
            position = float(base_exposure[symbol])

            standalone_exposure = {name: 0.0 for name in symbols}
            standalone_exposure[symbol] = position
            standalone_snapshot = RiskEngine(standalone_exposure).snapshot_from_returns(returns, config)

            without_exposure = dict(base_exposure)
            without_exposure[symbol] = 0.0
            without_snapshot = RiskEngine(without_exposure).snapshot_from_returns(returns, config)

            bump = self._position_bump(position, bump_relative=bump_relative, bump_absolute=bump_absolute)
            plus_exposure = dict(base_exposure)
            plus_exposure[symbol] = position + bump
            plus_snapshot = RiskEngine(plus_exposure).snapshot_from_returns(returns, config)

            if abs(position) > 1e-9:
                minus_exposure = dict(base_exposure)
                minus_exposure[symbol] = position - bump
                minus_snapshot = RiskEngine(minus_exposure).snapshot_from_returns(returns, config)
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

        models = {
            model_name: ModelRiskAttribution(
                model=model_name,
                total_var=float(base.models[model_name].var),
                total_es=float(base.models[model_name].es),
                positions=model_payloads[model_name],
            )
            for model_name in ordered_model_names(base.models.keys())
        }
        return RiskAttributionSnapshot(alpha=base.alpha, sample_size=base.sample_size, models=models)

    def stress(self, pnl: pd.Series, alpha: float, scenarios: Iterable[StressScenario]) -> Dict[str, RiskTail]:
        return stress_report(pd.Series(pnl).astype(float), alpha, list(scenarios))

    def backtest(
        self,
        returns_wide: pd.DataFrame,
        window: int,
        config: RiskModelConfig,
        metadata: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        frame = self.build_portfolio_frame(returns_wide)
        symbols = self.portfolio_symbols(frame)

        if len(frame) <= window + 5:
            raise RuntimeError(f"Not enough observations ({len(frame)}) for window={window}")

        out_rows: list[dict[str, Any]] = []
        for i in range(window, len(frame)):
            window_df = frame.iloc[i - window : i]
            snapshot = self.evaluate_models(window_df["pnl"], window_df[symbols], config)

            row: dict[str, Any] = {
                "date": frame.index[i],
                "pnl": float(frame.iloc[i]["pnl"]),
                "alpha": float(config.alpha),
                "window": int(window),
            }

            for model_name, result in snapshot.models.items():
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
