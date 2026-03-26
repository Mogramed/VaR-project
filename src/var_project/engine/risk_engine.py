from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import pandas as pd

from var_project.portfolio.pnl import portfolio_pnl_from_returns, daily_from_intraday_pnl
from var_project.risk.expected_shortfall import historical_var_es, normal_parametric_var_es, RiskTail


@dataclass(frozen=True)
class SnapshotResult:
    alpha: float
    var: float
    es: float


class RiskEngine:
    """
    Minimal engine:
    - Takes returns (wide df)
    - Builds portfolio pnl
    - Computes VaR/ES (hist + normal param)
    """

    def __init__(self, positions_eur: Dict[str, float]):
        self.positions_eur = positions_eur

    def compute_intraday_pnl(self, returns_wide: pd.DataFrame) -> pd.Series:
        return portfolio_pnl_from_returns(returns_wide, self.positions_eur)

    def compute_daily_pnl(self, returns_wide: pd.DataFrame) -> pd.Series:
        intraday = self.compute_intraday_pnl(returns_wide)
        return daily_from_intraday_pnl(intraday)

    def snapshot_hist(self, pnl: pd.Series, alpha: float) -> SnapshotResult:
        tail = historical_var_es(pnl, alpha)
        return SnapshotResult(alpha=alpha, var=tail.var, es=tail.es)

    def snapshot_param_normal(self, pnl: pd.Series, alpha: float) -> SnapshotResult:
        tail = normal_parametric_var_es(pnl, alpha)
        return SnapshotResult(alpha=alpha, var=tail.var, es=tail.es)
