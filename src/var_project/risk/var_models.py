from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(pnl: pd.Series, alpha: float) -> float:
    """
    VaR historique sur P&L.
    Convention:
    - pnl > 0 : gain
    - pnl < 0 : perte
    VaR = quantile alpha des pertes (positif).
    """
    x = pnl.dropna().astype(float).values
    losses = -x  # pertes positives
    return float(np.quantile(losses, float(alpha)))


def expected_shortfall(pnl: pd.Series, alpha: float) -> float:
    """
    ES (CVaR) = moyenne des pertes au-delà de la VaR.
    """
    x = pnl.dropna().astype(float).values
    losses = -x
    var = float(np.quantile(losses, float(alpha)))
    tail = losses[losses >= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())
