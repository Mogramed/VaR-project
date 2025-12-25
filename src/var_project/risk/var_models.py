from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, stats


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


def parametric_var(pnl: pd.Series, alpha: float) -> float:
    """
    VaR paramétrique (Normal) sur P&L :
    VaR = -(mu + sigma * z) en perte positive
    où z = norm.ppf(1-alpha) (queue gauche du P&L).
    """
    x = pnl.dropna().astype(float).values
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))

    z = float(norm.ppf(1.0 - float(alpha)))                 # ex alpha=0.99 => p=0.01 => z négatif
    q_pnl = mu + sigma * z                                  # quantile P&L (souvent négatif)
    return float(-q_pnl)                                    # perte positive


def parametric_var_portfolio(weights: np.array, cov_matrix: np.ndarray, alpha: float = 0.99,
                             portfolio_value: float = 1.0) -> float:
    """
    VaR Paramétrique (Variance-Covariance) pour un portefeuille.
    Reference: VAR Course.pdf (Page 41) - Var(ui + uj) = ...
    """
    # Calcul de la variance du portefeuille : w.T * Cov * w
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_std = np.sqrt(port_variance)

    # Z-score pour la loi normale (ex: 2.33 pour 99%)
    z_score = stats.norm.ppf(alpha)

    return port_std * z_score * portfolio_value