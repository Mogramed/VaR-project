from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ewma_cov(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    returns: array shape (T, N) de rendements arithmétiques
    EWMA covariance: S_t = lam S_{t-1} + (1-lam) r_t r_t'
    (on suppose mean ~ 0 sur daily FX)
    """
    if returns.ndim != 2:
        raise ValueError("returns doit être 2D (T,N)")
    T, N = returns.shape
    if T < 5:
        raise ValueError("T trop petit")
    if not (0.0 < lam < 1.0):
        raise ValueError("lam dans (0,1)")

    # init par covariance sample sur les premières obs (robuste)
    init_T = min(30, T)
    S = np.cov(returns[:init_T].T, ddof=1)
    if S.shape != (N, N):
        S = np.atleast_2d(S)

    for t in range(T):
        r = returns[t].reshape(N, 1)
        S = lam * S + (1.0 - lam) * (r @ r.T)
    return S


def ewma_var_es(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    lam: float = 0.94,
) -> tuple[float, float]:
    """
    VaR/ES EWMA sous hypothèse Normal sur le portefeuille:
    PnL = w' r
    sigma_p^2 = w' S w

    VaR_loss = z_alpha * sigma_p
    ES_loss  = sigma_p * phi(z_alpha) / (1-alpha)
    """
    S = ewma_cov(returns, lam=lam)
    sigma_p = float(np.sqrt(weights.T @ S @ weights))

    z = float(norm.ppf(float(alpha)))      # ex 0.95 -> 1.645
    var = z * sigma_p

    q = 1.0 - float(alpha)
    es = sigma_p * float(norm.pdf(z)) / q

    return float(var), float(es)
