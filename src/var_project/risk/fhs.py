from __future__ import annotations

import numpy as np


def ewma_sigma_portfolio(pnl: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    Calcule sigma_t (vol) EWMA de la série pnl (1D).
    Retourne un array sigma (même taille).
    """
    x = np.asarray(pnl, dtype=float)
    if x.ndim != 1 or len(x) < 5:
        raise ValueError("pnl doit être 1D et assez long")
    if not (0.0 < lam < 1.0):
        raise ValueError("lam dans (0,1)")

    # init variance avec variance sample des 30 premières obs (ou moins)
    init_T = min(30, len(x))
    var = np.var(x[:init_T], ddof=1)
    var = max(var, 1e-12)

    sigmas = np.empty_like(x)
    for t in range(len(x)):
        # update avec observation précédente (standard RiskMetrics-like)
        r2 = x[t - 1] ** 2 if t > 0 else x[0] ** 2
        var = lam * var + (1.0 - lam) * r2
        sigmas[t] = np.sqrt(max(var, 1e-12))
    return sigmas


def fhs_var_es(
    pnl_train: np.ndarray,
    alpha: float,
    lam: float = 0.94,
) -> tuple[float, float]:
    """
    FHS sur PnL:
    - sigma_t via EWMA sur pnl_train
    - résidus z_t = pnl_t / sigma_t
    - VaR/ES sur pertes standardisées, puis re-scale au sigma dernier jour

    Retourne (VaR_loss, ES_loss) en EUR (positif).
    """
    x = np.asarray(pnl_train, dtype=float)
    if len(x) < 20:
        raise ValueError("fenêtre trop courte pour FHS")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha dans (0,1)")

    sigma = ewma_sigma_portfolio(x, lam=lam)
    z = x / sigma
    loss_z = -z  # pertes standardisées (positives)

    var_z = float(np.quantile(loss_z, alpha))
    tail = loss_z[loss_z >= var_z]
    es_z = float(tail.mean()) if len(tail) else var_z

    sigma_next = float(sigma[-1])  # vol "courante" (fin de fenêtre)
    var = var_z * sigma_next
    es = es_z * sigma_next
    return float(var), float(es)
