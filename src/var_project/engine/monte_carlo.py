from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def mc_var_es(
    returns: pd.DataFrame,
    positions: Dict[str, float],
    alpha: float,
    n_sims: int = 20000,
    dist: str = "normal",      # "normal" ou "t"
    df_t: int = 6,             # utilisé si dist="t"
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo VaR/ES sur un portefeuille linéaire:
      pnl = sum_i positions[i] * returns[i]

    returns: DataFrame (rows=time, cols=symbols), rendements arithmétiques (daily)
    alpha: ex 0.99 => VaR au quantile 99% de la distribution des pertes
    """
    if returns.empty:
        raise ValueError("returns est vide")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha doit être dans (0,1)")
    if n_sims <= 0:
        raise ValueError("n_sims doit être > 0")

    cols = list(returns.columns)
    missing = [c for c in cols if c not in positions]
    if missing:
        raise ValueError(f"positions manquantes pour: {missing}")

    r = returns.dropna().copy()
    if len(r) < 10:
        raise ValueError("Pas assez de points pour estimer une covariance stable")

    mu = r.mean().to_numpy(dtype=float)                 # (n_assets,)
    cov = r.cov().to_numpy(dtype=float)                 # (n_assets, n_assets)
    n = cov.shape[0]

    # Cholesky robuste (jitter si matrice quasi-singulière)
    chol = None
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            chol = np.linalg.cholesky(cov + np.eye(n) * eps)
            break
        except np.linalg.LinAlgError:
            continue
    if chol is None:
        raise ValueError("Covariance non décomposable (même avec jitter).")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_sims, n))

    dist = dist.lower().strip()
    if dist == "normal":
        sims = z @ chol.T + mu
    elif dist in ("t", "student", "student-t", "student_t"):
        if df_t <= 2:
            raise ValueError("df_t doit être > 2 (variance finie)")
        g = rng.chisquare(df_t, size=(n_sims, 1))
        z_t = z / np.sqrt(g / df_t)
        sims = z_t @ chol.T + mu
    else:
        raise ValueError("dist doit être 'normal' ou 't'")

    w = np.array([positions[c] for c in cols], dtype=float)  # notionnels EUR
    pnl = sims @ w
    loss = -pnl

    var = float(np.quantile(loss, alpha))
    tail = loss[loss >= var]
    es = float(tail.mean()) if len(tail) > 0 else var

    return var, es
