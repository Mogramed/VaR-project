from __future__ import annotations

import math
import numpy as np
from scipy.stats import chi2
import pandas as pd


def exceptions(pnl: pd.Series, var_value: float) -> pd.Series:
    """
    Retourne un booléen par date : True si perte > VaR (exception).
    pnl : gains/pertes (EUR). pnl < 0 => perte.
    var_value : VaR positive (EUR) calculée sur la distribution des pertes.
    """
    pnl = pnl.dropna().astype(float)
    losses = -pnl  # pertes positives
    return losses > float(var_value)


def kupiec_test(num_exceptions: int, n: int, alpha: float) -> dict:
    """
    Test de Kupiec (Unconditional Coverage).
    H0: probabilité d'exception = (1-alpha)

    Retourne LR_uc et p_value (approx chi2 à 1 ddl).
    """
    if n <= 0:
        raise ValueError("n doit être > 0")
    if not (0 < alpha < 1):
        raise ValueError("alpha doit être entre 0 et 1")

    p = 1.0 - float(alpha)   # proba théorique d'exception
    x = int(num_exceptions)

    # Évite log(0)
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)

    # proportion observée
    phat = x / n
    phat = min(max(phat, eps), 1 - eps)

    # log-likelihood sous H0 et sous H1
    ll_h0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll_h1 = (n - x) * math.log(1 - phat) + x * math.log(phat)

    LR_uc = -2 * (ll_h0 - ll_h1)

    # p-value approx: chi-square(1)
    # p = 1 - CDF_chi2(LR, df=1)
    # CDF chi2 df=1 => via erfc(sqrt(LR/2))
    p_value = float(chi2.sf(LR_uc, df=1))

    return {"LR_uc": float(LR_uc), "p_value": float(p_value), "p": float(p), "phat": float(phat)}


def kupiec_pof_test(exceptions_count: int, total_obs: int, confidence_level: float = 0.99) -> dict:
    """
    Test de Kupiec (Proportion of Failures).
    Vérifie si le nombre d'exceptions observées est cohérent avec le niveau de confiance théorique.
    H0: Le modèle est correct.
    """
    p_hat = exceptions_count / total_obs
    p = 1 - confidence_level

    # Eviter division par zéro
    if p_hat == 0:
        return {"reject_h0": False, "lr": 0.0}

    # Ratio de vraisemblance (Likelihood Ratio)
    # Formule standard Kupiec 1995
    ln_part1 = (total_obs - exceptions_count) * np.log(1 - p) + exceptions_count * np.log(p)
    ln_part2 = (total_obs - exceptions_count) * np.log(1 - p_hat) + exceptions_count * np.log(p_hat)

    lr_stat = -2 * (ln_part1 - ln_part2)

    # Seuil critique Chi-2 à 1 degré de liberté (souvent 3.84 pour 95% de confiance sur le test lui-même)
    critical_value = chi2.ppf(0.95, df=1)

    return {
        "lr_stat": lr_stat,
        "critical_value": critical_value,
        "reject_h0": lr_stat > critical_value,  # Si True, le modèle est rejeté (trop ou trop peu d'exceptions)
        "actual_freq": p_hat,
        "expected_freq": p
    }



def basel_traffic_light(num_exceptions_250: int) -> str:
    """
    Règle simplifiée “traffic light” sur 250 jours :
    - Vert : 0 à 4
    - Orange : 5 à 9
    - Rouge : 10+
    (version pédagogique)
    """
    x = int(num_exceptions_250)
    if x <= 4:
        return "GREEN"
    if x <= 9:
        return "AMBER"
    return "RED"

