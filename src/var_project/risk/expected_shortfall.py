from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from var_project.core.math_utils import check_alpha, nanquantile, normal_pdf, normal_ppf, safe_std


@dataclass(frozen=True)
class RiskTail:
    var: float
    es: float


def historical_var_es(pnl, alpha: float) -> RiskTail:
    """
    Convention:
    - pnl is P&L (positive good, negative loss)
    - VaR returned as positive loss number
    """
    check_alpha(alpha)
    x = np.asarray(pnl, dtype=float).reshape(-1)
    losses = -x
    var = nanquantile(losses, alpha)
    tail = losses[losses >= var]
    es = float(np.mean(tail)) if tail.size else float(var)
    return RiskTail(var=float(var), es=float(es))


def normal_parametric_var_es(pnl, alpha: float) -> RiskTail:
    """
    Gaussian parametric VaR/ES on pnl distribution.
    (Uses built-in normal approximation, no scipy.)
    """
    check_alpha(alpha)
    x = np.asarray(pnl, dtype=float).reshape(-1)
    mu = float(np.nanmean(x))
    sigma = safe_std(x, ddof=1)

    # Work on losses = -pnl
    # loss ~ N(-mu, sigma)
    z = normal_ppf(alpha)
    var = (-mu) + sigma * z

    # ES for normal losses: ES = mean + sigma * pdf(z)/(1-alpha)
    es = (-mu) + sigma * (normal_pdf(z) / (1.0 - alpha))
    return RiskTail(var=float(var), es=float(es))
