from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import t as student_t

from var_project.core.math_utils import check_alpha, normal_pdf, normal_ppf
from var_project.risk.expected_shortfall import RiskTail, normal_parametric_var_es

try:
    from arch import arch_model
except Exception:  # pragma: no cover - fallback is tested instead
    arch_model = None


def _clean_pnl(pnl) -> np.ndarray:
    values = np.asarray(pnl, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _standardized_t_var_es(alpha: float, nu: float) -> tuple[float, float]:
    scale = math.sqrt(nu / (nu - 2.0))
    q_raw = float(student_t.ppf(alpha, df=nu))
    q_std = q_raw / scale
    tail_std = float(student_t.pdf(q_raw, df=nu)) * (nu + q_raw * q_raw)
    tail_std /= (nu - 1.0) * (1.0 - alpha) * scale
    return q_std, tail_std


def garch_var_es(
    pnl,
    alpha: float,
    *,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    mean: str = "constant",
) -> RiskTail:
    """
    One-step-ahead VaR / ES from a univariate GARCH(p, q) fit on portfolio PnL.

    Falls back to the Gaussian parametric model if the fit cannot be estimated
    robustly or if the optional `arch` dependency is unavailable.
    """

    check_alpha(alpha)
    x = _clean_pnl(pnl)
    if x.size < max(30, p + q + 5):
        return normal_parametric_var_es(x, alpha)

    if arch_model is None:
        return normal_parametric_var_es(x, alpha)

    arch_dist = "t" if str(dist).lower() in {"t", "student", "student_t", "student-t"} else "normal"
    arch_mean = "Zero" if str(mean).lower() == "zero" else "Constant"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                x,
                mean=arch_mean,
                vol="GARCH",
                p=int(p),
                q=int(q),
                dist=arch_dist,
                rescale=False,
            )
            result = model.fit(disp="off", update_freq=0)

        forecast = result.forecast(horizon=1, reindex=False)
        mu = float(np.asarray(forecast.mean.iloc[-1]).reshape(-1)[0])
        variance = float(np.asarray(forecast.variance.iloc[-1]).reshape(-1)[0])
        if not np.isfinite(mu) or not np.isfinite(variance) or variance < 0.0:
            raise ValueError("Non-finite GARCH forecast moments.")

        sigma = math.sqrt(max(variance, 1e-18))
        if arch_dist == "t":
            nu = float(result.params.get("nu", float("nan")))
            if not np.isfinite(nu) or nu <= 2.05:
                raise ValueError("Estimated Student-t nu is invalid for ES.")
            q_alpha, tail_alpha = _standardized_t_var_es(alpha, nu)
        else:
            q_alpha = normal_ppf(alpha)
            tail_alpha = normal_pdf(q_alpha) / (1.0 - alpha)

        var = (-mu) + sigma * q_alpha
        es = (-mu) + sigma * tail_alpha
        var = max(float(var), 0.0)
        es = max(float(es), var)
        return RiskTail(var=var, es=es)
    except Exception:
        return normal_parametric_var_es(x, alpha)
