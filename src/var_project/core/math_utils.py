from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable

import numpy as np


def check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")


def normal_ppf(p: float) -> float:
    """Inverse CDF of N(0,1) without scipy."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    return NormalDist().inv_cdf(p)


def normal_pdf(x: float) -> float:
    """PDF of N(0,1) without scipy."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def as_1d_float_array(x: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float).reshape(-1)
    return arr


def nanquantile(x: Iterable[float] | np.ndarray, q: float) -> float:
    arr = as_1d_float_array(x)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def safe_std(x: Iterable[float] | np.ndarray, ddof: int = 1) -> float:
    arr = as_1d_float_array(x)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=ddof))
