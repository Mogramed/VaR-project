from __future__ import annotations

import warnings

import numpy as np

from var_project.risk.garch import garch_var_es
from var_project.risk.expected_shortfall import normal_parametric_var_es


def test_garch_var_es_returns_positive_tail_metrics():
    rng = np.random.default_rng(7)
    pnl = rng.normal(loc=0.0, scale=14.0, size=240)

    tail = garch_var_es(pnl, alpha=0.95, dist="normal")

    assert tail.var >= 0.0
    assert tail.es >= tail.var


def test_garch_var_es_supports_student_t_innovations():
    rng = np.random.default_rng(21)
    pnl = rng.standard_t(df=6, size=260) * 11.0

    tail = garch_var_es(pnl, alpha=0.99, dist="t")

    assert tail.var >= 0.0
    assert tail.es >= tail.var


def test_garch_var_es_silences_non_convergence_and_falls_back(monkeypatch):
    class _ForecastSlice:
        def __init__(self, value: float):
            self._value = value

        @property
        def iloc(self):
            return self

        def __getitem__(self, index: int):
            return np.array([self._value], dtype=float)

    class _Forecast:
        def __init__(self):
            self.mean = _ForecastSlice(0.0)
            self.variance = _ForecastSlice(1.0)

    class _Result:
        convergence_flag = 4
        params = {}

        def forecast(self, horizon: int, reindex: bool = False):
            return _Forecast()

    class _Model:
        def fit(self, **kwargs):
            if kwargs.get("show_warning") is not False:
                warnings.warn("optimizer failed", RuntimeWarning)
            return _Result()

    pnl = np.linspace(-12.0, 14.0, 260)
    expected = normal_parametric_var_es(pnl, 0.95)

    monkeypatch.setattr("var_project.risk.garch.arch_model", lambda *args, **kwargs: _Model())

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tail = garch_var_es(pnl, alpha=0.95, dist="t")

    assert captured == []
    assert tail.var == expected.var
    assert tail.es == expected.es
