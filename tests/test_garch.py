from __future__ import annotations

import numpy as np

from var_project.risk.garch import garch_var_es


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
