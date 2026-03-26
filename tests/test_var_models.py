import numpy as np
import pandas as pd
import pytest

from var_project.risk.var_models import historical_var, parametric_var


def test_var_positive_on_losses():
    rng = np.random.default_rng(0)
    pnl = pd.Series(rng.normal(loc=0.0, scale=50.0, size=500))
    v1 = historical_var(pnl, 0.95)
    v2 = parametric_var(pnl, 0.95)
    assert v1 >= 0.0
    assert v2 >= 0.0


def test_var_increases_with_alpha():
    rng = np.random.default_rng(1)
    pnl = pd.Series(rng.normal(loc=0.0, scale=50.0, size=1000))
    v95 = historical_var(pnl, 0.95)
    v99 = historical_var(pnl, 0.99)
    assert v99 >= v95
