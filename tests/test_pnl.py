import pandas as pd
import numpy as np

from var_project.portfolio.pnl import portfolio_pnl_from_returns


def test_portfolio_pnl_linear():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    returns = pd.DataFrame({"EURUSD": [0.01, -0.02, 0.0], "USDJPY": [0.0, 0.01, 0.02]}, index=idx)
    pos = {"EURUSD": 10_000.0, "USDJPY": 10_000.0}
    pnl = portfolio_pnl_from_returns(returns, pos)

    # day1: 100
    assert abs(pnl.iloc[0] - 100.0) < 1e-9
