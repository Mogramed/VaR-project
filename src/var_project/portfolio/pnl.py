from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def align_returns(returns_by_symbol: Dict[str, pd.DataFrame], time_col: str = "time") -> pd.DataFrame:
    """
    Input: {sym: df(time, ret)}
    Output: wide df indexed by time, columns=symbols
    """
    frames = []
    for sym, df in returns_by_symbol.items():
        d = df.copy()
        if time_col in d.columns:
            d[time_col] = pd.to_datetime(d[time_col])
            d = d.set_index(time_col)
        d = d.rename(columns={"ret": sym, "return": sym, "returns": sym})
        if sym not in d.columns:
            # try first numeric column
            num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
            if not num_cols:
                raise ValueError(f"No returns column found for {sym}")
            d = d.rename(columns={num_cols[0]: sym})
        frames.append(d[[sym]])

    wide = pd.concat(frames, axis=1, join="inner").sort_index()
    return wide


def portfolio_pnl_from_returns(returns_wide: pd.DataFrame, positions_eur: Dict[str, float]) -> pd.Series:
    """
    Simple PnL approximation: pnl_t = sum_i notional_i * return_i,t
    (works when notional is in EUR exposure notionnel)
    """
    cols = [c for c in returns_wide.columns if c in positions_eur]
    w = np.array([positions_eur[c] for c in cols], dtype=float)
    mat = returns_wide[cols].to_numpy(dtype=float)
    pnl = mat @ w
    return pd.Series(pnl, index=returns_wide.index, name="pnl")


def daily_from_intraday_pnl(pnl_intraday: pd.Series) -> pd.Series:
    """
    Aggregate intraday pnl to daily pnl (sum).
    """
    s = pnl_intraday.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("1D").sum().dropna()
