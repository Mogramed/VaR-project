from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(bars: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    bars: colonnes attendues: time + price_col
    Retourne: time, price, log_return (ln(Pt/Pt-1))
    """
    if "time" not in bars.columns:
        raise ValueError("bars doit contenir une colonne 'time'")
    if price_col not in bars.columns:
        raise ValueError(f"bars doit contenir la colonne '{price_col}'")

    df = bars.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    df["price"] = df[price_col].astype(float)
    df["log_return"] = np.log(df["price"]).diff()

    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df[["time", "price", "log_return"]]


def _bars_per_day_from_timeframe(timeframe: str) -> int:
    tf = timeframe.upper().strip()
    minutes_map = {
        "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5,
        "M10": 10, "M15": 15, "M30": 30,
        "H1": 60, "H2": 120, "H4": 240,
        "D1": 1440,
    }
    if tf not in minutes_map:
        raise ValueError(f"Timeframe inconnue: {timeframe}")
    minutes = minutes_map[tf]
    return int(1440 / minutes)


def intraday_to_daily_log_returns(
    intraday_returns: pd.DataFrame,
    timeframe: str,
    min_coverage: float = 0.90,
) -> pd.DataFrame:
    """
    Transforme une série intraday (time, log_return) en log_return journalier:

    - daily_log_return = somme des log_returns sur la journée (UTC)
    - coverage = nb_bars_observées / nb_bars_attendues (ex: M5 => 288)

    On conserve seulement les jours avec coverage >= min_coverage.
    """
    if "time" not in intraday_returns.columns or "log_return" not in intraday_returns.columns:
        raise ValueError("intraday_returns doit contenir 'time' et 'log_return'")

    df = intraday_returns.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "log_return"]).sort_values("time").reset_index(drop=True)

    expected = _bars_per_day_from_timeframe(timeframe)

    df["date"] = df["time"].dt.floor("D")

    daily = (
        df.groupby("date")
          .agg(
              daily_log_return=("log_return", "sum"),
              bars=("log_return", "size"),
          )
          .reset_index()
    )
    daily["expected_bars"] = expected
    daily["coverage"] = daily["bars"] / daily["expected_bars"]

    daily = daily[daily["coverage"] >= float(min_coverage)].reset_index(drop=True)
    return daily[["date", "daily_log_return", "bars", "expected_bars", "coverage"]]
