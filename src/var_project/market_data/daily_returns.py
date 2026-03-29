from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from var_project.market_data.store import load_csv
from var_project.market_data.transforms import intraday_to_daily_log_returns


def load_daily_simple_returns_from_processed(
    root: Path,
    symbols: list[str],
    timeframe: str,
    days: int,
    min_coverage: float,
) -> pd.DataFrame:
    """
    Build aligned daily simple returns from processed intraday log-return CSVs.
    Returns a DataFrame with columns: date, <symbol1>, <symbol2>, ...
    """
    processed_dir = root / "data" / "processed"
    daily_frames: list[pd.DataFrame] = []

    for sym in symbols:
        path = processed_dir / f"{sym}_{timeframe}_{days}d_returns.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed returns: {path}")

        df = load_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "log_return"]).reset_index(drop=True)

        daily = intraday_to_daily_log_returns(
            df[["time", "log_return"]],
            timeframe=timeframe,
            min_coverage=min_coverage,
        )
        daily[f"{sym}_ret"] = np.expm1(daily["daily_log_return"].astype(float))
        daily_frames.append(daily[["date", f"{sym}_ret"]])

    merged = None
    for daily in daily_frames:
        merged = daily if merged is None else merged.merge(daily, on="date", how="inner")

    if merged is None:
        return pd.DataFrame(columns=["date", *symbols])

    merged = merged.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame({"date": merged["date"]})
    for sym in symbols:
        out[sym] = merged[f"{sym}_ret"].astype(float)

    return out
