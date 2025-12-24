from __future__ import annotations

import pandas as pd


def basic_clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimal:
    - garde colonnes essentielles si elles existent
    - supprime doublons de time
    - trie par time
    """
    if "time" not in bars.columns:
        raise ValueError("bars doit contenir 'time'")

    df = bars.copy()
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def data_quality_report(bars: pd.DataFrame, expected_minutes: int = 5) -> dict:
    """
    Petit rapport qualité:
    - nb lignes
    - nb doublons time
    - nb trous (delta > expected_minutes)
    """
    if "time" not in bars.columns:
        raise ValueError("bars doit contenir 'time'")

    df = bars.copy().sort_values("time")
    n_rows = len(df)
    n_dupes = int(df["time"].duplicated().sum())

    # calc deltas
    t = pd.to_datetime(df["time"], utc=True)
    delta = t.diff().dt.total_seconds().dropna()
    holes = int((delta > expected_minutes * 60).sum())

    return {
        "rows": n_rows,
        "duplicate_times": n_dupes,
        "holes_count": holes,
        "expected_minutes": expected_minutes,
    }
