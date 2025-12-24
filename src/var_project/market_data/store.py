from __future__ import annotations
from pathlib import Path
import pandas as pd

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_raw_bars(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)

def load_raw_bars(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_processed_returns(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
