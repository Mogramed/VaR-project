from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict, List
import re

@dataclass(frozen=True)
class CompareSummary:
    n: int
    exceptions: Dict[str, int]


def summarize_compare_csv(df: pd.DataFrame) -> CompareSummary:
    exc_cols = [c for c in df.columns if c.startswith("exc_")]
    exc = {c.replace("exc_", ""): int(df[c].sum()) for c in exc_cols}
    return CompareSummary(n=len(df), exceptions=exc)


def _parse_alpha_from_name(stem: str) -> Optional[float]:
    m = re.search(r"alpha(\d{2})", stem)
    if not m:
        return None
    return int(m.group(1)) / 100.0


def _infer_models(df: pd.DataFrame) -> List[str]:
    models = []
    for c in df.columns:
        if c.startswith("var_"):
            models.append(c.replace("var_", ""))
    # stable ordering
    preferred = ["hist", "param", "mc", "ewma", "fhs"]
    models = sorted(set(models), key=lambda x: (preferred.index(x) if x in preferred else 999, x))
    return models


def _count_exceptions(df: pd.DataFrame, model: str) -> int:
    exc_col = f"exc_{model}"
    var_col = f"var_{model}"

    if exc_col in df.columns:
        return int(pd.to_numeric(df[exc_col], errors="coerce").fillna(0).astype(int).sum())

    # fallback: compute from pnl and var
    if "pnl" in df.columns and var_col in df.columns:
        pnl = pd.to_numeric(df["pnl"], errors="coerce")
        var = pd.to_numeric(df[var_col], errors="coerce")
        mask = (pnl < -var)
        return int(mask.fillna(False).sum())

    return 0