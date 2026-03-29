from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from var_project.reporting.metrics import _infer_models, _count_exceptions


def plot_compare(df: pd.DataFrame, out_png: Path) -> None:
    """
    Plot pnl + var curves if present.
    Expected columns: date, pnl, var_hist, var_param, ...
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")

    plt.figure()
    if "pnl" in d.columns:
        plt.plot(d["date"], d["pnl"], label="PnL")

    for c in ["var_hist", "var_param", "var_mc", "var_ewma", "var_garch", "var_fhs"]:
        if c in d.columns:
            plt.plot(d["date"], -d[c], label=f"-{c}")  # show as pnl threshold

    plt.legend()
    plt.title("PnL vs VaR thresholds")
    plt.xlabel("Date")
    plt.ylabel("EUR")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_exceptions(df: pd.DataFrame, out_path: Path) -> None:
    models = _infer_models(df)
    counts = [_count_exceptions(df, m) for m in models]
    if not models:
        return

    plt.figure()
    plt.bar(models, counts)
    plt.title("Exceptions count (backtest window)")
    plt.xlabel("Model")
    plt.ylabel("Exceptions")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pnl_vs_var(df: pd.DataFrame, out_path: Path, model: str = "hist") -> None:
    if "date" not in df.columns or "pnl" not in df.columns or f"var_{model}" not in df.columns:
        return

    x = pd.to_datetime(df["date"], errors="coerce", utc=True)
    pnl = pd.to_numeric(df["pnl"], errors="coerce")
    var = pd.to_numeric(df[f"var_{model}"], errors="coerce")

    # plot losses as positive line for readability
    loss = (-pnl).clip(lower=0)

    plt.figure()
    plt.plot(x, loss, label="Loss (positive)")
    plt.plot(x, var, label=f"VaR ({model})")
    plt.title(f"Loss vs VaR ({model})")
    plt.xlabel("Date")
    plt.ylabel("EUR")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
