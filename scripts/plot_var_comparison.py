from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Chemin du fichier compare_*.csv")
    args = parser.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # pertes positives pour comparer facilement
    loss = -df["pnl"].astype(float)

    plt.figure()
    plt.plot(df["date"], loss, label="Loss (-PnL)")

    if "var_hist" in df.columns:
        plt.plot(df["date"], df["var_hist"].astype(float), label="VaR Historical")
    if "var_param" in df.columns:
        plt.plot(df["date"], df["var_param"].astype(float), label="VaR Parametric")
    if "var_mc" in df.columns:
        plt.plot(df["date"], df["var_mc"].astype(float), label="VaR Monte Carlo")

    title = "Portfolio Loss vs VaR"
    if "var_mc" in df.columns:
        title += " (Hist / Param / MC)"
    else:
        title += " (Hist / Param)"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("EUR (loss / VaR)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

    out_path = path.with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
