from __future__ import annotations

"""
One command to:
1) run compare_var_models.py
2) run score_var_models.py on the produced CSV
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="M5")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--window", type=int, default=250)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    compare_script = root / "scripts" / "compare_var_models.py"
    score_script = root / "scripts" / "score_var_models.py"

    # 1) compare
    subprocess.check_call([
        sys.executable, str(compare_script),
        "--timeframe", args.timeframe,
        "--days", str(args.days),
        "--alpha", str(args.alpha),
        "--window", str(args.window),
    ])

    csv_name = f"compare_{args.timeframe}_{args.days}d_alpha{int(args.alpha*100)}.csv"
    csv_path = root / "reports" / "backtests" / csv_name

    # 2) score
    subprocess.check_call([
        sys.executable, str(score_script),
        "--csv", str(csv_path),
        "--alpha", str(args.alpha),
    ])


if __name__ == "__main__":
    main()
