from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.var_project.reporting.render import render_daily_markdown


def find_latest(path: Path, pattern: str) -> Optional[Path]:
    files = list(path.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="compare_*.csv (optional). If omitted, latest is used.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    # 1) compare csv
    if args.csv:
        compare_csv = Path(args.csv)
        if not compare_csv.is_absolute():
            compare_csv = root / compare_csv
    else:
        compare_csv = find_latest(root / "reports" / "backtests", "compare_*.csv")
        if compare_csv is None:
            raise FileNotFoundError("No compare_*.csv found in reports/backtests")

    # 2) snapshot dir + limits
    snapshot_dir = root / "data" / "snapshots"
    limits_yaml = root / "config" / "risk_limits.yaml"

    out_dir = root / "reports" / "daily"
    md = render_daily_markdown(
        compare_csv=compare_csv,
        out_dir=out_dir,
        snapshot_dir=snapshot_dir,
        risk_limits_yaml=limits_yaml,
    )
    print(f"[OK] Daily report: {md}")


if __name__ == "__main__":
    main()
