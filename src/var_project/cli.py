from __future__ import annotations

import argparse
import logging
import logging.config
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml


# -------------------------
# Repo root + config loading
# -------------------------
def find_repo_root() -> Path:
    candidates = [Path(__file__).resolve(), Path.cwd().resolve()]
    for start in candidates:
        for p in [start] + list(start.parents):
            if (p / "pyproject.toml").exists() and (p / "config" / "settings.yaml").exists():
                return p
    raise RuntimeError("Repo root not found (pyproject.toml + config/settings.yaml).")


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def setup_logging(root: Path) -> None:
    cfg_path = root / "config" / "logging.yaml"
    if not cfg_path.exists():
        logging.basicConfig(level=logging.INFO)
        return

    cfg = load_yaml(cfg_path)

    # Patch file handler path to absolute + ensure dir exists
    handlers = cfg.get("handlers", {})
    file_h = handlers.get("file")
    if isinstance(file_h, dict) and "filename" in file_h:
        rel = Path(str(file_h["filename"]))
        abs_path = (root / rel).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        file_h["filename"] = str(abs_path)

    logging.config.dictConfig(cfg)


# -------------------------
# Settings schema (minimal)
# -------------------------
@dataclass(frozen=True)
class Defaults:
    timeframes: List[str]
    history_days_list: List[int]
    alpha: float
    window: int


def read_defaults(root: Path) -> Defaults:
    s = load_yaml(root / "config" / "settings.yaml")
    data = (s.get("data") or {})

    # timeframes
    tfs = data.get("timeframes") or []
    if isinstance(tfs, str):
        tfs = [tfs]
    if not tfs:
        tf = data.get("timeframe") or "M5"
        tfs = [tf]

    # history_days_list
    days_list = data.get("history_days_list")
    if days_list is None:
        # fallback: single value
        one = data.get("history_days", 365)
        days_list = [int(one)]
    elif isinstance(days_list, int):
        days_list = [days_list]
    days_list = [int(x) for x in days_list]

    risk = (s.get("risk") or {})
    alpha = float(risk.get("alpha", 0.95))
    window = int(risk.get("window", 250))

    return Defaults(timeframes=tfs, history_days_list=days_list, alpha=alpha, window=window)


# -------------------------
# Script runner
# -------------------------
def run_script(root: Path, script: str, extra_args: List[str]) -> int:
    script_path = root / "scripts" / script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    cmd = [sys.executable, str(script_path)] + extra_args
    logging.getLogger("var_project.cli").info("RUN %s", " ".join(cmd))
    return subprocess.call(cmd)


def latest_file(dir_path: Path, pattern: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


# -------------------------
# CLI
# -------------------------
def main() -> None:
    root = find_repo_root()
    setup_logging(root)
    log = logging.getLogger("var_project.cli")

    defaults = read_defaults(root)
    backtests_dir = root / "reports" / "backtests"

    p = argparse.ArgumentParser(prog="var_project")
    sub = p.add_subparsers(dest="cmd")  # not required

    # Single steps
    s = sub.add_parser("download-history")
    s.add_argument("--timeframe", default=None)
    s.add_argument("--days", type=int, default=None)

    s = sub.add_parser("build-returns")
    s.add_argument("--timeframe", default=None)
    s.add_argument("--days", type=int, default=None)

    s = sub.add_parser("compare")
    s.add_argument("--timeframe", default=None)
    s.add_argument("--days", type=int, default=None)
    s.add_argument("--alpha", type=float, default=None)
    s.add_argument("--window", type=int, default=None)

    s = sub.add_parser("score")
    s.add_argument("--csv", default=None)
    s.add_argument("--alpha", type=float, default=None)

    s = sub.add_parser("report")
    s.add_argument("--csv", default=None)

    s = sub.add_parser("live")
    s.add_argument("--once", action="store_true")
    s.add_argument("--interval", type=int, default=30)
    s.add_argument("--timeframe", default=None)
    s.add_argument("--days", type=int, default=None)

    # Pipeline
    s = sub.add_parser("pipeline")
    s.add_argument("--timeframe", default=None, help="If set, run pipeline for a single timeframe")
    s.add_argument("--days", type=int, default=None)
    s.add_argument("--alpha", type=float, default=None)
    s.add_argument("--window", type=int, default=None)
    s.add_argument("--skip-download", action="store_true")
    s.add_argument("--skip-returns", action="store_true")
    s.add_argument("--skip-compare", action="store_true")
    s.add_argument("--skip-score", action="store_true")
    s.add_argument("--skip-report", action="store_true")

    args, unknown = p.parse_known_args()

    # Default behaviour: no subcommand => pipeline
    if not args.cmd:
        log.info("No cmd provided -> defaulting to pipeline (from settings.yaml)")
        args = p.parse_args(["pipeline"] + unknown)
        unknown = []

    def tf_list() -> List[str]:
        if getattr(args, "timeframe", None):
            return [str(args.timeframe)]
        return list(defaults.timeframes)

    def days_list() -> List[int]:
        d = getattr(args, "days", None)
        if d is not None:
            return [int(d)]
        return list(defaults.history_days_list) if defaults.history_days_list else [365]

    def alpha() -> float:
        return float(getattr(args, "alpha", None) or defaults.alpha)

    def window() -> int:
        return int(getattr(args, "window", None) or defaults.window)

    # ----------------
    # Single-step cmds
    # ----------------
    if args.cmd == "download-history":
        extra: List[str] = []
        if args.timeframe:
            extra += ["--timeframe", str(args.timeframe)]
        if args.days is not None:
            extra += ["--days", str(int(args.days))]
        raise SystemExit(run_script(root, "download_history.py", unknown + extra))

    if args.cmd == "build-returns":
        extra = []
        if args.timeframe:
            extra += ["--timeframe", str(args.timeframe)]
        if args.days is not None:
            extra += ["--days", str(int(args.days))]
        raise SystemExit(run_script(root, "build_returns.py", unknown + extra))

    if args.cmd == "compare":
        tf = args.timeframe or defaults.timeframes[0]
        d = args.days if args.days is not None else (defaults.history_days_list[-1] if defaults.history_days_list else 365)
        a = args.alpha if args.alpha is not None else defaults.alpha
        w = args.window if args.window is not None else defaults.window
        extra = ["--timeframe", str(tf), "--days", str(int(d)), "--alpha", str(float(a)), "--window", str(int(w))]
        raise SystemExit(run_script(root, "compare_var_models.py", unknown + extra))

    if args.cmd == "score":
        csv = Path(args.csv) if args.csv else latest_file(backtests_dir, "compare_*.csv")
        if csv is None:
            raise FileNotFoundError("No compare_*.csv found in reports/backtests")
        if not csv.is_absolute():
            csv = root / csv
        a = args.alpha if args.alpha is not None else defaults.alpha
        raise SystemExit(run_script(root, "score_var_models.py", unknown + ["--csv", str(csv), "--alpha", str(float(a))]))

    if args.cmd == "report":
        csv = Path(args.csv) if args.csv else latest_file(backtests_dir, "compare_*.csv")
        if csv is None:
            raise FileNotFoundError("No compare_*.csv found in reports/backtests")
        if not csv.is_absolute():
            csv = root / csv
        raise SystemExit(run_script(root, "make_report.py", unknown + ["--csv", str(csv)]))

    if args.cmd == "live":
        extra = []
        if args.once:
            extra.append("--once")
        extra += ["--interval", str(args.interval)]
        if args.timeframe:
            extra += ["--timeframe", str(args.timeframe)]
        if args.days is not None:
            extra += ["--days", str(int(args.days))]
        raise SystemExit(run_script(root, "run_live_var.py", unknown + extra))

    # ----------
    # Pipeline
    # ----------
    if args.cmd == "pipeline":
        tfs = tf_list()
        ds = days_list()
        a = alpha()
        w = window()
        alpha_tag = int(round(a * 100))

        for tf in tfs:
            for d in ds:
                log.info("=== PIPELINE tf=%s days=%s alpha=%.2f window=%s ===", tf, d, a, w)

                if not args.skip_download:
                    code = run_script(root, "download_history.py", ["--timeframe", tf, "--days", str(d)] + unknown)
                    if code != 0:
                        raise SystemExit(code)

                if not args.skip_returns:
                    code = run_script(root, "build_returns.py", ["--timeframe", tf, "--days", str(d)] + unknown)
                    if code != 0:
                        raise SystemExit(code)

                compare_csv: Optional[Path] = None
                if not args.skip_compare:
                    extra = ["--timeframe", tf, "--days", str(d), "--alpha", str(a), "--window", str(w)]
                    code = run_script(root, "compare_var_models.py", extra + unknown)
                    if code != 0:
                        raise SystemExit(code)
                    # pick the compare file for this tf/days/alpha
                    pattern = f"compare_{tf}_{d}d_alpha{alpha_tag}*.csv"
                    compare_csv = latest_file(backtests_dir, pattern)

                if compare_csv is None:
                    # fallback (still safe)
                    compare_csv = latest_file(backtests_dir, "compare_*.csv")

                if compare_csv is None:
                    raise FileNotFoundError("Pipeline expected compare_*.csv in reports/backtests but found none.")

                if not args.skip_score:
                    code = run_script(root, "score_var_models.py", ["--csv", str(compare_csv), "--alpha", str(a)] + unknown)
                    if code != 0:
                        raise SystemExit(code)

                if not args.skip_report:
                    code = run_script(root, "make_report.py", ["--csv", str(compare_csv)] + unknown)
                    if code != 0:
                        raise SystemExit(code)

                log.info("PIPELINE OK for tf=%s days=%s. Latest compare: %s", tf, d, compare_csv.name)

        raise SystemExit(0)

    raise SystemExit(2)


if __name__ == "__main__":
    main()
