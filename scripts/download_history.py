from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from var_project.core.types import AppConfig, DataConfig, MT5Config
from var_project.connectors.mt5_connector import MT5Connector
from var_project.market_data.ingest import download_history


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config introuvable: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = DataConfig(**raw["data"])
    mt5 = MT5Config(**(raw.get("mt5") or {}))
    return AppConfig(
        base_currency=raw["base_currency"],
        symbols=raw["symbols"],
        data=data,
        mt5=mt5,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", type=str, default=None)
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    cfg_path = ROOT / "config" / "settings.yaml"
    cfg = load_config(cfg_path)

    # single timeframe / days if provided, else config
    timeframes = [args.timeframe] if args.timeframe else list(cfg.data.timeframes)
    days_list = [int(args.days)] if args.days is not None else list(cfg.data.history_days_list)

    mt5 = MT5Connector(cfg.mt5)
    mt5.init()

    try:
        out_dir = ROOT / "data" / "raw"
        for sym in cfg.symbols:
            for tf in timeframes:
                for days in days_list:
                    out = download_history(
                        mt5=mt5,
                        symbol=sym,
                        timeframe=tf,
                        history_days=int(days),
                        out_dir=out_dir,
                    )
                    print(f"[OK] Saved: {out}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
