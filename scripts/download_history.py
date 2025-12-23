from __future__ import annotations

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
    ROOT = Path(__file__).resolve().parents[1]
    cfg_path = ROOT / "config" / "settings.yaml"
    cfg = load_config(cfg_path)

    mt5 = MT5Connector(cfg.mt5)
    mt5.init()

    try:
        out_dir = ROOT / "data" / "raw"
        for sym in cfg.symbols:
            out = download_history(
                mt5=mt5,
                symbol=sym,
                timeframe=cfg.data.timeframe,
                history_days=cfg.data.history_days,
                out_dir=out_dir,
            )
            print(f"[OK] Saved: {out}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
