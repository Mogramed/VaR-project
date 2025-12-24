from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from var_project.core.types import AppConfig, DataConfig, MT5Config
from var_project.market_data.clean import basic_clean_bars, data_quality_report
from var_project.market_data.transforms import compute_log_returns
from var_project.market_data.store import load_csv, save_processed_returns


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = DataConfig(**raw["data"])
    mt5 = MT5Config(**(raw.get("mt5") or {}))
    return AppConfig(
        base_currency=raw["base_currency"],
        symbols=raw["symbols"],
        data=data,
        mt5=mt5,
    )


def main():
    ROOT = Path(__file__).resolve().parents[1]
    cfg = load_config(ROOT / "config" / "settings.yaml")

    raw_dir = ROOT / "data" / "raw"
    out_dir = ROOT / "data" / "processed"
    # boucle sur les éléments de la config (plusieurs timeframes et history days)
    for sym in cfg.symbols:
        for tf in cfg.data.timeframes:
            for days in cfg.data.history_days_list:
                raw_path = raw_dir / f"{sym}_{tf}_{days}d.csv"
                if not raw_path.exists():
                    print(f"[SKIP] Missing raw file: {raw_path}")
                    continue

                bars = load_csv(raw_path)
                bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
                bars = bars.dropna(subset=["time"]).reset_index(drop=True)

                expected_minutes = 5 if tf.startswith("M") else 60 if tf.startswith("H") else 1440
                report = data_quality_report(bars, expected_minutes=expected_minutes)
                print(f"[QUALITY] {sym} {tf} {days}d: {report}")

                bars = basic_clean_bars(bars)
                rets = compute_log_returns(bars, price_col="close")

                out_path = out_dir / f"{sym}_{tf}_{days}d_returns.csv"
                save_processed_returns(rets, out_path)
                print(f"[OK] Saved returns: {out_path}")


if __name__ == "__main__":
    main()
