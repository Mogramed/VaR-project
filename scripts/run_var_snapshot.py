from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np

from var_project.core.types import AppConfig, DataConfig, MT5Config
from var_project.market_data.store import load_csv
from var_project.market_data.transforms import intraday_to_daily_log_returns
from var_project.risk.var_models import historical_var, expected_shortfall


def load_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw


def get_positions(raw_cfg: dict, symbols: list[str]) -> dict[str, float]:
    """
    Retourne un dict {symbol: notional_eur}.
    Si absent dans config, défaut: 10k EUR long sur chaque symbole.
    """
    portfolio = raw_cfg.get("portfolio", {}) or {}
    positions = portfolio.get("positions_eur", None)

    if not positions:
        return {s: 10_000.0 for s in symbols}

    # On force float + on ignore symboles non listés
    out = {}
    for s in symbols:
        if s in positions:
            out[s] = float(positions[s])
        else:
            out[s] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default=None, help="Ex: M5, H1")
    parser.add_argument("--days", type=int, default=None, help="Ex: 30, 365")
    parser.add_argument("--min-coverage", type=float, default=0.90, help="coverage journalier minimum")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    cfg_path = ROOT / "config" / "settings.yaml"
    raw_cfg = load_config(cfg_path)

    symbols = raw_cfg["symbols"]
    timeframes = raw_cfg["data"]["timeframes"]
    days_list = raw_cfg["data"]["history_days_list"]

    tf = args.timeframe or timeframes[0]
    days = args.days or days_list[-1]  # par défaut : la plus grande fenêtre

    positions = get_positions(raw_cfg, symbols)

    processed_dir = ROOT / "data" / "processed"

    # --- 1) Daily log returns par symbole ---
    daily_map = {}
    for sym in symbols:
        path = processed_dir / f"{sym}_{tf}_{days}d_returns.csv"
        if not path.exists():
            raise FileNotFoundError(f"Fichier returns introuvable: {path}")

        df = load_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "log_return"]).reset_index(drop=True)

        daily = intraday_to_daily_log_returns(df[["time", "log_return"]], timeframe=tf, min_coverage=args.min_coverage)
        daily = daily.rename(columns={"daily_log_return": f"{sym}_daily_log_return"})
        daily_map[sym] = daily

        print(f"[INFO] {sym} {tf} {days}d -> daily rows kept: {len(daily)}")

    # --- 2) Alignement par date (intersection) ---
    merged = None
    for sym in symbols:
        d = daily_map[sym][["date", f"{sym}_daily_log_return"]]
        merged = d if merged is None else merged.merge(d, on="date", how="inner")

    if merged is None or len(merged) < 50:
        raise RuntimeError(f"Trop peu de jours après alignement: {0 if merged is None else len(merged)}")

    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"[INFO] Aligned daily sample size (days): {len(merged)}")

    # --- 3) Portfolio P&L (EUR) ---
    # simple_return = exp(log_return) - 1
    pnl = np.zeros(len(merged), dtype=float)

    gross = sum(abs(v) for v in positions.values()) or 1.0

    for sym in symbols:
        notional = float(positions.get(sym, 0.0))
        r = np.expm1(merged[f"{sym}_daily_log_return"].astype(float).values)  # simple return
        pnl += notional * r

    pnl_s = pd.Series(pnl, index=merged["date"], name="portfolio_pnl_eur")

    # --- 4) VaR & ES ---
    for alpha in (0.95, 0.99):
        var = historical_var(pnl_s, alpha)
        es = expected_shortfall(pnl_s, alpha)
        print(f"[RISK] alpha={alpha:.2f} | VaR={var:,.2f} EUR | ES={es:,.2f} EUR")

    # Quelques stats utiles
    print(f"[STATS] mean PnL={pnl_s.mean():.2f} EUR | std={pnl_s.std(ddof=1):.2f} EUR | gross_notional={gross:,.2f} EUR")
    print("[POSITIONS]", positions)


if __name__ == "__main__":
    main()
