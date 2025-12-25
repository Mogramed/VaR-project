from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np

from var_project.market_data.store import load_csv
from var_project.market_data.transforms import intraday_to_daily_log_returns
from var_project.risk.var_models import historical_var
from var_project.risk.backtesting import exceptions, kupiec_test, basel_traffic_light


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_positions(cfg: dict, symbols: list[str]) -> dict[str, float]:
    portfolio = cfg.get("portfolio", {}) or {}
    positions = portfolio.get("positions_eur", None)
    if not positions:
        return {s: 10_000.0 for s in symbols}
    return {s: float(positions.get(s, 0.0)) for s in symbols}


def build_daily_portfolio_pnl(
    ROOT: Path,
    symbols: list[str],
    timeframe: str,
    days: int,
    min_coverage: float,
    positions: dict[str, float],
) -> pd.Series:
    processed_dir = ROOT / "data" / "processed"

    daily_map = {}
    for sym in symbols:
        path = processed_dir / f"{sym}_{timeframe}_{days}d_returns.csv"
        df = load_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "log_return"]).reset_index(drop=True)

        daily = intraday_to_daily_log_returns(df[["time", "log_return"]], timeframe=timeframe, min_coverage=min_coverage)
        daily = daily.rename(columns={"daily_log_return": f"{sym}_daily_log_return"})
        daily_map[sym] = daily[["date", f"{sym}_daily_log_return"]]

    merged = None
    for sym in symbols:
        merged = daily_map[sym] if merged is None else merged.merge(daily_map[sym], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    pnl = np.zeros(len(merged), dtype=float)
    for sym in symbols:
        notional = float(positions.get(sym, 0.0))
        r = np.expm1(merged[f"{sym}_daily_log_return"].astype(float).values)
        pnl += notional * r

    return pd.Series(pnl, index=merged["date"], name="portfolio_pnl_eur")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="M5")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--window", type=int, default=250, help="fenêtre rolling pour estimer la VaR")
    parser.add_argument("--alpha", type=float, default=0.95, help="niveau (ex: 0.95)")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    cfg = load_yaml(ROOT / "config" / "settings.yaml")

    symbols = cfg["symbols"]
    positions = get_positions(cfg, symbols)

    pnl = build_daily_portfolio_pnl(
        ROOT=ROOT,
        symbols=symbols,
        timeframe=args.timeframe,
        days=args.days,
        min_coverage=args.min_coverage,
        positions=positions,
    )

    if len(pnl) <= args.window + 5:
        raise RuntimeError(f"Pas assez de jours ({len(pnl)}) pour une fenêtre {args.window}")

    # Rolling backtest
    dates = pnl.index
    var_series = []
    exc_series = []

    for i in range(args.window, len(pnl)):
        train = pnl.iloc[i - args.window : i]
        var_t = historical_var(train, args.alpha)
        var_series.append(var_t)

        # exception jour i ?
        pnl_today = pnl.iloc[i]
        is_exc = (-pnl_today) > var_t
        exc_series.append(bool(is_exc))

    test_dates = dates[args.window:]
    var_s = pd.Series(var_series, index=test_dates, name="VaR")
    exc_s = pd.Series(exc_series, index=test_dates, name="Exception")

    n = len(exc_s)
    x = int(exc_s.sum())
    rate = x / n
    print(f"[BACKTEST] alpha={args.alpha:.2f} | window={args.window} | n_test={n}")
    print(f"[BACKTEST] exceptions={x} | rate={rate:.4f} | expected={1-args.alpha:.4f}")

    k = kupiec_test(x, n, args.alpha)
    print(f"[KUPIEC] LR_uc={k['LR_uc']:.3f} | p_value={k['p_value']:.4f} | phat={k['phat']:.4f}")

    # Traffic light sur les 250 derniers jours de test (si assez long)
    last_250 = exc_s.iloc[-250:] if len(exc_s) >= 250 else exc_s
    x250 = int(last_250.sum())
    print(f"[TL] exceptions_last_{len(last_250)}={x250} => {basel_traffic_light(x250)}")

    # (Optionnel) sauvegarde d’un CSV backtest
    out = pd.concat([pnl.loc[test_dates].rename("PnL"), var_s, exc_s], axis=1)
    out_path = ROOT / "reports" / "backtests" / f"backtest_{args.timeframe}_{args.days}d_alpha{int(args.alpha*100)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=True)
    print(f"[OK] Saved backtest: {out_path}")


if __name__ == "__main__":
    main()
