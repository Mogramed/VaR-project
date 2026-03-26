from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import logging
import logging.config


import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.var_project.core.types import AppConfig, DataConfig, MT5Config
from src.var_project.connectors.mt5_connector import MT5Connector
from src.var_project.market_data.store import load_csv
from src.var_project.market_data.transforms import intraday_to_daily_log_returns
from src.var_project.risk.var_models import historical_var, expected_shortfall, parametric_var
from src.var_project.engine.monte_carlo import mc_var_es
from src.var_project.risk.ewma import ewma_var_es
from src.var_project.engine.live_loop import run_live_loop, LiveLoopConfig


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


def get_positions(cfg_dict: dict, symbols: list[str]) -> dict[str, float]:
    positions = (cfg_dict.get("portfolio", {}) or {}).get("positions_eur", None)
    if not positions:
        return {s: 10_000.0 for s in symbols}
    return {s: float(positions.get(s, 0.0)) for s in symbols}


def parametric_es(pnl: pd.Series, alpha: float) -> float:
    x = pnl.dropna().astype(float).values
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    q = 1.0 - float(alpha)
    z = float(norm.ppf(q))  # négatif
    phi = float(norm.pdf(z))
    es_pnl = mu - sigma * (phi / q)      # E[PnL | PnL <= quantile]
    return float(-es_pnl)                # perte positive


def load_risk_limits(root: Path) -> dict:
    path = root / "config" / "risk_limits.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def traffic_light(live_loss: float, var_value: float, yellow: float = 1.0, red: float = 1.3) -> str:
    """
    Ratio-based: loss/var.
    - GREEN: live_loss <= yellow * VaR
    - AMBER: live_loss <= red * VaR
    - RED:   live_loss >  red * VaR
    """
    if var_value <= 0:
        return "UNKNOWN"
    ratio = live_loss / var_value
    if ratio <= yellow:
        return "GREEN"
    if ratio <= red:
        return "AMBER"
    return "RED"


def update_processed_returns_from_bars(
    mt5: MT5Connector,
    symbol: str,
    timeframe: str,
    days: int,
    out_path: Path,
    n_update_bars: int,
) -> None:
    """
    Fetch dernières bougies -> calc log_returns -> merge dans data/processed/*_returns.csv
    """
    bars = mt5.fetch_last_n_bars(symbol, timeframe=timeframe, n_bars=n_update_bars)

    if bars.empty or len(bars) < 3:
        return

    bars = bars.sort_values("time").reset_index(drop=True)

    close = bars["close"].astype(float).to_numpy()
    log_ret = np.diff(np.log(close))
    times = bars.loc[1:, "time"].reset_index(drop=True)

    new_df = pd.DataFrame({"time": times, "log_return": log_ret})
    new_df["time"] = pd.to_datetime(new_df["time"], utc=True, errors="coerce")
    new_df = new_df.dropna(subset=["time", "log_return"]).reset_index(drop=True)

    if out_path.exists():
        old = load_csv(out_path)
        old["time"] = pd.to_datetime(old["time"], utc=True, errors="coerce")
        old = old.dropna(subset=["time", "log_return"]).reset_index(drop=True)
        all_df = pd.concat([old, new_df], ignore_index=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_df = new_df

    all_df = (
        all_df.sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    # garde environ "days" de données (approx via bars_per_day)
    max_rows = int(mt5.bars_per_day(timeframe) * days) + 10_000
    if len(all_df) > max_rows:
        all_df = all_df.iloc[-max_rows:].reset_index(drop=True)

    all_df.to_csv(out_path, index=False)


def load_daily_simple_returns(
    ROOT: Path,
    symbols: list[str],
    timeframe: str,
    days: int,
    min_coverage: float,
) -> pd.DataFrame:
    """
    Idée identique à compare_var_models.py :
    processed log_returns -> daily log -> daily simple returns alignés.
    """
    processed_dir = ROOT / "data" / "processed"
    daily_map = []

    for sym in symbols:
        path = processed_dir / f"{sym}_{timeframe}_{days}d_returns.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed returns: {path}")

        df = load_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "log_return"]).reset_index(drop=True)

        daily = intraday_to_daily_log_returns(
            df[["time", "log_return"]],
            timeframe=timeframe,
            min_coverage=min_coverage,
        )
        daily[f"{sym}_ret"] = np.expm1(daily["daily_log_return"].astype(float))
        daily_map.append(daily[["date", f"{sym}_ret"]])

    merged = None
    for d in daily_map:
        merged = d if merged is None else merged.merge(d, on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame({"date": merged["date"]})
    for sym in symbols:
        out[sym] = merged[f"{sym}_ret"].astype(float)

    return out

def zone_from_ratio(r: float, green_max, orange_max) -> str:
    if r <= green_max:
        return "GREEN"
    if r <= orange_max:
        return "ORANGE"
    return "RED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")

    ap.add_argument("--timeframe", type=str, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.95)

    ap.add_argument("--n-update-bars", type=int, default=1500)

    # MC options
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--dist", type=str, default="normal", choices=["normal", "t"])
    ap.add_argument("--df-t", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    log = logging.getLogger("var_project.live")
    cfg_path = ROOT / "config" / "settings.yaml"
    cfg = load_config(cfg_path)
    cfg_dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    limits_cfg = load_risk_limits(ROOT)

    limits = yaml.safe_load((ROOT / "config" / "risk_limits.yaml").read_text(encoding="utf-8"))
    zones_cfg = limits["live_zones"]
    green_max = float(zones_cfg["green_max_ratio"])
    orange_max = float(zones_cfg["orange_max_ratio"])

    symbols = cfg.symbols
    positions = get_positions(cfg_dict, symbols)

    timeframe = (args.timeframe or cfg.data.timeframe)
    days = int(args.days or cfg.data.history_days)

    mt5 = MT5Connector(cfg.mt5)
    mt5.init()
    try:
        # Prépare symbols (MarketWatch)
        for s in symbols:
            mt5.ensure_symbol(s)

        def step():
            # 1) update returns (incremental)
            for s in symbols:
                out_path = ROOT / "data" / "processed" / f"{s}_{timeframe}_{days}d_returns.csv"
                update_processed_returns_from_bars(
                    mt5=mt5,
                    symbol=s,
                    timeframe=timeframe,
                    days=days,
                    out_path=out_path,
                    n_update_bars=args.n_update_bars,
                )

            # 2) daily aligned returns + pnl
            daily_rets = load_daily_simple_returns(
                ROOT=ROOT,
                symbols=symbols,
                timeframe=timeframe,
                days=days,
                min_coverage=args.min_coverage,
            )

            if len(daily_rets) <= args.window + 5:
                raise RuntimeError(f"Pas assez de jours ({len(daily_rets)}) pour window={args.window}")

            pnl = np.zeros(len(daily_rets), dtype=float)
            for s in symbols:
                pnl += float(positions.get(s, 0.0)) * daily_rets[s].to_numpy(dtype=float)

            daily_rets["pnl"] = pnl
            daily_rets = daily_rets.sort_values("date").reset_index(drop=True)

            pnl_train = daily_rets["pnl"].iloc[-args.window:]
            ret_train = daily_rets[symbols].iloc[-args.window:]

            # 3) VaR/ES snapshot
            var_hist = historical_var(pnl_train, args.alpha)
            es_hist = expected_shortfall(pnl_train, args.alpha)

            var_param = parametric_var(pnl_train, args.alpha)
            es_param = parametric_es(pnl_train, args.alpha)

            var_mc, es_mc = mc_var_es(
                returns=ret_train,
                positions=positions,
                alpha=args.alpha,
                n_sims=args.n_sims,
                dist=args.dist,
                df_t=args.df_t,
                seed=args.seed,
            )

            w = np.array([float(positions.get(s, 0.0)) for s in symbols], dtype=float)  # EUR notionals
            R = ret_train.to_numpy(dtype=float)  # (window, n_assets) simple returns

            var_ewma, es_ewma = ewma_var_es(
                returns=R,
                weights=w,
                alpha=float(args.alpha),
                lam=0.94,  # RiskMetrics classique (tu pourras le mettre en YAML après)
            )

            # 4) “live loss” proxy (intra-day) via mid vs yesterday close D1
            live_pnl = 0.0
            for s in symbols:
                mid = mt5.mid_price(s)
                d1 = mt5.fetch_last_n_bars(s, timeframe="D1", n_bars=2)
                if len(d1) >= 2:
                    prev_close = float(d1.iloc[-2]["close"])
                    ret_today = (mid / prev_close) - 1.0
                    live_pnl += float(positions.get(s, 0.0)) * ret_today

            live_loss = max(0.0, -live_pnl)

            # --- Limits / traffic light ---
            # fallback ratios if no yaml
            yellow_mult = float((limits_cfg.get("multipliers", {}) or {}).get("yellow", 1.0))
            red_mult = float((limits_cfg.get("multipliers", {}) or {}).get("red", 1.3))

            zone_hist = traffic_light(live_loss, var_hist, yellow=yellow_mult, red=red_mult)
            zone_ewma = traffic_light(live_loss, var_ewma, yellow=yellow_mult, red=red_mult)

            now = datetime.now(timezone.utc)

            ratio_hist = (live_loss / var_hist) if var_hist > 0 else 0.0
            ratio_ewma = (live_loss / var_ewma) if var_ewma > 0 else 0.0

            print(
                f"[LIVE] {now.isoformat(timespec='seconds')} "
                f"| VaR{int(args.alpha * 100)} hist={var_hist:.2f} param={var_param:.2f} mc={var_mc:.2f} ewma={var_ewma:.2f} "
                f"| live_pnl≈{live_pnl:.2f} live_loss≈{live_loss:.2f} "
                f"| zone(hist)={zone_from_ratio(ratio_hist, green_max, orange_max)} zone(ewma)={zone_from_ratio(ratio_ewma, green_max, orange_max)}"
            )

            snap = {
                "time_utc": now.isoformat(),
                "alpha": float(args.alpha),
                "timeframe": timeframe,
                "days": int(days),
                "window": int(args.window),
                "positions_eur": positions,
                "var": {"hist": var_hist, "param": var_param, "mc": var_mc, "ewma": var_ewma},
                "es": {"hist": es_hist, "param": es_param, "mc": es_mc, "ewma": es_ewma},
                "live_loss_proxy": float(live_loss),
                "breach_hist": bool(live_loss > var_hist),
                "limits" : {"multipliers": {"yellow": yellow_mult, "red": red_mult}, "zone_hist": zone_hist,"zone_ewma": zone_ewma}
            }

            out_dir = ROOT / "data" / "snapshots"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"live_snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"
            out_file.write_text(json.dumps(snap, indent=2), encoding="utf-8")

        run_live_loop(step, LiveLoopConfig(interval_seconds=args.interval, once=args.once))

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
