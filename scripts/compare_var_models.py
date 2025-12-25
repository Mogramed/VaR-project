from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
from scipy.stats import norm

from var_project.market_data.store import load_csv
from var_project.market_data.transforms import intraday_to_daily_log_returns
from var_project.risk.var_models import historical_var, expected_shortfall, parametric_var
from var_project.engine.monte_carlo import mc_var_es
from var_project.risk.ewma import ewma_var_es
from var_project.risk.fhs import fhs_var_es



def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_positions(cfg: dict, symbols: list[str]) -> dict[str, float]:
    positions = (cfg.get("portfolio", {}) or {}).get("positions_eur", None)
    if not positions:
        return {s: 10_000.0 for s in symbols}
    return {s: float(positions.get(s, 0.0)) for s in symbols}


def parametric_es(pnl: pd.Series, alpha: float) -> float:
    """
    ES paramétrique sous hypothèse PnL ~ Normal(mu, sigma^2).
    On regarde la queue gauche du PnL (pertes).
    ES_loss = -E[PnL | PnL <= q], où q = quantile_{1-alpha}(PnL)
    """
    x = pnl.dropna().astype(float).values
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    q = 1.0 - float(alpha)
    z = float(norm.ppf(q))             # négatif
    phi = float(norm.pdf(z))
    # E[X | X <= mu + sigma z] = mu - sigma * phi/q
    es_pnl = mu - sigma * (phi / q)
    return float(-es_pnl)              # perte positive


def load_daily_simple_returns(
    ROOT: Path,
    symbols: list[str],
    timeframe: str,
    days: int,
    min_coverage: float,
) -> pd.DataFrame:
    """
    Charge les returns intraday (log_return), agrège en daily log_return,
    convertit en daily simple return: exp(daily_log_return)-1
    et aligne les symboles sur les mêmes dates (inner join).
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

        # simple daily return
        daily[f"{sym}_ret"] = np.expm1(daily["daily_log_return"].astype(float))
        daily_map.append(daily[["date", f"{sym}_ret"]])

    merged = None
    for d in daily_map:
        merged = d if merged is None else merged.merge(d, on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)

    # rename columns to symbols
    out = pd.DataFrame({"date": merged["date"]})
    for sym in symbols:
        out[sym] = merged[f"{sym}_ret"].astype(float)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="M5")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.95)

    # Monte Carlo
    parser.add_argument("--n-sims", type=int, default=20000)
    parser.add_argument("--dist", type=str, default="normal", choices=["normal", "t"])
    parser.add_argument("--df-t", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    cfg = load_yaml(ROOT / "config" / "settings.yaml")
    symbols = cfg["symbols"]
    positions = get_positions(cfg, symbols)
    weights = np.array([positions[s] for s in symbols], dtype=float)

    # 1) Daily returns (par symbole) alignés
    daily_rets = load_daily_simple_returns(
        ROOT=ROOT,
        symbols=symbols,
        timeframe=args.timeframe,
        days=args.days,
        min_coverage=args.min_coverage,
    )

    if len(daily_rets) <= args.window + 5:
        raise RuntimeError(f"Pas assez de jours ({len(daily_rets)}) pour window={args.window}")

    # 2) PnL portefeuille
    pnl = np.zeros(len(daily_rets), dtype=float)
    for sym in symbols:
        pnl += float(positions.get(sym, 0.0)) * daily_rets[sym].to_numpy(dtype=float)

    daily_rets["pnl"] = pnl
    daily_rets = daily_rets.sort_values("date").reset_index(drop=True)

    # 3) Rolling compare (hist / param / MC)
    out_rows = []
    for i in range(args.window, len(daily_rets)):
        window_df = daily_rets.iloc[i - args.window : i].copy()
        pnl_train = window_df["pnl"]
        pnl_today = float(daily_rets.loc[i, "pnl"])

        # returns matrix (assets) for MC
        ret_train = window_df[symbols]

        # Historical
        var_hist = historical_var(pnl_train, args.alpha)
        es_hist = expected_shortfall(pnl_train, args.alpha)

        # Parametric (Normal on PnL)
        var_param = parametric_var(pnl_train, args.alpha)
        es_param = parametric_es(pnl_train, args.alpha)

        # Monte Carlo (Normal or t)
        var_mc, es_mc = mc_var_es(
            returns=ret_train,
            positions=positions,
            alpha=args.alpha,
            n_sims=args.n_sims,
            dist=args.dist,
            df_t=args.df_t,
            seed=args.seed,
        )

        # EWMA
        var_ewma, es_ewma = ewma_var_es(
            returns=ret_train.to_numpy(dtype=float),
            weights=weights,
            alpha=args.alpha,
            lam=0.94,
        )

        #FHS
        var_fhs, es_fhs = fhs_var_es(
            pnl_train=pnl_train.to_numpy(dtype=float),
            alpha=args.alpha,
            lam=0.94,
        )

        out_rows.append({
            "date": daily_rets.loc[i, "date"],
            "pnl": pnl_today,

            "var_hist": var_hist,
            "es_hist": es_hist,
            "exc_hist": int((-pnl_today) > var_hist),

            "var_param": var_param,
            "es_param": es_param,
            "exc_param": int((-pnl_today) > var_param),

            "var_mc": var_mc,
            "es_mc": es_mc,
            "exc_mc": int((-pnl_today) > var_mc),

            "var_ewma": var_ewma,
            "es_ewma": es_ewma,
            "exc_ewma": int((-pnl_today) > var_ewma),

            "var_fhs": var_fhs,
            "es_fhs": es_fhs,
            "exc_fhs": int((-pnl_today) > var_fhs),

        })

    df = pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)

    print(f"[COMPARE] n_test={len(df)} alpha={args.alpha} window={args.window}")
    print(
        f"[COMPARE] exceptions_hist={df['exc_hist'].sum()} | "
        f"exceptions_param={df['exc_param'].sum()} | "
        f"exceptions_mc={df['exc_mc'].sum()} | "
        f"exceptions_ewma={df['exc_ewma'].sum()} |"
        f"exceptions_fhs={df['exc_fhs'].sum()}"
    )

    print(f"[MC] n_sims={args.n_sims} dist={args.dist} df_t={args.df_t} seed={args.seed}")

    out_path = ROOT / "reports" / "backtests" / (
        f"compare_{args.timeframe}_{args.days}d_alpha{int(args.alpha * 100)}_mc{args.dist}_ewma_fhs.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
