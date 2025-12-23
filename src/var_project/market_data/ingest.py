from __future__ import annotations

from pathlib import Path

from var_project.connectors.mt5_connector import MT5Connector
from var_project.market_data.store import save_raw_bars


def download_history(
    mt5: MT5Connector,
    symbol: str,
    timeframe: str,
    history_days: int,
    out_dir: Path,
) -> Path:
    # Convertit history_days -> nombre de bougies à télécharger
    bars_per_day = mt5.bars_per_day(timeframe)
    n_bars = history_days * bars_per_day

    # Sécurité : éviter de demander trop d’un coup
    n_bars = min(n_bars, 200_000)

    # Méthode stable : prend les N dernières bougies
    df = mt5.fetch_last_n_bars(symbol, timeframe, n_bars)

    filename = f"{symbol}_{timeframe}_{history_days}d.csv"
    out_path = out_dir / filename
    save_raw_bars(df, out_path)
    return out_path
