from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class PortfolioSpec:
    base_ccy: str = "EUR"
    default_notional_eur: float = 10_000.0
    positions_eur: Optional[Dict[str, float]] = None


def build_positions_eur(symbols: Iterable[str], portfolio_cfg: dict | None) -> Dict[str, float]:
    """
    Returns dict symbol -> notional in EUR.

    settings.yaml supports:
    portfolio:
      positions_eur:
        EURUSD: 10000
        USDJPY: 10000
    """
    symbols = [s for s in symbols]
    if not portfolio_cfg:
        return {s: 10_000.0 for s in symbols}

    pos = (portfolio_cfg.get("positions_eur") or None)
    if not pos:
        return {s: float(portfolio_cfg.get("default_notional_eur", 10_000.0)) for s in symbols}

    out = {}
    for s in symbols:
        out[s] = float(pos.get(s, 0.0))
    return out


def gross_notional(positions_eur: Dict[str, float]) -> float:
    return float(sum(abs(v) for v in positions_eur.values()))
