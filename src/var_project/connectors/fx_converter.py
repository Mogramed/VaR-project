from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


def split_fx_symbol(symbol: str) -> Tuple[str, str]:
    """
    EURUSD -> ('EUR','USD')
    USDJPY -> ('USD','JPY')
    """
    s = symbol.strip().upper()
    if len(s) != 6:
        raise ValueError(f"Unsupported FX symbol format: {symbol}")
    return s[:3], s[3:]


@dataclass(frozen=True)
class FxRates:
    """
    Store mid rates as: 1 BASE = rate QUOTE
    Example: EURUSD=1.10 means 1 EUR = 1.10 USD
    """
    mid: Dict[str, float]  # symbol -> mid price

    def to_graph(self) -> Dict[str, Dict[str, float]]:
        g: Dict[str, Dict[str, float]] = {}
        for sym, px in self.mid.items():
            b, q = split_fx_symbol(sym)
            g.setdefault(b, {})[q] = float(px)
            g.setdefault(q, {})[b] = 1.0 / float(px) if px else 0.0
        return g

    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> float:
        """
        Convert amount from_ccy -> to_ccy using available pairs.
        Uses BFS to find any path.
        """
        f = from_ccy.upper()
        t = to_ccy.upper()
        if f == t:
            return float(amount)

        g = self.to_graph()
        if f not in g or t not in g:
            raise ValueError(f"No conversion path available ({f}->{t}). Available: {list(g.keys())}")

        # BFS over currencies
        q = deque([f])
        prev: Dict[str, Optional[str]] = {f: None}
        prev_rate: Dict[str, float] = {f: 1.0}

        while q:
            cur = q.popleft()
            if cur == t:
                break
            for nxt, rate in g.get(cur, {}).items():
                if nxt not in prev and rate:
                    prev[nxt] = cur
                    prev_rate[nxt] = prev_rate[cur] * rate
                    q.append(nxt)

        if t not in prev_rate:
            raise ValueError(f"No conversion path found ({f}->{t}) with given rates.")

        return float(amount) * prev_rate[t]
