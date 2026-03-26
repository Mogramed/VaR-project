from __future__ import annotations

from typing import Dict, Any


def mid_from_tick(tick: Any) -> float:
    """
    tick can be:
    - dict with bid/ask
    - MT5 tick object with .bid/.ask
    """
    bid = getattr(tick, "bid", None) if not isinstance(tick, dict) else tick.get("bid")
    ask = getattr(tick, "ask", None) if not isinstance(tick, dict) else tick.get("ask")
    if bid is None or ask is None:
        raise ValueError("tick must contain bid/ask")
    return (float(bid) + float(ask)) / 2.0


def last_close_from_bars(df) -> float:
    if df is None or len(df) == 0:
        raise ValueError("Empty bars dataframe")
    return float(df["close"].iloc[-1])
