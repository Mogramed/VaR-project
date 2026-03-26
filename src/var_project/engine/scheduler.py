from __future__ import annotations

import re


_TIMEFRAME_RE = re.compile(r"^([MHD])(\d+)$", re.IGNORECASE)


def timeframe_to_minutes(tf: str) -> int:
    """
    M5 -> 5
    H1 -> 60
    D1 -> 1440
    """
    m = _TIMEFRAME_RE.match(tf.strip())
    if not m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    unit = m.group(1).upper()
    n = int(m.group(2))
    if unit == "M":
        return n
    if unit == "H":
        return 60 * n
    if unit == "D":
        return 1440 * n
    raise ValueError(f"Unsupported timeframe: {tf}")
