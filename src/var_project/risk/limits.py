from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass(frozen=True)
class LimitBreach:
    metric: str
    value: float
    limit: float
    level: str  # e.g. "WARN" / "BREACH"


def check_limits(metrics: Dict[str, float], limits_cfg: Dict[str, Any]) -> List[LimitBreach]:
    """
    limits.yaml example:
      var_95_eur:
        warn: 100
        breach: 150
      var_99_eur:
        warn: 150
        breach: 200
    """
    breaches: List[LimitBreach] = []
    for metric, rule in (limits_cfg or {}).items():
        if metric not in metrics:
            continue
        v = float(metrics[metric])
        warn = rule.get("warn")
        breach = rule.get("breach")

        if breach is not None and v >= float(breach):
            breaches.append(LimitBreach(metric, v, float(breach), "BREACH"))
        elif warn is not None and v >= float(warn):
            breaches.append(LimitBreach(metric, v, float(warn), "WARN"))

    return breaches
