from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from var_project.risk.expected_shortfall import historical_var_es, RiskTail


@dataclass(frozen=True)
class StressScenario:
    name: str
    vol_multiplier: float = 1.0
    shock_pnl: float = 0.0  # additive shock in EUR


def apply_stress_to_pnl(pnl: pd.Series, scenario: StressScenario) -> pd.Series:
    x = pnl.astype(float).copy()
    mu = float(x.mean())
    # scale around mean (simple)
    x = mu + scenario.vol_multiplier * (x - mu)
    x = x + scenario.shock_pnl
    return x


def stress_report(pnl: pd.Series, alpha: float, scenarios: list[StressScenario]) -> Dict[str, RiskTail]:
    out: Dict[str, RiskTail] = {}
    for sc in scenarios:
        stressed = apply_stress_to_pnl(pnl, sc)
        out[sc.name] = historical_var_es(stressed, alpha)
    return out
