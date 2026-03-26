# src/var_project/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd


@dataclass(frozen=True)
class MT5Config:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None


@dataclass(frozen=True)
class DataConfig:
    timeframes: List[str]
    history_days_list: List[int]
    storage_format: str = "csv"     # paquets plus tard
    timezone: str = "UTC"

    def __post_init__(self):
        if not self.timeframes:
            raise ValueError("DataConfig.timeframes must be non-empty")
        if not self.history_days_list:
            raise ValueError("DataConfig.history_days_list must be non-empty")

    # --- Backward compatible aliases ---
    @property
    def timeframe(self) -> str:
        if not self.timeframes:
            raise ValueError("config.data.timeframes is empty")
        return str(self.timeframes[1])

    @property
    def history_days(self) -> int:
        if not self.history_days_list:
            raise ValueError("data.history_days_list est vide dans settings.yaml")
        return int(self.history_days_list[0])


@dataclass(frozen=True)
class AppConfig:
    base_currency: str
    symbols: list[str]
    data: DataConfig
    mt5: MT5Config
