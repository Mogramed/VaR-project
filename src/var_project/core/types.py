from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class MT5Config:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None


@dataclass(frozen=True)
class DataConfig:
    timeframe: str
    history_days: int
    storage_format: str = "csv"  # paquets plus tard
    timezone: str = "UTC"


@dataclass(frozen=True)
class AppConfig:
    base_currency: str
    symbols: list[str]
    data: DataConfig
    mt5: MT5Config
