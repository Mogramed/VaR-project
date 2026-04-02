# src/var_project/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class MT5Config:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    path: Optional[str] = None
    timeout_ms: Optional[int] = None
    portable: bool = False
    agent_base_url: Optional[str] = None
    agent_api_key: Optional[str] = None
    execution_enabled: bool = True
    magic: int = 420001
    deviation_points: int = 20
    comment_prefix: str = "var_risk_desk"
    live_enabled: bool = True
    live_poll_seconds: float = 2.0
    live_history_poll_seconds: float = 30.0
    live_history_lookback_minutes: int = 180
    live_event_buffer_size: int = 500
    live_stale_after_seconds: float = 6.0


@dataclass(frozen=True)
class DataConfig:
    timeframes: List[str]
    history_days_list: List[int]
    market_history_days: int | None = None
    market_retention_days: dict[str, int] = field(default_factory=dict)
    tick_retention_days: int = 30
    tick_archive_dir: str = "data/market_ticks"
    tick_archive_format: str = "parquet"
    storage_format: str = "csv"     # paquets plus tard
    timezone: str = "UTC"

    def __post_init__(self):
        if not self.timeframes:
            raise ValueError("DataConfig.timeframes must be non-empty")
        if not self.history_days_list:
            raise ValueError("DataConfig.history_days_list must be non-empty")
        if self.market_history_days is not None and int(self.market_history_days) <= 0:
            raise ValueError("DataConfig.market_history_days must be positive when provided")
        if int(self.tick_retention_days) <= 0:
            raise ValueError("DataConfig.tick_retention_days must be positive")
        invalid_retention = [
            tf for tf, days in dict(self.market_retention_days or {}).items() if int(days) <= 0
        ]
        if invalid_retention:
            raise ValueError(f"DataConfig.market_retention_days contains invalid values for {invalid_retention}")

    # --- Backward compatible aliases ---
    @property
    def timeframe(self) -> str:
        if not self.timeframes:
            raise ValueError("config.data.timeframes is empty")
        return str(self.timeframes[0])

    @property
    def history_days(self) -> int:
        if not self.history_days_list:
            raise ValueError("data.history_days_list est vide dans settings.yaml")
        return int(self.history_days_list[0])

    @property
    def backfill_days(self) -> int:
        configured = self.market_history_days
        if configured is not None:
            return int(configured)
        return max(int(item) for item in self.history_days_list)

    @property
    def retention_days_by_timeframe(self) -> dict[str, int]:
        configured = {
            str(timeframe).upper(): int(days)
            for timeframe, days in dict(self.market_retention_days or {}).items()
        }
        if configured:
            return configured
        backfill = int(self.backfill_days)
        return {"H1": backfill, "D1": backfill, "M1": min(backfill, 180)}


@dataclass(frozen=True)
class AppConfig:
    base_currency: str
    symbols: list[str]
    data: DataConfig
    mt5: MT5Config
