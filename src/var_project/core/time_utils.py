from __future__ import annotations
from datetime import datetime, timedelta

def now_naive() -> datetime:
    # MT5 aime les datetime naïfs (sans tzinfo)
    return datetime.now()

def days_ago_naive(days: int) -> datetime:
    return now_naive() - timedelta(days=days)
