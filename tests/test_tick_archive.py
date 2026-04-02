from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from var_project.storage.tick_archive import archive_ticks, summarize_tick_archive


def test_tick_archive_writes_dedupes_and_summarizes(tmp_path: Path) -> None:
    root = tmp_path / "ticks"
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    batch = [
        {"time_utc": start.isoformat(), "bid": 1.1000, "ask": 1.1002, "last": 1.1001},
        {"time_utc": (start + timedelta(minutes=1)).isoformat(), "bid": 1.1001, "ask": 1.1003, "last": 1.1002},
        {"time_utc": (start + timedelta(minutes=1)).isoformat(), "bid": 1.1001, "ask": 1.1003, "last": 1.1002},
    ]

    first = archive_ticks(root=root, symbol="EURUSD", ticks=batch, retention_days=30)
    second = archive_ticks(root=root, symbol="EURUSD", ticks=batch[:2], retention_days=30)
    summary = summarize_tick_archive(root, symbols=["EURUSD"], stale_after_seconds=3600.0)

    assert first["rows"] == 2
    assert second["rows"] == 2
    assert summary["row_count"] == 2
    assert summary["symbol_count"] == 1
    assert summary["coverage_status"] == "healthy"
    assert summary["microstructure"]["avg_spread_bps"] is not None
    assert summary["symbols"][0]["tick_count_5m"] >= 2
    assert summary["symbols"][0]["realized_vol_30m"] is not None
