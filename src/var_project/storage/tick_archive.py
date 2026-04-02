from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from var_project.storage.serialization import coerce_datetime


def _to_utc(value: Any) -> datetime | None:
    dt = coerce_datetime(value)
    if dt is None:
        return None
    return dt.astimezone(timezone.utc)


def _tick_mid(bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    if bid is not None and bid > 0:
        return bid
    if ask is not None and ask > 0:
        return ask
    return None


def normalize_ticks(symbol: str, ticks: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    normalized_symbol = str(symbol).upper()
    for tick in ticks:
        payload = dict(tick or {})
        time_utc = _to_utc(payload.get("time_utc") or payload.get("time"))
        if time_utc is None:
            continue
        bid = None if payload.get("bid") is None else float(payload.get("bid"))
        ask = None if payload.get("ask") is None else float(payload.get("ask"))
        last = None if payload.get("last") is None else float(payload.get("last"))
        mid = _tick_mid(bid, ask, last)
        spread = None if bid is None or ask is None else float(ask - bid)
        spread_bps = None if mid in {None, 0.0} or spread is None else float((spread / mid) * 10_000.0)
        rows.append(
            {
                "symbol": normalized_symbol,
                "time_utc": pd.Timestamp(time_utc),
                "bid": bid,
                "ask": ask,
                "last": last,
                "mid": mid,
                "spread": spread,
                "spread_bps": spread_bps,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "time_utc", "bid", "ask", "last", "mid", "spread", "spread_bps"])
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["symbol", "time_utc", "bid", "ask", "last"]).sort_values("time_utc")
    return frame.reset_index(drop=True)


def _partition_path(root: Path, symbol: str, timestamp: pd.Timestamp) -> Path:
    dt = timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC")
    return (
        root
        / f"symbol={str(symbol).upper()}"
        / f"date={dt.strftime('%Y-%m-%d')}"
        / f"hour={dt.strftime('%H')}"
        / "ticks.parquet"
    )


def _partition_start(parquet_path: Path) -> datetime | None:
    try:
        symbol_dir = parquet_path.parent.parent.parent.name
        date_dir = parquet_path.parent.parent.name
        hour_dir = parquet_path.parent.name
        if not symbol_dir.startswith("symbol=") or not date_dir.startswith("date=") or not hour_dir.startswith("hour="):
            return None
        day = date_dir.split("=", 1)[1]
        hour = hour_dir.split("=", 1)[1]
        return datetime.fromisoformat(f"{day}T{hour}:00:00+00:00")
    except Exception:
        return None


def _recent_partition_paths(root: Path, *, symbol: str, since: datetime) -> list[Path]:
    paths: list[Path] = []
    for parquet_path in (root / f"symbol={symbol.upper()}").glob("date=*/hour=*/ticks.parquet"):
        start = _partition_start(parquet_path)
        if start is None:
            continue
        if start + timedelta(hours=1) < since:
            continue
        paths.append(parquet_path)
    return sorted(paths)


def _load_recent_ticks(root: Path, *, symbol: str, since: datetime) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for parquet_path in _recent_partition_paths(root, symbol=symbol, since=since):
        try:
            frame = pd.read_parquet(
                parquet_path,
                columns=["time_utc", "bid", "ask", "last", "mid", "spread", "spread_bps"],
            )
        except Exception:
            continue
        if frame.empty:
            continue
        frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["time_utc"]).sort_values("time_utc")
        frame = frame[frame["time_utc"] >= pd.Timestamp(since)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["time_utc", "bid", "ask", "last", "mid", "spread", "spread_bps"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["time_utc", "bid", "ask", "last"]).sort_values("time_utc")
    return merged.reset_index(drop=True)


def _realized_vol_annualized(frame: pd.DataFrame, *, lookback_minutes: int) -> float | None:
    if frame.empty or "mid" not in frame:
        return None
    subset = frame.dropna(subset=["mid"]).copy()
    if len(subset) < 2:
        return None
    positive = subset["mid"].astype(float) > 0
    subset = subset[positive]
    if len(subset) < 2:
        return None
    log_returns = np.log(subset["mid"].astype(float) / subset["mid"].astype(float).shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if log_returns.empty:
        return None
    window_days = max(float(lookback_minutes) / 1440.0, 1.0 / 1440.0)
    realized_variance_daily = float(np.square(log_returns).sum()) / window_days
    annualized = math.sqrt(max(realized_variance_daily, 0.0) * 252.0)
    return float(annualized)


def _symbol_tick_quality(frame: pd.DataFrame, *, now: datetime, stale_after_seconds: float) -> str:
    if frame.empty:
        return "incomplete"
    latest_time = pd.to_datetime(frame["time_utc"].max(), utc=True, errors="coerce")
    if pd.isna(latest_time):
        return "incomplete"
    age_seconds = max((now - latest_time.to_pydatetime()).total_seconds(), 0.0)
    latest_row = frame.iloc[-1]
    if age_seconds > stale_after_seconds:
        return "stale"
    if pd.isna(latest_row.get("bid")) or pd.isna(latest_row.get("ask")):
        return "incomplete"
    return "healthy"


def _regime_from_metrics(*, spread_bps: float | None, realized_vol_30m: float | None) -> str:
    if spread_bps is None and realized_vol_30m is None:
        return "incomplete"
    if (spread_bps is not None and spread_bps >= 10.0) or (realized_vol_30m is not None and realized_vol_30m >= 0.35):
        return "stressed"
    if (spread_bps is not None and spread_bps >= 3.0) or (realized_vol_30m is not None and realized_vol_30m >= 0.18):
        return "volatile"
    return "normal"


def _purge_old_partitions(root: Path, *, retention_days: int) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(retention_days), 1))
    deleted_files = 0
    deleted_dirs = 0
    for parquet_path in root.glob("symbol=*/date=*/hour=*/ticks.parquet"):
        date_part = parquet_path.parent.parent.name
        if not date_part.startswith("date="):
            continue
        try:
            partition_day = datetime.fromisoformat(date_part.split("=", 1)[1]).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if partition_day >= cutoff.replace(hour=0, minute=0, second=0, microsecond=0):
            continue
        if parquet_path.exists():
            parquet_path.unlink()
            deleted_files += 1
        hour_dir = parquet_path.parent
        date_dir = hour_dir.parent
        symbol_dir = date_dir.parent
        for directory in (hour_dir, date_dir, symbol_dir):
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
                    deleted_dirs += 1
            except OSError:
                continue
    return {"deleted_files": deleted_files, "deleted_dirs": deleted_dirs}


def archive_ticks(
    *,
    root: Path,
    symbol: str,
    ticks: Iterable[Mapping[str, Any]],
    retention_days: int,
    register_artifact: Callable[[Path, dict[str, Any]], Any] | None = None,
    artifact_base_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    frame = normalize_ticks(symbol, ticks)
    if frame.empty:
        purge = _purge_old_partitions(root, retention_days=retention_days)
        return {
            "symbol": str(symbol).upper(),
            "rows": 0,
            "partitions": 0,
            "latest_tick_at": None,
            "oldest_tick_at": None,
            "purged": purge,
        }

    partitions = 0
    written_rows = 0
    base_details = dict(artifact_base_details or {})
    for _, group in frame.groupby(frame["time_utc"].dt.strftime("%Y-%m-%d %H"), sort=True):
        target = _partition_path(root, str(symbol).upper(), group["time_utc"].iloc[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = pd.read_parquet(target)
            merged = pd.concat([existing, group], ignore_index=True)
        else:
            merged = group.copy()
        merged = merged.drop_duplicates(subset=["symbol", "time_utc", "bid", "ask", "last"]).sort_values("time_utc")
        merged.to_parquet(target, index=False)
        partitions += 1
        written_rows += len(group)
        if register_artifact is not None:
            register_artifact(
                target,
                {
                    **base_details,
                    "symbol": str(symbol).upper(),
                    "row_count": int(len(merged)),
                    "partition_date": str(group["time_utc"].dt.strftime("%Y-%m-%d").iloc[0]),
                    "partition_hour": str(group["time_utc"].dt.strftime("%H").iloc[0]),
                    "oldest_tick_at": group["time_utc"].min().isoformat(),
                    "latest_tick_at": group["time_utc"].max().isoformat(),
                    "retention_days": int(retention_days),
                },
            )

    purge = _purge_old_partitions(root, retention_days=retention_days)
    return {
        "symbol": str(symbol).upper(),
        "rows": int(len(frame)),
        "written_rows": int(written_rows),
        "partitions": int(partitions),
        "oldest_tick_at": frame["time_utc"].min().isoformat(),
        "latest_tick_at": frame["time_utc"].max().isoformat(),
        "purged": purge,
    }


def summarize_tick_archive(
    root: Path,
    *,
    symbols: Iterable[str] | None = None,
    stale_after_seconds: float = 120.0,
) -> dict[str, Any]:
    normalized_symbols = {
        str(symbol).upper() for symbol in symbols or [] if str(symbol or "").strip()
    }
    symbol_stats: dict[str, dict[str, Any]] = {}
    total_rows = 0
    partition_count = 0
    oldest_tick_at: datetime | None = None
    latest_tick_at: datetime | None = None
    now = datetime.now(timezone.utc)

    for parquet_path in root.glob("symbol=*/date=*/hour=*/ticks.parquet"):
        symbol_dir = parquet_path.parent.parent.parent.name
        if not symbol_dir.startswith("symbol="):
            continue
        symbol = symbol_dir.split("=", 1)[1].upper()
        if normalized_symbols and symbol not in normalized_symbols:
            continue
        try:
            metadata = pq.read_metadata(parquet_path)
        except Exception:
            continue
        rows = int(metadata.num_rows)
        partition_count += 1
        total_rows += rows
        frame = pd.read_parquet(parquet_path, columns=["time_utc"])
        if frame.empty:
            continue
        frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["time_utc"])
        if frame.empty:
            continue
        part_oldest = frame["time_utc"].min().to_pydatetime()
        part_latest = frame["time_utc"].max().to_pydatetime()
        oldest_tick_at = part_oldest if oldest_tick_at is None or part_oldest < oldest_tick_at else oldest_tick_at
        latest_tick_at = part_latest if latest_tick_at is None or part_latest > latest_tick_at else latest_tick_at
        bucket = symbol_stats.setdefault(
            symbol,
            {
                "symbol": symbol,
                "row_count": 0,
                "partition_count": 0,
                "oldest_tick_at": None,
                "latest_tick_at": None,
            },
        )
        bucket["row_count"] = int(bucket["row_count"] + rows)
        bucket["partition_count"] = int(bucket["partition_count"] + 1)
        bucket["oldest_tick_at"] = (
            part_oldest.isoformat()
            if bucket["oldest_tick_at"] is None or part_oldest < datetime.fromisoformat(str(bucket["oldest_tick_at"]))
            else bucket["oldest_tick_at"]
        )
        bucket["latest_tick_at"] = (
            part_latest.isoformat()
            if bucket["latest_tick_at"] is None or part_latest > datetime.fromisoformat(str(bucket["latest_tick_at"]))
            else bucket["latest_tick_at"]
        )

    enriched_symbols: list[dict[str, Any]] = []
    quality_counts = {"healthy": 0, "stale": 0, "incomplete": 0}
    aggregate_spreads: list[float] = []
    aggregate_vols_30m: list[float] = []
    widest_symbol: str | None = None
    widest_spread_bps: float | None = None
    regime = "incomplete"
    for symbol in sorted(symbol_stats):
        recent = _load_recent_ticks(root, symbol=symbol, since=now - timedelta(hours=1))
        latest_row = None if recent.empty else recent.iloc[-1]
        spread_bps = None
        spread = None
        mid = None
        if latest_row is not None:
            spread = None if pd.isna(latest_row.get("spread")) else float(latest_row.get("spread"))
            spread_bps = None if pd.isna(latest_row.get("spread_bps")) else float(latest_row.get("spread_bps"))
            mid = None if pd.isna(latest_row.get("mid")) else float(latest_row.get("mid"))
        realized_5m = _realized_vol_annualized(recent[recent["time_utc"] >= pd.Timestamp(now - timedelta(minutes=5))], lookback_minutes=5)
        realized_30m = _realized_vol_annualized(recent[recent["time_utc"] >= pd.Timestamp(now - timedelta(minutes=30))], lookback_minutes=30)
        realized_1h = _realized_vol_annualized(recent, lookback_minutes=60)
        tick_quality = _symbol_tick_quality(recent, now=now, stale_after_seconds=stale_after_seconds)
        quality_counts[tick_quality] = int(quality_counts.get(tick_quality, 0) + 1)
        symbol_regime = _regime_from_metrics(spread_bps=spread_bps, realized_vol_30m=realized_30m)
        if spread_bps is not None:
            aggregate_spreads.append(spread_bps)
            if widest_spread_bps is None or spread_bps > widest_spread_bps:
                widest_spread_bps = spread_bps
                widest_symbol = symbol
        if realized_30m is not None:
            aggregate_vols_30m.append(realized_30m)
        if symbol_regime == "stressed":
            regime = "stressed"
        elif symbol_regime == "volatile" and regime != "stressed":
            regime = "volatile"
        elif symbol_regime == "normal" and regime == "incomplete":
            regime = "normal"
        enriched_symbols.append(
            {
                **symbol_stats[symbol],
                "mid": mid,
                "spread": spread,
                "spread_bps": spread_bps,
                "tick_count_5m": int(len(recent[recent["time_utc"] >= pd.Timestamp(now - timedelta(minutes=5))])),
                "tick_count_30m": int(len(recent[recent["time_utc"] >= pd.Timestamp(now - timedelta(minutes=30))])),
                "tick_count_1h": int(len(recent)),
                "realized_vol_5m": realized_5m,
                "realized_vol_30m": realized_30m,
                "realized_vol_1h": realized_1h,
                "tick_quality": tick_quality,
                "regime": symbol_regime,
            }
        )

    coverage_status = "incomplete"
    if enriched_symbols:
        if quality_counts.get("healthy", 0) == len(enriched_symbols):
            coverage_status = "healthy"
        elif quality_counts.get("healthy", 0) > 0:
            coverage_status = "thin_history"
        elif quality_counts.get("stale", 0) > 0:
            coverage_status = "stale"

    avg_spread_bps = None if not aggregate_spreads else float(sum(aggregate_spreads) / len(aggregate_spreads))
    avg_realized_vol_30m = None if not aggregate_vols_30m else float(sum(aggregate_vols_30m) / len(aggregate_vols_30m))

    return {
        "enabled": root.exists(),
        "format": "parquet",
        "partition_count": int(partition_count),
        "row_count": int(total_rows),
        "symbol_count": int(len(symbol_stats)),
        "oldest_tick_at": None if oldest_tick_at is None else oldest_tick_at.isoformat(),
        "latest_tick_at": None if latest_tick_at is None else latest_tick_at.isoformat(),
        "coverage_status": coverage_status,
        "tick_quality": {
            "status": "healthy" if quality_counts.get("healthy", 0) == len(enriched_symbols) and enriched_symbols else "degraded",
            "healthy_symbols": int(quality_counts.get("healthy", 0)),
            "stale_symbols": int(quality_counts.get("stale", 0)),
            "incomplete_symbols": int(quality_counts.get("incomplete", 0)),
            "stale_after_seconds": float(stale_after_seconds),
        },
        "microstructure": {
            "avg_spread_bps": avg_spread_bps,
            "widest_spread_bps": widest_spread_bps,
            "widest_symbol": widest_symbol,
            "realized_vol_30m": avg_realized_vol_30m,
            "regime": regime,
        },
        "symbols": enriched_symbols,
    }
