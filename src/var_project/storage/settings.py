from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sqlite_url_for_path(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


@dataclass(frozen=True)
class StorageSettings:
    database_url: str
    database_path: Path | None
    analytics_dir: Path
    reports_dir: Path
    snapshots_dir: Path
    analytics_format: str = "csv"

    @classmethod
    def from_root(cls, root: Path, raw_config: Mapping[str, Any] | None = None) -> "StorageSettings":
        cfg = dict(raw_config or {})
        storage_cfg = dict(cfg.get("storage") or {})
        data_cfg = dict(cfg.get("data") or {})

        database_url = os.getenv("VAR_PROJECT_DATABASE_URL") or storage_cfg.get("database_url")
        database_path = None
        if not database_url:
            database_path = resolve_path(root, str(storage_cfg.get("database_path", "data/app/var_risk_desk.db")))
            database_url = sqlite_url_for_path(database_path)

        analytics_dir = resolve_path(root, str(storage_cfg.get("analytics_dir", "reports/backtests")))
        reports_dir = resolve_path(root, str(storage_cfg.get("reports_dir", "reports/daily")))
        snapshots_dir = resolve_path(root, str(storage_cfg.get("snapshots_dir", "data/snapshots")))
        analytics_format = str(storage_cfg.get("analytics_format") or data_cfg.get("storage_format") or "csv").lower()

        return cls(
            database_url=str(database_url),
            database_path=database_path,
            analytics_dir=analytics_dir,
            reports_dir=reports_dir,
            snapshots_dir=snapshots_dir,
            analytics_format=analytics_format,
        )
