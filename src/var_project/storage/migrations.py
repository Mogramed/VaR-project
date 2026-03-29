from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from alembic import command
from alembic.config import Config

from var_project.storage.settings import StorageSettings


def _load_runtime_config(root: Path) -> dict[str, Any]:
    config_path = root / "config" / "settings.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def build_alembic_config(root: Path, *, database_url: str | None = None) -> Config:
    runtime_root = root.resolve()
    migrations_root = runtime_root if (runtime_root / "alembic.ini").exists() else Path(__file__).resolve().parents[3]
    raw_config = _load_runtime_config(runtime_root)
    config = Config(str(migrations_root / "alembic.ini"))
    config.set_main_option("script_location", str(migrations_root / "alembic"))
    config.set_main_option(
        "sqlalchemy.url",
        str(database_url or StorageSettings.from_root(runtime_root, raw_config).database_url),
    )
    return config


def upgrade_database(root: Path, *, revision: str = "head", database_url: str | None = None) -> dict[str, Any]:
    runtime_root = root.resolve()
    raw_config = _load_runtime_config(runtime_root)
    settings = StorageSettings.from_root(runtime_root, raw_config)
    if settings.database_path is not None:
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    config = build_alembic_config(root, database_url=database_url)
    command.upgrade(config, revision)
    return {
        "database_url": config.get_main_option("sqlalchemy.url"),
        "revision": revision,
        "script_location": config.get_main_option("script_location"),
    }
