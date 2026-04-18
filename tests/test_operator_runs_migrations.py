from __future__ import annotations

from pathlib import Path
import sqlite3

from alembic import command
import pytest

from var_project.core.settings import load_settings
from var_project.storage import AppStorage, upgrade_database
from var_project.storage.migrations import build_alembic_config
from var_project.storage.schema_checks import validate_operator_runs_schema

from support import write_settings


def _create_drifted_operator_runs_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL,
                CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
            );
            INSERT INTO alembic_version (version_num) VALUES ('20260405_0008');

            CREATE TABLE operator_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NULL,
                portfolio_slug VARCHAR(160) NULL,
                action VARCHAR(32) NOT NULL,
                request_id VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL,
                stage VARCHAR(64) NOT NULL,
                request_payload_json JSON NULL,
                artifact_refs_json JSON NULL,
                result_json JSON NULL,
                error_code VARCHAR(64) NULL,
                error_message VARCHAR(1024) NULL,
                hint VARCHAR(1024) NULL,
                reused_run_id INTEGER NULL,
                started_at DATETIME NULL,
                finished_at DATETIME NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );

            CREATE INDEX ix_operator_runs_request_id ON operator_runs (request_id);
            """
        )
        connection.commit()


def _insert_duplicate_operator_runs_request_ids(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO operator_runs (
                portfolio_id,
                portfolio_slug,
                action,
                request_id,
                status,
                stage,
                request_payload_json,
                artifact_refs_json,
                result_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None,
                "fx_eur_20k",
                "sync",
                "duplicate-request-id",
                "queued",
                "queued",
                "{}",
                "{}",
                "{}",
                "2026-04-18T09:00:00Z",
                "2026-04-18T09:00:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO operator_runs (
                portfolio_id,
                portfolio_slug,
                action,
                request_id,
                status,
                stage,
                request_payload_json,
                artifact_refs_json,
                result_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None,
                "fx_eur_20k",
                "sync",
                "duplicate-request-id",
                "running",
                "starting",
                "{}",
                "{}",
                "{}",
                "2026-04-18T09:01:00Z",
                "2026-04-18T09:01:00Z",
            ),
        )
        connection.commit()


def test_operator_runs_migration_upgrade_is_idempotent_on_fresh_database(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)

    upgrade_database(root)
    upgrade_database(root)

    storage = AppStorage.from_root(root, load_settings(root))
    storage.initialize(create_schema=False)
    assert validate_operator_runs_schema(storage.engine) == []

    run_id = storage.create_operator_run(
        portfolio_id=None,
        portfolio_slug="fx_eur_20k",
        action="sync",
        request_id="fresh-run-queue-task-1",
        status="queued",
        stage="queued",
        request_payload={"portfolio_slug": "fx_eur_20k"},
        queue_task_id="celery-task-1",
    )
    run = storage.operator_run_by_id(run_id)
    assert run is not None
    assert run["queue_task_id"] == "celery-task-1"


def test_operator_runs_migration_repairs_existing_drifted_database(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    db_path = root / "data" / "app" / "test_platform.db"
    _create_drifted_operator_runs_database(db_path)

    upgrade_database(root)

    storage = AppStorage.from_root(root, load_settings(root))
    storage.initialize(create_schema=False)
    assert validate_operator_runs_schema(storage.engine) == []

    run_id = storage.create_operator_run(
        portfolio_id=None,
        portfolio_slug="fx_eur_20k",
        action="backtest",
        request_id="drifted-run-queue-task-1",
        status="queued",
        stage="queued",
        request_payload={"portfolio_slug": "fx_eur_20k"},
        queue_task_id="celery-task-drift-1",
    )
    run = storage.operator_run_by_id(run_id)
    assert run is not None
    assert run["queue_task_id"] == "celery-task-drift-1"

    with sqlite3.connect(db_path) as connection:
        index_rows = connection.execute("PRAGMA index_list('operator_runs')").fetchall()
    index_by_name = {row[1]: row for row in index_rows}
    assert "ux_operator_runs_request_id" in index_by_name
    assert int(index_by_name["ux_operator_runs_request_id"][2]) == 1


def test_operator_runs_migration_downgrade_upgrade_cycle_is_non_blocking(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)

    upgrade_database(root)
    alembic_config = build_alembic_config(root)
    command.downgrade(alembic_config, "20260405_0008")
    command.upgrade(alembic_config, "head")

    storage = AppStorage.from_root(root, load_settings(root))
    storage.initialize(create_schema=False)
    assert validate_operator_runs_schema(storage.engine) == []


def test_operator_runs_migration_reports_duplicate_request_ids_with_clear_error(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    db_path = root / "data" / "app" / "test_platform.db"
    _create_drifted_operator_runs_database(db_path)
    _insert_duplicate_operator_runs_request_ids(db_path)

    with pytest.raises(RuntimeError, match="Cannot enforce unique operator_runs.request_id"):
        upgrade_database(root)
