from __future__ import annotations

from sqlalchemy import create_engine, text

from var_project.storage.schema_checks import validate_storage_schema


def test_validate_storage_schema_reports_missing_critical_tables() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    try:
        issues = validate_storage_schema(engine)
    finally:
        engine.dispose()

    assert "Missing table 'portfolios'." in issues
    assert "Missing table 'artifacts'." in issues
    assert "Missing table 'operator_runs'." in issues


def test_validate_storage_schema_reports_missing_operator_run_indexes() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE portfolios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        slug TEXT,
                        name TEXT,
                        base_currency TEXT,
                        symbols_json TEXT,
                        positions_json TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE artifacts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        artifact_type TEXT,
                        format TEXT,
                        path TEXT,
                        size_bytes INTEGER,
                        details_json TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE operator_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER NULL,
                        portfolio_slug TEXT NULL,
                        action TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        status_reason TEXT NULL,
                        request_payload_json TEXT NULL,
                        artifact_refs_json TEXT NULL,
                        result_json TEXT NULL,
                        error_code TEXT NULL,
                        error_message TEXT NULL,
                        hint TEXT NULL,
                        queue_task_id TEXT NULL,
                        reused_run_id INTEGER NULL,
                        started_at TEXT NULL,
                        finished_at TEXT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            )
        issues = validate_storage_schema(engine)
    finally:
        engine.dispose()

    assert "Missing index 'ix_operator_runs_queue_task_id' on 'operator_runs'." in issues
    assert "Missing unique constraint/index for 'operator_runs.request_id'." in issues
