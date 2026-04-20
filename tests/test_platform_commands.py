from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest
import yaml
from fastapi.testclient import TestClient

from var_project.api import create_app
from var_project.api.service import DeskApiService
from var_project.bootstrap import seed_demo_environment
from var_project.storage import upgrade_database

from support import write_processed_returns, write_settings


def _assert_schema_invalid_error(response) -> dict:
    assert response.status_code == 503
    body = response.json()
    assert isinstance(body.get("detail"), dict)
    detail = body["detail"]
    assert detail.get("error_code") == "schema_invalid"
    assert "alembic upgrade head" in str(detail.get("hint") or "")
    return detail


def test_db_upgrade_enables_health_and_jobs_status(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")

    upgrade_database(root)

    client = TestClient(create_app(repo_root=root))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["dependencies"]["database"]["schema_ready"] is True
    dependencies = client.get("/health/dependencies")
    assert dependencies.status_code == 200
    assert "market_data" in dependencies.json()["dependencies"]

    jobs = client.get("/jobs/status")
    assert jobs.status_code == 200
    body = jobs.json()
    assert body["database_ready"] is True
    assert set(body["jobs"]) == {"snapshot", "backtest", "live_refresh", "report"}
    assert body["jobs"]["snapshot"]["state"] in {"pending", "due", "ok"}
    assert body["jobs"]["live_refresh"]["enabled"] is False
    assert body["jobs"]["live_refresh"]["state"] == "disabled"


def test_operator_action_refuses_when_operator_runs_table_is_missing(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")

    db_path = root / "data" / "app" / "test_platform.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT,
                name TEXT,
                base_currency TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_type TEXT,
                format TEXT,
                path TEXT
            )
            """
        )
        connection.commit()

    client = TestClient(create_app(repo_root=root))
    health = client.get("/health")
    assert health.status_code == 200
    health_body = health.json()
    db_detail = health_body["dependencies"]["database"]
    assert health_body["status"] == "unhealthy"
    assert db_detail["schema_ready"] is False
    assert "Missing table 'operator_runs'." in list(db_detail.get("issues") or [])
    assert "alembic upgrade head" in str(db_detail["detail"])
    assert "alembic upgrade head" in str(db_detail["hint"])

    _assert_schema_invalid_error(client.post("/operator/actions/backtest", json={"portfolio_slug": "fx_eur_20k"}))
    _assert_schema_invalid_error(client.get("/operator/runs"))
    _assert_schema_invalid_error(client.get("/operator/runs/1"))


def test_operator_action_refuses_when_operator_runs_queue_task_id_column_is_missing(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")

    db_path = root / "data" / "app" / "test_platform.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT,
                name TEXT,
                base_currency TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_type TEXT,
                format TEXT,
                path TEXT
            )
            """
        )
        connection.execute(
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
                reused_run_id INTEGER NULL,
                started_at TEXT NULL,
                finished_at TEXT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()

    client = TestClient(create_app(repo_root=root))
    health = client.get("/health")
    assert health.status_code == 200
    health_body = health.json()
    db_detail = health_body["dependencies"]["database"]
    assert health_body["status"] == "unhealthy"
    assert db_detail["schema_ready"] is False
    assert "Missing column 'operator_runs.queue_task_id'." in list(db_detail.get("issues") or [])
    assert "alembic upgrade head" in str(db_detail["detail"])
    assert "alembic upgrade head" in str(db_detail["hint"])

    _assert_schema_invalid_error(client.post("/operator/actions/backtest", json={"portfolio_slug": "fx_eur_20k"}))


def test_health_unhealthy_when_alembic_revision_is_obsolete(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")
    upgrade_database(root)

    db_path = root / "data" / "app" / "test_platform.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE alembic_version SET version_num = ?", ("20260405_0008",))
        connection.commit()

    client = TestClient(create_app(repo_root=root))
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    db_dependency = body["dependencies"]["database"]

    assert body["status"] == "unhealthy"
    assert db_dependency["schema_ready"] is False
    assert db_dependency["current_revision"] == "20260405_0008"
    assert db_dependency["expected_revision"] != "20260405_0008"
    assert "Alembic revision mismatch" in str(db_dependency["detail"])
    assert "alembic upgrade head" in str(db_dependency["hint"])

    operator_error = _assert_schema_invalid_error(
        client.post("/operator/actions/backtest", json={"portfolio_slug": "fx_eur_20k"})
    )
    assert operator_error["current_revision"] == "20260405_0008"
    assert operator_error["expected_revision"] == db_dependency["expected_revision"]
    _assert_schema_invalid_error(client.get("/operator/runs"))


def test_seed_demo_environment_populates_platform_state(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")

    result = seed_demo_environment(root)

    assert result["portfolio_count"] == 1
    seeded = result["seeded"][0]
    assert Path(seeded["report_markdown"]).exists()
    assert seeded["snapshot_id"] > 0
    assert seeded["backtest_run_id"] > 0
    assert seeded["validation_run_id"] > 0
    assert seeded["execution_preview_symbol"] is None

    service = DeskApiService(root)
    assert service.latest_report_content(portfolio_slug=seeded["portfolio_slug"]) is not None
    assert service.recent_decisions(portfolio_slug=seeded["portfolio_slug"])


def test_seed_demo_environment_allows_substep_demo_decision_delta(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(
        root,
        portfolios=[
            {
                "name": "FX_MICRO",
                "configured_exposure": {"EURUSD": 5_000.0, "USDJPY": 7_500.0},
            }
        ],
    )
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")

    result = seed_demo_environment(root)

    assert result["portfolio_count"] == 1
    seeded = result["seeded"][0]
    assert seeded["decision_id"] is not None

    service = DeskApiService(root)
    decisions = service.recent_decisions(portfolio_slug=seeded["portfolio_slug"])
    assert decisions
    assert float(decisions[0]["requested_exposure_change"]) == pytest.approx(500.0)


def test_seed_demo_environment_fails_fast_on_incompatible_backtest_config(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")

    settings_path = root / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    settings["risk"]["window"] = 250
    settings["risk"]["estimation_window_days"] = 500
    settings["risk"]["minimum_valid_days"] = 250
    settings["risk"]["validation_window_days"] = 500
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="tracked history"):
        seed_demo_environment(root)
