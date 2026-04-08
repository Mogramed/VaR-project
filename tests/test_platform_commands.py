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


def test_db_upgrade_enables_health_and_jobs_status(tmp_path: Path) -> None:
    root = tmp_path
    write_settings(root)
    write_processed_returns(root, "EURUSD")
    write_processed_returns(root, "USDJPY")

    upgrade_database(root)

    client = TestClient(create_app(repo_root=root))

    health = client.get("/health")
    assert health.status_code == 200
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
    response = client.post("/operator/actions/backtest", json={"portfolio_slug": "fx_eur_20k"})

    assert response.status_code == 503
    body = response.json()
    assert isinstance(body.get("detail"), dict)
    assert body["detail"].get("error_code") == "runtime_error"


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
