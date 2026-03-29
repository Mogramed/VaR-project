from __future__ import annotations

from pathlib import Path

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

    jobs = client.get("/jobs/status")
    assert jobs.status_code == 200
    body = jobs.json()
    assert body["database_ready"] is True
    assert set(body["jobs"]) == {"snapshot", "backtest", "report"}
    assert body["jobs"]["snapshot"]["state"] in {"pending", "due", "ok"}


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
