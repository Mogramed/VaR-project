from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from test_api import _write_processed_returns, _write_settings
from var_project.api import create_app
from var_project.observability.metrics import build_metrics_payload
from var_project.storage.serialization import utcnow


def test_metrics_endpoint_exposes_runtime_metrics(tmp_path: Path) -> None:
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))

    health_response = client.get("/health")
    assert health_response.status_code == 200

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "text/plain" in metrics_response.headers.get("content-type", "")

    payload = metrics_response.text
    assert "var_api_http_requests_total" in payload
    assert 'route="/health"' in payload
    assert "var_operator_runs_window_total" in payload
    assert "var_market_data_sync_runs_window_total" in payload
    assert "var_market_data_sync_last_status" in payload
    assert "var_reconciliation_mismatches_total" in payload
    assert "var_mt5_bridge_state" in payload


def test_request_id_roundtrip_header(tmp_path: Path) -> None:
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))

    response = client.get("/health", headers={"X-Request-ID": "req-test-observability-001"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "req-test-observability-001"


def test_metrics_scrape_is_read_only_for_operator_runs(tmp_path: Path) -> None:
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    app = create_app(repo_root=root, bootstrap_storage=True)
    client = TestClient(app)

    # Initialize service singleton.
    assert client.get("/health").status_code == 200
    service = app.state._desk_api_service
    assert service is not None

    portfolio = service.runtime._resolve_portfolio_context(None)
    portfolio_id = service.runtime._resolve_portfolio_id(portfolio["slug"])
    run_id = service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug=portfolio["slug"],
        action="snapshot",
        request_id="req-observability-readonly-001",
        status="running",
        stage="running_snapshot",
        request_payload={"portfolio_slug": portfolio["slug"]},
    )
    service.storage.update_operator_run(
        run_id,
        status="running",
        stage="running_snapshot",
        started_at=utcnow() - timedelta(hours=5),
    )

    before = service.storage.operator_run_by_id(run_id)
    assert before is not None
    assert str(before.get("status")) == "running"

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200

    after = service.storage.operator_run_by_id(run_id)
    assert after is not None
    assert str(after.get("status")) == "running"


def test_metrics_refresh_uses_live_state_fallback_when_summary_cache_is_empty() -> None:
    class _FakeStorage:
        def list_operator_runs(self, *, portfolio_slug: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
            return []

        def list_market_data_sync_runs(self, *, portfolio_slug: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
            return []

        def recent_audit_events(
            self,
            *,
            limit: int = 1,
            portfolio_slug: str | None = None,
            object_type: str | None = None,
        ) -> list[dict[str, Any]]:
            return []

    class _FakeMt5:
        def cached_live_state(
            self,
            *,
            portfolio_slug: str | None = None,
            detail_level: str = "summary",
        ) -> dict[str, Any] | None:
            return None

        def live_state(
            self,
            *,
            portfolio_slug: str | None = None,
            detail_level: str = "summary",
            force_refresh: bool = False,
        ) -> dict[str, Any]:
            return {"connected": True, "degraded": False, "stale": False, "fallback_snapshot_used": False, "bridge_consecutive_failures": 0}

    class _FakeService:
        portfolios = [{"slug": "mt5_live_portfolio"}]
        storage = _FakeStorage()
        mt5 = _FakeMt5()

    payload, content_type = build_metrics_payload(service=_FakeService())
    text_payload = payload.decode("utf-8")

    assert "text/plain" in content_type
    assert 'var_mt5_bridge_state{portfolio_slug="mt5_live_portfolio",state="connected"} 1.0' in text_payload
