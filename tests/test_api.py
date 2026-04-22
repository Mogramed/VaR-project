from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import text

from var_project.api import create_app
from var_project.api.service import DeskApiService
from var_project.risk.expected_shortfall import historical_var_es
from var_project.storage.serialization import utcnow
from test_mt5_execution_api import FakeMT5Connector, FailingMT5Connector


class PnlDriftConnector(FakeMT5Connector):
    position_profit: float = 25.0

    @classmethod
    def reset(cls) -> None:
        super().reset()
        cls.position_profit = 25.0

    def positions_get(self, symbol: str | None = None) -> list[dict[str, object]]:
        rows = super().positions_get(symbol=symbol)
        for row in rows:
            row["profit"] = float(type(self).position_profit)
        return rows



def _write_settings(
    root: Path,
    *,
    portfolios: list[dict[str, object]] | None = None,
    desk_name: str = "FX Risk Desk",
    portfolio_mode: str | None = None,
    market_history_days: int | None = None,
) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    default_portfolio = {
        "name": "FX_EUR_20k",
        "configured_exposure": {"EURUSD": 10_000, "USDJPY": 10_000},
    }
    if portfolio_mode is not None:
        default_portfolio["mode"] = portfolio_mode
    settings = {
        "base_currency": "EUR",
        "symbols": ["EURUSD", "USDJPY"],
        "portfolio": default_portfolio,
        "data": {
            "timeframes": ["H1"],
            "history_days_list": [60],
            **({} if market_history_days is None else {"market_history_days": int(market_history_days)}),
            "storage_format": "csv",
            "timezone": "Europe/Paris",
            "min_coverage": 0.90,
        },
        "risk": {
            "alpha": 0.95,
            "window": 20,
            "ewma": {"lambda": 0.94},
            "fhs": {"lambda": 0.94},
            "mc": {"n_sims": 250, "dist": "normal", "df_t": 6, "seed": 7},
        },
        "mt5": {"login": None, "password": None, "server": None},
        "storage": {
            "database_path": "data/app/test_api.db",
            "analytics_dir": "reports/backtests",
            "reports_dir": "reports/daily",
            "snapshots_dir": "data/snapshots",
        },
        "desk": {"name": desk_name},
    }
    if portfolios:
        settings["portfolios"] = portfolios
    (config_dir / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    risk_limits = {
        "model_limits_eur": {
            "hist": {"var": 300.0, "es": 360.0},
            "param": {"var": 320.0, "es": 380.0},
            "mc": {"var": 320.0, "es": 380.0},
            "ewma": {"var": 320.0, "es": 380.0},
            "garch": {"var": 320.0, "es": 380.0},
            "fhs": {"var": 320.0, "es": 380.0},
        },
        "risk_budget": {
            "utilisation_warn": 0.85,
            "utilisation_breach": 1.00,
            "target_buffer": 0.95,
            "position_tolerance": 0.05,
            "preferred_model": "best_validation",
            "symbol_weights": {"EURUSD": 0.5, "USDJPY": 0.5},
        },
        "capital_management": {
            "reserve_ratio": 0.10,
            "rebalance_min_gap": 10.0,
            "preferred_model": "best_validation",
        },
        "risk_decision": {
            "decision_mode": "advisory",
            "reference_model": "best_validation",
            "warn_threshold": 0.85,
            "breach_threshold": 1.00,
            "min_fill_ratio": 0.25,
            "allow_risk_reducing_override": True,
        },
    }
    (config_dir / "risk_limits.yaml").write_text(yaml.safe_dump(risk_limits, sort_keys=False), encoding="utf-8")


def _write_processed_returns(root: Path, symbol: str, timeframe: str = "H1", days: int = 60) -> None:
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    bars_per_day = 24
    n_days = 45
    n_bars = n_days * bars_per_day
    times = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    x = np.arange(n_bars)

    if symbol == "EURUSD":
        log_returns = 0.0002 + 0.0007 * np.sin(x / 13.0) + 0.0002 * np.cos(x / 7.0)
    else:
        log_returns = -0.0001 + 0.0008 * np.cos(x / 11.0) - 0.00015 * np.sin(x / 5.0)

    frame = pd.DataFrame({"time": times, "log_return": log_returns})
    frame.to_csv(processed_dir / f"{symbol}_{timeframe}_{days}d_returns.csv", index=False)


def test_api_runs_snapshot_backtest_and_report(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))

    root_discovery = client.get("/")
    assert root_discovery.status_code == 200
    assert root_discovery.json()["docs"] == "/docs"
    assert root_discovery.json()["health"] == "/health"
    assert root_discovery.json()["readiness"] == "/health/readiness"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["dependencies"]["database"]["schema_ready"] is True
    dependencies = client.get("/health/dependencies")
    assert dependencies.status_code == 200
    assert "mt5_live" in dependencies.json()["dependencies"]
    readiness = client.get("/health/readiness")
    assert readiness.status_code == 200
    readiness_body = readiness.json()
    assert readiness_body["status"] in {"ready", "degraded", "not_ready"}
    assert readiness_body["portfolio_slug"] == "fx_eur_20k"
    assert "database" in readiness_body["checks"]
    assert "mt5_live" in readiness_body["checks"]

    jobs_status = client.get("/jobs/status")
    assert jobs_status.status_code == 200
    assert jobs_status.json()["database_ready"] is True
    assert "operator_runs" in jobs_status.json()

    portfolios = client.get("/portfolios")
    assert portfolios.status_code == 200
    assert portfolios.json()[0]["slug"] == "fx_eur_20k"

    desks = client.get("/desks")
    assert desks.status_code == 200
    assert desks.json()[0]["slug"] == "fx_risk_desk"

    snapshot = client.post(
        "/snapshots/run",
        json={"timeframe": "H1", "days": 60, "alpha": 0.95, "window": 20},
    )
    assert snapshot.status_code == 200
    snapshot_body = snapshot.json()
    assert Path(snapshot_body["artifact_path"]).exists()

    latest_snapshot = client.get("/snapshots/latest", params={"source": "historical"})
    assert latest_snapshot.status_code == 200
    assert latest_snapshot.json()["source"] == "historical"
    assert "attribution" in latest_snapshot.json()["payload"]
    assert "risk_budget" in latest_snapshot.json()["payload"]
    latest_snapshot_auto = client.get("/snapshots/latest", params={"source": "auto"})
    assert latest_snapshot_auto.status_code == 200
    assert latest_snapshot_auto.json()["source"] == "historical"

    initial_decision = client.post(
        "/decisions/evaluate",
        json={"symbol": "EURUSD", "exposure_change": 1_000.0, "note": "pre-validation"},
    )
    assert initial_decision.status_code == 200
    initial_decision_body = initial_decision.json()
    assert initial_decision_body["model_used"] == "hist"
    assert initial_decision_body["decision"] in {"ACCEPT", "REDUCE", "REJECT"}

    backtest = client.post(
        "/backtests/run",
        json={"timeframe": "H1", "days": 60, "alpha": 0.95, "window": 20, "n_sims": 250, "seed": 7},
    )
    assert backtest.status_code == 200
    backtest_body = backtest.json()
    assert Path(backtest_body["compare_csv"]).exists()
    assert Path(backtest_body["validation_json"]).exists()
    assert backtest_body["best_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}

    validation_service = DeskApiService(root, bootstrap_storage=True)
    validation_result = validation_service.run_validation(compare_path=backtest_body["compare_csv"], alpha=0.95)
    assert Path(validation_result["validation_json"]).exists()
    assert validation_result["best_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}

    latest_backtest = client.get("/backtests/latest")
    assert latest_backtest.status_code == 200
    assert latest_backtest.json()["n_rows"] > 0

    latest_backtest_frame = client.get("/backtests/frame/latest", params={"limit": 50})
    assert latest_backtest_frame.status_code == 200
    assert latest_backtest_frame.json()["rows"]
    assert latest_backtest_frame.json()["portfolio_slug"] == "fx_eur_20k"
    assert str(latest_backtest_frame.json()["rows"][0]["date"]).startswith("2024-")

    latest_validation = client.get("/validations/latest")
    assert latest_validation.status_code == 200
    assert latest_validation.json()["best_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}

    latest_comparison = client.get("/models/compare/latest")
    assert latest_comparison.status_code == 200
    comparison_body = latest_comparison.json()
    assert comparison_body["champion_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}
    assert comparison_body["ranking"]
    first_ranking_row = comparison_body["ranking"][0]
    assert "es_acerbi_status" in first_ranking_row
    assert "es_acerbi_p_value" in first_ranking_row
    assert "es_acerbi_observations" in first_ranking_row

    latest_attribution = client.get("/snapshots/attribution/latest", params={"source": "historical"})
    assert latest_attribution.status_code == 200
    attribution_body = latest_attribution.json()
    assert attribution_body["models"]["hist"]["positions"]["EURUSD"]["standalone_var"] >= 0.0
    hist_attribution = attribution_body["models"]["hist"]
    assert hist_attribution["concentration_var"] is not None
    assert hist_attribution["concentration_var"]["hhi"] is not None
    assert hist_attribution["concentration_var"]["top1_share"] is not None
    assert hist_attribution["diversification_ratio_var"] is not None

    latest_budget = client.get("/snapshots/budget/latest", params={"source": "historical"})
    assert latest_budget.status_code == 200
    budget_body = latest_budget.json()
    assert budget_body["preferred_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}
    assert budget_body["models"]["hist"]["positions"]["EURUSD"]["target_var_budget"] >= 0.0

    latest_capital = client.get("/capital/latest")
    assert latest_capital.status_code == 200
    capital_body = latest_capital.json()
    assert capital_body["total_capital_budget_eur"] >= capital_body["total_capital_consumed_eur"]
    assert capital_body["allocations"]["EURUSD"]["target_capital_eur"] >= 0.0

    rebalance = client.post("/capital/rebalance", json={"total_budget_eur": 420.0, "reserve_ratio": 0.05})
    assert rebalance.status_code == 200
    rebalance_body = rebalance.json()
    assert rebalance_body["total_capital_budget_eur"] == 420.0
    assert rebalance_body["budget"]["reserve_ratio"] == 0.05

    capital_history = client.get("/capital/history", params={"limit": 5})
    assert capital_history.status_code == 200
    assert capital_history.json()

    portfolio_capital = client.get("/portfolios/fx_eur_20k/capital")
    assert portfolio_capital.status_code == 200
    assert portfolio_capital.json()["portfolio_slug"] == "fx_eur_20k"

    decision = client.post(
        "/decisions/evaluate",
        json={"symbol": "EURUSD", "exposure_change": 3_000.0, "note": "post-validation"},
    )
    assert decision.status_code == 200
    decision_body = decision.json()
    assert decision_body["decision"] in {"ACCEPT", "REDUCE", "REJECT"}
    assert decision_body["model_used"] == backtest_body["best_model"]
    assert decision_body["pre_trade"]["var"] >= 0.0

    recent_decisions = client.get("/decisions/recent", params={"limit": 5})
    assert recent_decisions.status_code == 200
    assert recent_decisions.json()

    alerts = client.get("/alerts", params={"limit": 10})
    assert alerts.status_code == 200
    assert isinstance(alerts.json(), list)

    report = client.post("/reports/run", json={})
    assert report.status_code == 200
    report_body = report.json()
    assert Path(report_body["report_markdown"]).exists()

    latest_report_payload = client.get("/reports/latest")
    assert latest_report_payload.status_code == 200
    assert latest_report_payload.json()["portfolio_slug"] == "fx_eur_20k"
    assert latest_report_payload.json()["report_id"] is not None
    assert latest_report_payload.json()["content"].startswith("# Risk Report")
    assert (
        "Preferred snapshot source: **historical**" in latest_report_payload.json()["content"]
        or "Preferred snapshot source: **mt5_live_bridge**" in latest_report_payload.json()["content"]
        or "Preferred snapshot source: **mt5_live**" in latest_report_payload.json()["content"]
    )
    assert "## Portfolio Snapshot" in latest_report_payload.json()["content"]
    assert "## Decision History" in latest_report_payload.json()["content"]
    assert "## Capital History" in latest_report_payload.json()["content"]
    assert "## Audit Trail" in latest_report_payload.json()["content"]
    if report_body["chart_paths"]:
        chart_name = Path(report_body["chart_paths"][0]).name
        chart_asset = client.get(f"/reports/charts/{chart_name}")
        assert chart_asset.status_code == 200
        assert chart_asset.headers.get("content-type", "").startswith("image/")
        chart_with_report_id = client.get(
            f"/reports/charts/{chart_name}",
            params={"report_id": latest_report_payload.json()["report_id"]},
        )
        assert chart_with_report_id.status_code == 200

    missing_chart_asset = client.get("/reports/charts/does-not-exist.png")
    assert missing_chart_asset.status_code == 404

    latest_report = client.get("/artifacts/latest/daily_report")
    assert latest_report.status_code == 200
    assert latest_report.json()["path"] == report_body["report_markdown"]

    decision_history = client.get("/reports/decision-history", params={"limit": 5})
    assert decision_history.status_code == 200
    assert decision_history.json()

    capital_report_history = client.get("/reports/capital-history", params={"limit": 5})
    assert capital_report_history.status_code == 200
    assert capital_report_history.json()

    audit = client.get("/audit/recent", params={"limit": 10})
    assert audit.status_code == 200
    assert audit.json()


def test_api_backtest_rejects_incompatible_fixture_window_early(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    settings_path = root / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    settings["risk"]["window"] = 250
    settings["risk"]["estimation_window_days"] = 500
    settings["risk"]["minimum_valid_days"] = 250
    settings["risk"]["validation_window_days"] = 500
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))
    response = client.post("/backtests/run", json={})

    assert response.status_code == 400
    assert "tracked history" in response.json()["detail"]


def test_trade_exposure_validation_boundaries(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    client = TestClient(
        create_app(
            repo_root=root,
            mt5_connector_factory=FakeMT5Connector,
            bootstrap_storage=True,
        )
    )

    for invalid in (0.0, 1.0, 999.0, 1001.0):
        decision = client.post(
            "/decisions/evaluate",
            json={"symbol": "EURUSD", "exposure_change": invalid, "note": f"decision invalid {invalid}"},
        )
        assert decision.status_code == 400
        assert "1,000 EUR" in str(decision.json().get("detail"))

        preview = client.post(
            "/execution/preview",
            json={"symbol": "EURUSD", "exposure_change": invalid, "note": f"preview invalid {invalid}"},
        )
        assert preview.status_code == 400
        assert "1,000 EUR" in str(preview.json().get("detail"))

    for valid in (1000.0, -1000.0):
        decision = client.post(
            "/decisions/evaluate",
            json={"symbol": "EURUSD", "exposure_change": valid, "note": f"decision valid {valid}"},
        )
        assert decision.status_code == 200

        preview = client.post(
            "/execution/preview",
            json={"symbol": "EURUSD", "exposure_change": valid, "note": f"preview valid {valid}"},
        )
        assert preview.status_code == 200


def test_operator_actions_enqueue_and_complete(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))

    response = client.post("/operator/actions/backtest", json={"portfolio_slug": "fx_eur_20k"})
    assert response.status_code == 202
    run_body = response.json()
    assert run_body["action"] == "backtest"
    assert run_body["request_id"]
    assert int(run_body["queued_timeout_seconds"]) > 0
    assert int(run_body["running_timeout_seconds"]) > 0
    assert int(run_body["sla_seconds"]) == int(run_body["running_timeout_seconds"])
    assert int(run_body["poll_after_ms"]) > 0
    assert run_body["interruptible"] is True

    run_id = int(run_body["id"])
    latest = client.get(f"/operator/runs/{run_id}")
    assert latest.status_code == 200
    latest_body = latest.json()
    assert latest_body["status"] in {"queued", "running", "succeeded"}

    for _ in range(20):
        latest = client.get(f"/operator/runs/{run_id}")
        latest_body = latest.json()
        if latest_body["status"] in {"succeeded", "failed"}:
            break

    if latest_body["status"] not in {"succeeded", "failed"}:
        service = DeskApiService(root, bootstrap_storage=True)
        latest_body = service.process_operator_run(run_id)

    assert latest_body["status"] == "succeeded"
    assert latest_body["stage"] == "completed"
    assert latest_body["interruptible"] is False
    assert latest_body["poll_after_ms"] is None
    assert latest_body["artifact_refs"]["compare_artifact_id"] > 0
    assert latest_body["result"]["backtest"]["best_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}


def test_operator_run_can_be_interrupted(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    run = service.enqueue_operator_action(
        action="backtest",
        request_payload={"portfolio_slug": "fx_eur_20k"},
    )
    run_id = int(run["id"])

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))
    interrupt_response = client.post(
        f"/operator/runs/{run_id}/interrupt",
        params={"reason": "operator requested interruption"},
    )
    assert interrupt_response.status_code == 200
    body = interrupt_response.json()
    assert body["id"] == run_id
    assert body["status"] == "failed"
    assert body["status_reason"] == "interrupted"
    assert body["error_code"] == "operator_interrupted"
    assert "interrupted" in str(body["error_message"]).lower()
    assert body["interruptible"] is False

    second_interrupt = client.post(
        f"/operator/runs/{run_id}/interrupt",
        params={"reason": "second interruption should be idempotent"},
    )
    assert second_interrupt.status_code == 200
    second_body = second_interrupt.json()
    assert second_body["id"] == run_id
    assert second_body["status"] == "failed"
    assert second_body["status_reason"] == "interrupted"
    assert second_body["error_code"] == "operator_interrupted"
    assert second_body["interruptible"] is False

    latest = client.get(f"/operator/runs/{run_id}")
    assert latest.status_code == 200
    assert latest.json()["status"] == "failed"
    assert latest.json()["status_reason"] == "interrupted"
    assert latest.json()["error_code"] == "operator_interrupted"


def test_interrupt_operator_run_does_not_override_succeeded_state_on_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    run = service.enqueue_operator_action(
        action="backtest",
        request_payload={"portfolio_slug": "fx_eur_20k"},
    )
    run_id = int(run["id"])
    original_operator_run_by_id = service.storage.operator_run_by_id
    first_read = {"done": False}

    def _operator_run_by_id_with_race(target_run_id: int):
        current = original_operator_run_by_id(target_run_id)
        if first_read["done"] or target_run_id != run_id or current is None:
            return current
        first_read["done"] = True
        service.storage.update_operator_run(
            run_id,
            status="succeeded",
            stage="completed",
            result={"backtest": {"best_model": "hist"}},
            artifact_refs={"compare_artifact_id": 1},
            finished_at=utcnow(),
        )
        return current

    monkeypatch.setattr(service.storage, "operator_run_by_id", _operator_run_by_id_with_race)
    interrupted = service.interrupt_operator_run(
        run_id,
        reason="operator interrupt request collided with worker completion",
    )

    assert interrupted is not None
    assert interrupted["id"] == run_id
    assert interrupted["status"] == "succeeded"
    assert interrupted["stage"] == "completed"
    assert interrupted.get("error_code") in {None, ""}

    latest = service.operator_run(run_id)
    assert latest is not None
    assert latest["status"] == "succeeded"
    assert latest["stage"] == "completed"
    assert latest.get("error_code") in {None, ""}


def test_operator_interruption_is_not_overwritten_by_processing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    run = service.enqueue_operator_action(
        action="backtest",
        request_payload={"portfolio_slug": "fx_eur_20k"},
    )
    run_id = int(run["id"])

    def _interrupt_then_fail(**_: object):
        service.interrupt_operator_run(run_id, reason="manual interruption during processing")
        raise RuntimeError("simulated failure after interruption")

    monkeypatch.setattr(service.analytics, "run_backtest", _interrupt_then_fail)
    processed = service.process_operator_run(run_id)

    assert processed["status"] == "failed"
    assert processed["status_reason"] == "interrupted"
    assert processed["error_code"] == "operator_interrupted"
    assert "interrupted" in str(processed["error_message"]).lower()
    assert processed["interruptible"] is False


def test_operator_run_fails_fast_when_mt5_unavailable_for_strict_live(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    service.runtime.market_data.mt5_configured = lambda: False  # type: ignore[method-assign]
    run = service.enqueue_operator_action(
        action="sync",
        request_payload={"portfolio_slug": "fx_eur_20k"},
    )
    processed = service.process_operator_run(int(run["id"]))

    assert processed["status"] == "failed"
    assert processed["status_reason"] == "mt5_unavailable"
    assert processed["error_code"] == "mt5_live_unavailable"
    assert "VAR_PROJECT_MT5_AGENT_BASE_URL" in str(processed.get("hint") or "")


def test_live_portfolio_context_drops_configured_fallback_positions(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    service = DeskApiService(root, bootstrap_storage=True)

    portfolio = service.runtime._resolve_portfolio_context("fx_eur_20k")
    assert portfolio["configured_holdings"] == []
    assert portfolio["configured_exposure"] == {}
    assert portfolio["positions"] == {}

    listed = next(item for item in service.list_portfolios() if item["slug"] == "fx_eur_20k")
    assert dict(listed.get("positions") or {}) == {}


def test_live_portfolio_context_accepts_legacy_slug_alias_after_rename(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    settings_path = root / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    settings["portfolio"]["name"] = "MT5_Live_Portfolio"
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    service = DeskApiService(root, bootstrap_storage=True)
    portfolio = service.runtime._resolve_portfolio_context("fx_eur_20k")

    assert portfolio["slug"] == "mt5_live_portfolio"
    assert portfolio["mode"] == "live_mt5"


def test_list_portfolios_prefers_runtime_slug_over_legacy_singleton_row(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    settings_path = root / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    settings["portfolio"]["name"] = "MT5_Live_Portfolio"
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    service = DeskApiService(root, bootstrap_storage=True)
    with service.runtime.storage.session_factory() as session:
        session.execute(
            text("UPDATE portfolios SET slug = :slug, name = :name"),
            {"slug": "fx_eur_20k", "name": "FX_EUR_20k"},
        )
        session.commit()

    listed = service.list_portfolios()

    assert [item["slug"] for item in listed] == ["mt5_live_portfolio"]
    assert listed[0]["mode"] == "live_mt5"


def test_live_snapshot_fails_when_mt5_returns_empty_holdings(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    FakeMT5Connector.reset()
    FakeMT5Connector.positions_lots = {}

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))
    snapshot = client.post("/snapshots/run", json={})

    assert snapshot.status_code == 503
    detail = str(snapshot.json().get("detail") or "")
    assert "mt5_live_unavailable" in detail
    assert "MT5-only" in detail


def test_offline_portfolio_live_holdings_still_use_configured_fallback(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="offline_fixture")
    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FailingMT5Connector, bootstrap_storage=True))

    live_holdings = client.get("/portfolio/live-holdings")
    assert live_holdings.status_code == 200
    holdings_payload = live_holdings.json()
    assert {item["symbol"] for item in holdings_payload} == {"EURUSD", "USDJPY"}


def test_operator_enqueue_reaps_stale_run_and_creates_new_one(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    payload = {"portfolio_slug": "fx_eur_20k"}
    normalized_payload = service._normalize_operator_payload(action="backtest", request_payload=payload)
    stale_timeout = service._operator_running_ttl_seconds("backtest")
    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")
    stale_run_id = service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="backtest",
        request_id="stale_run_test_1",
        status="running",
        stage="running_backtest",
        request_payload=normalized_payload,
        started_at=utcnow() - timedelta(seconds=stale_timeout + 60),
    )

    fresh_run = service.enqueue_operator_action(action="backtest", request_payload=payload)
    stale_run = service.operator_run(stale_run_id)

    assert stale_run is not None
    assert stale_run["status"] == "failed"
    assert stale_run["status_reason"] == "timeout"
    assert stale_run["error_code"] == "timeout_stale_run"
    assert stale_run.get("elapsed_seconds") is not None
    assert fresh_run["id"] != stale_run_id
    assert fresh_run["status"] in {"queued", "running"}


def test_operator_reap_marks_stale_queued_run_as_abandoned(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    payload = {"portfolio_slug": "fx_eur_20k"}
    normalized_payload = service._normalize_operator_payload(action="sync", request_payload=payload)
    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")
    stale_run_id = service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="sync",
        request_id="stale_run_test_queued_1",
        status="queued",
        stage="accepted",
        request_payload=normalized_payload,
    )
    stale_timeout = service._operator_queued_ttl_seconds("sync")
    stale_anchor = (utcnow() - timedelta(seconds=stale_timeout + 30)).isoformat()
    with service.storage.engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE operator_runs
                SET created_at = :created_at,
                    updated_at = :updated_at
                WHERE id = :run_id
                """
            ),
            {
                "created_at": stale_anchor,
                "updated_at": stale_anchor,
                "run_id": stale_run_id,
            },
        )

    stale_updates = service.reap_stale_operator_runs(
        portfolio_slug="fx_eur_20k",
        action="sync",
        limit=20,
    )
    stale_run = service.operator_run(stale_run_id)

    assert any(int(item["id"]) == stale_run_id for item in stale_updates)
    assert stale_run is not None
    assert stale_run["status"] == "failed"
    assert stale_run["status_reason"] == "abandoned"
    assert stale_run["error_code"] == "abandoned_stale_run"
    assert "ttl=" in str(stale_run.get("error_message") or "")


def test_operator_runs_supports_status_reason_filter(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    service = DeskApiService(root, bootstrap_storage=True)
    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")

    timeout_run_id = service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="backtest",
        request_id="status-reason-timeout-1",
        status="failed",
        stage="failed",
        status_reason="timeout",
        error_code="timeout_stale_run",
    )
    legacy_timeout_run_id = service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="snapshot",
        request_id="status-reason-timeout-legacy-1",
        status="failed",
        stage="failed",
        error_code="timeout_stale_run",
    )
    service.storage.create_operator_run(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        action="sync",
        request_id="status-reason-abandoned-1",
        status="failed",
        stage="failed",
        status_reason="abandoned",
        error_code="abandoned_stale_run",
    )

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))
    response = client.get(
        "/operator/runs",
        params={"portfolio_slug": "fx_eur_20k", "status_reason": "timeout", "limit": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert body
    assert {item["status_reason"] for item in body} == {"timeout"}
    assert any(int(item["id"]) == timeout_run_id for item in body)
    assert any(int(item["id"]) == legacy_timeout_run_id for item in body)


def test_operator_run_claim_is_single_owner(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    run = service.enqueue_operator_action(action="sync", request_payload={"portfolio_slug": "fx_eur_20k"})
    run_id = int(run["id"])

    claimed = service.storage.claim_operator_run(run_id, stage="starting", started_at=utcnow())
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["stage"] == "starting"

    duplicate_claim = service.storage.claim_operator_run(run_id, stage="starting", started_at=utcnow())
    assert duplicate_claim is None

    processed = service.process_operator_run(run_id)
    assert processed["status"] == "running"


def test_latest_backtest_frame_rejects_epoch_timeline(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    service = DeskApiService(root, bootstrap_storage=True)
    compare_path = root / "reports" / "backtests" / "compare_epoch.csv"
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": [1_717_401, 1_717_402],
            "pnl": [120.0, 110.0],
            "var_hist": [90.0, 92.0],
            "var_garch": [95.0, 94.0],
            "var_fhs": [91.0, 90.0],
        }
    ).to_csv(compare_path, index=False)

    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")
    artifact_id = service.storage.register_artifact(
        compare_path,
        artifact_type="backtest_compare",
        details={"portfolio_slug": "fx_eur_20k"},
    )
    service.storage.record_backtest_run(
        portfolio_id=portfolio_id,
        artifact_id=artifact_id,
        timeframe="H1",
        days=60,
        alpha=0.95,
        window=20,
        n_rows=2,
        summary={},
    )

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))
    response = client.get("/backtests/frame/latest", params={"portfolio_slug": "fx_eur_20k"})
    assert response.status_code == 503
    assert "epoch-style" in str(response.json()["detail"]).lower()


def test_api_supports_multi_portfolio_desk_overview(tmp_path: Path):
    root = tmp_path
    _write_settings(
        root,
        portfolios=[
            {"name": "FX_ALPHA", "symbols": ["EURUSD", "USDJPY"], "configured_exposure": {"EURUSD": 8_000, "USDJPY": 12_000}},
            {"name": "FX_BETA", "symbols": ["EURUSD", "USDJPY"], "configured_exposure": {"EURUSD": 15_000, "USDJPY": -4_000}},
        ],
        desk_name="Desk Paris",
    )
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")

    client = TestClient(create_app(repo_root=root, bootstrap_storage=True))

    portfolios = client.get("/portfolios")
    assert portfolios.status_code == 200
    assert {item["slug"] for item in portfolios.json()} == {"fx_alpha", "fx_beta"}

    alpha_snapshot = client.post("/snapshots/run", json={"portfolio_slug": "fx_alpha", "timeframe": "H1", "days": 60, "window": 20})
    beta_snapshot = client.post("/snapshots/run", json={"portfolio_slug": "fx_beta", "timeframe": "H1", "days": 60, "window": 20})
    assert alpha_snapshot.status_code == 200
    assert beta_snapshot.status_code == 200

    alpha_backtest = client.post("/backtests/run", json={"portfolio_slug": "fx_alpha", "timeframe": "H1", "days": 60, "window": 20})
    beta_backtest = client.post("/backtests/run", json={"portfolio_slug": "fx_beta", "timeframe": "H1", "days": 60, "window": 20})
    assert alpha_backtest.status_code == 200
    assert beta_backtest.status_code == 200

    alpha_frame = client.get("/backtests/frame/latest", params={"portfolio_slug": "fx_alpha", "limit": 10})
    beta_frame = client.get("/backtests/frame/latest", params={"portfolio_slug": "fx_beta", "limit": 10})
    assert alpha_frame.status_code == 200
    assert beta_frame.status_code == 200
    assert alpha_frame.json()["portfolio_slug"] == "fx_alpha"
    assert beta_frame.json()["portfolio_slug"] == "fx_beta"

    desks = client.get("/desks")
    assert desks.status_code == 200
    assert desks.json()[0]["slug"] == "desk_paris"

    overview = client.get("/desks/desk_paris/overview")
    assert overview.status_code == 200
    overview_body = overview.json()
    assert overview_body["desk_name"] == "Desk Paris"
    assert len(overview_body["portfolios"]) == 2
    assert {item["portfolio_slug"] for item in overview_body["portfolios"]} == {"fx_alpha", "fx_beta"}


def test_risk_summary_fallback_tolerates_empty_data_quality_snapshot(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    service = DeskApiService(root, mt5_connector_factory=FailingMT5Connector, bootstrap_storage=True)
    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")

    service.runtime.storage.record_snapshot(
        {
            "time_utc": "2026-04-04T09:00:00+00:00",
            "source": "historical",
            "alpha": 0.95,
            "timeframe": "H1",
            "days": 60,
            "window": 20,
            "sample_size": 45,
            "var": {"hist": 12.0},
            "es": {"hist": 18.0},
            "risk_surface": {},
            "headline_risk": [],
            "stress_surface": {},
            "data_quality": {},
            "model_diagnostics": {},
            "risk_nowcast": {},
            "microstructure": {},
            "tick_quality": {},
            "pnl_explain": {},
        },
        portfolio_id=portfolio_id,
        source="historical",
    )

    def fail_live_state(*, portfolio_slug=None):
        raise RuntimeError("bridge unavailable")

    service.mt5.live_state = fail_live_state  # type: ignore[method-assign]
    payload = service.risk_summary()

    assert payload is not None
    assert payload["data_quality"]["status"] == "thin_history"
    assert payload["data_quality"]["available_observations"] == 45
    assert payload["data_quality"]["minimum_valid_days"] >= 20


def test_risk_summary_fallback_uses_snapshot_when_live_state_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path
    _write_settings(root)
    service = DeskApiService(root, mt5_connector_factory=FailingMT5Connector, bootstrap_storage=True)
    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")

    service.runtime.storage.record_snapshot(
        {
            "time_utc": "2026-04-04T09:00:00+00:00",
            "source": "historical",
            "alpha": 0.95,
            "timeframe": "H1",
            "days": 60,
            "window": 20,
            "sample_size": 45,
            "var": {"hist": 11.0},
            "es": {"hist": 16.0},
            "risk_surface": {},
            "headline_risk": [],
            "stress_surface": {},
            "data_quality": {},
            "model_diagnostics": {},
            "risk_nowcast": {},
            "microstructure": {},
            "tick_quality": {},
            "pnl_explain": {},
        },
        portfolio_id=portfolio_id,
        source="historical",
    )

    def slow_live_state(*, portfolio_slug=None, detail_level="summary", force_refresh=False):
        time.sleep(0.6)
        return {"risk_summary": {"source": "mt5_live_bridge"}}

    monkeypatch.setenv("VAR_PROJECT_API_LIVE_READ_TIMEOUT_MS", "100")
    service.mt5.live_state = slow_live_state  # type: ignore[method-assign]

    started = time.perf_counter()
    payload = service.risk_summary()
    elapsed = time.perf_counter() - started

    assert payload is not None
    assert payload["source"] == "historical"
    assert elapsed < 0.45


def test_latest_capital_fallback_when_live_state_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path
    _write_settings(root)
    service = DeskApiService(root, mt5_connector_factory=FailingMT5Connector, bootstrap_storage=True)
    portfolio_id = service.runtime._resolve_portfolio_id("fx_eur_20k")

    service.runtime.storage.record_capital_snapshot(
        {
            "portfolio_slug": "fx_eur_20k",
            "reference_model": "hist",
            "snapshot_timestamp": "2026-04-04T09:00:00+00:00",
            "total_capital_budget_eur": 1_000.0,
            "total_capital_consumed_eur": 400.0,
            "total_capital_reserved_eur": 100.0,
            "total_capital_remaining_eur": 500.0,
            "headroom_ratio": 0.5,
            "allocations": {
                "EURUSD": {"target_capital_eur": 300.0},
                "USDJPY": {"target_capital_eur": 300.0},
            },
        },
        portfolio_id=portfolio_id,
        source="historical",
    )

    def slow_live_state(*, portfolio_slug=None, detail_level="summary", force_refresh=False):
        time.sleep(0.6)
        return {"capital_usage": {"source": "mt5_live_bridge"}}

    monkeypatch.setenv("VAR_PROJECT_API_LIVE_READ_TIMEOUT_MS", "100")
    service.mt5.live_state = slow_live_state  # type: ignore[method-assign]

    started = time.perf_counter()
    payload = service.latest_capital(portfolio_slug="fx_eur_20k")
    elapsed = time.perf_counter() - started

    assert payload["portfolio_slug"] == "fx_eur_20k"
    assert payload["source"] == "historical"
    assert payload["total_capital_budget_eur"] >= payload["total_capital_consumed_eur"]
    assert elapsed < 0.45


def test_health_readiness_refresh_timeout_does_not_reuse_cached_ready_state(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    service = DeskApiService(root, mt5_connector_factory=FailingMT5Connector, bootstrap_storage=True)

    def cached_ready_state(*, portfolio_slug=None, detail_level="summary"):
        return {
            "status": "ok",
            "connected": True,
            "degraded": False,
            "stale": False,
            "fallback_snapshot_used": False,
            "generated_at": utcnow().isoformat(),
        }

    def slow_live_state(*, portfolio_slug=None, detail_level="summary", force_refresh=False):
        time.sleep(0.6)
        return {
            "status": "ok",
            "connected": True,
            "degraded": False,
            "stale": False,
            "fallback_snapshot_used": False,
            "generated_at": utcnow().isoformat(),
        }

    service.mt5.cached_live_state = cached_ready_state  # type: ignore[method-assign]
    service.mt5.live_state = slow_live_state  # type: ignore[method-assign]

    payload = service.health_readiness(refresh_live=True, max_wait_ms=100)
    mt5_check = payload["checks"]["mt5_live"]

    assert payload["status"] in {"degraded", "not_ready"}
    assert mt5_check["status"] in {"degraded", "not_ready"}
    assert mt5_check["value"]["timed_out"] is True
    assert "Request timed out" in str(mt5_check["detail"])


def test_api_exposes_mt5_market_sync_and_reconciliation(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    FakeMT5Connector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    sync = client.post("/market-data/sync", json={})
    assert sync.status_code == 200
    sync_body = sync.json()
    assert sync_body["status"] in {"ok", "incomplete"}
    assert sync_body["instrument_count"] >= 2

    status = client.get("/market-data/status")
    assert status.status_code == 200
    assert status.json()["portfolio_mode"] == "live_mt5"
    assert status.json()["symbols"] == ["EURUSD", "USDJPY"]
    assert status.json()["live_bridge_status"] in {"ok", "degraded"}

    instruments = client.get("/instruments")
    assert instruments.status_code == 200
    assert {item["symbol"] for item in instruments.json()} == {"EURUSD", "USDJPY"}

    live_holdings = client.get("/portfolio/live-holdings")
    assert live_holdings.status_code == 200
    assert live_holdings.json()

    live_exposure = client.get("/portfolio/live-exposure")
    assert live_exposure.status_code == 200
    assert live_exposure.json()["items"]

    live_state = client.get("/mt5/live/state")
    assert live_state.status_code == 200
    assert live_state.json()["risk_summary"]["source"] == "mt5_live_bridge"
    assert live_state.json()["capital_usage"]["snapshot_source"] == "mt5_live_bridge"
    persisted_live_snapshot = client.get("/snapshots/latest", params={"source": "mt5_live_bridge"})
    assert persisted_live_snapshot.status_code == 200
    assert persisted_live_snapshot.json()["payload"]["metadata"]["live_sequence"] == live_state.json()["sequence"]

    # Force a broker-side drift after the live snapshot has been persisted so reconciliation
    # compares the current MT5 state against a stale desk snapshot.
    FakeMT5Connector.positions_lots["EURUSD"] = 0.25

    history_orders = client.get("/mt5/history/orders", params={"limit": 20})
    assert history_orders.status_code == 200
    assert history_orders.json()

    history_deals = client.get("/mt5/history/deals", params={"limit": 20})
    assert history_deals.status_code == 200
    assert history_deals.json()

    reconciliation = client.get("/reconciliation/summary")
    assert reconciliation.status_code == 200
    reconciliation_body = reconciliation.json()
    assert reconciliation_body["holdings"]
    assert reconciliation_body["manual_event_count"] >= 1
    assert len(reconciliation_body["mismatches"]) == 2
    assert reconciliation_body["status_counts"]
    assert reconciliation_body["summary_severity"] in {"ok", "warn", "critical"}
    assert set(reconciliation_body["severity_counts"].keys()) >= {"ok", "warn", "critical"}
    assert isinstance(reconciliation_body["critical_mismatch_count"], int)
    assert isinstance(reconciliation_body["warning_mismatch_count"], int)
    assert reconciliation_body["live_window_minutes"] >= 1
    assert reconciliation_body["heal_window_days"] == 30
    assert isinstance(reconciliation_body.get("history_backfill_applied"), bool)
    assert reconciliation_body["active_incident_count"] >= 1
    assert reconciliation_body["resolved_incident_count"] >= 0
    assert reconciliation_body["autoresolved_count"] >= 0
    mismatch_row = next(item for item in reconciliation_body["mismatches"] if item["status"] != "match")
    mismatch_symbol = mismatch_row["symbol"]
    assert mismatch_row["acknowledged"] is False
    assert mismatch_row["severity"] in {"warn", "critical"}
    assert mismatch_row["probable_cause"]
    assert mismatch_row["exposure_difference_abs_eur"] >= 0.0
    assert mismatch_row["exposure_difference_relative"] >= 0.0
    assert mismatch_row["volume_difference_abs_lots"] >= 0.0

    reconciliation_history = client.get("/reconciliation/history", params={"limit": 20})
    assert reconciliation_history.status_code == 200
    history_payload = reconciliation_history.json()
    assert history_payload
    assert history_payload[0]["summary_severity"] in {"ok", "warn", "critical"}
    assert isinstance(history_payload[0]["top_symbols"], list)
    filtered_history = client.get(
        "/reconciliation/history",
        params={"severity": history_payload[0]["summary_severity"], "limit": 1},
    )
    assert filtered_history.status_code == 200
    filtered_payload = filtered_history.json()
    assert len(filtered_payload) <= 1
    if filtered_payload:
        assert filtered_payload[0]["summary_severity"] == history_payload[0]["summary_severity"]
    invalid_filter = client.get("/reconciliation/history", params={"severity": "fatal"})
    assert invalid_filter.status_code == 400
    assert "Unsupported reconciliation severity" in invalid_filter.json()["detail"]

    acknowledge = client.post(
        "/reconciliation/acknowledge",
        json={
            "symbol": mismatch_symbol,
            "reason": "operator_reviewed",
            "operator_note": "known broker drift",
        },
    )
    assert acknowledge.status_code == 200
    acknowledge_body = acknowledge.json()
    assert acknowledge_body["acknowledged"] is True
    assert acknowledge_body["symbol"] == mismatch_symbol
    assert acknowledge_body["acknowledgement"]["reason"] == "operator_reviewed"
    assert acknowledge_body["incident_status"] == "acknowledged"

    reconciliation_after_ack = client.get("/reconciliation/summary")
    assert reconciliation_after_ack.status_code == 200
    acknowledged_row = next(
        item for item in reconciliation_after_ack.json()["mismatches"] if item["symbol"] == mismatch_symbol
    )
    assert acknowledged_row["acknowledged"] is True
    assert acknowledged_row["acknowledged_reason"] == "operator_reviewed"
    assert acknowledged_row["acknowledged_note"] == "known broker drift"
    assert acknowledged_row["incident_status"] == "acknowledged"
    assert reconciliation_after_ack.json()["incident_status_counts"]["acknowledged"] >= 1
    assert reconciliation_after_ack.json()["incidents"]

    incidents = client.get("/reconciliation/incidents")
    assert incidents.status_code == 200
    incident_row = next(item for item in incidents.json() if item["symbol"] == mismatch_symbol)
    assert incident_row["incident_status"] == "acknowledged"

    filtered_incidents = client.get(
        "/reconciliation/incidents",
        params={"symbol": mismatch_symbol, "incident_status": "acknowledged", "include_resolved": False},
    )
    assert filtered_incidents.status_code == 200
    assert len(filtered_incidents.json()) == 1
    assert filtered_incidents.json()[0]["symbol"] == mismatch_symbol

    investigating = client.post(
        "/reconciliation/incidents/update",
        json={
            "symbol": mismatch_symbol,
            "reason": "broker_fill_in_progress",
            "operator_note": "desk is waiting for broker reconciliation",
            "incident_status": "investigating",
        },
    )
    assert investigating.status_code == 200
    assert investigating.json()["incident_status"] == "investigating"

    FakeMT5Connector.positions_lots[mismatch_symbol] = 0.0

    resolved = client.post(
        "/reconciliation/incidents/update",
        json={
            "symbol": mismatch_symbol,
            "reason": "resolved_after_reconciliation",
            "operator_note": "desk confirmed final broker state",
            "incident_status": "resolved",
            "resolution_note": "broker and desk are now aligned",
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["incident_status"] == "resolved"
    assert resolved.json()["acknowledgement"]["resolution_note"] == "broker and desk are now aligned"
    assert resolved.json()["baseline_snapshot_id"] is not None

    reconciliation_after_resolve = client.get("/reconciliation/summary")
    assert reconciliation_after_resolve.status_code == 200
    resolved_row = next(
        item for item in reconciliation_after_resolve.json()["mismatches"] if item["symbol"] == mismatch_symbol
    )
    assert resolved_row["status"] == "match"
    assert resolved_row["incident_status"] is None
    assert resolved_row["resolution_note"] is None
    assert resolved_row["resolved_at"] is None
    assert reconciliation_after_resolve.json()["active_incident_count"] == 0
    assert reconciliation_after_resolve.json()["resolved_incident_count"] >= 1

    incidents_after_resolve = client.get("/reconciliation/incidents")
    assert incidents_after_resolve.status_code == 200
    resolved_incident = next(item for item in incidents_after_resolve.json() if item["symbol"] == mismatch_symbol)
    assert resolved_incident["incident_status"] == "resolved"
    assert resolved_incident["resolution_note"] == "broker and desk are now aligned"
    assert resolved_incident["resolved_at"] is not None

    unresolved_after_resolve = client.get(
        "/reconciliation/incidents",
        params={"symbol": mismatch_symbol, "include_resolved": False},
    )
    assert unresolved_after_resolve.status_code == 200
    assert unresolved_after_resolve.json() == []

    latest_live_snapshot = client.get("/snapshots/latest", params={"source": "mt5_live_bridge"})
    assert latest_live_snapshot.status_code == 200
    assert latest_live_snapshot.json()["payload"]["metadata"]["reconciliation_baseline_accepted"] is True

    snapshot = client.post("/snapshots/run", json={})
    assert snapshot.status_code == 200


def test_reconciliation_nominal_summary_without_drift(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    FakeMT5Connector.reset()
    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    live_state = client.get("/mt5/live/state")
    assert live_state.status_code == 200

    summary = client.get("/reconciliation/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["summary_severity"] in {"ok", "warn"}
    assert payload["critical_mismatch_count"] == 0
    assert payload["warning_mismatch_count"] >= 0


def test_reconciliation_detects_pnl_drift(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    PnlDriftConnector.reset()
    client = TestClient(create_app(repo_root=root, mt5_connector_factory=PnlDriftConnector, bootstrap_storage=True))

    baseline_live_state = client.get("/mt5/live/state")
    assert baseline_live_state.status_code == 200

    PnlDriftConnector.position_profit = 300.0
    summary = client.get("/reconciliation/summary")
    assert summary.status_code == 200
    payload = summary.json()
    pnl_mismatch = next((item for item in payload["mismatches"] if item["status"] == "pnl_drift"), None)
    assert pnl_mismatch is not None
    assert pnl_mismatch["severity"] in {"warn", "critical"}
    assert pnl_mismatch["pnl_difference_abs_eur"] is not None
    assert pnl_mismatch["pnl_difference_abs_eur"] > 0.0
    assert pnl_mismatch["probable_cause"] in {"mark_to_market_or_fee_drift", "valuation_drift"}


def test_api_mt5_transaction_history_supports_filters_pagination_and_csv(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    FakeMT5Connector.reset()

    timestamp = int(datetime(2026, 3, 30, 9, 0, tzinfo=timezone.utc).timestamp())
    FakeMT5Connector.order_history.append(
        {
            "ticket": 902,
            "position_id": 2,
            "symbol": "USDJPY",
            "type": 1,
            "state": 4,
            "volume_initial": 0.07,
            "volume_current": 0.00,
            "price_open": 156.11,
            "price_current": 156.07,
            "comment": "var_risk_desk:auto",
            "magic": 420001,
            "time_setup": timestamp,
            "time_done": timestamp + 60,
        }
    )
    FakeMT5Connector.deal_history.append(
        {
            "ticket": 802,
            "order": 902,
            "position_id": 2,
            "symbol": "USDJPY",
            "type": 1,
            "entry": 1,
            "volume": 0.07,
            "price": 156.08,
            "profit": -4.5,
            "commission": -0.3,
            "swap": 0.0,
            "fee": 0.0,
            "reason": 0,
            "comment": "var_risk_desk:auto",
            "magic": 420001,
            "time": timestamp + 60,
        }
    )
    for index in range(260):
        ts = timestamp + 120 + index
        FakeMT5Connector.deal_history.append(
            {
                "ticket": 9000 + index,
                "order": 9500 + index,
                "position_id": 100 + index,
                "symbol": "EURUSD",
                "type": 0,
                "entry": 1,
                "volume": 0.01,
                "price": 1.08 + (index * 0.00001),
                "profit": float(index) * 0.1,
                "commission": -0.01,
                "swap": 0.0,
                "fee": 0.0,
                "reason": 0,
                "comment": "var_risk_desk:auto",
                "magic": 420001,
                "time": ts,
            }
        )

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    first_page = client.get("/mt5/history/transactions", params={"page_size": 1, "page": 1, "type": "all"})
    assert first_page.status_code == 200
    first_page_body = first_page.json()
    assert first_page_body["total"] >= 4
    assert len(first_page_body["items"]) == 1
    assert first_page_body["has_next"] is True

    second_page = client.get("/mt5/history/transactions", params={"page_size": 1, "page": 2, "type": "all"})
    assert second_page.status_code == 200
    second_page_body = second_page.json()
    assert len(second_page_body["items"]) == 1
    assert second_page_body["items"][0]["id"] != first_page_body["items"][0]["id"]

    ascending = client.get("/mt5/history/transactions", params={"sort": "time_asc", "page_size": 200})
    assert ascending.status_code == 200
    ascending_items = ascending.json()["items"]
    ascending_times = [
        pd.to_datetime(item["time_utc"], utc=True)
        for item in ascending_items
        if item.get("time_utc")
    ]
    assert ascending_times == sorted(ascending_times)

    orders_usdjpy = client.get("/mt5/history/orders", params={"symbol": "USDJPY", "limit": 20})
    assert orders_usdjpy.status_code == 200
    assert orders_usdjpy.json()
    assert all(item["symbol"] == "USDJPY" for item in orders_usdjpy.json())

    deals_usdjpy = client.get("/mt5/history/deals", params={"symbol": "USDJPY", "limit": 20})
    assert deals_usdjpy.status_code == 200
    assert deals_usdjpy.json()
    assert all(item["symbol"] == "USDJPY" for item in deals_usdjpy.json())

    deals_only = client.get("/mt5/history/transactions", params={"type": "deal", "symbol": "USDJPY"})
    assert deals_only.status_code == 200
    deals_only_body = deals_only.json()
    assert deals_only_body["items"]
    assert all(item["kind"] == "deal" for item in deals_only_body["items"])
    assert all(item["symbol"] == "USDJPY" for item in deals_only_body["items"])

    manual_only = client.get("/mt5/history/transactions", params={"type": "manual"})
    assert manual_only.status_code == 200
    assert manual_only.json()["items"]
    assert all(item["is_manual"] is True for item in manual_only.json()["items"])

    desk_only = client.get("/mt5/history/transactions", params={"type": "desk"})
    assert desk_only.status_code == 200
    assert desk_only.json()["items"]
    assert all(item["is_manual"] is False for item in desk_only.json()["items"])

    csv_response = client.get("/mt5/history/transactions/export", params={"symbol": "USDJPY", "max_rows": 10})
    assert csv_response.status_code == 200
    assert "text/csv" in csv_response.headers["content-type"]
    assert "attachment; filename=" in csv_response.headers.get("content-disposition", "")
    csv_text = csv_response.text
    assert "kind,symbol,ticket" in csv_text
    assert "USDJPY" in csv_text

    csv_many_rows = client.get(
        "/mt5/history/transactions/export",
        params={"symbol": "EURUSD", "type": "deal", "max_rows": 250},
    )
    assert csv_many_rows.status_code == 200
    csv_many_lines = [line for line in csv_many_rows.text.splitlines() if line.strip()]
    assert len(csv_many_lines) == 251

    invalid_type = client.get("/mt5/history/transactions", params={"type": "unsupported"})
    assert invalid_type.status_code == 400
    assert "type must be one of" in invalid_type.json()["detail"]

    invalid_sort = client.get("/mt5/history/transactions", params={"sort": "invalid"})
    assert invalid_sort.status_code == 400
    assert "sort must be one of" in invalid_sort.json()["detail"]

    invalid_date = client.get("/mt5/history/transactions", params={"date_from": "not-a-date"})
    assert invalid_date.status_code == 400
    assert "date_from must be a valid ISO-8601 datetime" in invalid_date.json()["detail"]

    invalid_csv_date = client.get("/mt5/history/transactions/export", params={"date_to": "not-a-date"})
    assert invalid_csv_date.status_code == 400
    assert "date_to must be a valid ISO-8601 datetime" in invalid_csv_date.json()["detail"]

    invalid_window = client.get(
        "/mt5/history/transactions",
        params={"date_from": "2026-04-21T10:00:00Z", "date_to": "2026-04-20T10:00:00Z"},
    )
    assert invalid_window.status_code == 400
    assert "date_from must be less than or equal to date_to" in invalid_window.json()["detail"]

    invalid_order_window = client.get(
        "/mt5/history/orders",
        params={"date_from": "2026-04-21T10:00:00Z", "date_to": "2026-04-20T10:00:00Z"},
    )
    assert invalid_order_window.status_code == 400

    invalid_deal_window = client.get(
        "/mt5/history/deals",
        params={"date_from": "2026-04-21T10:00:00Z", "date_to": "2026-04-20T10:00:00Z"},
    )
    assert invalid_deal_window.status_code == 400


def test_live_state_backfills_market_data_without_manual_sync(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    FakeMT5Connector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    live_state = client.get("/mt5/live/state")
    assert live_state.status_code == 200
    assert live_state.json()["connected"] is True
    live_codes = {str(item.get("code") or "") for item in list(live_state.json().get("operator_alerts") or [])}
    assert "VALIDATION_SURFACE_SAMPLE_THIN" not in live_codes
    assert "VALIDATION_HORIZON_SAMPLE_THIN" not in live_codes

    status = client.get("/market-data/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["instrument_count"] >= 2
    assert status_body["missing_symbols"] == []
    assert status_body["missing_bars"] == []
    assert status_body["retention_tiers"]["H1"] >= 60
    assert status_body["tick_archive"]["row_count"] >= 0
    assert status_body["coverage_status"] in {"healthy", "thin_history", "stale", "incomplete"}

    risk_summary = client.get("/risk/summary")
    assert risk_summary.status_code == 200
    risk_summary_body = risk_summary.json()
    assert risk_summary_body["headline_risk"]
    assert risk_summary_body["risk_nowcast"]["live_1d_99"]["nowcast_var"] is not None
    assert risk_summary_body["concentration"] is not None
    assert risk_summary_body["concentration"]["var"]["top3_share"] is not None

    risk_contributions = client.get("/risk/contributions")
    assert risk_contributions.status_code == 200
    risk_contributions_body = risk_contributions.json()
    assert risk_contributions_body["models"]
    assert risk_contributions_body["models"]["hist"]["asset_classes"]
    assert risk_contributions_body["models"]["hist"]["concentration_es"] is not None
    assert risk_contributions_body["models"]["hist"]["concentration_es"]["effective_count"] is not None

    instruments = client.get("/instruments")
    assert instruments.status_code == 200
    assert {item["symbol"] for item in instruments.json()} == {"EURUSD", "USDJPY"}

    active_alerts = client.get("/alerts/active", params={"limit": 10, "portfolio_slug": "fx_eur_20k"})
    assert active_alerts.status_code == 200
    active_alerts_body = active_alerts.json()
    assert isinstance(active_alerts_body, list)
    if active_alerts_body:
        assert all(item.get("is_active") is True for item in active_alerts_body)
        assert {str(item.get("code") or "") for item in active_alerts_body}.issubset(live_codes)

    latest_snapshot = client.get("/snapshots/latest", params={"source": "mt5_live"})
    assert latest_snapshot.status_code == 200
    assert latest_snapshot.json()["source"] == "mt5_live"
    latest_snapshot_auto = client.get("/snapshots/latest", params={"source": "auto"})
    assert latest_snapshot_auto.status_code == 200
    assert latest_snapshot_auto.json()["source"] in {"mt5_live_bridge", "mt5_live"}

    backtest = client.post("/backtests/run", json={})
    assert backtest.status_code == 200

    latest_capital = client.get("/capital/latest")
    assert latest_capital.status_code == 200
    assert latest_capital.json()["snapshot_source"] == "mt5_live_bridge"

    live_capital_history = client.get(
        "/capital/history",
        params={"limit": 5, "source": "mt5_live_bridge"},
    )
    assert live_capital_history.status_code == 200
    assert live_capital_history.json()
    assert all(item["snapshot_source"] == "mt5_live_bridge" for item in live_capital_history.json())

    live_report_capital_history = client.get(
        "/reports/capital-history",
        params={"limit": 5, "source": "mt5_live_bridge"},
    )
    assert live_report_capital_history.status_code == 200
    assert live_report_capital_history.json()
    assert all(item["snapshot_source"] == "mt5_live_bridge" for item in live_report_capital_history.json())

    refreshed_live_state = client.get("/mt5/live/state")
    assert refreshed_live_state.status_code == 200
    live_report = client.get("/reports/latest")
    assert live_report.status_code == 200
    assert "Preferred snapshot source: **mt5_live_bridge**" in live_report.json()["content"]
    assert "## Portfolio Snapshot" in live_report.json()["content"]
    latest_live_report_artifact = client.get("/artifacts/latest/daily_report")
    assert latest_live_report_artifact.status_code == 200
    assert latest_live_report_artifact.json()["details"]["auto_generated"] is True


def test_mt5_live_state_handles_default_portfolio_alias_and_unknown_slug(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    FakeMT5Connector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    default_alias = client.get("/mt5/live/state", params={"portfolio_slug": "default", "detail_level": "summary"})
    assert default_alias.status_code == 200
    assert default_alias.json()["portfolio_slug"] == "fx_eur_20k"

    unknown = client.get("/mt5/live/state", params={"portfolio_slug": "does_not_exist", "detail_level": "summary"})
    assert unknown.status_code == 404


def test_strict_live_ignores_historical_source_override_on_live_routes(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FakeMT5Connector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    warmup = client.get("/mt5/live/state")
    assert warmup.status_code == 200
    assert warmup.json()["connected"] is True

    snapshot = client.get("/snapshots/latest", params={"source": "historical"})
    assert snapshot.status_code in {200, 404}
    if snapshot.status_code == 200:
        assert snapshot.json()["source"] in {"mt5_live_bridge", "mt5_live"}

    capital = client.get("/capital/latest", params={"source": "historical"})
    assert capital.status_code in {200, 503}
    if capital.status_code == 200:
        assert str(capital.json()["snapshot_source"]).startswith("mt5_live")


def test_live_stress_uses_mt5_holdings(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FakeMT5Connector.reset()
    FakeMT5Connector.positions_lots = {"EURUSD": 0.01, "USDJPY": 0.01}

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))
    stress = client.post(
        "/snapshots/stress",
        json={
            "scenarios": [
                {"name": "Small shock", "vol_multiplier": 1.2, "shock_pnl": -5.0},
            ]
        },
    )
    assert stress.status_code == 200
    stress_body = stress.json()

    service = DeskApiService(root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True)
    portfolio = service.runtime._resolve_portfolio_context(None)
    with service.runtime._mt5_gateway() as live:
        holdings = [item.to_dict() for item in live.holdings(symbols=None)]
    bundle = service.runtime._compute_portfolio_state_for_holdings(
        portfolio=portfolio,
        holdings=holdings,
        timeframe=service.runtime._default_timeframe(),
        days=service.runtime._default_days(),
        min_coverage=float(service.runtime.data_defaults["min_coverage"]),
        config=service.runtime._build_risk_model_config(None, None, None, None, None),
        window=int(service.runtime.risk_defaults["window"]),
        snapshot_source="mt5_live_bridge",
    )
    expected = historical_var_es(bundle["sample"]["pnl"], 0.95)

    assert stress_body["portfolio_slug"] == "fx_eur_20k"
    assert stress_body["baseline_var"] == expected.var
    assert stress_body["baseline_es"] == expected.es
    assert stress_body["headline_risk"]
    assert stress_body["historical_extremes"]
    assert stress_body["scenarios"][0]["risk_surface"]
    assert stress_body["attribution"]["models"]["hist"]["asset_classes"]
    assert stress_body["scenarios"][0]["attribution"]["models"]["hist"]["positions"]


def test_get_stress_endpoint_returns_default_report(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FakeMT5Connector.reset()
    FakeMT5Connector.positions_lots = {"EURUSD": 0.01, "USDJPY": 0.01}

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))
    stress = client.get("/snapshots/stress", params={"portfolio_slug": "fx_eur_20k"})

    assert stress.status_code == 200
    stress_body = stress.json()
    assert stress_body["portfolio_slug"] == "fx_eur_20k"
    assert stress_body["headline_risk"]
    assert stress_body["historical_extremes"]
    assert stress_body["scenarios"]


def test_live_stress_baseline_respects_requested_alpha(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FakeMT5Connector.reset()
    FakeMT5Connector.positions_lots = {"EURUSD": 0.01, "USDJPY": 0.01}

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))
    stress = client.post(
        "/snapshots/stress",
        json={
            "alpha": 0.99,
            "scenarios": [
                {"name": "Small shock", "vol_multiplier": 1.2, "shock_pnl": -5.0},
            ],
        },
    )
    assert stress.status_code == 200
    stress_body = stress.json()

    service = DeskApiService(root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True)
    portfolio = service.runtime._resolve_portfolio_context(None)
    with service.runtime._mt5_gateway() as live:
        holdings = [item.to_dict() for item in live.holdings(symbols=None)]
    bundle = service.runtime._compute_portfolio_state_for_holdings(
        portfolio=portfolio,
        holdings=holdings,
        timeframe=service.runtime._default_timeframe(),
        days=service.runtime._default_days(),
        min_coverage=float(service.runtime.data_defaults["min_coverage"]),
        config=service.runtime._build_risk_model_config(0.99, None, None, None, None),
        window=int(service.runtime.risk_defaults["window"]),
        snapshot_source="mt5_live_bridge",
    )
    expected = historical_var_es(bundle["sample"]["pnl"], 0.99)

    assert stress_body["alpha"] == 0.99
    assert stress_body["baseline_var"] == pytest.approx(expected.var, rel=0.05)
    assert stress_body["baseline_es"] == pytest.approx(expected.es, rel=0.05)


def test_market_data_sync_backfills_richer_history_for_var(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    FakeMT5Connector.reset()

    service = DeskApiService(root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]

    status = service.runtime.market_data.sync_market_data(
        portfolio_slug=portfolio_slug,
        days=60,
        timeframes=["H1"],
    )
    latest_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
    assert latest_sync is not None
    details = dict(latest_sync.get("details") or {})

    assert details["requested_days"] == 60
    assert details["stored_history_days"] == 60
    assert details["history_reconciliation_days"] <= 30
    assert status["stored_history_days"] == 60
    assert details["coverage"]["EURUSD"]["H1"]["stored_bars"] >= 60 * 24
    assert details["coverage"]["USDJPY"]["H1"]["stored_bars"] >= 60 * 24
    assert details["tick_archive"]["summary"]["row_count"] >= 1
    assert status["tick_archive"]["row_count"] >= 1


class HistoryWindowTrackingConnector(FakeMT5Connector):
    order_history_calls: list[dict[str, str]] = []
    deal_history_calls: list[dict[str, str]] = []

    @classmethod
    def reset(cls) -> None:
        super().reset()
        cls.order_history_calls = []
        cls.deal_history_calls = []

    def history_orders_get(
        self,
        date_from,
        date_to,
        *,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ):
        type(self).order_history_calls.append(
            {
                "date_from": pd.to_datetime(date_from, utc=True).isoformat(),
                "date_to": pd.to_datetime(date_to, utc=True).isoformat(),
            }
        )
        return super().history_orders_get(
            date_from,
            date_to,
            symbol=symbol,
            ticket=ticket,
            position=position,
        )

    def history_deals_get(
        self,
        date_from,
        date_to,
        *,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ):
        type(self).deal_history_calls.append(
            {
                "date_from": pd.to_datetime(date_from, utc=True).isoformat(),
                "date_to": pd.to_datetime(date_to, utc=True).isoformat(),
            }
        )
        return super().history_deals_get(
            date_from,
            date_to,
            symbol=symbol,
            ticket=ticket,
            position=position,
        )


class FailDealHistoryOnceConnector(FakeMT5Connector):
    fail_deal_history_once: bool = True

    @classmethod
    def reset(cls) -> None:
        super().reset()
        cls.fail_deal_history_once = True

    def history_deals_get(
        self,
        date_from,
        date_to,
        *,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ):
        if type(self).fail_deal_history_once:
            type(self).fail_deal_history_once = False
            raise RuntimeError("simulated-deal-history-outage")
        return super().history_deals_get(
            date_from,
            date_to,
            symbol=symbol,
            ticket=ticket,
            position=position,
        )


def test_market_data_sync_records_checkpoint_and_stage_events(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    FakeMT5Connector.reset()

    service = DeskApiService(root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]

    service.runtime.market_data.sync_market_data(
        portfolio_slug=portfolio_slug,
        days=60,
        timeframes=["H1"],
    )
    latest_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
    assert latest_sync is not None
    details = dict(latest_sync.get("details") or {})
    checkpoint = dict(details.get("checkpoint") or {})
    stage_events = list(details.get("stage_events") or [])

    assert checkpoint.get("stage") == "completed"
    assert checkpoint.get("status") == "ok"
    completed = set(checkpoint.get("completed_stages") or [])
    assert {
        "mt5_connect",
        "capture_state",
        "sync_instruments_bars",
        "sync_order_history",
        "sync_deal_history",
        "sync_ticks_and_snapshot",
    }.issubset(completed)
    assert details.get("sync_started_at") is not None
    assert details.get("sync_completed_at") is not None
    assert details["history_windows"]["orders"]["mode"] == "bootstrap"
    assert details["history_windows"]["deals"]["mode"] == "bootstrap"
    assert len(stage_events) >= 6
    assert all(event.get("duration_ms") is not None for event in stage_events)


def test_market_data_sync_defaults_to_operational_timeframes(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    FakeMT5Connector.reset()

    service = DeskApiService(root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]

    status = service.runtime.market_data.sync_market_data(portfolio_slug=portfolio_slug, days=60)
    latest_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)

    assert latest_sync is not None
    details = dict(latest_sync.get("details") or {})
    assert details["timeframes"] == service.runtime.market_data.startup_sync_timeframes()
    assert "M1" not in details["timeframes"]
    assert set(details["coverage"]["EURUSD"]) == set(details["timeframes"])
    assert set(details["coverage"]["USDJPY"]) == set(details["timeframes"])
    assert status["stored_history_days"] == 60


def test_market_data_sync_writes_single_row_per_run(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    FakeMT5Connector.reset()

    service = DeskApiService(root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]

    def _sync_count() -> int:
        with service.runtime.storage.engine.connect() as connection:
            result = connection.execute(
                text("SELECT COUNT(*) FROM market_data_sync_runs WHERE portfolio_slug = :slug"),
                {"slug": portfolio_slug},
            )
            return int(result.scalar_one())

    before = _sync_count()
    service.runtime.market_data.sync_market_data(portfolio_slug=portfolio_slug, days=60, timeframes=["H1"])
    after_first = _sync_count()
    service.runtime.market_data.sync_market_data(portfolio_slug=portfolio_slug, days=60, timeframes=["H1"])
    after_second = _sync_count()

    assert after_first == before + 1
    assert after_second == after_first + 1
    latest_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
    assert latest_sync is not None
    assert str(latest_sync["status"]) in {"ok", "incomplete"}


def test_market_data_status_closes_stale_running_sync(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    FakeMT5Connector.reset()

    service = DeskApiService(root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]
    portfolio_id = service.runtime._resolve_portfolio_id(portfolio_slug)

    stale_run_id = service.runtime.storage.record_market_data_sync(
        portfolio_id=portfolio_id,
        portfolio_slug=portfolio_slug,
        mode="live_mt5",
        status="running",
        details={
            "symbols": ["EURUSD", "USDJPY"],
            "timeframes": ["H1"],
            "requested_days": 60,
        },
    )
    service.runtime.storage.update_market_data_sync(
        stale_run_id,
        synced_at=utcnow() - timedelta(hours=2),
    )

    status = service.runtime.market_data.market_data_status(portfolio_slug=portfolio_slug)
    latest_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)

    assert latest_sync is not None
    assert status["status"] == "incomplete"
    assert latest_sync["status"] == "incomplete"
    errors = list(dict(latest_sync.get("details") or {}).get("errors") or [])
    assert any(str(item.get("code") or "") == "stale_market_sync_run" for item in errors)


def test_market_data_sync_runs_endpoint_lists_and_filters_recent_runs(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    FailDealHistoryOnceConnector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FailDealHistoryOnceConnector, bootstrap_storage=True))
    first_sync = client.post("/market-data/sync", json={"days": 60, "timeframes": ["H1"]})
    second_sync = client.post("/market-data/sync", json={"days": 60, "timeframes": ["H1"]})
    assert first_sync.status_code == 200
    assert second_sync.status_code == 200

    runs_resp = client.get("/market-data/sync/runs", params={"limit": 5})
    assert runs_resp.status_code == 200
    runs = runs_resp.json()
    assert len(runs) >= 2
    assert all("checkpoint" in dict(item.get("details") or {}) for item in runs)
    assert any(str(item.get("status")) == "incomplete" for item in runs)
    assert any(str(item.get("status")) == "ok" for item in runs)

    only_ok_resp = client.get("/market-data/sync/runs", params=[("status", "ok"), ("limit", "5")])
    assert only_ok_resp.status_code == 200
    only_ok_runs = only_ok_resp.json()
    assert only_ok_runs
    assert all(str(item.get("status")) == "ok" for item in only_ok_runs)

    only_ok_upper_resp = client.get("/market-data/sync/runs", params=[("status", "OK"), ("limit", "5")])
    assert only_ok_upper_resp.status_code == 200
    only_ok_upper_runs = only_ok_upper_resp.json()
    assert only_ok_upper_runs
    assert all(str(item.get("status")) == "ok" for item in only_ok_upper_runs)

    limited_resp = client.get("/market-data/sync/runs", params={"limit": 1})
    assert limited_resp.status_code == 200
    assert len(limited_resp.json()) == 1

    blank_filter_resp = client.get("/market-data/sync/runs", params=[("status", "")])
    assert blank_filter_resp.status_code == 200
    assert blank_filter_resp.json() == []


def test_market_data_sync_uses_incremental_history_windows_after_bootstrap(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    HistoryWindowTrackingConnector.reset()

    service = DeskApiService(root, mt5_connector_factory=HistoryWindowTrackingConnector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]

    service.runtime.market_data.sync_market_data(
        portfolio_slug=portfolio_slug,
        days=60,
        timeframes=["H1"],
    )
    service.runtime.market_data.sync_market_data(
        portfolio_slug=portfolio_slug,
        days=60,
        timeframes=["H1"],
    )

    assert len(HistoryWindowTrackingConnector.order_history_calls) >= 2
    assert len(HistoryWindowTrackingConnector.deal_history_calls) >= 2
    first_order_from = pd.to_datetime(HistoryWindowTrackingConnector.order_history_calls[-2]["date_from"], utc=True)
    second_order_from = pd.to_datetime(HistoryWindowTrackingConnector.order_history_calls[-1]["date_from"], utc=True)
    first_deal_from = pd.to_datetime(HistoryWindowTrackingConnector.deal_history_calls[-2]["date_from"], utc=True)
    second_deal_from = pd.to_datetime(HistoryWindowTrackingConnector.deal_history_calls[-1]["date_from"], utc=True)

    assert second_order_from > first_order_from
    assert second_deal_from > first_deal_from

    latest_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
    assert latest_sync is not None
    details = dict(latest_sync.get("details") or {})
    assert details["history_windows"]["orders"]["mode"] == "incremental"
    assert details["history_windows"]["deals"]["mode"] == "incremental"


def test_market_data_sync_bootstrap_history_honors_reconciliation_window(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    HistoryWindowTrackingConnector.reset()

    service = DeskApiService(root, mt5_connector_factory=HistoryWindowTrackingConnector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]
    portfolio_id = service.runtime._resolve_portfolio_id(portfolio_slug)
    unresolved_created_at = utcnow() - timedelta(days=14, hours=6)

    service.runtime.storage.record_execution_result(
        {
            "created_at": unresolved_created_at.isoformat(),
            "time_utc": unresolved_created_at.isoformat(),
            "portfolio_slug": portfolio_slug,
            "symbol": "EURUSD",
            "status": "EXECUTED",
            "reconciliation_status": "pending_broker",
            "requested_exposure_change": 1_000.0,
            "approved_exposure_change": 1_000.0,
            "executed_exposure_change": 0.0,
            "mt5_result": {"order": 91_000_001, "deal": 81_000_001},
        },
        portfolio_id=portfolio_id,
    )

    service.runtime.market_data.sync_market_data(
        portfolio_slug=portfolio_slug,
        days=1,
        timeframes=["H1"],
    )
    latest_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)

    assert latest_sync is not None
    assert HistoryWindowTrackingConnector.order_history_calls
    assert HistoryWindowTrackingConnector.deal_history_calls

    details = dict(latest_sync.get("details") or {})
    sync_started = pd.to_datetime(details["sync_started_at"], utc=True).to_pydatetime()
    expected_reconciliation_days = max(
        7,
        min(
            int(max((sync_started - unresolved_created_at).total_seconds(), 0.0) // 86400) + 1,
            30,
        ),
    )

    last_order_call = HistoryWindowTrackingConnector.order_history_calls[-1]
    last_deal_call = HistoryWindowTrackingConnector.deal_history_calls[-1]
    order_from = pd.to_datetime(last_order_call["date_from"], utc=True).to_pydatetime()
    order_to = pd.to_datetime(last_order_call["date_to"], utc=True).to_pydatetime()
    deal_from = pd.to_datetime(last_deal_call["date_from"], utc=True).to_pydatetime()
    deal_to = pd.to_datetime(last_deal_call["date_to"], utc=True).to_pydatetime()

    order_lookback_days = (order_to - order_from).total_seconds() / 86400.0
    deal_lookback_days = (deal_to - deal_from).total_seconds() / 86400.0
    assert order_lookback_days >= float(expected_reconciliation_days) - 0.01
    assert deal_lookback_days >= float(expected_reconciliation_days) - 0.01
    assert int(details.get("history_reconciliation_days") or 0) >= expected_reconciliation_days


def test_market_data_sync_recovers_after_partial_history_failure_without_duplicates(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5", market_history_days=365)
    FailDealHistoryOnceConnector.reset()

    service = DeskApiService(root, mt5_connector_factory=FailDealHistoryOnceConnector, bootstrap_storage=True)
    portfolio_slug = service.runtime.portfolio["slug"]

    first_status = service.runtime.market_data.sync_market_data(
        portfolio_slug=portfolio_slug,
        days=60,
        timeframes=["H1"],
    )
    assert first_status["status"] == "incomplete"
    first_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
    assert first_sync is not None
    first_details = dict(first_sync.get("details") or {})
    assert any(item.get("scope") == "history_deals" for item in list(first_details.get("errors") or []))

    with service.runtime.storage.engine.connect() as connection:
        first_orders = int(connection.execute(text("SELECT COUNT(*) FROM mt5_order_history")).scalar_one())
        first_deals = int(connection.execute(text("SELECT COUNT(*) FROM mt5_deal_history")).scalar_one())

    second_status = service.runtime.market_data.sync_market_data(
        portfolio_slug=portfolio_slug,
        days=60,
        timeframes=["H1"],
    )
    second_sync = service.runtime.storage.latest_market_data_sync(portfolio_slug=portfolio_slug)
    assert second_sync is not None
    second_details = dict(second_sync.get("details") or {})

    with service.runtime.storage.engine.connect() as connection:
        second_orders = int(connection.execute(text("SELECT COUNT(*) FROM mt5_order_history")).scalar_one())
        second_deals = int(connection.execute(text("SELECT COUNT(*) FROM mt5_deal_history")).scalar_one())

    assert second_status["status"] == "ok"
    assert second_orders == first_orders
    assert second_deals >= first_deals
    resume_from = dict(second_details.get("resume_from") or {})
    assert resume_from.get("status") == "incomplete"
