from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from var_project.api import create_app
from var_project.api.service import DeskApiService
from var_project.risk.expected_shortfall import historical_var_es
from test_mt5_execution_api import FakeMT5Connector


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

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["dependencies"]["database"]["schema_ready"] is True
    assert "mt5_live" in health.json()["dependencies"]

    jobs_status = client.get("/jobs/status")
    assert jobs_status.status_code == 200
    assert jobs_status.json()["database_ready"] is True

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

    latest_validation = client.get("/validations/latest")
    assert latest_validation.status_code == 200
    assert latest_validation.json()["best_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}

    latest_comparison = client.get("/models/compare/latest")
    assert latest_comparison.status_code == 200
    comparison_body = latest_comparison.json()
    assert comparison_body["champion_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}
    assert comparison_body["ranking"]

    latest_attribution = client.get("/snapshots/attribution/latest", params={"source": "historical"})
    assert latest_attribution.status_code == 200
    attribution_body = latest_attribution.json()
    assert attribution_body["models"]["hist"]["positions"]["EURUSD"]["standalone_var"] >= 0.0

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
        json={"symbol": "EURUSD", "exposure_change": 2_500.0, "note": "post-validation"},
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
    assert latest_report_payload.json()["content"].startswith("# Risk Report")
    assert "Preferred snapshot source: **historical**" in latest_report_payload.json()["content"]
    assert "## Portfolio Snapshot" in latest_report_payload.json()["content"]
    assert "## Decision History" in latest_report_payload.json()["content"]
    assert "## Capital History" in latest_report_payload.json()["content"]
    assert "## Audit Trail" in latest_report_payload.json()["content"]

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
    mismatch_row = next(item for item in reconciliation_body["mismatches"] if item["status"] != "match")
    mismatch_symbol = mismatch_row["symbol"]
    assert mismatch_row["acknowledged"] is False

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


def test_live_state_backfills_market_data_without_manual_sync(tmp_path: Path):
    root = tmp_path
    _write_settings(root, portfolio_mode="live_mt5")
    FakeMT5Connector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    live_state = client.get("/mt5/live/state")
    assert live_state.status_code == 200
    assert live_state.json()["connected"] is True

    status = client.get("/market-data/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["instrument_count"] >= 2
    assert status_body["missing_symbols"] == []
    assert status_body["missing_bars"] == []
    assert status_body["retention_tiers"]["H1"] >= 60
    assert status_body["tick_archive"]["row_count"] >= 1
    assert status_body["coverage_status"] in {"healthy", "thin_history", "stale", "incomplete"}

    risk_summary = client.get("/risk/summary")
    assert risk_summary.status_code == 200
    risk_summary_body = risk_summary.json()
    assert risk_summary_body["headline_risk"]
    assert risk_summary_body["risk_nowcast"]["live_1d_99"]["nowcast_var"] is not None

    risk_contributions = client.get("/risk/contributions")
    assert risk_contributions.status_code == 200
    risk_contributions_body = risk_contributions.json()
    assert risk_contributions_body["models"]
    assert risk_contributions_body["models"]["hist"]["asset_classes"]

    instruments = client.get("/instruments")
    assert instruments.status_code == 200
    assert {item["symbol"] for item in instruments.json()} == {"EURUSD", "USDJPY"}
    latest_snapshot = client.get("/snapshots/latest", params={"source": "mt5_live"})
    assert latest_snapshot.status_code == 200
    assert latest_snapshot.json()["source"] == "mt5_live"

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
    assert stress_body["baseline_var"] == expected.var
    assert stress_body["baseline_es"] == expected.es


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
    assert details["stored_history_days"] == 365
    assert status["stored_history_days"] == 365
    assert details["coverage"]["EURUSD"]["H1"]["bars"] >= 365 * 24
    assert details["coverage"]["USDJPY"]["H1"]["bars"] >= 365 * 24
    assert details["tick_archive"]["summary"]["row_count"] >= 1
    assert status["tick_archive"]["row_count"] >= 1
