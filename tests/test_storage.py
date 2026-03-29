from __future__ import annotations

from pathlib import Path

import pandas as pd

from var_project.alerts.engine import alerts_from_validation_summary
from var_project.storage import AppStorage
from var_project.validation.model_validation import validate_compare_frame


def _compare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "pnl": [-12, 4, -15, 3, -11, 5, -13, 2, -10, 1],
            "var_hist": [10] * 10,
            "es_hist": [12] * 10,
            "exc_hist": [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            "var_param": [14] * 10,
            "es_param": [16] * 10,
            "exc_param": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "portfolio": ["FX_EUR_20k"] * 10,
            "base_currency": ["EUR"] * 10,
            "symbols": ["EURUSD,USDJPY"] * 10,
            "positions_eur_json": ['{"EURUSD": 10000, "USDJPY": 10000}'] * 10,
            "timeframe": ["H1"] * 10,
            "days": [365] * 10,
        }
    )


def test_app_storage_persists_and_reads_platform_records(tmp_path: Path):
    root = tmp_path
    storage = AppStorage.from_root(root, {"storage": {"database_path": "data/app/test.db"}})
    storage.initialize(create_schema=True)

    assert storage.settings.database_path is not None
    assert storage.settings.database_path.exists()

    portfolio_id = storage.upsert_portfolio(
        name="FX_EUR_20k",
        base_currency="EUR",
        symbols=["EURUSD", "USDJPY"],
        positions={"EURUSD": 10_000.0, "USDJPY": 10_000.0},
        slug="fx_eur_20k",
    )

    compare = _compare_frame()
    compare_path = root / "reports" / "backtests" / "compare_test.csv"
    compare_artifact_id = storage.write_dataframe_artifact(
        compare,
        compare_path,
        artifact_type="backtest_compare",
        index=False,
    )
    storage.record_backtest_run(
        portfolio_id=portfolio_id,
        artifact_id=compare_artifact_id,
        timeframe="H1",
        days=365,
        alpha=0.95,
        window=250,
        n_rows=len(compare),
        summary={"exception_counts": {"hist": 4, "param": 1}},
    )

    summary = validate_compare_frame(compare, alpha=0.95)
    validation_run_id = storage.record_validation_run(
        summary,
        portfolio_id=portfolio_id,
        source_artifact_id=compare_artifact_id,
    )
    storage.record_alerts(
        alerts_from_validation_summary(summary),
        portfolio_id=portfolio_id,
        validation_run_id=validation_run_id,
    )

    snapshot = {
        "time_utc": "2026-03-27T09:00:00+00:00",
        "alpha": 0.95,
        "timeframe": "H1",
        "days": 365,
        "window": 250,
        "positions_eur": {"EURUSD": 10_000.0, "USDJPY": 10_000.0},
        "var": {"hist": 250.0, "ewma": 180.0},
        "es": {"hist": 270.0, "ewma": 190.0},
        "live_loss_proxy": 260.0,
        "breach_hist": True,
        "limits": {"zone_hist": "RED", "zone_ewma": "AMBER"},
        "alerts": [],
    }
    snapshot_path = root / "data" / "snapshots" / "live_snapshot_test.json"
    snapshot_artifact_id = storage.write_json_artifact(snapshot, snapshot_path, artifact_type="live_snapshot")
    storage.record_snapshot(
        snapshot,
        portfolio_id=portfolio_id,
        artifact_id=snapshot_artifact_id,
        source="live",
    )
    decision_id = storage.record_decision(
        {
            "time_utc": "2026-03-27T09:05:00+00:00",
            "symbol": "EURUSD",
            "decision": "REDUCE",
            "requested_delta_position_eur": 5_000.0,
            "approved_delta_position_eur": 3_000.0,
            "suggested_delta_position_eur": 3_000.0,
            "resulting_position_eur": 13_000.0,
            "model_used": "hist",
            "reasons": ["Too close to budget"],
            "pre_trade": {"var": 180.0, "es": 220.0, "headroom_var": 20.0, "headroom_es": 30.0, "gross_notional": 20_000.0, "position_eur": 10_000.0, "status": "OK"},
            "post_trade": {"var": 198.0, "es": 240.0, "headroom_var": 2.0, "headroom_es": 10.0, "gross_notional": 23_000.0, "position_eur": 13_000.0, "status": "WARN"},
        },
        portfolio_id=portfolio_id,
    )
    capital_id = storage.record_capital_snapshot(
        {
            "portfolio_slug": "fx_eur_20k",
            "base_currency": "EUR",
            "reference_model": "hist",
            "snapshot_source": "historical",
            "snapshot_timestamp": "2026-03-27T09:10:00+00:00",
            "total_capital_budget_eur": 300.0,
            "total_capital_consumed_eur": 220.0,
            "total_capital_reserved_eur": 30.0,
            "total_capital_remaining_eur": 50.0,
            "headroom_ratio": 0.1667,
            "status": "WARN",
            "budget": {
                "reference_model": "hist",
                "total_budget_eur": 300.0,
                "reserve_ratio": 0.10,
                "reserved_capital_eur": 30.0,
                "model_budgets": {"hist": 300.0},
                "symbol_budgets": {"EURUSD": 135.0, "USDJPY": 135.0},
            },
            "models": {"hist": {"model": "hist", "budget_eur": 300.0, "consumed_eur": 220.0, "remaining_eur": 80.0, "utilization": 0.7333, "status": "OK"}},
            "allocations": {
                "EURUSD": {"symbol": "EURUSD", "weight": 0.5, "target_capital_eur": 135.0, "consumed_capital_eur": 120.0, "reserved_capital_eur": 15.0, "remaining_capital_eur": 0.0, "utilization": 0.8889, "action": "HOLD", "status": "WARN"}
            },
            "recommendations": [],
        },
        portfolio_id=portfolio_id,
        source="historical",
    )
    execution_result_id = storage.record_execution_result(
        {
            "time_utc": "2026-03-27T09:15:00+00:00",
            "portfolio_slug": "fx_eur_20k",
            "symbol": "USDJPY",
            "requested_delta_position_eur": -1500.0,
            "approved_delta_position_eur": -1200.0,
            "executed_delta_position_eur": -1200.0,
            "status": "EXECUTED",
            "requested_volume_lots": 0.12,
            "approved_volume_lots": 0.10,
            "submitted_volume_lots": 0.10,
            "filled_volume_lots": 0.08,
            "remaining_volume_lots": 0.02,
            "fill_ratio": 0.8,
            "broker_status": "filled",
            "position_id": 10_501,
            "slippage_points": 1.5,
            "reconciliation_status": "partial_fill",
            "mt5_result": {"order": 910, "deal": 810, "retcode": 10009},
            "fills": [
                {
                    "symbol": "USDJPY",
                    "order_ticket": 910,
                    "deal_ticket": 810,
                    "position_id": 10_501,
                    "side": "SELL",
                    "entry": "in",
                    "volume_lots": 0.08,
                    "price": 156.42,
                    "profit": 0.0,
                    "commission": -0.4,
                    "swap": 0.0,
                    "fee": 0.0,
                    "reason": "dealer",
                    "comment": "desk submit",
                    "is_manual": False,
                    "slippage_points": 1.5,
                    "time_utc": "2026-03-27T09:15:01+00:00",
                    "raw": {"ticket": 810},
                }
            ],
        },
        portfolio_id=portfolio_id,
        decision_id=decision_id,
    )
    audit_id = storage.record_audit_event(
        actor="api",
        action_type="execution.submit",
        object_type="execution_result",
        object_id=execution_result_id,
        payload={"portfolio_slug": "fx_eur_20k"},
        portfolio_id=portfolio_id,
    )

    report_path = root / "reports" / "daily" / "compare_test.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Report", encoding="utf-8")
    storage.register_artifact(report_path, artifact_type="daily_report")

    latest_backtest = storage.latest_backtest_run(portfolio_slug="fx_eur_20k")
    latest_validation = storage.latest_validation_run(portfolio_slug="fx_eur_20k")
    latest_snapshot = storage.latest_snapshot(source="live", portfolio_slug="fx_eur_20k")
    recent_alerts = storage.recent_alerts(limit=10)
    recent_decisions = storage.recent_decisions(limit=10, portfolio_slug="fx_eur_20k")
    latest_capital = storage.latest_capital_snapshot(source="historical", portfolio_slug="fx_eur_20k")
    capital_history = storage.capital_history(limit=10, portfolio_slug="fx_eur_20k")
    recent_execution_results = storage.recent_execution_results(limit=10, portfolio_slug="fx_eur_20k")
    recent_execution_fills = storage.recent_execution_fills(limit=10, portfolio_slug="fx_eur_20k")
    recent_audit = storage.recent_audit_events(limit=10, portfolio_slug="fx_eur_20k")
    latest_report = storage.latest_artifact("daily_report")
    portfolios = storage.list_portfolios()

    assert latest_backtest is not None
    assert latest_backtest["artifact_id"] == compare_artifact_id
    assert latest_validation is not None
    assert latest_validation["best_model"] == summary.best_model
    assert latest_snapshot is not None
    assert latest_snapshot["payload"]["var"]["hist"] == 250.0
    assert recent_alerts
    assert recent_decisions
    assert recent_decisions[0]["id"] == decision_id
    assert latest_capital is not None
    assert latest_capital["id"] == capital_id
    assert capital_history
    assert recent_execution_results
    assert recent_execution_results[0]["id"] == execution_result_id
    assert recent_execution_results[0]["fill_ratio"] == 0.8
    assert recent_execution_fills
    assert recent_execution_fills[0]["execution_result_id"] == execution_result_id
    assert recent_audit
    assert recent_audit[0]["id"] == audit_id
    assert latest_report is not None
    assert latest_report["path"] == str(report_path.resolve())
    assert portfolios[0]["slug"] == "fx_eur_20k"


def test_app_storage_persists_mt5_market_cache(tmp_path: Path):
    root = tmp_path
    storage = AppStorage.from_root(root, {"storage": {"database_path": "data/app/test_mt5_cache.db"}})
    storage.initialize(create_schema=True)

    portfolio_id = storage.upsert_portfolio(
        name="FX_EUR_20k",
        base_currency="EUR",
        symbols=["EURUSD", "XAUUSD"],
        positions={"EURUSD": 10_000.0, "XAUUSD": 5_000.0},
        slug="fx_eur_20k",
    )

    sync_id = storage.record_market_data_sync(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        mode="live_mt5",
        status="running",
        details={"symbols": ["EURUSD", "XAUUSD"], "timeframes": ["H1"], "days": 60},
    )
    storage.upsert_instrument(
        {
            "symbol": "EURUSD",
            "asset_class": "fx",
            "contract_size": 100_000.0,
            "base_currency": "EUR",
            "quote_currency": "USD",
            "profit_currency": "USD",
            "margin_currency": "EUR",
            "tick_size": 0.0001,
            "tick_value": 10.0,
            "volume_min": 0.01,
            "volume_max": 50.0,
            "volume_step": 0.01,
            "trading_mode": "full_access",
            "raw": {"description": "Euro vs US Dollar"},
        }
    )
    storage.sync_market_bars(
        symbol="EURUSD",
        timeframe="H1",
        bars=[
            {"time": "2026-03-28T08:00:00+00:00", "open": 1.08, "high": 1.09, "low": 1.07, "close": 1.085, "tick_volume": 100.0},
            {"time": "2026-03-28T09:00:00+00:00", "open": 1.085, "high": 1.091, "low": 1.084, "close": 1.089, "tick_volume": 110.0},
        ],
        sync_run_id=sync_id,
    )
    storage.sync_mt5_order_history(
        [
            {
                "ticket": 901,
                "symbol": "EURUSD",
                "side": "BUY",
                "order_type": "market",
                "state": "filled",
                "volume_initial": 0.05,
                "volume_current": 0.0,
                "price_open": 1.089,
                "price_current": 1.089,
                "comment": "manual rebalance",
                "is_manual": True,
                "time_setup_utc": "2026-03-28T09:00:00+00:00",
                "time_done_utc": "2026-03-28T09:00:02+00:00",
            }
        ],
        sync_run_id=sync_id,
        portfolio_id=portfolio_id,
    )
    storage.sync_mt5_deal_history(
        [
            {
                "ticket": 801,
                "order_ticket": 901,
                "symbol": "EURUSD",
                "side": "BUY",
                "entry": "in",
                "volume": 0.05,
                "price": 1.089,
                "profit": 12.0,
                "commission": -0.5,
                "swap": 0.0,
                "fee": 0.0,
                "reason": "manual",
                "comment": "manual rebalance",
                "is_manual": True,
                "time_utc": "2026-03-28T09:00:02+00:00",
            }
        ],
        sync_run_id=sync_id,
        portfolio_id=portfolio_id,
    )
    storage.record_market_data_sync(
        portfolio_id=portfolio_id,
        portfolio_slug="fx_eur_20k",
        mode="live_mt5",
        status="ok",
        details={"coverage": {"EURUSD": {"H1": {"bars": 2}}}},
    )

    instruments = storage.list_instruments(symbols=["EURUSD"])
    latest_sync = storage.latest_market_data_sync(portfolio_slug="fx_eur_20k")
    bars = storage.market_bars(symbol="EURUSD", timeframe="H1")
    latest_bar_times = storage.latest_market_bar_times(symbols=["EURUSD"], timeframe="H1")
    orders = storage.recent_mt5_order_history(limit=5, portfolio_slug="fx_eur_20k")
    deals = storage.recent_mt5_deal_history(limit=5, portfolio_slug="fx_eur_20k")

    assert instruments[0]["symbol"] == "EURUSD"
    assert latest_sync is not None
    assert latest_sync["status"] == "ok"
    assert len(bars) == 2
    assert latest_bar_times["EURUSD"] == "2026-03-28T09:00:00+00:00"
    assert orders[0]["ticket"] == 901
    assert orders[0]["is_manual"] is True
    assert deals[0]["ticket"] == 801
