from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from var_project.api import create_app
from var_project.core.exceptions import MT5ConnectionError


def _write_settings(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "base_currency": "EUR",
        "symbols": ["EURUSD", "USDJPY"],
        "portfolio": {
            "name": "FX_EUR_20k",
            "positions_eur": {"EURUSD": 10_000, "USDJPY": 10_000},
        },
        "data": {
            "timeframes": ["H1"],
            "history_days_list": [60],
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
            "garch": {"enabled": True, "p": 1, "q": 1, "dist": "t", "mean": "constant"},
        },
        "mt5": {
            "login": None,
            "password": None,
            "server": None,
            "execution_enabled": True,
            "magic": 420001,
            "deviation_points": 20,
            "comment_prefix": "var_risk_desk",
        },
        "storage": {
            "database_path": "data/app/test_mt5_api.db",
            "analytics_dir": "reports/backtests",
            "reports_dir": "reports/daily",
            "snapshots_dir": "data/snapshots",
        },
        "desk": {"name": "FX Risk Desk"},
    }
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


class FakeMT5Connector:
    class _MT5Module:
        TRADE_ACTION_DEAL = 1
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        ORDER_TIME_GTC = 0
        ORDER_FILLING_FOK = 0
        ORDER_FILLING_IOC = 1
        ORDER_FILLING_RETURN = 2

    positions_lots: dict[str, float] = {"EURUSD": 0.10, "USDJPY": 0.11}
    next_ticket: int = 1000
    next_deal: int = 5000
    force_order_check_reject: bool = False
    terminal_trade_allowed: bool = True
    terminal_tradeapi_disabled: bool = False
    order_history: list[dict[str, object]] = []
    deal_history: list[dict[str, object]] = []

    def __init__(self, config):
        self.config = config
        self._mt5 = self._MT5Module()
        self._initialized = False

    @classmethod
    def reset(cls) -> None:
        cls.positions_lots = {"EURUSD": 0.10, "USDJPY": 0.11}
        cls.next_ticket = 1000
        cls.next_deal = 5000
        cls.force_order_check_reject = False
        cls.terminal_trade_allowed = True
        cls.terminal_tradeapi_disabled = False
        timestamp = int(datetime(2026, 3, 28, 9, 0, tzinfo=timezone.utc).timestamp())
        cls.order_history = [
            {
                "ticket": 901,
                "position_id": 0,
                "symbol": "EURUSD",
                "type": 0,
                "state": 4,
                "volume_initial": 0.05,
                "volume_current": 0.00,
                "price_open": 1.0875,
                "price_current": 1.0890,
                "comment": "manual rebalance",
                "time_setup": timestamp,
                "time_done": timestamp + 60,
            }
        ]
        cls.deal_history = [
            {
                "ticket": 801,
                "order": 901,
                "position_id": 0,
                "symbol": "EURUSD",
                "type": 0,
                "entry": 1,
                "volume": 0.05,
                "price": 1.0880,
                "profit": 12.5,
                "commission": -0.5,
                "swap": 0.0,
                "fee": 0.0,
                "reason": 0,
                "comment": "manual rebalance",
                "time": timestamp + 60,
            }
        ]

    def init(self) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise MT5ConnectionError("Fake MT5 connector not initialized.")

    def ensure_symbol(self, symbol: str) -> None:
        if symbol.upper() not in {"EURUSD", "USDJPY"}:
            raise MT5ConnectionError(f"Unknown symbol {symbol}.")

    def terminal_info(self) -> dict[str, object]:
        self._ensure_initialized()
        return {
            "company": "MetaQuotes",
            "path": "C:/Program Files/MetaTrader 5/terminal64.exe",
            "data_path": "C:/Users/Test/AppData/Roaming/MetaQuotes/Terminal",
            "commondata_path": "C:/Users/Public/MetaQuotes/Terminal/Common",
            "trade_allowed": self.terminal_trade_allowed,
            "tradeapi_disabled": self.terminal_tradeapi_disabled,
        }

    def account_info(self) -> dict[str, object]:
        self._ensure_initialized()
        margin = self._current_margin()
        return {
            "login": 123456,
            "name": "Demo Trader",
            "server": "MetaQuotes-Demo",
            "currency": "EUR",
            "leverage": 100,
            "balance": 100_000.0,
            "equity": 100_250.0,
            "profit": 250.0,
            "margin": margin,
            "margin_free": 100_000.0 - margin,
            "margin_level": 500.0,
            "trade_allowed": True,
        }

    def symbol_info(self, symbol: str) -> dict[str, object]:
        self._ensure_initialized()
        self.ensure_symbol(symbol)
        if symbol.upper() == "EURUSD":
            return {
                "currency_base": "EUR",
                "trade_contract_size": 100_000.0,
                "volume_min": 0.01,
                "volume_max": 50.0,
                "volume_step": 0.01,
                "filling_mode": 2,
            }
        return {
            "currency_base": "USD",
            "trade_contract_size": 100_000.0,
            "volume_min": 0.01,
            "volume_max": 50.0,
            "volume_step": 0.01,
            "filling_mode": 2,
        }

    def symbol_info_tick(self, symbol: str) -> dict[str, object]:
        self._ensure_initialized()
        self.ensure_symbol(symbol)
        if symbol.upper() == "EURUSD":
            return {"bid": 1.0898, "ask": 1.0900, "time": 1_711_620_000}
        return {"bid": 156.18, "ask": 156.22, "time": 1_711_620_000}

    def bars_per_day(self, timeframe: str) -> int:
        if timeframe.upper() != "H1":
            raise ValueError(f"Unsupported timeframe {timeframe}")
        return 24

    def fetch_last_n_bars(self, symbol: str, timeframe: str, n_bars: int, chunk_size: int = 5000) -> pd.DataFrame:
        self._ensure_initialized()
        self.ensure_symbol(symbol)
        count = max(int(n_bars), 24 * 40)
        times = pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC")
        x = np.arange(count)
        if symbol.upper() == "EURUSD":
            close = 1.08 + 0.0004 * np.sin(x / 7.0) + x * 0.00001
        else:
            close = 2400.0 + 2.5 * np.cos(x / 9.0) + x * 0.03
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + 0.0002 if symbol.upper() == "EURUSD" else np.maximum(open_, close) + 0.8
        low = np.minimum(open_, close) - 0.0002 if symbol.upper() == "EURUSD" else np.minimum(open_, close) - 0.8
        return pd.DataFrame(
            {
                "time": times,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "tick_volume": np.full(count, 120.0),
                "spread": np.full(count, 2.0),
                "real_volume": np.full(count, 0.0),
            }
        )

    def positions_get(self, symbol: str | None = None) -> list[dict[str, object]]:
        self._ensure_initialized()
        rows: list[dict[str, object]] = []
        for item_symbol, signed_lots in self.positions_lots.items():
            if symbol and item_symbol != symbol.upper():
                continue
            if abs(signed_lots) <= 1e-9:
                continue
            tick = self.symbol_info_tick(item_symbol)
            rows.append(
                {
                    "ticket": 10_000 + len(rows),
                    "symbol": item_symbol,
                    "type": 0 if signed_lots > 0 else 1,
                    "volume": abs(signed_lots),
                    "price_open": tick["ask"],
                    "price_current": tick["bid"] if signed_lots > 0 else tick["ask"],
                    "profit": 25.0,
                    "swap": 0.0,
                    "comment": "demo",
                    "time": 1_711_620_000,
                }
            )
        return rows

    def orders_get(self, symbol: str | None = None) -> list[dict[str, object]]:
        self._ensure_initialized()
        return []

    def history_orders_get(self, date_from: datetime, date_to: datetime, *, symbol: str | None = None) -> list[dict[str, object]]:
        self._ensure_initialized()
        normalized = None if symbol is None else symbol.upper()
        return [
            dict(item)
            for item in type(self).order_history
            if normalized is None or str(item.get("symbol")).upper() == normalized
        ]

    def history_deals_get(self, date_from: datetime, date_to: datetime, *, symbol: str | None = None) -> list[dict[str, object]]:
        self._ensure_initialized()
        normalized = None if symbol is None else symbol.upper()
        return [
            dict(item)
            for item in type(self).deal_history
            if normalized is None or str(item.get("symbol")).upper() == normalized
        ]

    def order_check(self, request: dict[str, object]) -> dict[str, object]:
        self._ensure_initialized()
        volume = float(request["volume"])
        margin = round(volume * 1_500.0, 2)
        margin_free = 100_000.0 - self._current_margin() - margin
        if self.force_order_check_reject or margin_free < 0.0:
            return {"retcode": 10019, "comment": "Insufficient margin", "margin": margin, "margin_free": margin_free}
        return {"retcode": 0, "comment": "Done", "margin": margin, "margin_free": margin_free}

    def order_send(self, request: dict[str, object]) -> dict[str, object]:
        self._ensure_initialized()
        symbol = str(request["symbol"]).upper()
        signed_lots = float(request["volume"]) if int(request["type"]) == 0 else -float(request["volume"])
        self.positions_lots[symbol] = float(self.positions_lots.get(symbol, 0.0) + signed_lots)
        self.next_ticket += 1
        self.next_deal += 1
        timestamp = int(datetime(2026, 3, 29, 9, 0, tzinfo=timezone.utc).timestamp())
        type(self).order_history.append(
            {
                "ticket": self.next_ticket,
                "position_id": 10_000 + len(type(self).order_history),
                "symbol": symbol,
                "type": request["type"],
                "state": 4,
                "volume_initial": request["volume"],
                "volume_current": 0.0,
                "price_open": request["price"],
                "price_current": request["price"],
                "comment": request.get("comment"),
                "time_setup": timestamp,
                "time_done": timestamp + 2,
            }
        )
        type(self).deal_history.append(
            {
                "ticket": self.next_deal,
                "order": self.next_ticket,
                "position_id": 10_000 + len(type(self).deal_history),
                "symbol": symbol,
                "type": request["type"],
                "entry": 1,
                "volume": request["volume"],
                "price": request["price"],
                "profit": 0.0,
                "commission": 0.0,
                "swap": 0.0,
                "fee": 0.0,
                "reason": 0,
                "comment": request.get("comment"),
                "time": timestamp + 2,
            }
        )
        return {
            "retcode": 10009,
            "comment": "Request completed",
            "order": self.next_ticket,
            "deal": self.next_deal,
            "price": request["price"],
            "volume": request["volume"],
        }

    @classmethod
    def _current_margin(cls) -> float:
        return round(sum(abs(volume) for volume in cls.positions_lots.values()) * 1_000.0, 2)


def test_mt5_execution_preview_and_submit_flow(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FakeMT5Connector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    status = client.get("/mt5/status")
    assert status.status_code == 200
    assert status.json()["connected"] is True
    assert status.json()["execution_enabled"] is True

    account = client.get("/mt5/account")
    assert account.status_code == 200
    assert account.json()["currency"] == "EUR"

    positions = client.get("/mt5/positions")
    assert positions.status_code == 200
    assert {item["symbol"] for item in positions.json()} == {"EURUSD", "USDJPY"}

    orders = client.get("/mt5/orders")
    assert orders.status_code == 200
    assert orders.json() == []

    live_state = client.get("/mt5/live/state")
    assert live_state.status_code == 200
    live_state_body = live_state.json()
    assert live_state_body["connected"] is True
    assert live_state_body["holdings"]
    assert live_state_body["reconciliation"]["mismatches"]
    assert live_state_body["risk_summary"]["reference_model"] in {"hist", "param", "mc", "ewma", "garch", "fhs"}
    assert live_state_body["capital_usage"]["snapshot_source"] == "mt5_live_bridge"
    assert live_state_body["operator_alerts"]
    assert {item["code"] for item in live_state_body["operator_alerts"]} & {
        "MT5_MANUAL_EVENTS",
        "DESK_BROKER_DRIFT",
    }
    persisted_live_snapshot = client.get("/snapshots/latest", params={"source": "mt5_live_bridge"})
    assert persisted_live_snapshot.status_code == 200
    assert persisted_live_snapshot.json()["source"] == "mt5_live_bridge"
    assert persisted_live_snapshot.json()["payload"]["metadata"]["live_sequence"] == live_state_body["sequence"]

    live_events = client.get("/mt5/live/events", params={"after": 0, "limit": 5, "wait_seconds": 0.1})
    assert live_events.status_code == 200
    assert live_events.json()

    preview = client.post(
        "/execution/preview",
        json={"symbol": "EURUSD", "exposure_change": 1_000.0, "note": "demo preview"},
    )
    assert preview.status_code == 200
    preview_body = preview.json()
    assert preview_body["guard"]["margin_ok"] is True
    assert preview_body["guard"]["submit_allowed"] is True
    assert preview_body["guard"]["volume_lots"] > 0.0

    submit = client.post(
        "/execution/submit",
        json={"symbol": "EURUSD", "exposure_change": 1_000.0, "note": "demo execution"},
    )
    assert submit.status_code == 200
    submit_body = submit.json()
    assert submit_body["status"] in {"EXECUTED", "PLACED"}
    assert submit_body["executed_exposure_change"] > 0.0
    assert submit_body["mt5_result"]["retcode"] == 10009
    assert submit_body["fill_ratio"] is not None
    assert submit_body["broker_status"] in {"filled", "placed"}
    assert submit_body["reconciliation_status"] in {"match", "partial_fill", "pending_broker"}
    assert submit_body["fills"]

    recent = client.get("/execution/recent", params={"limit": 5})
    assert recent.status_code == 200
    assert recent.json()

    recent_fills = client.get("/execution/fills/recent", params={"limit": 5})
    assert recent_fills.status_code == 200
    assert recent_fills.json()

    positions_after = client.get("/mt5/positions")
    assert positions_after.status_code == 200
    eurusd_after = next(item for item in positions_after.json() if item["symbol"] == "EURUSD")
    assert eurusd_after["signed_exposure_base_ccy"] > preview_body["live_positions"][0]["signed_exposure_base_ccy"]

    latest_live_snapshot = client.get("/snapshots/latest", params={"source": "mt5_live"})
    assert latest_live_snapshot.status_code == 200
    assert latest_live_snapshot.json()["source"] == "mt5_live"

    latest_capital = client.get("/capital/latest")
    assert latest_capital.status_code == 200
    assert latest_capital.json()["portfolio_slug"] == "fx_eur_20k"


def test_mt5_execution_guard_blocks_on_margin_reject(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FakeMT5Connector.reset()
    FakeMT5Connector.force_order_check_reject = True

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FakeMT5Connector, bootstrap_storage=True))

    preview = client.post(
        "/execution/preview",
        json={"symbol": "EURUSD", "exposure_change": 1_000.0, "note": "blocked"},
    )
    assert preview.status_code == 200
    preview_body = preview.json()
    assert preview_body["guard"]["decision"] == "REJECT"
    assert preview_body["guard"]["margin_ok"] is False

    submit = client.post(
        "/execution/submit",
        json={"symbol": "EURUSD", "exposure_change": 1_000.0, "note": "blocked"},
    )
    assert submit.status_code == 200
    submit_body = submit.json()
    assert submit_body["status"] in {"BLOCKED", "REJECTED"}
    assert submit_body["executed_exposure_change"] == 0.0


class FillingModeFallbackConnector(FakeMT5Connector):
    def symbol_info(self, symbol: str) -> dict[str, object]:
        payload = dict(super().symbol_info(symbol))
        payload["filling_mode"] = 1
        return payload

    def order_check(self, request: dict[str, object]) -> dict[str, object]:
        self._ensure_initialized()
        if int(request.get("type_filling", -1)) != self._mt5.ORDER_FILLING_FOK:
            return {"retcode": 10030, "comment": "Unsupported filling mode"}
        return super().order_check(request)


def test_mt5_execution_preview_recovers_with_fill_mode_fallback(tmp_path: Path):
    root = tmp_path
    _write_settings(root)
    _write_processed_returns(root, "EURUSD")
    _write_processed_returns(root, "USDJPY")
    FillingModeFallbackConnector.reset()

    client = TestClient(create_app(repo_root=root, mt5_connector_factory=FillingModeFallbackConnector, bootstrap_storage=True))

    preview = client.post(
        "/execution/preview",
        json={"symbol": "EURUSD", "exposure_change": -5_001.0, "note": "fill fallback"},
    )
    assert preview.status_code == 200
    payload = preview.json()
    assert payload["guard"]["decision"] in {"ACCEPT", "REDUCE"}
    assert payload["guard"]["submit_allowed"] is True
    assert payload["order_request"]["type_filling"] == 0
    assert payload["order_check"]["retcode"] == 0
