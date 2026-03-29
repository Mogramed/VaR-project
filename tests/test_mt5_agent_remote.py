from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from var_project.core.exceptions import MT5ConnectionError
from var_project.core.settings import get_mt5_config, load_settings
from var_project.execution.mt5_agent import create_mt5_agent_app
from var_project.execution.mt5_remote import RemoteMT5Connector

from test_mt5_execution_api import FakeMT5Connector, _write_settings


def test_mt5_agent_endpoints_with_api_key(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_settings(root)
    FakeMT5Connector.reset()
    monkeypatch.setenv("VAR_PROJECT_MT5_AGENT_API_KEY", "secret-key")

    client = TestClient(create_mt5_agent_app(repo_root=root, connector_factory=FakeMT5Connector))

    unauthorized = client.get("/terminal-info")
    assert unauthorized.status_code == 401

    headers = {"X-MT5-Agent-Key": "secret-key"}
    health = client.get("/health")
    assert health.status_code == 200

    terminal = client.get("/terminal-info", headers=headers)
    assert terminal.status_code == 200
    assert terminal.json()["company"] == "MetaQuotes"

    account = client.get("/account-info", headers=headers)
    assert account.status_code == 200
    assert account.json()["currency"] == "EUR"

    positions = client.get("/positions", headers=headers)
    assert positions.status_code == 200
    assert len(positions.json()) == 2

    live_state = client.get("/live/state", headers=headers)
    assert live_state.status_code == 200
    live_state_body = live_state.json()
    assert live_state_body["status"] in {"ok", "degraded"}
    assert live_state_body["holdings"]
    assert "EURUSD" in live_state_body["symbols"]

    live_events = client.get("/live/events", headers=headers, params={"after": 0, "limit": 10, "wait_seconds": 0.1})
    assert live_events.status_code == 200
    assert live_events.json()

    history_orders = client.get(
        "/history/orders",
        headers=headers,
        params={"date_from": "2026-03-28T00:00:00+00:00", "date_to": "2026-03-29T00:00:00+00:00"},
    )
    assert history_orders.status_code == 200
    assert history_orders.json()

    history_deals = client.get(
        "/history/deals",
        headers=headers,
        params={"date_from": "2026-03-28T00:00:00+00:00", "date_to": "2026-03-29T00:00:00+00:00"},
    )
    assert history_deals.status_code == 200
    assert history_deals.json()

    order_check = client.post(
        "/order-check",
        headers=headers,
        json={
            "request": {
                "symbol": "EURUSD",
                "volume": 0.05,
                "type": 0,
                "price": 1.09,
            }
        },
    )
    assert order_check.status_code == 200
    assert order_check.json()["retcode"] == 0


def test_remote_mt5_connector_uses_agent_contract(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_settings(root)
    FakeMT5Connector.reset()
    monkeypatch.setenv("VAR_PROJECT_MT5_AGENT_API_KEY", "secret-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-MT5-Agent-Key"] == "secret-key"
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/terminal-info":
            return httpx.Response(200, json={"company": "MetaQuotes", "trade_allowed": True, "tradeapi_disabled": False})
        if path == "/account-info":
            return httpx.Response(200, json={"currency": "EUR", "equity": 100_000.0, "margin_free": 90_000.0})
        if path == "/symbol-info/EURUSD":
            return httpx.Response(
                200,
                json={
                    "currency_base": "EUR",
                    "trade_contract_size": 100_000.0,
                    "volume_min": 0.01,
                    "volume_max": 50.0,
                    "volume_step": 0.01,
                    "filling_mode": 2,
                },
            )
        if path == "/symbol-tick/EURUSD":
            return httpx.Response(200, json={"bid": 1.0898, "ask": 1.09, "time": 1_711_620_000})
        if path == "/positions":
            return httpx.Response(200, json=[])
        if path == "/orders":
            return httpx.Response(200, json=[])
        if path == "/live/state":
            return httpx.Response(
                200,
                json={
                    "sequence": 3,
                    "status": "ok",
                    "connected": True,
                    "degraded": False,
                    "stale": False,
                    "generated_at": "2026-03-29T09:00:00+00:00",
                    "last_success_at": "2026-03-29T09:00:00+00:00",
                    "last_error": None,
                    "poll_interval_seconds": 2.0,
                    "history_poll_interval_seconds": 30.0,
                    "history_lookback_minutes": 180,
                    "symbols": ["EURUSD"],
                    "terminal_status": {"connected": True, "ready": True, "execution_enabled": True, "message": "ok", "raw": {}},
                    "account": {"currency": "EUR", "equity": 100_000.0, "balance": 100_000.0, "profit": 0.0, "margin": 0.0, "margin_free": 100_000.0, "raw": {}},
                    "ticks": {"EURUSD": {"symbol": "EURUSD", "bid": 1.0898, "ask": 1.09, "last": 1.0899, "time_utc": "2026-03-29T09:00:00+00:00", "raw": {}}},
                    "holdings": [],
                    "pending_orders": [],
                    "order_history": [],
                    "deal_history": [],
                },
            )
        if path == "/live/events":
            return httpx.Response(
                200,
                json=[
                    {
                        "sequence": 3,
                        "kind": "snapshot",
                        "timestamp_utc": "2026-03-29T09:00:00+00:00",
                        "change_summary": {"symbols_added": ["EURUSD"]},
                        "state": {
                            "sequence": 3,
                            "status": "ok",
                            "connected": True,
                            "degraded": False,
                            "stale": False,
                            "generated_at": "2026-03-29T09:00:00+00:00",
                            "last_success_at": "2026-03-29T09:00:00+00:00",
                            "last_error": None,
                            "poll_interval_seconds": 2.0,
                            "history_poll_interval_seconds": 30.0,
                            "history_lookback_minutes": 180,
                            "symbols": ["EURUSD"],
                            "terminal_status": {"connected": True, "ready": True, "execution_enabled": True, "message": "ok", "raw": {}},
                            "account": {"currency": "EUR", "equity": 100_000.0, "balance": 100_000.0, "profit": 0.0, "margin": 0.0, "margin_free": 100_000.0, "raw": {}},
                            "ticks": {"EURUSD": {"symbol": "EURUSD", "bid": 1.0898, "ask": 1.09, "last": 1.0899, "time_utc": "2026-03-29T09:00:00+00:00", "raw": {}}},
                            "holdings": [],
                            "pending_orders": [],
                            "order_history": [],
                            "deal_history": [],
                        },
                    }
                ],
            )
        if path == "/history/orders":
            return httpx.Response(
                200,
                json=[
                    {
                        "ticket": 901,
                        "symbol": "EURUSD",
                        "type": 0,
                        "state": 4,
                        "volume_initial": 0.05,
                        "volume_current": 0.0,
                        "price_open": 1.0890,
                        "price_current": 1.0890,
                        "comment": "manual rebalance",
                        "time_setup": 1_711_620_000,
                        "time_done": 1_711_620_060,
                    }
                ],
            )
        if path == "/history/deals":
            return httpx.Response(
                200,
                json=[
                    {
                        "ticket": 801,
                        "order": 901,
                        "symbol": "EURUSD",
                        "type": 0,
                        "entry": 1,
                        "volume": 0.05,
                        "price": 1.0890,
                        "profit": 12.0,
                        "commission": -0.5,
                        "swap": 0.0,
                        "fee": 0.0,
                        "reason": 0,
                        "comment": "manual rebalance",
                        "time": 1_711_620_060,
                    }
                ],
            )
        if path == "/order-check":
            return httpx.Response(200, json={"retcode": 0, "comment": "Done"})
        if path == "/order-send":
            return httpx.Response(200, json={"retcode": 10009, "comment": "Request completed"})
        return httpx.Response(404, json={"detail": f"Unhandled {path}"})

    transport = httpx.MockTransport(handler)
    config = replace(
        get_mt5_config(load_settings(root)),
        agent_base_url="http://mt5-agent.local",
        agent_api_key="secret-key",
    )

    connector = RemoteMT5Connector(config, transport=transport)
    connector.init()
    try:
        assert connector.account_info()["currency"] == "EUR"
        assert connector.symbol_info("EURUSD")["currency_base"] == "EUR"
        assert connector.live_state()["status"] == "ok"
        assert connector.live_events(after=0, limit=5, wait_seconds=0.1)[0]["kind"] == "snapshot"
        assert connector.history_orders_get(datetime.fromisoformat("2026-03-28T00:00:00+00:00"), datetime.fromisoformat("2026-03-29T00:00:00+00:00"))[0]["ticket"] == 901
        assert connector.history_deals_get(datetime.fromisoformat("2026-03-28T00:00:00+00:00"), datetime.fromisoformat("2026-03-29T00:00:00+00:00"))[0]["ticket"] == 801
        assert connector.order_check({"symbol": "EURUSD", "volume": 0.05})["retcode"] == 0
        assert connector.order_send({"symbol": "EURUSD", "volume": 0.05})["retcode"] == 10009
    finally:
        connector.shutdown()


class FlakyTerminalConnector(FakeMT5Connector):
    init_calls: int = 0
    shutdown_calls: int = 0
    fail_terminal_once: bool = False

    @classmethod
    def reset(cls) -> None:
        super().reset()
        cls.init_calls = 0
        cls.shutdown_calls = 0
        cls.fail_terminal_once = False

    def init(self) -> None:
        type(self).init_calls += 1
        super().init()

    def shutdown(self) -> None:
        type(self).shutdown_calls += 1
        super().shutdown()

    def terminal_info(self) -> dict[str, object]:
        if type(self).fail_terminal_once:
            type(self).fail_terminal_once = False
            raise MT5ConnectionError("terminal_info() a echoue: (-10004, 'No IPC connection')")
        return super().terminal_info()


def test_mt5_agent_reuses_session_and_recovers_from_ipc_error(tmp_path: Path) -> None:
    root = tmp_path
    _write_settings(root)
    FlakyTerminalConnector.reset()
    FlakyTerminalConnector.fail_terminal_once = True

    with TestClient(create_mt5_agent_app(repo_root=root, connector_factory=FlakyTerminalConnector)) as client:
        account = client.get("/account-info")
        assert account.status_code == 200

        positions = client.get("/positions")
        assert positions.status_code == 200

        terminal = client.get("/terminal-info")
        assert terminal.status_code == 200
        assert terminal.json()["company"] == "MetaQuotes"

    assert FlakyTerminalConnector.init_calls == 2
    assert FlakyTerminalConnector.shutdown_calls == 2
