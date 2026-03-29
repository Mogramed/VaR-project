from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from var_project.connectors.mt5_connector import MT5Connector
from var_project.core.exceptions import MT5ConnectionError
from var_project.core.settings import find_repo_root, get_mt5_config, load_settings


class MT5AgentRequest(BaseModel):
    request: dict[str, Any] = Field(default_factory=dict)


class MT5AgentRuntime:
    def __init__(self, *, config, connector_factory) -> None:
        self.config = config
        self.connector_factory = connector_factory
        self._lock = RLock()
        self._connector = None

    def close(self) -> None:
        with self._lock:
            self._reset_locked()

    def execute(self, operation):
        with self._lock:
            connector = self._ensure_connector_locked()
            try:
                return operation(connector)
            except MT5ConnectionError:
                self._reset_locked()
                connector = self._ensure_connector_locked()
                return operation(connector)

    def _ensure_connector_locked(self):
        if self._connector is None:
            connector = self.connector_factory(self.config)
            connector.init()
            self._connector = connector
        return self._connector

    def _reset_locked(self) -> None:
        if self._connector is not None:
            self._connector.shutdown()
        self._connector = None


def create_mt5_agent_app(repo_root: Path | None = None, connector_factory=None) -> FastAPI:
    root = (repo_root or find_repo_root()).resolve()
    raw_config = load_settings(root)
    mt5_config = get_mt5_config(raw_config)
    runtime = MT5AgentRuntime(config=mt5_config, connector_factory=connector_factory or MT5Connector)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="VaR Risk Desk MT5 Agent",
        version="0.1.0",
        description="Windows-side MetaTrader 5 bridge for guarded demo execution.",
        lifespan=lifespan,
    )
    app.state.mt5_config = mt5_config
    app.state.mt5_runtime = runtime

    def authorize(x_mt5_agent_key: str | None = Header(default=None)) -> None:
        expected = app.state.mt5_config.agent_api_key
        if expected and x_mt5_agent_key != expected:
            raise HTTPException(status_code=401, detail="Invalid MT5 agent key.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "execution_enabled": bool(app.state.mt5_config.execution_enabled),
            "agent_mode": "windows_mt5_bridge",
        }

    @app.get("/terminal-info")
    def terminal_info(_: None = Depends(authorize)) -> dict[str, Any]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.terminal_info())
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/account-info")
    def account_info(_: None = Depends(authorize)) -> dict[str, Any]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.account_info())
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/symbol-info/{symbol}")
    def symbol_info(symbol: str, _: None = Depends(authorize)) -> dict[str, Any]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.symbol_info(symbol))
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/symbol-tick/{symbol}")
    def symbol_tick(symbol: str, _: None = Depends(authorize)) -> dict[str, Any]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.symbol_info_tick(symbol))
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/positions")
    def positions(
        symbol: str | None = Query(default=None),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.positions_get(symbol=symbol))
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/orders")
    def orders(
        symbol: str | None = Query(default=None),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.orders_get(symbol=symbol))
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/history/orders")
    def history_orders(
        date_from: str = Query(...),
        date_to: str = Query(...),
        symbol: str | None = Query(default=None),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            start = datetime.fromisoformat(date_from)
            end = datetime.fromisoformat(date_to)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date range.") from exc
        try:
            return app.state.mt5_runtime.execute(
                lambda connector: connector.history_orders_get(start, end, symbol=symbol)
            )
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/history/deals")
    def history_deals(
        date_from: str = Query(...),
        date_to: str = Query(...),
        symbol: str | None = Query(default=None),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            start = datetime.fromisoformat(date_from)
            end = datetime.fromisoformat(date_to)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date range.") from exc
        try:
            return app.state.mt5_runtime.execute(
                lambda connector: connector.history_deals_get(start, end, symbol=symbol)
            )
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/order-check")
    def order_check(payload: MT5AgentRequest, _: None = Depends(authorize)) -> dict[str, Any]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.order_check(payload.request))
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/order-send")
    def order_send(payload: MT5AgentRequest, _: None = Depends(authorize)) -> dict[str, Any]:
        try:
            return app.state.mt5_runtime.execute(lambda connector: connector.order_send(payload.request))
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app
