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
from var_project.execution.mt5_bridge import MT5EventBridge


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
    live_bridge = MT5EventBridge(
        runtime=runtime,
        config=mt5_config,
        base_currency=str(raw_config.get("base_currency", "EUR")),
        seed_symbols=raw_config.get("symbols") or [],
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            live_bridge.start()
            live_bridge.prime()
            yield
        finally:
            live_bridge.stop()
            runtime.close()

    app = FastAPI(
        title="VaR Risk Desk MT5 Agent",
        version="0.1.0",
        description="Windows-side MetaTrader 5 bridge for guarded demo execution.",
        lifespan=lifespan,
    )
    app.state.mt5_config = mt5_config
    app.state.mt5_runtime = runtime
    app.state.mt5_live_bridge = live_bridge

    def authorize(x_mt5_agent_key: str | None = Header(default=None)) -> None:
        expected = app.state.mt5_config.agent_api_key
        if expected and x_mt5_agent_key != expected:
            raise HTTPException(status_code=401, detail="Invalid MT5 agent key.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        live_state = app.state.mt5_live_bridge.current_state()
        return {
            "status": "ok",
            "execution_enabled": bool(app.state.mt5_config.execution_enabled),
            "agent_mode": "windows_mt5_bridge",
            "live_bridge_enabled": bool(app.state.mt5_config.live_enabled),
            "live_bridge_status": live_state.get("status"),
            "live_sequence": live_state.get("sequence"),
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

    @app.get("/bars/{symbol}")
    def bars(
        symbol: str,
        timeframe: str = Query(...),
        n_bars: int = Query(..., ge=1, le=500000),
        chunk_size: int = Query(default=5000, ge=1, le=500000),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            frame = app.state.mt5_runtime.execute(
                lambda connector: connector.fetch_last_n_bars(
                    symbol,
                    timeframe,
                    n_bars,
                    chunk_size=chunk_size,
                )
            )
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return frame.to_dict(orient="records")

    @app.get("/ticks/{symbol}")
    def ticks(
        symbol: str,
        date_from: str = Query(...),
        date_to: str = Query(...),
        flags: int | None = Query(default=None),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            start = datetime.fromisoformat(date_from)
            end = datetime.fromisoformat(date_to)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid tick date range.") from exc
        try:
            frame = app.state.mt5_runtime.execute(
                lambda connector: connector.fetch_ticks_range(
                    symbol,
                    start,
                    end,
                    flags=flags,
                )
            )
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return frame.to_dict(orient="records")

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
        ticket: int | None = Query(default=None),
        position: int | None = Query(default=None),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            start = datetime.fromisoformat(date_from)
            end = datetime.fromisoformat(date_to)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date range.") from exc
        try:
            return app.state.mt5_runtime.execute(
                lambda connector: connector.history_orders_get(
                    start,
                    end,
                    symbol=symbol,
                    ticket=ticket,
                    position=position,
                )
            )
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/history/deals")
    def history_deals(
        date_from: str = Query(...),
        date_to: str = Query(...),
        symbol: str | None = Query(default=None),
        ticket: int | None = Query(default=None),
        position: int | None = Query(default=None),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        try:
            start = datetime.fromisoformat(date_from)
            end = datetime.fromisoformat(date_to)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date range.") from exc
        try:
            return app.state.mt5_runtime.execute(
                lambda connector: connector.history_deals_get(
                    start,
                    end,
                    symbol=symbol,
                    ticket=ticket,
                    position=position,
                )
            )
        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/live/state")
    def live_state(_: None = Depends(authorize)) -> dict[str, Any]:
        return app.state.mt5_live_bridge.current_state()

    @app.get("/live/events")
    def live_events(
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        wait_seconds: float = Query(default=15.0, ge=0.0, le=60.0),
        _: None = Depends(authorize),
    ) -> list[dict[str, Any]]:
        return app.state.mt5_live_bridge.events_after(after, limit=limit, wait_seconds=wait_seconds)

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
