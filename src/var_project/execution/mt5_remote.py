from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from var_project.core.exceptions import MT5ConnectionError
from var_project.core.types import MT5Config


class _RemoteMT5Module:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_RETURN = 2


class RemoteMT5Connector:
    def __init__(
        self,
        config: MT5Config,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self._client: httpx.Client | None = None
        self._initialized = False
        self._mt5 = _RemoteMT5Module()

    def init(self) -> None:
        if not self.config.agent_base_url:
            raise MT5ConnectionError("VAR_PROJECT_MT5_AGENT_BASE_URL is not configured.")

        headers: dict[str, str] = {}
        if self.config.agent_api_key:
            headers["X-MT5-Agent-Key"] = self.config.agent_api_key

        timeout = 10.0 if self.config.timeout_ms is None else max(float(self.config.timeout_ms) / 1000.0, 1.0)
        self._client = httpx.Client(
            base_url=str(self.config.agent_base_url),
            timeout=timeout,
            headers=headers,
            transport=self.transport,
        )
        self._initialized = True
        try:
            self._request("GET", "/health")
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._initialized = False

    def terminal_info(self) -> dict[str, Any]:
        return dict(self._request("GET", "/terminal-info"))

    def account_info(self) -> dict[str, Any]:
        return dict(self._request("GET", "/account-info"))

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/symbol-info/{symbol.upper()}"))

    def symbol_info_tick(self, symbol: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/symbol-tick/{symbol.upper()}"))

    def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {} if symbol is None else {"symbol": str(symbol).upper()}
        payload = self._request("GET", "/positions", params=params)
        return [dict(item) for item in payload]

    def orders_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {} if symbol is None else {"symbol": str(symbol).upper()}
        payload = self._request("GET", "/orders", params=params)
        return [dict(item) for item in payload]

    def history_orders_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        }
        if symbol is not None:
            params["symbol"] = str(symbol).upper()
        payload = self._request("GET", "/history/orders", params=params)
        return [dict(item) for item in payload]

    def history_deals_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        }
        if symbol is not None:
            params["symbol"] = str(symbol).upper()
        payload = self._request("GET", "/history/deals", params=params)
        return [dict(item) for item in payload]

    def order_check(self, request: dict[str, Any]) -> dict[str, Any]:
        return dict(self._request("POST", "/order-check", json={"request": dict(request)}))

    def order_send(self, request: dict[str, Any]) -> dict[str, Any]:
        return dict(self._request("POST", "/order-send", json={"request": dict(request)}))

    def ensure_symbol(self, symbol: str) -> None:
        self.symbol_info(symbol)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        if not self._initialized or self._client is None:
            raise MT5ConnectionError("Remote MT5 connector not initialized.")
        try:
            response = self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:
            raise MT5ConnectionError(f"MT5 agent request failed: {exc}") from exc

        if response.status_code >= 400:
            detail: str
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if isinstance(payload, dict) and payload.get("detail"):
                detail = str(payload["detail"])
            else:
                detail = response.text.strip() or f"HTTP {response.status_code}"
            raise MT5ConnectionError(detail)

        try:
            return response.json()
        except ValueError as exc:
            raise MT5ConnectionError("MT5 agent returned a non-JSON response.") from exc
