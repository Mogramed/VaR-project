from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Any

import httpx
import pandas as pd

from var_project.core.exceptions import MT5ConnectionError
from var_project.core.types import MT5Config


class _RemoteMT5Module:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_RETURN = 2


class RemoteMT5Connector:
    _RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
    _NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 409, 410, 422}

    def __init__(
        self,
        config: MT5Config,
        *,
        account_id: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.account_id = None if account_id in {None, "", "null"} else str(account_id).strip()
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
        ticket: int | None = None,
        position: int | None = None,
    ) -> list[dict[str, Any]]:
        start, end = self._normalized_datetime_range(date_from, date_to)
        params = {
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
        }
        if symbol is not None:
            params["symbol"] = str(symbol).upper()
        if ticket is not None:
            params["ticket"] = int(ticket)
        if position is not None:
            params["position"] = int(position)
        payload = self._request("GET", "/history/orders", params=params)
        return [dict(item) for item in payload]

    def history_deals_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> list[dict[str, Any]]:
        start, end = self._normalized_datetime_range(date_from, date_to)
        params = {
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
        }
        if symbol is not None:
            params["symbol"] = str(symbol).upper()
        if ticket is not None:
            params["ticket"] = int(ticket)
        if position is not None:
            params["position"] = int(position)
        payload = self._request("GET", "/history/deals", params=params)
        return [dict(item) for item in payload]

    def order_check(self, request: dict[str, Any]) -> dict[str, Any]:
        return dict(self._request("POST", "/order-check", json={"request": dict(request)}))

    def order_send(self, request: dict[str, Any]) -> dict[str, Any]:
        # Never retry order_send automatically: a network timeout may happen after the broker accepted
        # the order, and retrying could duplicate live execution.
        return dict(
            self._request(
                "POST",
                "/order-send",
                json={"request": dict(request)},
                allow_retry=False,
            )
        )

    def live_state(self) -> dict[str, Any]:
        return dict(self._request("GET", "/live/state"))

    def live_events(
        self,
        *,
        after: int = 0,
        limit: int = 100,
        wait_seconds: float = 15.0,
    ) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/live/events",
            params={
                "after": int(after),
                "limit": int(limit),
                "wait_seconds": float(wait_seconds),
            },
            timeout=max(float(wait_seconds) + 5.0, 10.0),
        )
        return [dict(item) for item in payload]

    def ensure_symbol(self, symbol: str) -> None:
        self.symbol_info(symbol)

    def bars_per_day(self, timeframe: str) -> int:
        tf = str(timeframe).upper().strip()
        minutes_map = {
            "M1": 1,
            "M2": 2,
            "M3": 3,
            "M4": 4,
            "M5": 5,
            "M10": 10,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H2": 120,
            "H4": 240,
            "D1": 1440,
        }
        if tf not in minutes_map:
            raise ValueError(f"Timeframe inconnu: {timeframe}")
        return int(1440 / minutes_map[tf])

    def fetch_last_n_bars(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int,
        chunk_size: int = 5000,
    ) -> pd.DataFrame:
        payload = self._request(
            "GET",
            f"/bars/{str(symbol).upper()}",
            params={
                "timeframe": str(timeframe).upper(),
                "n_bars": int(n_bars),
                "chunk_size": int(chunk_size),
            },
        )
        frame = pd.DataFrame(list(payload or []))
        if frame.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        return frame

    def fetch_bars_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        start, end = self._normalized_datetime_range(date_from, date_to)
        payload = self._request(
            "GET",
            f"/bars-range/{str(symbol).upper()}",
            params={
                "timeframe": str(timeframe).upper(),
                "date_from": start.isoformat(),
                "date_to": end.isoformat(),
            },
        )
        frame = pd.DataFrame(list(payload or []))
        if frame.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        return frame

    def fetch_ticks_range(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
        *,
        flags: int | None = None,
    ) -> pd.DataFrame:
        start, end = self._normalized_datetime_range(date_from, date_to)
        params = {
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
        }
        if flags is not None:
            params["flags"] = int(flags)
        payload = self._request("GET", f"/ticks/{str(symbol).upper()}", params=params, timeout=60.0)
        frame = pd.DataFrame(list(payload or []))
        if frame.empty:
            return pd.DataFrame(columns=["time_utc", "bid", "ask", "last", "volume", "time_msc", "flags"])
        if "time_utc" in frame.columns:
            frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
        return frame

    @staticmethod
    def _normalized_datetime_range(
        date_from: datetime,
        date_to: datetime,
        *,
        minimum_seconds: float = 1.0,
    ) -> tuple[datetime, datetime]:
        def _to_utc(value: datetime) -> datetime:
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        start = _to_utc(date_from)
        end = _to_utc(date_to)
        if end > start:
            return start, end
        if start == end:
            return start, start + timedelta(seconds=max(float(minimum_seconds), 0.001))
        return end, start

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
        allow_retry: bool = True,
    ) -> Any:
        if not self._initialized or self._client is None:
            raise MT5ConnectionError("Remote MT5 connector not initialized.")
        request_params = dict(params or {})
        if self.account_id not in {None, ""} and "account_id" not in request_params:
            request_params["account_id"] = str(self.account_id)
        effective_params = request_params or None
        configured_attempts = max(int(getattr(self.config, "reconnect_attempts", 2) or 2), 1)
        max_attempts = configured_attempts if allow_retry else 1
        backoff_seconds = max(float(getattr(self.config, "reconnect_backoff_seconds", 0.25) or 0.0), 0.0)
        for attempt in range(max_attempts):
            try:
                response = self._client.request(method, path, params=effective_params, json=json, timeout=timeout)
            except httpx.HTTPError as exc:
                can_retry = bool(allow_retry) and self._is_retryable_http_error(exc)
                if can_retry and attempt < (max_attempts - 1):
                    sleep_for = backoff_seconds * (2**attempt)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    continue
                raise MT5ConnectionError(f"MT5 agent request failed: {exc}") from exc

            if response.status_code >= 400:
                detail = self._error_detail_from_response(response)
                can_retry = bool(allow_retry) and self._is_retryable_http_status(response.status_code, detail)
                if can_retry and attempt < (max_attempts - 1):
                    sleep_for = backoff_seconds * (2**attempt)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    continue
                raise MT5ConnectionError(detail)

            try:
                return response.json()
            except ValueError as exc:
                raise MT5ConnectionError("MT5 agent returned a non-JSON response.") from exc

        raise MT5ConnectionError("MT5 agent request failed after retry budget exhausted.")

    @staticmethod
    def _is_retryable_http_error(exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
            return True
        message = str(exc).lower()
        retryable_markers = (
            "timeout",
            "temporar",
            "connection",
            "network",
            "broken pipe",
            "econnreset",
            "econnrefused",
            "unavailable",
            "busy",
        )
        return any(marker in message for marker in retryable_markers)

    @classmethod
    def _is_retryable_http_status(cls, status_code: int, detail: str) -> bool:
        if int(status_code) in cls._NON_RETRYABLE_HTTP_STATUS_CODES:
            return False
        if int(status_code) in cls._RETRYABLE_HTTP_STATUS_CODES:
            return True
        if int(status_code) >= 500:
            return True
        message = str(detail).lower()
        return any(
            marker in message
            for marker in (
                "timeout",
                "temporar",
                "unavailable",
                "busy",
                "rate limit",
            )
        )

    @staticmethod
    def _error_detail_from_response(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])
        return response.text.strip() or f"HTTP {response.status_code}"
