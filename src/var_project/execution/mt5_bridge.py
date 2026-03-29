from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from threading import Condition, Event, Thread
from typing import Any, Iterable, Mapping

from var_project.core.exceptions import MT5ConnectionError
from var_project.core.types import MT5Config
from var_project.execution.mt5_live import MT5LiveGateway


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _normalize_symbol_list(symbols: Iterable[str] | None) -> list[str]:
    return sorted({str(symbol).upper() for symbol in symbols or [] if str(symbol).strip()})


def build_empty_live_state(
    *,
    config: MT5Config,
    seed_symbols: Iterable[str] | None = None,
    status: str = "starting",
    connected: bool = False,
    degraded: bool = False,
    stale: bool = False,
    last_error: str | None = None,
    last_success_at: str | None = None,
) -> dict[str, Any]:
    return {
        "sequence": 0,
        "source": "mt5_agent_bridge",
        "status": status,
        "connected": connected,
        "degraded": degraded,
        "stale": stale,
        "generated_at": _utcnow_iso(),
        "last_success_at": last_success_at,
        "last_error": last_error,
        "poll_interval_seconds": float(config.live_poll_seconds),
        "history_poll_interval_seconds": float(config.live_history_poll_seconds),
        "history_lookback_minutes": int(config.live_history_lookback_minutes),
        "symbols": _normalize_symbol_list(seed_symbols),
        "terminal_status": None,
        "account": None,
        "ticks": {},
        "holdings": [],
        "pending_orders": [],
        "order_history": [],
        "deal_history": [],
    }


def _normalize_tick_payload(symbol: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(symbol).upper(),
        "bid": None if payload.get("bid") is None else float(payload.get("bid")),
        "ask": None if payload.get("ask") is None else float(payload.get("ask")),
        "last": None if payload.get("last") is None else float(payload.get("last")),
        "time_utc": payload.get("time_utc"),
        "raw": dict(payload),
    }


def collect_live_state_from_connector(
    connector: Any,
    *,
    config: MT5Config,
    base_currency: str,
    seed_symbols: Iterable[str] | None = None,
    history_lookback_minutes: int = 180,
    order_history: list[dict[str, Any]] | None = None,
    deal_history: list[dict[str, Any]] | None = None,
    refresh_history: bool = True,
    history_limit: int = 200,
) -> dict[str, Any]:
    live = MT5LiveGateway(connector, config=config, base_currency=base_currency)
    terminal_status = live.terminal_status().to_dict()
    account = live.account_snapshot().to_dict()
    holdings = [item.to_dict() for item in live.holdings(symbols=None)]
    pending_orders = [item.to_dict() for item in live.pending_orders(symbols=None)]

    tracked_symbols = {
        str(symbol).upper()
        for symbol in list(seed_symbols or [])
        + [item.get("symbol") for item in holdings]
        + [item.get("symbol") for item in pending_orders]
        if str(symbol or "").strip()
    }

    orders_payload = list(order_history or [])
    deals_payload = list(deal_history or [])
    if refresh_history:
        end = _utcnow()
        start = end - timedelta(minutes=int(history_lookback_minutes))
        orders_payload = [item.to_dict() for item in live.order_history(date_from=start, date_to=end, symbols=None)]
        deals_payload = [item.to_dict() for item in live.deal_history(date_from=start, date_to=end, symbols=None)]
    if history_limit > 0:
        orders_payload = list(orders_payload)[-history_limit:]
        deals_payload = list(deals_payload)[-history_limit:]

    for section in (orders_payload, deals_payload):
        for item in section:
            symbol = str(item.get("symbol") or "").upper()
            if symbol:
                tracked_symbols.add(symbol)

    ticks: dict[str, dict[str, Any]] = {}
    for symbol in sorted(tracked_symbols):
        try:
            ticks[symbol] = _normalize_tick_payload(symbol, dict(connector.symbol_info_tick(symbol)))
        except MT5ConnectionError:
            continue

    return {
        "source": "mt5_agent_bridge",
        "generated_at": _utcnow_iso(),
        "symbols": sorted(tracked_symbols),
        "terminal_status": terminal_status,
        "account": account,
        "ticks": ticks,
        "holdings": holdings,
        "pending_orders": pending_orders,
        "order_history": orders_payload,
        "deal_history": deals_payload,
    }


def _fingerprint_state(state: Mapping[str, Any]) -> str:
    payload = {
        "status": state.get("status"),
        "connected": state.get("connected"),
        "degraded": state.get("degraded"),
        "symbols": state.get("symbols"),
        "terminal_status": state.get("terminal_status"),
        "account": state.get("account"),
        "ticks": state.get("ticks"),
        "holdings": state.get("holdings"),
        "pending_orders": state.get("pending_orders"),
        "order_history": state.get("order_history"),
        "deal_history": state.get("deal_history"),
        "last_error": state.get("last_error"),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _change_summary(previous: Mapping[str, Any] | None, current: Mapping[str, Any]) -> dict[str, Any]:
    prev_symbols = set(previous.get("symbols") or []) if previous else set()
    curr_symbols = set(current.get("symbols") or [])
    return {
        "symbols_added": sorted(curr_symbols - prev_symbols),
        "symbols_removed": sorted(prev_symbols - curr_symbols),
        "open_positions": len(list(current.get("holdings") or [])),
        "pending_orders": len(list(current.get("pending_orders") or [])),
        "order_history": len(list(current.get("order_history") or [])),
        "deal_history": len(list(current.get("deal_history") or [])),
    }


class MT5EventBridge:
    def __init__(
        self,
        *,
        runtime: Any,
        config: MT5Config,
        base_currency: str,
        seed_symbols: Iterable[str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.base_currency = str(base_currency or "EUR").upper()
        self.seed_symbols = _normalize_symbol_list(seed_symbols)
        self._condition = Condition()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._sequence = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=max(int(config.live_event_buffer_size), 10))
        self._state = build_empty_live_state(config=config, seed_symbols=self.seed_symbols)
        self._fingerprint = _fingerprint_state(self._state)
        self._cached_order_history: list[dict[str, Any]] = []
        self._cached_deal_history: list[dict[str, Any]] = []
        self._last_history_refresh_at: datetime | None = None
        self._last_success_at: datetime | None = None

    def start(self) -> None:
        if not self.config.live_enabled:
            return
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name="mt5-live-bridge", daemon=True)
            self._thread.start()

    def prime(self) -> None:
        if not self.config.live_enabled:
            return
        with self._condition:
            if self._sequence > 0:
                return
        self._capture(force_event=True)

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def current_state(self) -> dict[str, Any]:
        self.prime()
        with self._condition:
            return deepcopy(self._state)

    def events_after(self, sequence: int, *, limit: int = 100, wait_seconds: float = 15.0) -> list[dict[str, Any]]:
        self.prime()
        deadline = None if wait_seconds <= 0 else _utcnow().timestamp() + float(wait_seconds)
        with self._condition:
            while not self._stop_event.is_set():
                batch = [deepcopy(item) for item in self._events if int(item["sequence"]) > int(sequence)]
                if batch:
                    return batch[: max(int(limit), 1)]
                if deadline is None:
                    break
                remaining = deadline - _utcnow().timestamp()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return []

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._capture(force_event=False)
            if self._stop_event.wait(timeout=max(float(self.config.live_poll_seconds), 0.5)):
                break

    def _capture(self, *, force_event: bool) -> None:
        now = _utcnow()
        refresh_history = self._should_refresh_history(now)
        previous_state: dict[str, Any] | None = None
        with self._condition:
            previous_state = deepcopy(self._state)

        try:
            payload = self.runtime.execute(
                lambda connector: collect_live_state_from_connector(
                    connector,
                    config=self.config,
                    base_currency=self.base_currency,
                    seed_symbols=self.seed_symbols,
                    history_lookback_minutes=int(self.config.live_history_lookback_minutes),
                    order_history=self._cached_order_history,
                    deal_history=self._cached_deal_history,
                    refresh_history=refresh_history,
                )
            )
            if refresh_history:
                self._cached_order_history = list(payload.get("order_history") or [])
                self._cached_deal_history = list(payload.get("deal_history") or [])
                self._last_history_refresh_at = now
            payload.update(
                {
                    "status": "ok",
                    "connected": True,
                    "degraded": False,
                    "stale": False,
                    "last_success_at": payload.get("generated_at"),
                    "last_error": None,
                    "poll_interval_seconds": float(self.config.live_poll_seconds),
                    "history_poll_interval_seconds": float(self.config.live_history_poll_seconds),
                    "history_lookback_minutes": int(self.config.live_history_lookback_minutes),
                }
            )
            self._last_success_at = now
            self._store_state(payload, previous=previous_state, force_event=force_event, kind="snapshot")
        except Exception as exc:
            stale = True
            connected = False
            if self._last_success_at is not None:
                age = max((_utcnow() - self._last_success_at).total_seconds(), 0.0)
                stale = age >= float(self.config.live_stale_after_seconds)
            degraded = True
            degraded_state = build_empty_live_state(
                config=self.config,
                seed_symbols=self.seed_symbols,
                status="degraded",
                connected=connected,
                degraded=degraded,
                stale=stale,
                last_error=str(exc),
                last_success_at=None if self._last_success_at is None else self._last_success_at.isoformat(),
            )
            with self._condition:
                degraded_state["sequence"] = self._sequence
                degraded_state["holdings"] = deepcopy(self._state.get("holdings") or [])
                degraded_state["pending_orders"] = deepcopy(self._state.get("pending_orders") or [])
                degraded_state["order_history"] = deepcopy(self._state.get("order_history") or [])
                degraded_state["deal_history"] = deepcopy(self._state.get("deal_history") or [])
                degraded_state["ticks"] = deepcopy(self._state.get("ticks") or {})
                degraded_state["terminal_status"] = deepcopy(self._state.get("terminal_status"))
                degraded_state["account"] = deepcopy(self._state.get("account"))
            self._store_state(degraded_state, previous=previous_state, force_event=True, kind="connection_error")

    def _store_state(
        self,
        state: dict[str, Any],
        *,
        previous: Mapping[str, Any] | None,
        force_event: bool,
        kind: str,
    ) -> None:
        fingerprint = _fingerprint_state(state)
        should_emit = force_event or fingerprint != self._fingerprint
        with self._condition:
            state["sequence"] = self._sequence
            self._state = deepcopy(state)
            if should_emit:
                self._sequence += 1
                self._state["sequence"] = self._sequence
                event = {
                    "sequence": self._sequence,
                    "kind": kind,
                    "timestamp_utc": _utcnow_iso(),
                    "change_summary": _change_summary(previous, self._state),
                    "state": deepcopy(self._state),
                }
                self._events.append(event)
                self._fingerprint = fingerprint
                self._condition.notify_all()

    def _should_refresh_history(self, now: datetime) -> bool:
        if self._last_history_refresh_at is None:
            return True
        return (now - self._last_history_refresh_at).total_seconds() >= float(self.config.live_history_poll_seconds)
