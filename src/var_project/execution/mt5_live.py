from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from var_project.core.exceptions import MT5ConnectionError
from var_project.core.types import MT5Config


ORDER_CHECK_OK = 0
ORDER_SEND_DONE = 10009
ORDER_SEND_PLACED = 10008


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _position_side(type_value: Any) -> str:
    return "BUY" if int(type_value or 0) == 0 else "SELL"


def _signed_from_side(side: str) -> float:
    return 1.0 if side.upper() == "BUY" else -1.0


def _normalize_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(_coerce_float(value), tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _asset_class_from_symbol(symbol: str, info: Mapping[str, Any]) -> str:
    normalized = str(symbol).upper()
    base = str(info.get("currency_base") or "").upper()
    profit = str(info.get("currency_profit") or "").upper()
    path = str(info.get("path") or info.get("description") or "").lower()
    if normalized.startswith(("XAU", "XAG", "XPT", "XPD")) or "metal" in path:
        return "metals"
    if len(normalized) == 6 and base and profit:
        return "fx"
    if "index" in path or "indices" in path or normalized.startswith(("US", "DE", "FR", "UK")):
        return "indices_cfd"
    return "cfd"


def _trading_mode_name(value: Any) -> str | None:
    if value is None:
        return None
    mapping = {
        0: "disabled",
        1: "long_only",
        2: "short_only",
        3: "close_only",
        4: "full_access",
    }
    try:
        key = int(value)
    except (TypeError, ValueError):
        return str(value)
    return mapping.get(key, str(key))


def _manual_event(comment: str | None, *, prefix: str) -> bool:
    if comment is None:
        return True
    normalized = str(comment).strip().lower()
    expected = str(prefix or "").strip().lower()
    if not expected:
        return False
    return not normalized.startswith(expected)


@dataclass(frozen=True)
class MT5TerminalStatus:
    connected: bool
    ready: bool
    execution_enabled: bool
    trade_allowed: bool | None
    tradeapi_disabled: bool | None
    company: str | None
    terminal_path: str | None
    data_path: str | None
    commondata_path: str | None
    message: str
    timestamp_utc: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "ready": self.ready,
            "execution_enabled": self.execution_enabled,
            "trade_allowed": self.trade_allowed,
            "tradeapi_disabled": self.tradeapi_disabled,
            "company": self.company,
            "terminal_path": self.terminal_path,
            "data_path": self.data_path,
            "commondata_path": self.commondata_path,
            "message": self.message,
            "timestamp_utc": self.timestamp_utc,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class MT5AccountSnapshot:
    login: int | None
    name: str | None
    server: str | None
    currency: str | None
    leverage: int | None
    balance: float
    equity: float
    profit: float
    margin: float
    margin_free: float
    margin_level: float | None
    trade_allowed: bool | None
    timestamp_utc: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "login": self.login,
            "name": self.name,
            "server": self.server,
            "currency": self.currency,
            "leverage": self.leverage,
            "balance": self.balance,
            "equity": self.equity,
            "profit": self.profit,
            "margin": self.margin,
            "margin_free": self.margin_free,
            "margin_level": self.margin_level,
            "trade_allowed": self.trade_allowed,
            "timestamp_utc": self.timestamp_utc,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class MT5Position:
    ticket: int | None
    symbol: str
    side: str
    volume_lots: float
    signed_position_eur: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    comment: str | None
    time_utc: str | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "volume_lots": self.volume_lots,
            "signed_position_eur": self.signed_position_eur,
            "price_open": self.price_open,
            "price_current": self.price_current,
            "profit": self.profit,
            "swap": self.swap,
            "comment": self.comment,
            "time_utc": self.time_utc,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class MT5PendingOrder:
    ticket: int | None
    symbol: str
    side: str
    volume_initial: float
    volume_current: float
    price_open: float
    comment: str | None
    time_setup_utc: str | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "volume_initial": self.volume_initial,
            "volume_current": self.volume_current,
            "price_open": self.price_open,
            "comment": self.comment,
            "time_setup_utc": self.time_setup_utc,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class InstrumentDefinition:
    symbol: str
    asset_class: str
    contract_size: float | None
    base_currency: str | None
    quote_currency: str | None
    profit_currency: str | None
    margin_currency: str | None
    tick_size: float | None
    tick_value: float | None
    volume_min: float | None
    volume_max: float | None
    volume_step: float | None
    trading_mode: str | None
    source: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "contract_size": self.contract_size,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "profit_currency": self.profit_currency,
            "margin_currency": self.margin_currency,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "trading_mode": self.trading_mode,
            "source": self.source,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class HoldingSnapshot:
    symbol: str
    asset_class: str
    side: str
    volume_lots: float
    signed_position_eur: float
    signed_units: float | None
    contract_size: float | None
    base_currency: str | None
    profit_currency: str | None
    margin_currency: str | None
    mark_price: float | None
    market_value_base_ccy: float | None
    exposure_base_ccy: float | None
    unrealized_pnl_base_ccy: float | None
    profit: float
    source: str | None
    comment: str | None
    time_utc: str | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "side": self.side,
            "volume_lots": self.volume_lots,
            "signed_position_eur": self.signed_position_eur,
            "signed_units": self.signed_units,
            "contract_size": self.contract_size,
            "base_currency": self.base_currency,
            "profit_currency": self.profit_currency,
            "margin_currency": self.margin_currency,
            "mark_price": self.mark_price,
            "market_value_base_ccy": self.market_value_base_ccy,
            "exposure_base_ccy": self.exposure_base_ccy,
            "unrealized_pnl_base_ccy": self.unrealized_pnl_base_ccy,
            "profit": self.profit,
            "source": self.source,
            "comment": self.comment,
            "time_utc": self.time_utc,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class OrderHistoryEntry:
    ticket: int | None
    position_id: int | None
    symbol: str
    side: str | None
    order_type: str | None
    state: str | None
    volume_initial: float | None
    volume_current: float | None
    price_open: float | None
    price_current: float | None
    comment: str | None
    is_manual: bool
    time_setup_utc: str | None
    time_done_utc: str | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "state": self.state,
            "volume_initial": self.volume_initial,
            "volume_current": self.volume_current,
            "price_open": self.price_open,
            "price_current": self.price_current,
            "comment": self.comment,
            "is_manual": self.is_manual,
            "time_setup_utc": self.time_setup_utc,
            "time_done_utc": self.time_done_utc,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class DealHistoryEntry:
    ticket: int | None
    order_ticket: int | None
    position_id: int | None
    symbol: str
    side: str | None
    entry: str | None
    volume: float | None
    price: float | None
    profit: float | None
    commission: float | None
    swap: float | None
    fee: float | None
    reason: str | None
    comment: str | None
    is_manual: bool
    time_utc: str | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "order_ticket": self.order_ticket,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry": self.entry,
            "volume": self.volume,
            "price": self.price,
            "profit": self.profit,
            "commission": self.commission,
            "swap": self.swap,
            "fee": self.fee,
            "reason": self.reason,
            "comment": self.comment,
            "is_manual": self.is_manual,
            "time_utc": self.time_utc,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class ExecutionGuardDecision:
    decision: str
    risk_decision: str
    requested_delta_position_eur: float
    approved_delta_position_eur: float
    executable_delta_position_eur: float
    suggested_delta_position_eur: float | None
    model_used: str
    side: str | None
    volume_lots: float
    price: float | None
    execution_enabled: bool
    submit_allowed: bool
    margin_ok: bool
    margin_required: float | None
    free_margin_after: float | None
    order_check_retcode: int | None
    order_check_comment: str | None
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "risk_decision": self.risk_decision,
            "requested_delta_position_eur": self.requested_delta_position_eur,
            "approved_delta_position_eur": self.approved_delta_position_eur,
            "executable_delta_position_eur": self.executable_delta_position_eur,
            "suggested_delta_position_eur": self.suggested_delta_position_eur,
            "model_used": self.model_used,
            "side": self.side,
            "volume_lots": self.volume_lots,
            "price": self.price,
            "execution_enabled": self.execution_enabled,
            "submit_allowed": self.submit_allowed,
            "margin_ok": self.margin_ok,
            "margin_required": self.margin_required,
            "free_margin_after": self.free_margin_after,
            "order_check_retcode": self.order_check_retcode,
            "order_check_comment": self.order_check_comment,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ExecutionPreview:
    time_utc: str
    portfolio_slug: str
    symbol: str
    terminal_status: MT5TerminalStatus
    account: MT5AccountSnapshot
    live_positions: list[MT5Position]
    pending_orders: list[MT5PendingOrder]
    risk_decision: dict[str, Any]
    guard: ExecutionGuardDecision
    order_request: dict[str, Any]
    order_check: dict[str, Any]
    pre_capital: dict[str, Any]
    post_capital: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_utc": self.time_utc,
            "portfolio_slug": self.portfolio_slug,
            "symbol": self.symbol,
            "terminal_status": self.terminal_status.to_dict(),
            "account": self.account.to_dict(),
            "live_positions": [item.to_dict() for item in self.live_positions],
            "pending_orders": [item.to_dict() for item in self.pending_orders],
            "risk_decision": dict(self.risk_decision),
            "guard": self.guard.to_dict(),
            "order_request": dict(self.order_request),
            "order_check": dict(self.order_check),
            "pre_capital": dict(self.pre_capital),
            "post_capital": dict(self.post_capital),
        }


@dataclass(frozen=True)
class ExecutionResult:
    time_utc: str
    portfolio_slug: str
    symbol: str
    status: str
    requested_delta_position_eur: float
    approved_delta_position_eur: float
    executed_delta_position_eur: float
    terminal_status: MT5TerminalStatus
    account_before: MT5AccountSnapshot
    account_after: MT5AccountSnapshot | None
    guard: ExecutionGuardDecision
    risk_decision: dict[str, Any]
    order_request: dict[str, Any]
    order_check: dict[str, Any]
    mt5_result: dict[str, Any]
    positions_after: list[MT5Position]
    post_capital: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_utc": self.time_utc,
            "portfolio_slug": self.portfolio_slug,
            "symbol": self.symbol,
            "status": self.status,
            "requested_delta_position_eur": self.requested_delta_position_eur,
            "approved_delta_position_eur": self.approved_delta_position_eur,
            "executed_delta_position_eur": self.executed_delta_position_eur,
            "terminal_status": self.terminal_status.to_dict(),
            "account_before": self.account_before.to_dict(),
            "account_after": None if self.account_after is None else self.account_after.to_dict(),
            "guard": self.guard.to_dict(),
            "risk_decision": dict(self.risk_decision),
            "order_request": dict(self.order_request),
            "order_check": dict(self.order_check),
            "mt5_result": dict(self.mt5_result),
            "positions_after": [item.to_dict() for item in self.positions_after],
            "post_capital": dict(self.post_capital),
        }


class MT5LiveGateway:
    def __init__(self, connector: Any, *, config: MT5Config, base_currency: str = "EUR"):
        self.connector = connector
        self.config = config
        self.base_currency = str(base_currency or "EUR").upper()
        self._symbol_cache: dict[str, dict[str, Any]] = {}
        self._tick_cache: dict[str, dict[str, Any]] = {}

    def terminal_status(self) -> MT5TerminalStatus:
        info = dict(self.connector.terminal_info())
        trade_allowed = None if info.get("trade_allowed") is None else bool(info.get("trade_allowed"))
        tradeapi_disabled = None if info.get("tradeapi_disabled") is None else bool(info.get("tradeapi_disabled"))
        connected = True
        execution_enabled = bool(self.config.execution_enabled)
        ready = connected and execution_enabled and bool(trade_allowed) and not bool(tradeapi_disabled)

        reasons: list[str] = []
        if not execution_enabled:
            reasons.append("Execution is disabled in VAR_PROJECT_MT5_EXECUTION_ENABLED / settings.")
        if trade_allowed is False:
            reasons.append("MetaTrader terminal trading is not allowed.")
        if tradeapi_disabled is True:
            reasons.append("MetaTrader Python/API trading is disabled.")
        message = "MT5 terminal ready." if not reasons else " ".join(reasons)

        return MT5TerminalStatus(
            connected=connected,
            ready=ready,
            execution_enabled=execution_enabled,
            trade_allowed=trade_allowed,
            tradeapi_disabled=tradeapi_disabled,
            company=None if info.get("company") is None else str(info.get("company")),
            terminal_path=None if info.get("path") is None else str(info.get("path")),
            data_path=None if info.get("data_path") is None else str(info.get("data_path")),
            commondata_path=None if info.get("commondata_path") is None else str(info.get("commondata_path")),
            message=message,
            timestamp_utc=_utcnow_iso(),
            raw=info,
        )

    def account_snapshot(self) -> MT5AccountSnapshot:
        info = dict(self.connector.account_info())
        return MT5AccountSnapshot(
            login=None if info.get("login") is None else _coerce_int(info.get("login")),
            name=None if info.get("name") is None else str(info.get("name")),
            server=None if info.get("server") is None else str(info.get("server")),
            currency=None if info.get("currency") is None else str(info.get("currency")),
            leverage=None if info.get("leverage") is None else _coerce_int(info.get("leverage")),
            balance=_coerce_float(info.get("balance")),
            equity=_coerce_float(info.get("equity")),
            profit=_coerce_float(info.get("profit")),
            margin=_coerce_float(info.get("margin")),
            margin_free=_coerce_float(info.get("margin_free")),
            margin_level=None if info.get("margin_level") is None else _coerce_float(info.get("margin_level")),
            trade_allowed=None if info.get("trade_allowed") is None else bool(info.get("trade_allowed")),
            timestamp_utc=_utcnow_iso(),
            raw=info,
        )

    def positions(self, *, symbols: Iterable[str] | None = None) -> list[MT5Position]:
        allowed = {str(symbol).upper() for symbol in symbols or []}
        rows = self.connector.positions_get()
        positions: list[MT5Position] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if allowed and symbol not in allowed:
                continue
            side = _position_side(row.get("type"))
            signed = _signed_from_side(side) * _coerce_float(row.get("volume")) * self.notional_eur_per_lot(symbol)
            time_epoch = row.get("time_update") or row.get("time")
            time_utc = None
            if time_epoch is not None:
                time_utc = datetime.fromtimestamp(_coerce_float(time_epoch), tz=timezone.utc).isoformat()
            positions.append(
                MT5Position(
                    ticket=None if row.get("ticket") is None else _coerce_int(row.get("ticket")),
                    symbol=symbol,
                    side=side,
                    volume_lots=_coerce_float(row.get("volume")),
                    signed_position_eur=signed,
                    price_open=_coerce_float(row.get("price_open")),
                    price_current=_coerce_float(row.get("price_current")),
                    profit=_coerce_float(row.get("profit")),
                    swap=_coerce_float(row.get("swap")),
                    comment=None if row.get("comment") is None else str(row.get("comment")),
                    time_utc=time_utc,
                    raw=dict(row),
                )
            )
        return positions

    def pending_orders(self, *, symbols: Iterable[str] | None = None) -> list[MT5PendingOrder]:
        allowed = {str(symbol).upper() for symbol in symbols or []}
        rows = self.connector.orders_get()
        orders: list[MT5PendingOrder] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if allowed and symbol not in allowed:
                continue
            side = _position_side(row.get("type"))
            time_epoch = row.get("time_setup") or row.get("time_done") or row.get("time")
            time_utc = None
            if time_epoch is not None:
                time_utc = datetime.fromtimestamp(_coerce_float(time_epoch), tz=timezone.utc).isoformat()
            orders.append(
                MT5PendingOrder(
                    ticket=None if row.get("ticket") is None else _coerce_int(row.get("ticket")),
                    symbol=symbol,
                    side=side,
                    volume_initial=_coerce_float(row.get("volume_initial")),
                    volume_current=_coerce_float(row.get("volume_current")),
                    price_open=_coerce_float(row.get("price_open")),
                    comment=None if row.get("comment") is None else str(row.get("comment")),
                    time_setup_utc=time_utc,
                    raw=dict(row),
                )
            )
        return orders

    def instrument_definition(self, symbol: str) -> InstrumentDefinition:
        normalized = str(symbol).upper()
        info = self._symbol_info(normalized)
        return InstrumentDefinition(
            symbol=normalized,
            asset_class=_asset_class_from_symbol(normalized, info),
            contract_size=None if info.get("trade_contract_size") is None else _coerce_float(info.get("trade_contract_size")),
            base_currency=None if info.get("currency_base") is None else str(info.get("currency_base")).upper(),
            quote_currency=None if info.get("currency_profit") is None else str(info.get("currency_profit")).upper(),
            profit_currency=None if info.get("currency_profit") is None else str(info.get("currency_profit")).upper(),
            margin_currency=None if info.get("currency_margin") is None else str(info.get("currency_margin")).upper(),
            tick_size=None if info.get("trade_tick_size") is None else _coerce_float(info.get("trade_tick_size")),
            tick_value=None if info.get("trade_tick_value") is None else _coerce_float(info.get("trade_tick_value")),
            volume_min=None if info.get("volume_min") is None else _coerce_float(info.get("volume_min")),
            volume_max=None if info.get("volume_max") is None else _coerce_float(info.get("volume_max")),
            volume_step=None if info.get("volume_step") is None else _coerce_float(info.get("volume_step")),
            trading_mode=_trading_mode_name(info.get("trade_mode")),
            source="mt5",
            raw=dict(info),
        )

    def holdings(self, *, symbols: Iterable[str] | None = None) -> list[HoldingSnapshot]:
        allowed = {str(symbol).upper() for symbol in symbols or []}
        rows = self.connector.positions_get()
        holdings: list[HoldingSnapshot] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if allowed and symbol not in allowed:
                continue
            side = _position_side(row.get("type"))
            info = self.instrument_definition(symbol)
            tick = self._tick(symbol)
            mark_price = _coerce_float(tick.get("bid")) if side == "BUY" else _coerce_float(tick.get("ask"))
            signed_units = _signed_from_side(side) * _coerce_float(row.get("volume")) * _coerce_float(info.contract_size, 0.0)
            exposure_base_ccy = _signed_from_side(side) * _coerce_float(row.get("volume")) * self.notional_eur_per_lot(symbol)
            holdings.append(
                HoldingSnapshot(
                    symbol=symbol,
                    asset_class=info.asset_class,
                    side=side,
                    volume_lots=_coerce_float(row.get("volume")),
                    signed_position_eur=_signed_from_side(side) * _coerce_float(row.get("volume")) * self.notional_eur_per_lot(symbol),
                    signed_units=None if abs(signed_units) <= 1e-9 else signed_units,
                    contract_size=info.contract_size,
                    base_currency=info.base_currency,
                    profit_currency=info.profit_currency,
                    margin_currency=info.margin_currency,
                    mark_price=None if mark_price <= 0 else mark_price,
                    market_value_base_ccy=None if mark_price <= 0 or abs(signed_units) <= 1e-9 else signed_units * mark_price,
                    exposure_base_ccy=exposure_base_ccy,
                    unrealized_pnl_base_ccy=_coerce_float(row.get("profit")),
                    profit=_coerce_float(row.get("profit")),
                    source="mt5_live",
                    comment=None if row.get("comment") is None else str(row.get("comment")),
                    time_utc=_normalize_timestamp(row.get("time_update") or row.get("time")),
                    raw=dict(row),
                )
            )
        return holdings

    def order_history(
        self,
        *,
        date_from: datetime,
        date_to: datetime,
        symbols: Iterable[str] | None = None,
    ) -> list[OrderHistoryEntry]:
        allowed = {str(symbol).upper() for symbol in symbols or []}
        prefix = str(self.config.comment_prefix or "")
        rows = self.connector.history_orders_get(date_from, date_to)
        entries: list[OrderHistoryEntry] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if allowed and symbol not in allowed:
                continue
            side = None if row.get("type") is None else _position_side(row.get("type"))
            entries.append(
                OrderHistoryEntry(
                    ticket=None if row.get("ticket") is None else _coerce_int(row.get("ticket")),
                    position_id=None if row.get("position_id") is None else _coerce_int(row.get("position_id")),
                    symbol=symbol,
                    side=side,
                    order_type=None if row.get("type") is None else str(row.get("type")),
                    state=None if row.get("state") is None else str(row.get("state")),
                    volume_initial=None if row.get("volume_initial") is None else _coerce_float(row.get("volume_initial")),
                    volume_current=None if row.get("volume_current") is None else _coerce_float(row.get("volume_current")),
                    price_open=None if row.get("price_open") is None else _coerce_float(row.get("price_open")),
                    price_current=None if row.get("price_current") is None else _coerce_float(row.get("price_current")),
                    comment=None if row.get("comment") is None else str(row.get("comment")),
                    is_manual=_manual_event(None if row.get("comment") is None else str(row.get("comment")), prefix=prefix),
                    time_setup_utc=_normalize_timestamp(row.get("time_setup") or row.get("time_setup_msc")),
                    time_done_utc=_normalize_timestamp(row.get("time_done") or row.get("time_done_msc")),
                    raw=dict(row),
                )
            )
        return entries

    def deal_history(
        self,
        *,
        date_from: datetime,
        date_to: datetime,
        symbols: Iterable[str] | None = None,
    ) -> list[DealHistoryEntry]:
        allowed = {str(symbol).upper() for symbol in symbols or []}
        prefix = str(self.config.comment_prefix or "")
        rows = self.connector.history_deals_get(date_from, date_to)
        entries: list[DealHistoryEntry] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if allowed and symbol not in allowed:
                continue
            side = None if row.get("type") is None else _position_side(row.get("type"))
            entries.append(
                DealHistoryEntry(
                    ticket=None if row.get("ticket") is None else _coerce_int(row.get("ticket")),
                    order_ticket=None if row.get("order") is None else _coerce_int(row.get("order")),
                    position_id=None if row.get("position_id") is None else _coerce_int(row.get("position_id")),
                    symbol=symbol,
                    side=side,
                    entry=None if row.get("entry") is None else str(row.get("entry")),
                    volume=None if row.get("volume") is None else _coerce_float(row.get("volume")),
                    price=None if row.get("price") is None else _coerce_float(row.get("price")),
                    profit=None if row.get("profit") is None else _coerce_float(row.get("profit")),
                    commission=None if row.get("commission") is None else _coerce_float(row.get("commission")),
                    swap=None if row.get("swap") is None else _coerce_float(row.get("swap")),
                    fee=None if row.get("fee") is None else _coerce_float(row.get("fee")),
                    reason=None if row.get("reason") is None else str(row.get("reason")),
                    comment=None if row.get("comment") is None else str(row.get("comment")),
                    is_manual=_manual_event(None if row.get("comment") is None else str(row.get("comment")), prefix=prefix),
                    time_utc=_normalize_timestamp(row.get("time") or row.get("time_msc")),
                    raw=dict(row),
                )
            )
        return entries

    def positions_map_eur(self, *, symbols: Iterable[str]) -> dict[str, float]:
        tracked = {str(symbol).upper() for symbol in symbols}
        positions = {symbol: 0.0 for symbol in tracked}
        for item in self.positions(symbols=tracked):
            positions[item.symbol] = float(positions.get(item.symbol, 0.0) + item.signed_position_eur)
        return positions

    def build_market_order(self, *, symbol: str, delta_position_eur: float, note: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized_symbol = str(symbol).upper()
        if abs(float(delta_position_eur)) <= 1e-9:
            raise ValueError("delta_position_eur must be non-zero.")

        info = self._symbol_info(normalized_symbol)
        tick = dict(self.connector.symbol_info_tick(normalized_symbol))
        mt5 = self.connector._mt5  # noqa: SLF001 - wrapper intentionally exposes underlying package after init

        side = "BUY" if float(delta_position_eur) > 0.0 else "SELL"
        price = _coerce_float(tick.get("ask")) if side == "BUY" else _coerce_float(tick.get("bid"))
        if price <= 0.0:
            raise MT5ConnectionError(f"No tradable price available for {normalized_symbol}.")

        eur_per_lot = self.notional_eur_per_lot(normalized_symbol)
        raw_volume = abs(float(delta_position_eur)) / eur_per_lot
        volume = self._round_lot_volume(normalized_symbol, raw_volume)

        min_volume = _coerce_float(info.get("volume_min"), 0.0)
        if volume < min_volume - 1e-12:
            minimum_notional = eur_per_lot * min_volume
            raise ValueError(
                f"Requested EUR delta is below the broker minimum size for {normalized_symbol} "
                f"({minimum_notional:,.2f} {self.base_currency})."
            )

        fill_candidates = self._fill_candidates(info, mt5)
        fill_mode = fill_candidates[0]

        request = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL"),
            "symbol": normalized_symbol,
            "volume": volume,
            "type": getattr(mt5, "ORDER_TYPE_BUY" if side == "BUY" else "ORDER_TYPE_SELL"),
            "price": price,
            "deviation": int(self.config.deviation_points),
            "magic": int(self.config.magic),
            "comment": self._comment(note),
            "type_time": getattr(mt5, "ORDER_TIME_GTC"),
            "type_filling": fill_mode,
        }

        executable_delta = (_signed_from_side(side) * volume * eur_per_lot)
        meta = {
            "side": side,
            "price": price,
            "volume_lots": volume,
            "eur_per_lot": eur_per_lot,
            "executable_delta_position_eur": executable_delta,
            "requested_delta_position_eur": float(delta_position_eur),
            "minimum_notional_eur": eur_per_lot * min_volume if min_volume > 0 else 0.0,
            "fill_candidates": fill_candidates,
        }
        return request, meta

    def notional_eur_per_lot(self, symbol: str) -> float:
        info = self._symbol_info(symbol)
        base_ccy = str(info.get("currency_base") or str(symbol)[:3]).upper()
        contract_size = _coerce_float(info.get("trade_contract_size"), 100000.0)
        rate = self.fx_rate(base_ccy, self.base_currency)
        return contract_size * rate

    def fx_rate(self, from_currency: str, to_currency: str) -> float:
        source = str(from_currency).upper()
        target = str(to_currency).upper()
        if source == target:
            return 1.0

        direct_rate = self._direct_or_inverse_rate(source, target)
        if direct_rate is not None:
            return direct_rate

        pivots = [self.base_currency, "USD", "EUR", "JPY", "GBP", "CHF"]
        for pivot in pivots:
            pivot = pivot.upper()
            if pivot in {source, target}:
                continue
            leg_a = self._direct_or_inverse_rate(source, pivot)
            leg_b = self._direct_or_inverse_rate(pivot, target)
            if leg_a is not None and leg_b is not None:
                return leg_a * leg_b

        raise MT5ConnectionError(f"Unable to resolve FX rate from {source} to {target}.")

    def _direct_or_inverse_rate(self, source: str, target: str) -> float | None:
        direct = self._mid_if_exists(f"{source}{target}")
        if direct is not None and direct > 0.0:
            return direct

        inverse = self._mid_if_exists(f"{target}{source}")
        if inverse is not None and inverse > 0.0:
            return 1.0 / inverse
        return None

    def _mid_if_exists(self, symbol: str) -> float | None:
        try:
            tick = self._tick(symbol)
        except MT5ConnectionError:
            return None
        bid = _coerce_float(tick.get("bid"))
        ask = _coerce_float(tick.get("ask"))
        if bid > 0.0 and ask > 0.0:
            return 0.5 * (bid + ask)
        value = bid if bid > 0.0 else ask
        return None if value <= 0.0 else value

    def _symbol_info(self, symbol: str) -> dict[str, Any]:
        normalized = str(symbol).upper()
        cached = self._symbol_cache.get(normalized)
        if cached is not None:
            return cached
        info = dict(self.connector.symbol_info(normalized))
        self._symbol_cache[normalized] = info
        return info

    def _tick(self, symbol: str) -> dict[str, Any]:
        normalized = str(symbol).upper()
        cached = self._tick_cache.get(normalized)
        if cached is not None:
            return cached
        tick = dict(self.connector.symbol_info_tick(normalized))
        self._tick_cache[normalized] = tick
        return tick

    def _round_lot_volume(self, symbol: str, volume: float) -> float:
        info = self._symbol_info(symbol)
        step = _coerce_float(info.get("volume_step"), 0.01)
        min_volume = _coerce_float(info.get("volume_min"), step)
        max_volume = _coerce_float(info.get("volume_max"), volume)
        if step <= 0.0:
            step = 0.01
        normalized = int(volume / step) * step
        normalized = max(min_volume if volume >= min_volume else 0.0, normalized)
        if max_volume > 0.0:
            normalized = min(normalized, max_volume)
        digits = 0
        step_repr = f"{step:.10f}".rstrip("0")
        if "." in step_repr:
            digits = len(step_repr.split(".")[1])
        return round(normalized, digits)

    def _fill_candidates(self, info: Mapping[str, Any], mt5: Any) -> list[int]:
        allowed_flags = None if info.get("filling_mode") is None else int(info.get("filling_mode"))
        order_fok = int(getattr(mt5, "ORDER_FILLING_FOK", 0))
        order_ioc = int(getattr(mt5, "ORDER_FILLING_IOC", 1))
        order_return = int(getattr(mt5, "ORDER_FILLING_RETURN", 2))
        candidates: list[int] = []

        # MetaTrader symbol_info().filling_mode exposes allowed flags, not necessarily
        # the direct request enum value expected by order_check/order_send.
        if allowed_flags is not None:
            if allowed_flags & 1:
                candidates.append(order_fok)
            if allowed_flags & 2:
                candidates.append(order_ioc)

        # Keep safe fallbacks in case the broker accepts a broader execution policy.
        candidates.extend([order_return, order_ioc, order_fok])

        unique: list[int] = []
        seen: set[int] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique.append(candidate)
        return unique or [order_return]

    def _comment(self, note: str | None) -> str:
        prefix = str(self.config.comment_prefix or "var_risk_desk").strip() or "var_risk_desk"
        suffix = "" if note is None else f" {str(note).strip()}"
        return f"{prefix}{suffix}"[:31]
