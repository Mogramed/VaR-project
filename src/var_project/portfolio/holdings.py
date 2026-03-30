from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").upper().strip()


def _infer_asset_class(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    if normalized.startswith(("XAU", "XAG", "XPT", "XPD")):
        return "metals"
    if len(normalized) == 6 and normalized[:3].isalpha() and normalized[3:].isalpha():
        return "fx"
    if normalized.startswith(("US", "DE", "FR", "UK")):
        return "indices_cfd"
    return "cfd"


@dataclass(frozen=True)
class PortfolioHolding:
    symbol: str
    asset_class: str
    side: str
    volume_lots: float
    contract_size: float | None
    base_currency: str | None
    profit_currency: str | None
    margin_currency: str | None
    mark_price: float | None
    signed_units: float
    market_value_base_ccy: float
    exposure_base_ccy: float
    unrealized_pnl_base_ccy: float
    source: str = "configured"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "side": self.side,
            "volume_lots": self.volume_lots,
            "contract_size": self.contract_size,
            "base_currency": self.base_currency,
            "profit_currency": self.profit_currency,
            "margin_currency": self.margin_currency,
            "mark_price": self.mark_price,
            "signed_units": self.signed_units,
            "market_value_base_ccy": self.market_value_base_ccy,
            "exposure_base_ccy": self.exposure_base_ccy,
            "signed_exposure_base_ccy": self.exposure_base_ccy,
            "signed_position_eur": self.exposure_base_ccy,
            "unrealized_pnl_base_ccy": self.unrealized_pnl_base_ccy,
            "profit": self.unrealized_pnl_base_ccy,
            "source": self.source,
            "raw": dict(self.raw),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, fallback_base_currency: str = "EUR") -> "PortfolioHolding":
        symbol = _normalize_symbol(payload.get("symbol") or "")
        exposure = float(
            payload.get("exposure_base_ccy")
            or payload.get("signed_exposure_base_ccy")
            or payload.get("signed_position_eur")
            or payload.get("exposure_eur")
            or 0.0
        )
        side = str(payload.get("side") or ("BUY" if exposure > 0 else "SELL" if exposure < 0 else "FLAT")).upper()
        contract_size = None if payload.get("contract_size") is None else float(payload.get("contract_size"))
        mark_price = None if payload.get("mark_price") is None else float(payload.get("mark_price"))
        signed_units = float(payload.get("signed_units") or 0.0)
        if abs(signed_units) <= 1e-9 and contract_size is not None:
            signed_units = float(payload.get("volume_lots") or 0.0) * float(contract_size) * (1.0 if side == "BUY" else -1.0)
        return cls(
            symbol=symbol,
            asset_class=str(payload.get("asset_class") or _infer_asset_class(symbol)),
            side=side,
            volume_lots=float(payload.get("volume_lots") or 0.0),
            contract_size=contract_size,
            base_currency=None if payload.get("base_currency") is None else str(payload.get("base_currency")).upper(),
            profit_currency=None if payload.get("profit_currency") is None else str(payload.get("profit_currency")).upper(),
            margin_currency=(
                None if payload.get("margin_currency") is None else str(payload.get("margin_currency")).upper()
            ),
            mark_price=mark_price,
            signed_units=signed_units,
            market_value_base_ccy=float(payload.get("market_value_base_ccy") or exposure),
            exposure_base_ccy=exposure,
            unrealized_pnl_base_ccy=float(
                payload.get("unrealized_pnl_base_ccy")
                or payload.get("profit")
                or 0.0
            ),
            source=str(payload.get("source") or "configured"),
            raw=dict(payload.get("raw") or {"base_currency": fallback_base_currency}),
        )


def configured_holdings(
    symbols: Iterable[str],
    exposure_by_symbol: Mapping[str, Any] | None,
    *,
    base_currency: str = "EUR",
    source: str = "configured",
) -> list[PortfolioHolding]:
    exposures = {str(symbol).upper(): float(value) for symbol, value in dict(exposure_by_symbol or {}).items()}
    holdings: list[PortfolioHolding] = []
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        exposure = float(exposures.get(normalized, 0.0))
        side = "BUY" if exposure > 0.0 else "SELL" if exposure < 0.0 else "FLAT"
        holdings.append(
            PortfolioHolding(
                symbol=normalized,
                asset_class=_infer_asset_class(normalized),
                side=side,
                volume_lots=0.0,
                contract_size=None,
                base_currency=str(base_currency).upper(),
                profit_currency=str(base_currency).upper(),
                margin_currency=str(base_currency).upper(),
                mark_price=None,
                signed_units=0.0,
                market_value_base_ccy=exposure,
                exposure_base_ccy=exposure,
                unrealized_pnl_base_ccy=0.0,
                source=source,
                raw={"configured": True},
            )
        )
    return holdings


def normalize_holdings(
    holdings_or_exposure: Mapping[str, Any] | Iterable[Mapping[str, Any] | PortfolioHolding] | None,
    *,
    symbols: Iterable[str] | None = None,
    base_currency: str = "EUR",
    source: str = "configured",
) -> list[PortfolioHolding]:
    if holdings_or_exposure is None:
        return configured_holdings(symbols or [], {}, base_currency=base_currency, source=source)

    if isinstance(holdings_or_exposure, Mapping):
        return configured_holdings(
            symbols or holdings_or_exposure.keys(),
            holdings_or_exposure,
            base_currency=base_currency,
            source=source,
        )

    holdings: list[PortfolioHolding] = []
    for item in holdings_or_exposure:
        if isinstance(item, PortfolioHolding):
            holdings.append(item)
        else:
            holdings.append(PortfolioHolding.from_mapping(item, fallback_base_currency=base_currency))
    if not holdings and symbols is not None:
        return configured_holdings(symbols, {}, base_currency=base_currency, source=source)
    return holdings


def aggregate_exposure_by_symbol(
    holdings_or_exposure: Mapping[str, Any] | Iterable[Mapping[str, Any] | PortfolioHolding] | None,
    *,
    symbols: Iterable[str] | None = None,
    base_currency: str = "EUR",
) -> dict[str, float]:
    holdings = normalize_holdings(
        holdings_or_exposure,
        symbols=symbols,
        base_currency=base_currency,
    )
    exposure_by_symbol: dict[str, float] = {}
    for holding in holdings:
        exposure_by_symbol[holding.symbol] = float(
            exposure_by_symbol.get(holding.symbol, 0.0) + float(holding.exposure_base_ccy)
        )
    return exposure_by_symbol


def gross_exposure_base_ccy(
    holdings_or_exposure: Mapping[str, Any] | Iterable[Mapping[str, Any] | PortfolioHolding] | None,
    *,
    symbols: Iterable[str] | None = None,
    base_currency: str = "EUR",
) -> float:
    exposure_by_symbol = aggregate_exposure_by_symbol(
        holdings_or_exposure,
        symbols=symbols,
        base_currency=base_currency,
    )
    return float(sum(abs(value) for value in exposure_by_symbol.values()))


def holding_symbols(
    holdings_or_exposure: Mapping[str, Any] | Iterable[Mapping[str, Any] | PortfolioHolding] | None,
    *,
    symbols: Iterable[str] | None = None,
    base_currency: str = "EUR",
) -> list[str]:
    exposure_by_symbol = aggregate_exposure_by_symbol(
        holdings_or_exposure,
        symbols=symbols,
        base_currency=base_currency,
    )
    ordered = list(exposure_by_symbol.keys())
    if symbols is None:
        return ordered
    seen = {symbol for symbol in ordered}
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        if normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return ordered
