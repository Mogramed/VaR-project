from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _worst_status(statuses: Sequence[str]) -> str:
    normalized = {str(status).upper() for status in statuses}
    if "BREACH" in normalized:
        return "BREACH"
    if "WARN" in normalized:
        return "WARN"
    return "OK"


@dataclass(frozen=True)
class DeskDefinition:
    slug: str
    name: str
    base_currency: str
    portfolio_slugs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "base_currency": self.base_currency,
            "portfolio_slugs": list(self.portfolio_slugs),
        }


@dataclass(frozen=True)
class PortfolioCapitalSlice:
    portfolio_slug: str
    portfolio_name: str
    reference_model: str
    total_capital_budget_eur: float
    total_capital_consumed_eur: float
    total_capital_remaining_eur: float
    utilization: float | None
    status: str
    alert_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_slug": self.portfolio_slug,
            "portfolio_name": self.portfolio_name,
            "reference_model": self.reference_model,
            "total_capital_budget_eur": self.total_capital_budget_eur,
            "total_capital_consumed_eur": self.total_capital_consumed_eur,
            "total_capital_remaining_eur": self.total_capital_remaining_eur,
            "utilization": self.utilization,
            "status": self.status,
            "alert_count": self.alert_count,
        }


@dataclass(frozen=True)
class DeskSnapshot:
    desk_slug: str
    desk_name: str
    base_currency: str
    generated_at: str | None
    total_capital_budget_eur: float
    total_capital_consumed_eur: float
    total_capital_reserved_eur: float
    total_capital_remaining_eur: float
    worst_status: str
    portfolios: list[PortfolioCapitalSlice]

    def to_dict(self) -> dict[str, Any]:
        return {
            "desk_slug": self.desk_slug,
            "desk_name": self.desk_name,
            "base_currency": self.base_currency,
            "generated_at": self.generated_at,
            "total_capital_budget_eur": self.total_capital_budget_eur,
            "total_capital_consumed_eur": self.total_capital_consumed_eur,
            "total_capital_reserved_eur": self.total_capital_reserved_eur,
            "total_capital_remaining_eur": self.total_capital_remaining_eur,
            "worst_status": self.worst_status,
            "portfolios": [item.to_dict() for item in self.portfolios],
        }


def build_desk_snapshot(
    desk: Mapping[str, Any],
    capital_snapshots: Sequence[Mapping[str, Any]],
    portfolios: Mapping[str, Mapping[str, Any]],
    *,
    alerts_by_portfolio: Mapping[str, int] | None = None,
) -> DeskSnapshot:
    alerts_by_portfolio = dict(alerts_by_portfolio or {})
    slices: list[PortfolioCapitalSlice] = []
    total_budget = 0.0
    total_consumed = 0.0
    total_reserved = 0.0
    total_remaining = 0.0
    generated_at = None

    for snapshot in capital_snapshots:
        slug = str(snapshot.get("portfolio_slug", ""))
        portfolio = dict(portfolios.get(slug) or {})
        budget = float(snapshot.get("total_capital_budget_eur", 0.0))
        consumed = float(snapshot.get("total_capital_consumed_eur", 0.0))
        reserved = float(snapshot.get("total_capital_reserved_eur", 0.0))
        remaining = float(snapshot.get("total_capital_remaining_eur", 0.0))
        utilization = None if budget <= 0.0 else float(consumed / budget)
        generated_at = generated_at or snapshot.get("snapshot_timestamp") or snapshot.get("created_at")
        slices.append(
            PortfolioCapitalSlice(
                portfolio_slug=slug,
                portfolio_name=str(portfolio.get("name") or slug),
                reference_model=str(snapshot.get("reference_model") or "hist"),
                total_capital_budget_eur=budget,
                total_capital_consumed_eur=consumed,
                total_capital_remaining_eur=remaining,
                utilization=utilization,
                status=str(snapshot.get("status") or "OK"),
                alert_count=int(alerts_by_portfolio.get(slug, 0)),
            )
        )
        total_budget += budget
        total_consumed += consumed
        total_reserved += reserved
        total_remaining += remaining

    slices.sort(
        key=lambda item: (
            0 if item.status == "BREACH" else 1 if item.status == "WARN" else 2,
            -(item.utilization or 0.0),
            item.portfolio_slug,
        )
    )
    return DeskSnapshot(
        desk_slug=str(desk.get("slug") or "desk"),
        desk_name=str(desk.get("name") or "FX Risk Desk"),
        base_currency=str(desk.get("base_currency") or "EUR"),
        generated_at=generated_at,
        total_capital_budget_eur=float(total_budget),
        total_capital_consumed_eur=float(total_consumed),
        total_capital_reserved_eur=float(total_reserved),
        total_capital_remaining_eur=float(total_remaining),
        worst_status=_worst_status([item.status for item in slices]),
        portfolios=slices,
    )
