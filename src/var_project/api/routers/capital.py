from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    CapitalRebalanceRequest,
    CapitalUsageSnapshotResponse,
)
from var_project.api.service import DeskApiService


router = APIRouter(tags=["capital"])


@router.get("/capital/latest", response_model=CapitalUsageSnapshotResponse)
def latest_capital(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> CapitalUsageSnapshotResponse:
    try:
        capital = service.latest_capital(portfolio_slug=portfolio_slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CapitalUsageSnapshotResponse.model_validate(capital)


@router.get("/capital/history", response_model=list[CapitalUsageSnapshotResponse])
def capital_history(
    limit: int = Query(default=25, ge=1, le=200),
    source: str | None = Query(default=None),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[CapitalUsageSnapshotResponse]:
    return [
        CapitalUsageSnapshotResponse.model_validate(item)
        for item in service.capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)
    ]


@router.post("/capital/rebalance", response_model=CapitalUsageSnapshotResponse)
def rebalance_capital(
    payload: CapitalRebalanceRequest,
    service: DeskApiService = Depends(get_service),
) -> CapitalUsageSnapshotResponse:
    try:
        result = service.rebalance_capital(**payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CapitalUsageSnapshotResponse.model_validate(result)


@router.get("/portfolios/{portfolio_slug}/capital", response_model=CapitalUsageSnapshotResponse)
def portfolio_capital(portfolio_slug: str, service: DeskApiService = Depends(get_service)) -> CapitalUsageSnapshotResponse:
    try:
        result = service.portfolio_capital(portfolio_slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CapitalUsageSnapshotResponse.model_validate(result)
