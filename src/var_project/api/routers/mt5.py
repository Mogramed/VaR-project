from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    DealHistoryEntryResponse,
    HoldingSnapshotResponse,
    InstrumentDefinitionResponse,
    MarketDataSyncRequest,
    MarketDataSyncStatusResponse,
    MT5AccountSnapshotResponse,
    MT5PendingOrderResponse,
    MT5PositionResponse,
    MT5TerminalStatusResponse,
    OrderHistoryEntryResponse,
    PortfolioExposureResponse,
    ReconciliationSummaryResponse,
)
from var_project.api.service import DeskApiService
from var_project.core.exceptions import MT5ConnectionError


router = APIRouter(tags=["mt5"])


@router.get("/mt5/status", response_model=MT5TerminalStatusResponse)
def mt5_status(service: DeskApiService = Depends(get_service)) -> MT5TerminalStatusResponse:
    return MT5TerminalStatusResponse.model_validate(service.mt5_status())


@router.get("/mt5/account", response_model=MT5AccountSnapshotResponse)
def mt5_account(service: DeskApiService = Depends(get_service)) -> MT5AccountSnapshotResponse:
    try:
        payload = service.mt5_account()
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MT5AccountSnapshotResponse.model_validate(payload)


@router.get("/mt5/positions", response_model=list[MT5PositionResponse])
def mt5_positions(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[MT5PositionResponse]:
    try:
        payload = service.mt5_positions(portfolio_slug=portfolio_slug)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [MT5PositionResponse.model_validate(item) for item in payload]


@router.get("/mt5/orders", response_model=list[MT5PendingOrderResponse])
def mt5_orders(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[MT5PendingOrderResponse]:
    try:
        payload = service.mt5_orders(portfolio_slug=portfolio_slug)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [MT5PendingOrderResponse.model_validate(item) for item in payload]


@router.get("/mt5/history/orders", response_model=list[OrderHistoryEntryResponse])
def mt5_history_orders(
    limit: int = Query(default=100, ge=1, le=500),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[OrderHistoryEntryResponse]:
    try:
        payload = service.mt5_history_orders(portfolio_slug=portfolio_slug, limit=limit)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [OrderHistoryEntryResponse.model_validate(item) for item in payload]


@router.get("/mt5/history/deals", response_model=list[DealHistoryEntryResponse])
def mt5_history_deals(
    limit: int = Query(default=100, ge=1, le=500),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[DealHistoryEntryResponse]:
    try:
        payload = service.mt5_history_deals(portfolio_slug=portfolio_slug, limit=limit)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [DealHistoryEntryResponse.model_validate(item) for item in payload]


@router.get("/market-data/status", response_model=MarketDataSyncStatusResponse)
def market_data_status(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> MarketDataSyncStatusResponse:
    try:
        payload = service.market_data_status(portfolio_slug=portfolio_slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MarketDataSyncStatusResponse.model_validate(payload)


@router.post("/market-data/sync", response_model=MarketDataSyncStatusResponse)
def sync_market_data(
    payload: MarketDataSyncRequest,
    service: DeskApiService = Depends(get_service),
) -> MarketDataSyncStatusResponse:
    try:
        result = service.sync_market_data(**payload.model_dump())
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MarketDataSyncStatusResponse.model_validate(result)


@router.get("/instruments", response_model=list[InstrumentDefinitionResponse])
def instruments(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[InstrumentDefinitionResponse]:
    try:
        payload = service.list_instruments(portfolio_slug=portfolio_slug)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [InstrumentDefinitionResponse.model_validate(item) for item in payload]


@router.get("/portfolio/live-holdings", response_model=list[HoldingSnapshotResponse])
def portfolio_live_holdings(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[HoldingSnapshotResponse]:
    try:
        payload = service.live_holdings(portfolio_slug=portfolio_slug)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [HoldingSnapshotResponse.model_validate(item) for item in payload]


@router.get("/portfolio/live-exposure", response_model=PortfolioExposureResponse)
def portfolio_live_exposure(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> PortfolioExposureResponse:
    try:
        payload = service.live_exposure(portfolio_slug=portfolio_slug)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PortfolioExposureResponse.model_validate(payload)


@router.get("/reconciliation/summary", response_model=ReconciliationSummaryResponse)
def reconciliation_summary(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> ReconciliationSummaryResponse:
    try:
        payload = service.reconciliation_summary(portfolio_slug=portfolio_slug)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ReconciliationSummaryResponse.model_validate(payload)
