from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    DealHistoryEntryResponse,
    HoldingSnapshotResponse,
    InstrumentDefinitionResponse,
    MarketDataSyncRequest,
    MarketDataSyncStatusResponse,
    MT5AccountSnapshotResponse,
    MT5AnalyticsSeriesResponse,
    MT5LiveEventResponse,
    MT5LiveStateResponse,
    MT5PendingOrderResponse,
    MT5PositionResponse,
    MT5TerminalStatusResponse,
    OrderHistoryEntryResponse,
    PortfolioExposureResponse,
    ReconciliationSummaryResponse,
)
from var_project.api.service import DeskApiService
from var_project.core.exceptions import MT5ConnectionError
from var_project.execution.mt5_bridge import build_empty_live_state


router = APIRouter(tags=["mt5"])
MT5_LIVE_STREAM_WAIT_SECONDS = 5.0
MT5_LIVE_STREAM_RETRY_MS = 5000


def iter_mt5_live_stream(
    service: DeskApiService,
    *,
    portfolio_slug: str | None = None,
    detail_level: Literal["summary", "full", "inspector"] = "full",
):
    sequence = 0
    portfolio = service.runtime._resolve_portfolio_context(portfolio_slug)
    yield f"retry: {MT5_LIVE_STREAM_RETRY_MS}\n\n"
    while True:
        try:
            try:
                events = service.mt5_live_events(
                    portfolio_slug=portfolio_slug,
                    after=sequence,
                    limit=100,
                    wait_seconds=MT5_LIVE_STREAM_WAIT_SECONDS,
                    detail_level=detail_level,
                )
            except TypeError as exc:
                if "detail_level" not in str(exc):
                    raise
                events = service.mt5_live_events(
                    portfolio_slug=portfolio_slug,
                    after=sequence,
                    limit=100,
                    wait_seconds=MT5_LIVE_STREAM_WAIT_SECONDS,
                )
        except Exception as exc:
            sequence += 1
            degraded_state = build_empty_live_state(
                config=service.runtime.mt5_config,
                seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
                status="degraded",
                connected=False,
                degraded=True,
                stale=True,
                last_error=str(exc),
            )
            degraded_state["portfolio_slug"] = portfolio["slug"]
            degraded_state["portfolio_mode"] = portfolio.get("mode")
            degraded_state["generated_at"] = datetime.now(timezone.utc).isoformat()
            payload = {
                "sequence": sequence,
                "kind": "stream_error",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "change_summary": {},
                "state": degraded_state,
            }
            yield (
                f"id: {sequence}\n"
                f"data: {json.dumps(payload)}\n\n"
            )
            continue
        if not events:
            yield ": keep-alive\n\n"
            continue
        for event in events:
            sequence = max(sequence, int(event.get("sequence") or 0))
            yield (
                f"id: {sequence}\n"
                f"data: {json.dumps(event)}\n\n"
            )


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


@router.get("/mt5/live/state", response_model=MT5LiveStateResponse)
def mt5_live_state(
    portfolio_slug: str | None = Query(default=None),
    detail_level: Literal["summary", "full", "inspector"] = Query(default="full"),
    service: DeskApiService = Depends(get_service),
) -> MT5LiveStateResponse:
    try:
        payload = service.mt5_live_state(portfolio_slug=portfolio_slug, detail_level=detail_level)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MT5LiveStateResponse.model_validate(payload)


@router.get("/mt5/analytics/series", response_model=MT5AnalyticsSeriesResponse)
def mt5_analytics_series(
    portfolio_slug: str | None = Query(default=None),
    window_minutes: int = Query(default=240, ge=15, le=10080),
    max_points: int = Query(default=300, ge=50, le=2000),
    service: DeskApiService = Depends(get_service),
) -> MT5AnalyticsSeriesResponse:
    try:
        payload = service.mt5_analytics_series(
            portfolio_slug=portfolio_slug,
            window_minutes=window_minutes,
            max_points=max_points,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MT5AnalyticsSeriesResponse.model_validate(payload)


@router.get("/mt5/live/events", response_model=list[MT5LiveEventResponse])
def mt5_live_events(
    portfolio_slug: str | None = Query(default=None),
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    wait_seconds: float = Query(default=15.0, ge=0.0, le=60.0),
    detail_level: Literal["summary", "full", "inspector"] = Query(default="full"),
    service: DeskApiService = Depends(get_service),
) -> list[MT5LiveEventResponse]:
    try:
        payload = service.mt5_live_events(
            portfolio_slug=portfolio_slug,
            after=after,
            limit=limit,
            wait_seconds=wait_seconds,
            detail_level=detail_level,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [MT5LiveEventResponse.model_validate(item) for item in payload]


@router.get("/mt5/live/stream")
def mt5_live_stream(
    portfolio_slug: str | None = Query(default=None),
    detail_level: Literal["summary", "full", "inspector"] = Query(default="full"),
    service: DeskApiService = Depends(get_service),
) -> StreamingResponse:
    try:
        service.runtime._resolve_portfolio_context(portfolio_slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        iter_mt5_live_stream(service, portfolio_slug=portfolio_slug, detail_level=detail_level),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
