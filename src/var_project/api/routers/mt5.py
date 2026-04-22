from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Literal, Mapping

from fastapi import APIRouter, Depends, HTTPException, Header, Query, Response
from fastapi.responses import StreamingResponse

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    DealHistoryEntryResponse,
    HoldingSnapshotResponse,
    InstrumentDefinitionResponse,
    MarketDataSyncRequest,
    MarketDataSyncRunResponse,
    MarketDataSyncStatusResponse,
    MT5AccountSnapshotResponse,
    MT5AccountsResponse,
    MT5AnalyticsSeriesResponse,
    MT5LiveEventResponse,
    MT5LiveStateResponse,
    MT5PendingOrderResponse,
    MT5PositionResponse,
    MT5TransactionHistoryResponse,
    MT5TerminalStatusResponse,
    OrderHistoryEntryResponse,
    PortfolioExposureResponse,
    ReconciliationHistoryEntryResponse,
    ReconciliationSummaryResponse,
)
from var_project.api.service import DeskApiService
from var_project.core.exceptions import MT5ConnectionError
from var_project.execution.mt5_bridge import build_empty_live_state


router = APIRouter(tags=["mt5"])
MT5_LIVE_STREAM_WAIT_SECONDS = 5.0
MT5_LIVE_STREAM_RETRY_MS = 5000


def _effective_detail_level(
    detail_level: Literal["summary", "full", "inspector"],
) -> Literal["summary", "full"]:
    return "summary" if detail_level == "summary" else "full"


def _build_live_state_etag(payload: Mapping[str, object], *, detail_level: str) -> str:
    signature = json.dumps(
        {
            "detail_level": str(detail_level),
            "portfolio_slug": payload.get("portfolio_slug"),
            "account_id": payload.get("account_id"),
            "sequence": int(payload.get("sequence") or 0),
            "generated_at": payload.get("generated_at"),
            "last_success_at": payload.get("last_success_at"),
            "status": payload.get("status"),
            "connected": bool(payload.get("connected", False)),
            "degraded": bool(payload.get("degraded", False)),
            "stale": bool(payload.get("stale", False)),
            "truth_score": payload.get("truth_score"),
            "bridge_consecutive_failures": payload.get("bridge_consecutive_failures"),
            "bridge_last_event_kind": payload.get("bridge_last_event_kind"),
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return f'W/"{digest}"'


def _if_none_match_matches(*, if_none_match: str | None, etag: str) -> bool:
    if if_none_match in {None, ""}:
        return False
    header = str(if_none_match)
    candidates = [item.strip() for item in header.split(",") if item.strip()]
    if "*" in candidates:
        return True
    target = str(etag).strip()
    target_weak = target.removeprefix("W/").strip()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized == target:
            return True
        if normalized.removeprefix("W/").strip() == target_weak:
            return True
    return False


def _build_live_state_headers(
    *,
    payload: Mapping[str, object],
    etag: str,
    detail_level: str,
) -> dict[str, str]:
    sequence = int(payload.get("sequence") or 0)
    health = dict(payload.get("health") or {})
    next_poll = health.get("bridge_next_poll_delay_seconds")
    if next_poll in {None, "", "null"}:
        next_poll = payload.get("bridge_next_poll_delay_seconds")
    if next_poll in {None, "", "null"}:
        next_poll = payload.get("poll_interval_seconds")
    try:
        next_poll_seconds = max(float(next_poll), 0.1)
    except (TypeError, ValueError):
        next_poll_seconds = 1.0
    health_status = str(health.get("status") or payload.get("status") or "unknown")
    return {
        "ETag": etag,
        "Cache-Control": "no-cache, no-transform",
        "X-Live-Sequence": str(sequence),
        "X-Live-Detail-Level": str(detail_level),
        "X-Live-Health-Status": health_status,
        "X-Live-Next-Poll-Seconds": f"{next_poll_seconds:.3f}",
    }


def _resolve_stream_after_sequence(*, after: int | None, last_event_id: str | None) -> int:
    if after is not None:
        return max(int(after), 0)
    if last_event_id in {None, ""}:
        return 0
    try:
        parsed = int(str(last_event_id).strip())
    except ValueError:
        return 0
    return max(parsed, 0)


def _build_stream_error_payload(
    *,
    sequence: int,
    service: DeskApiService,
    portfolio: dict[str, object],
    account_id: str | None,
    error: Exception,
) -> dict[str, object]:
    detail = str(error)
    try:
        resolved_account_id = service.runtime.resolve_mt5_account_id(account_id)
    except ValueError:
        resolved_account_id = service.runtime.resolve_mt5_account_id(None)
    account_config = service.runtime.mt5_config_for_account(resolved_account_id)
    degraded_state = build_empty_live_state(
        config=account_config,
        seed_symbols=portfolio.get("watchlist_symbols") or portfolio["symbols"],
        status="degraded",
        connected=False,
        degraded=True,
        stale=True,
        last_error=detail,
    )
    degraded_state["account_id"] = resolved_account_id
    degraded_state["portfolio_slug"] = portfolio["slug"]
    degraded_state["portfolio_mode"] = portfolio.get("mode")
    degraded_state["generated_at"] = datetime.now(timezone.utc).isoformat()
    degraded_state["health"] = {
        "status": "offline",
        "message": f"Live stream backend unavailable. Last error: {detail}",
        "connected": False,
        "degraded": True,
        "stale": True,
        "market_closed": bool(degraded_state.get("market_closed", False)),
        "analytics_stale": True,
        "generated_age_seconds": 0.0,
        "last_success_age_seconds": None,
        "tick_quality_status": "incomplete",
        "nowcast_status": "degraded",
        "operational_truth": None,
        "truth_score": None,
        "error_retryable": True,
        "last_error": detail,
    }
    return {
        "sequence": sequence,
        "kind": "stream_error",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "change_summary": {},
        "state": degraded_state,
    }


def iter_mt5_live_stream(
    service: DeskApiService,
    *,
    portfolio_slug: str | None = None,
    detail_level: Literal["summary", "full", "inspector"] = "full",
    account_id: str | None = None,
    after: int = 0,
    emit_bootstrap: bool = False,
):
    sequence = max(int(after), 0)
    consecutive_failures = 0
    portfolio = service.runtime._resolve_portfolio_context(portfolio_slug)

    def _call_live_state() -> dict[str, object]:
        call_kwargs: dict[str, object] = {
            "portfolio_slug": portfolio_slug,
            "detail_level": detail_level,
        }
        if account_id not in {None, "", "null"}:
            call_kwargs["account_id"] = account_id
        try:
            return service.mt5_live_state(**call_kwargs)
        except TypeError as exc:
            message = str(exc)
            if "detail_level" not in message and "account_id" not in message:
                raise
            if "detail_level" in message and "account_id" in message:
                return service.mt5_live_state(portfolio_slug=portfolio_slug)
            if "detail_level" in message:
                if "account_id" in call_kwargs:
                    return service.mt5_live_state(
                        portfolio_slug=portfolio_slug,
                        account_id=account_id,
                    )
                return service.mt5_live_state(portfolio_slug=portfolio_slug)
            return service.mt5_live_state(
                portfolio_slug=portfolio_slug,
                detail_level=detail_level,
            )

    def _call_live_events() -> list[dict[str, object]]:
        call_kwargs: dict[str, object] = {
            "portfolio_slug": portfolio_slug,
            "after": sequence,
            "limit": 100,
            "wait_seconds": MT5_LIVE_STREAM_WAIT_SECONDS,
            "detail_level": detail_level,
        }
        if account_id not in {None, "", "null"}:
            call_kwargs["account_id"] = account_id
        try:
            return service.mt5_live_events(**call_kwargs)
        except TypeError as exc:
            message = str(exc)
            if "detail_level" not in message and "account_id" not in message:
                raise
            if "detail_level" in message and "account_id" in message:
                return service.mt5_live_events(
                    portfolio_slug=portfolio_slug,
                    after=sequence,
                    limit=100,
                    wait_seconds=MT5_LIVE_STREAM_WAIT_SECONDS,
                )
            if "detail_level" in message:
                if "account_id" in call_kwargs:
                    return service.mt5_live_events(
                        portfolio_slug=portfolio_slug,
                        after=sequence,
                        limit=100,
                        wait_seconds=MT5_LIVE_STREAM_WAIT_SECONDS,
                        account_id=account_id,
                    )
                return service.mt5_live_events(
                    portfolio_slug=portfolio_slug,
                    after=sequence,
                    limit=100,
                    wait_seconds=MT5_LIVE_STREAM_WAIT_SECONDS,
                )
            return service.mt5_live_events(
                portfolio_slug=portfolio_slug,
                after=sequence,
                limit=100,
                wait_seconds=MT5_LIVE_STREAM_WAIT_SECONDS,
                detail_level=detail_level,
            )

    yield f"retry: {MT5_LIVE_STREAM_RETRY_MS}\n\n"
    if emit_bootstrap:
        try:
            state = _call_live_state()
            bootstrap_sequence = max(sequence + 1, int(state.get("sequence") or 0))
            state = dict(state)
            state["sequence"] = bootstrap_sequence
            payload = {
                "sequence": bootstrap_sequence,
                "kind": "snapshot",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "change_summary": {},
                "state": state,
            }
            sequence = bootstrap_sequence
            yield (
                f"id: {sequence}\n"
                f"data: {json.dumps(payload)}\n\n"
            )
        except Exception as exc:
            sequence += 1
            payload = _build_stream_error_payload(
                sequence=sequence,
                service=service,
                portfolio=portfolio,
                account_id=account_id,
                error=exc,
            )
            yield (
                f"id: {sequence}\n"
                f"data: {json.dumps(payload)}\n\n"
            )
    while True:
        try:
            events = _call_live_events()
        except Exception as exc:
            sequence += 1
            payload = _build_stream_error_payload(
                sequence=sequence,
                service=service,
                portfolio=portfolio,
                account_id=account_id,
                error=exc,
            )
            yield (
                f"id: {sequence}\n"
                f"data: {json.dumps(payload)}\n\n"
            )
            consecutive_failures += 1
            sleep_seconds = min(
                MT5_LIVE_STREAM_WAIT_SECONDS,
                max(0.25, 0.5 * (2 ** min(consecutive_failures - 1, 4))),
            )
            time.sleep(float(sleep_seconds))
            continue
        consecutive_failures = 0
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
def mt5_status(
    account_id: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> MT5TerminalStatusResponse:
    try:
        payload = service.mt5_status(account_id=account_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MT5TerminalStatusResponse.model_validate(payload)


@router.get("/mt5/account", response_model=MT5AccountSnapshotResponse)
def mt5_account(
    account_id: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> MT5AccountSnapshotResponse:
    try:
        payload = service.mt5_account(account_id=account_id)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MT5AccountSnapshotResponse.model_validate(payload)


@router.get("/mt5/accounts", response_model=MT5AccountsResponse)
def mt5_accounts(service: DeskApiService = Depends(get_service)) -> MT5AccountsResponse:
    payload = service.mt5_accounts()
    return MT5AccountsResponse.model_validate(payload)


@router.get("/mt5/positions", response_model=list[MT5PositionResponse])
def mt5_positions(
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[MT5PositionResponse]:
    try:
        payload = service.mt5_positions(portfolio_slug=portfolio_slug, account_id=account_id)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [MT5PositionResponse.model_validate(item) for item in payload]


@router.get("/mt5/orders", response_model=list[MT5PendingOrderResponse])
def mt5_orders(
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[MT5PendingOrderResponse]:
    try:
        payload = service.mt5_orders(portfolio_slug=portfolio_slug, account_id=account_id)
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [MT5PendingOrderResponse.model_validate(item) for item in payload]


@router.get("/mt5/live/state", response_model=MT5LiveStateResponse)
def mt5_live_state(
    response: Response,
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    detail_level: Literal["summary", "full", "inspector"] = Query(default="full"),
    if_none_match: str | None = Header(default=None, alias="If-None-Match"),
    service: DeskApiService = Depends(get_service),
) -> MT5LiveStateResponse | Response:
    effective_detail_level = _effective_detail_level(detail_level)
    try:
        payload = service.mt5_live_state(
            portfolio_slug=portfolio_slug,
            detail_level=effective_detail_level,
            account_id=account_id,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload_dict = dict(payload)
    etag = _build_live_state_etag(payload_dict, detail_level=detail_level)
    headers = _build_live_state_headers(payload=payload_dict, etag=etag, detail_level=detail_level)
    if _if_none_match_matches(if_none_match=if_none_match, etag=etag):
        return Response(status_code=304, headers=headers)
    if response is not None:
        for key, value in headers.items():
            response.headers[key] = value
    return MT5LiveStateResponse.model_validate(payload_dict)


@router.get("/mt5/analytics/series", response_model=MT5AnalyticsSeriesResponse)
def mt5_analytics_series(
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    window_minutes: int = Query(default=240, ge=15, le=10080),
    max_points: int = Query(default=300, ge=50, le=2000),
    service: DeskApiService = Depends(get_service),
) -> MT5AnalyticsSeriesResponse:
    try:
        payload = service.mt5_analytics_series(
            portfolio_slug=portfolio_slug,
            window_minutes=window_minutes,
            max_points=max_points,
            account_id=account_id,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MT5AnalyticsSeriesResponse.model_validate(payload)


@router.get("/mt5/live/events", response_model=list[MT5LiveEventResponse])
def mt5_live_events(
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    wait_seconds: float = Query(default=15.0, ge=0.0, le=60.0),
    detail_level: Literal["summary", "full", "inspector"] = Query(default="full"),
    service: DeskApiService = Depends(get_service),
) -> list[MT5LiveEventResponse]:
    effective_detail_level = _effective_detail_level(detail_level)
    try:
        payload = service.mt5_live_events(
            portfolio_slug=portfolio_slug,
            after=after,
            limit=limit,
            wait_seconds=wait_seconds,
            detail_level=effective_detail_level,
            account_id=account_id,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [MT5LiveEventResponse.model_validate(item) for item in payload]


@router.get("/mt5/live/stream")
def mt5_live_stream(
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    detail_level: Literal["summary", "full", "inspector"] = Query(default="full"),
    after: int | None = Query(default=None, ge=0),
    bootstrap: bool = Query(default=True),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    service: DeskApiService = Depends(get_service),
) -> StreamingResponse:
    effective_detail_level = _effective_detail_level(detail_level)
    try:
        service.runtime._resolve_portfolio_context(portfolio_slug)
        service.runtime.resolve_mt5_account_id(account_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    resume_after = _resolve_stream_after_sequence(after=after, last_event_id=last_event_id)
    return StreamingResponse(
        iter_mt5_live_stream(
            service,
            portfolio_slug=portfolio_slug,
            detail_level=effective_detail_level,
            account_id=account_id,
            after=resume_after,
            emit_bootstrap=bool(bootstrap),
        ),
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
    account_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None, description="UTC start datetime (ISO-8601)."),
    date_to: str | None = Query(default=None, description="UTC end datetime (ISO-8601)."),
    symbol: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[OrderHistoryEntryResponse]:
    try:
        payload = service.mt5_history_orders(
            portfolio_slug=portfolio_slug,
            limit=limit,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [OrderHistoryEntryResponse.model_validate(item) for item in payload]


@router.get("/mt5/history/deals", response_model=list[DealHistoryEntryResponse])
def mt5_history_deals(
    limit: int = Query(default=100, ge=1, le=500),
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None, description="UTC start datetime (ISO-8601)."),
    date_to: str | None = Query(default=None, description="UTC end datetime (ISO-8601)."),
    symbol: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[DealHistoryEntryResponse]:
    try:
        payload = service.mt5_history_deals(
            portfolio_slug=portfolio_slug,
            limit=limit,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [DealHistoryEntryResponse.model_validate(item) for item in payload]


@router.get("/mt5/history/transactions", response_model=MT5TransactionHistoryResponse)
def mt5_history_transactions(
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None, description="UTC start datetime (ISO-8601)."),
    date_to: str | None = Query(default=None, description="UTC end datetime (ISO-8601)."),
    symbol: str | None = Query(default=None),
    type: str | None = Query(default=None, description="all|order|deal|manual|desk"),
    sort: str = Query(default="time_desc", description="time_desc|time_asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    service: DeskApiService = Depends(get_service),
) -> MT5TransactionHistoryResponse:
    try:
        payload = service.mt5_transaction_history(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
            type=type,
            sort=sort,
            page=page,
            page_size=page_size,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MT5TransactionHistoryResponse.model_validate(payload)


@router.get("/mt5/history/transactions/export")
def mt5_history_transactions_export(
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None, description="UTC start datetime (ISO-8601)."),
    date_to: str | None = Query(default=None, description="UTC end datetime (ISO-8601)."),
    symbol: str | None = Query(default=None),
    type: str | None = Query(default=None, description="all|order|deal|manual|desk"),
    sort: str = Query(default="time_desc", description="time_desc|time_asc"),
    max_rows: int = Query(default=5000, ge=1, le=10000),
    service: DeskApiService = Depends(get_service),
) -> Response:
    try:
        filename, payload = service.mt5_transaction_history_csv(
            portfolio_slug=portfolio_slug,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
            type=type,
            sort=sort,
            max_rows=max_rows,
        )
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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


@router.get("/market-data/sync/runs", response_model=list[MarketDataSyncRunResponse])
def list_market_data_sync_runs(
    portfolio_slug: str | None = Query(default=None, description="Portfolio slug. Defaults to active portfolio."),
    status: list[str] | None = Query(
        default=None,
        description="Optional status filter. Repeat query param for multiple values (e.g. status=ok&status=incomplete).",
    ),
    limit: int = Query(default=25, ge=1, le=200, description="Maximum number of runs to return."),
    service: DeskApiService = Depends(get_service),
) -> list[MarketDataSyncRunResponse]:
    try:
        payload = service.market_data_sync_runs(
            portfolio_slug=portfolio_slug,
            status=status,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [MarketDataSyncRunResponse.model_validate(item) for item in payload]


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


@router.get("/reconciliation/history", response_model=list[ReconciliationHistoryEntryResponse])
def reconciliation_history(
    portfolio_slug: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    service: DeskApiService = Depends(get_service),
) -> list[ReconciliationHistoryEntryResponse]:
    try:
        payload = service.reconciliation_history(
            portfolio_slug=portfolio_slug,
            severity=severity,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [ReconciliationHistoryEntryResponse.model_validate(item) for item in payload]
