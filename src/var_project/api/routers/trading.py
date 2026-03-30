from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    AlertSummary,
    AuditEventResponse,
    ExecutionFillResponse,
    ExecutionPreviewResponse,
    ExecutionRequest,
    ExecutionResultResponse,
    ReconciliationAcknowledgeRequest,
    ReconciliationAcknowledgeResponse,
    RiskDecisionResponse,
    TradeProposalRequest,
)
from var_project.api.service import DeskApiService
from var_project.core.exceptions import MT5ConnectionError


router = APIRouter(tags=["trading"])


@router.get("/alerts", response_model=list[AlertSummary])
def recent_alerts(limit: int = Query(default=25, ge=1, le=200), service: DeskApiService = Depends(get_service)) -> list[AlertSummary]:
    return [AlertSummary.model_validate(item) for item in service.recent_alerts(limit=limit)]


@router.post("/decisions/evaluate", response_model=RiskDecisionResponse)
def evaluate_trade_decision(payload: TradeProposalRequest, service: DeskApiService = Depends(get_service)) -> RiskDecisionResponse:
    try:
        result = service.evaluate_trade_decision(**payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RiskDecisionResponse.model_validate(result)


@router.get("/decisions/recent", response_model=list[RiskDecisionResponse])
def recent_decisions(
    limit: int = Query(default=20, ge=1, le=200),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[RiskDecisionResponse]:
    return [
        RiskDecisionResponse.model_validate(item)
        for item in service.recent_decisions(limit=limit, portfolio_slug=portfolio_slug)
    ]


@router.post("/execution/preview", response_model=ExecutionPreviewResponse)
def execution_preview(payload: ExecutionRequest, service: DeskApiService = Depends(get_service)) -> ExecutionPreviewResponse:
    try:
        result = service.preview_execution(**payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExecutionPreviewResponse.model_validate(result)


@router.post("/execution/submit", response_model=ExecutionResultResponse)
def execution_submit(payload: ExecutionRequest, service: DeskApiService = Depends(get_service)) -> ExecutionResultResponse:
    try:
        result = service.submit_execution(**payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MT5ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExecutionResultResponse.model_validate(result)


@router.get("/execution/recent", response_model=list[ExecutionResultResponse])
def recent_execution_results(
    limit: int = Query(default=20, ge=1, le=200),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[ExecutionResultResponse]:
    return [
        ExecutionResultResponse.model_validate(item)
        for item in service.recent_execution_results(limit=limit, portfolio_slug=portfolio_slug)
    ]


@router.get("/execution/fills/recent", response_model=list[ExecutionFillResponse])
def recent_execution_fills(
    limit: int = Query(default=50, ge=1, le=500),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[ExecutionFillResponse]:
    return [
        ExecutionFillResponse.model_validate(item)
        for item in service.recent_execution_fills(limit=limit, portfolio_slug=portfolio_slug)
    ]


@router.post("/reconciliation/acknowledge", response_model=ReconciliationAcknowledgeResponse)
def acknowledge_reconciliation(
    payload: ReconciliationAcknowledgeRequest,
    service: DeskApiService = Depends(get_service),
) -> ReconciliationAcknowledgeResponse:
    result = service.acknowledge_reconciliation_mismatch(
        portfolio_slug=payload.portfolio_slug,
        symbol=payload.symbol,
        reason=payload.reason,
        operator_note=payload.operator_note,
    )
    return ReconciliationAcknowledgeResponse.model_validate(result)


@router.get("/audit/recent", response_model=list[AuditEventResponse])
def recent_audit(
    limit: int = Query(default=50, ge=1, le=500),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[AuditEventResponse]:
    return [
        AuditEventResponse.model_validate(item)
        for item in service.recent_audit_events(limit=limit, portfolio_slug=portfolio_slug)
    ]
