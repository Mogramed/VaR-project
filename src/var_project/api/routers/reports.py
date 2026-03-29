from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    CapitalUsageSnapshotResponse,
    ReportContentResponse,
    ReportRunResponse,
    RiskDecisionResponse,
    RunReportRequest,
)
from var_project.api.service import DeskApiService


router = APIRouter(tags=["reports"])


@router.post("/reports/run", response_model=ReportRunResponse)
def run_report(payload: RunReportRequest, service: DeskApiService = Depends(get_service)) -> ReportRunResponse:
    try:
        result = service.run_report(compare_path=payload.compare_path, portfolio_slug=payload.portfolio_slug)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ReportRunResponse.model_validate(result)


@router.get("/reports/latest", response_model=ReportContentResponse)
def latest_report(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> ReportContentResponse:
    report = service.latest_report_content(portfolio_slug=portfolio_slug)
    if report is None:
        raise HTTPException(status_code=404, detail="No report found.")
    return ReportContentResponse.model_validate(report)


@router.get("/reports/decision-history", response_model=list[RiskDecisionResponse])
def report_decision_history(
    limit: int = Query(default=25, ge=1, le=200),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[RiskDecisionResponse]:
    return [
        RiskDecisionResponse.model_validate(item)
        for item in service.report_decision_history(limit=limit, portfolio_slug=portfolio_slug)
    ]


@router.get("/reports/capital-history", response_model=list[CapitalUsageSnapshotResponse])
def report_capital_history(
    limit: int = Query(default=25, ge=1, le=200),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[CapitalUsageSnapshotResponse]:
    return [
        CapitalUsageSnapshotResponse.model_validate(item)
        for item in service.report_capital_history(limit=limit, portfolio_slug=portfolio_slug)
    ]
