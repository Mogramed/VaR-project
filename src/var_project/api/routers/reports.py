from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

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
        result = service.run_report(
            compare_path=payload.compare_path,
            portfolio_slug=payload.portfolio_slug,
            account_id=payload.account_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ReportRunResponse.model_validate(result)


@router.get("/reports/latest", response_model=ReportContentResponse)
def latest_report(
    portfolio_slug: str | None = Query(default=None),
    report_id: int | None = Query(default=None, ge=1),
    account_id: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> ReportContentResponse:
    try:
        report = service.latest_report_content(portfolio_slug=portfolio_slug, report_id=report_id, account_id=account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="No report found.")
    return ReportContentResponse.model_validate(report)


@router.get("/reports/charts/{chart_name}")
def latest_report_chart(
    chart_name: str,
    portfolio_slug: str | None = Query(default=None),
    report_id: int | None = Query(default=None, ge=1),
    account_id: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> FileResponse:
    try:
        report = service.latest_report_content(portfolio_slug=portfolio_slug, report_id=report_id, account_id=account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="No report found.")

    normalized_name = Path(str(chart_name)).name
    if normalized_name != chart_name or not normalized_name:
        raise HTTPException(status_code=400, detail="Invalid chart name.")

    report_path = Path(str(report.get("report_markdown") or "")).resolve()
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report markdown is unavailable.")

    allowed_names = {Path(str(path)).name for path in list(report.get("chart_paths") or [])}
    candidate = (report_path.parent / normalized_name).resolve()
    if candidate.parent != report_path.parent:
        raise HTTPException(status_code=400, detail="Invalid chart path.")
    if normalized_name not in allowed_names or not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Report chart '{normalized_name}' not found.")
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
        raise HTTPException(status_code=400, detail="Unsupported chart asset format.")

    return FileResponse(candidate)


@router.get("/reports/decision-history", response_model=list[RiskDecisionResponse])
def report_decision_history(
    limit: int = Query(default=25, ge=1, le=200),
    portfolio_slug: str | None = Query(default=None),
    account_id: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[RiskDecisionResponse]:
    try:
        payload = service.report_decision_history(limit=limit, portfolio_slug=portfolio_slug, account_id=account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [RiskDecisionResponse.model_validate(item) for item in payload]


@router.get("/reports/capital-history", response_model=list[CapitalUsageSnapshotResponse])
def report_capital_history(
    limit: int = Query(default=25, ge=1, le=200),
    source: str | None = Query(default=None),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> list[CapitalUsageSnapshotResponse]:
    return [
        CapitalUsageSnapshotResponse.model_validate(item)
        for item in service.report_capital_history(limit=limit, source=source, portfolio_slug=portfolio_slug)
    ]
