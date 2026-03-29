from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    ArtifactSummary,
    BacktestFrameResponse,
    BacktestRunResponse,
    BacktestRunSummary,
    ModelComparisonResponse,
    RiskAttributionResponse,
    RiskBudgetResponse,
    RunBacktestRequest,
    RunSnapshotRequest,
    SnapshotRunResponse,
    SnapshotSummary,
    ValidationRunSummary,
)
from var_project.api.service import DeskApiService


router = APIRouter(tags=["analytics"])


@router.get("/snapshots/latest", response_model=SnapshotSummary)
def latest_snapshot(
    source: str | None = Query(default=None),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> SnapshotSummary:
    snapshot = service.latest_snapshot(source=source, portfolio_slug=portfolio_slug)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No snapshot found.")
    return SnapshotSummary.model_validate(snapshot)


@router.post("/snapshots/run", response_model=SnapshotRunResponse)
def run_snapshot(payload: RunSnapshotRequest, service: DeskApiService = Depends(get_service)) -> SnapshotRunResponse:
    try:
        result = service.run_snapshot(**payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SnapshotRunResponse.model_validate(result)


@router.get("/backtests/latest", response_model=BacktestRunSummary)
def latest_backtest(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> BacktestRunSummary:
    backtest = service.latest_backtest(portfolio_slug=portfolio_slug)
    if backtest is None:
        raise HTTPException(status_code=404, detail="No backtest run found.")
    return BacktestRunSummary.model_validate(backtest)


@router.post("/backtests/run", response_model=BacktestRunResponse)
def run_backtest(payload: RunBacktestRequest, service: DeskApiService = Depends(get_service)) -> BacktestRunResponse:
    try:
        result = service.run_backtest(**payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BacktestRunResponse.model_validate(result)


@router.get("/backtests/frame/latest", response_model=BacktestFrameResponse)
def latest_backtest_frame(
    limit: int = Query(default=400, ge=1, le=5000),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> BacktestFrameResponse:
    frame = service.latest_backtest_frame(limit=limit, portfolio_slug=portfolio_slug)
    if frame is None:
        raise HTTPException(status_code=404, detail="No backtest frame found.")
    return BacktestFrameResponse.model_validate(frame)


@router.get("/validations/latest", response_model=ValidationRunSummary)
def latest_validation(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> ValidationRunSummary:
    validation = service.latest_validation(portfolio_slug=portfolio_slug)
    if validation is None:
        raise HTTPException(status_code=404, detail="No validation run found.")
    return ValidationRunSummary.model_validate(validation)


@router.get("/models/compare/latest", response_model=ModelComparisonResponse)
def latest_model_comparison(
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> ModelComparisonResponse:
    comparison = service.latest_model_comparison(portfolio_slug=portfolio_slug)
    if comparison is None:
        raise HTTPException(status_code=404, detail="No model comparison available.")
    return ModelComparisonResponse.model_validate(comparison)


@router.get("/snapshots/attribution/latest", response_model=RiskAttributionResponse)
def latest_risk_attribution(
    source: str = Query(default="historical"),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> RiskAttributionResponse:
    attribution = service.latest_risk_attribution(source=source, portfolio_slug=portfolio_slug)
    if attribution is None:
        raise HTTPException(status_code=404, detail="No risk attribution available.")
    return RiskAttributionResponse.model_validate(attribution)


@router.get("/snapshots/budget/latest", response_model=RiskBudgetResponse)
def latest_risk_budget(
    source: str = Query(default="historical"),
    portfolio_slug: str | None = Query(default=None),
    service: DeskApiService = Depends(get_service),
) -> RiskBudgetResponse:
    budget = service.latest_risk_budget(source=source, portfolio_slug=portfolio_slug)
    if budget is None:
        raise HTTPException(status_code=404, detail="No risk budget available.")
    return RiskBudgetResponse.model_validate(budget)


@router.get("/artifacts/latest/{artifact_type}", response_model=ArtifactSummary)
def latest_artifact(artifact_type: str, service: DeskApiService = Depends(get_service)) -> ArtifactSummary:
    artifact = service.latest_artifact(artifact_type)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"No artifact found for type '{artifact_type}'.")
    return ArtifactSummary.model_validate(artifact)
