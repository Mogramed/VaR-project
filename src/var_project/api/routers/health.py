from __future__ import annotations

from fastapi import APIRouter, Depends

from var_project.api.dependencies import get_service
from var_project.api.schemas import HealthResponse, PortfolioSummary, WorkerStatusResponse
from var_project.api.service import DeskApiService


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(service: DeskApiService = Depends(get_service)) -> HealthResponse:
    return HealthResponse.model_validate(service.health())


@router.get("/jobs/status", response_model=WorkerStatusResponse)
def jobs_status(service: DeskApiService = Depends(get_service)) -> WorkerStatusResponse:
    return WorkerStatusResponse.model_validate(service.jobs_status())


@router.get("/portfolios", response_model=list[PortfolioSummary])
def list_portfolios(service: DeskApiService = Depends(get_service)) -> list[PortfolioSummary]:
    return [PortfolioSummary.model_validate(item) for item in service.list_portfolios()]
