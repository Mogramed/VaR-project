from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from var_project.api.dependencies import get_service
from var_project.api.schemas import DeskDefinitionResponse, DeskSnapshotResponse
from var_project.api.service import DeskApiService


router = APIRouter(tags=["desk"])


@router.get("/desks", response_model=list[DeskDefinitionResponse])
def list_desks(service: DeskApiService = Depends(get_service)) -> list[DeskDefinitionResponse]:
    return [DeskDefinitionResponse.model_validate(item) for item in service.list_desks()]


@router.get("/desks/{desk_slug}/overview", response_model=DeskSnapshotResponse)
def desk_overview(desk_slug: str, service: DeskApiService = Depends(get_service)) -> DeskSnapshotResponse:
    try:
        overview = service.desk_overview(desk_slug=desk_slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeskSnapshotResponse.model_validate(overview)
