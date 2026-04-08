from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.exc import OperationalError, ProgrammingError

from var_project.api.dependencies import get_service
from var_project.api.schemas import (
    MarketDataSyncRequest,
    OperatorRunResponse,
    RunBacktestRequest,
    RunReportRequest,
    RunSnapshotRequest,
)
from var_project.api.service import DeskApiService
from var_project.jobs.operator_queue import dispatch_operator_run, load_operator_queue_settings
from var_project.storage.serialization import utcnow


router = APIRouter(tags=["operator"])


def _raise_storage_not_ready(exc: Exception) -> None:
    raise HTTPException(
        status_code=503,
        detail={
            "detail": "Operator run storage is not ready. Run database migrations and retry.",
            "error_code": "operator_storage_not_ready",
            "hint": "Run `var-project db upgrade` then retry.",
        },
    ) from exc


def _enqueue_operator_run(
    *,
    action: str,
    payload: dict,
    background_tasks: BackgroundTasks,
    service: DeskApiService,
) -> OperatorRunResponse:
    queue_settings = load_operator_queue_settings()
    try:
        run = service.enqueue_operator_action(action=action, request_payload=payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": str(exc),
                "error_code": "invalid_request",
                "hint": "Check portfolio_slug, timeframe and days fields before retrying.",
            },
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "detail": str(exc),
                "error_code": "runtime_error",
                "hint": "Inspect backend logs and database migrations, then retry.",
            },
        ) from exc
    except (OperationalError, ProgrammingError) as exc:
        _raise_storage_not_ready(exc)

    if not bool(run.get("reused")) and str(run.get("status") or "").lower() in {"queued", "running"}:
        dispatch = None
        dispatch_error: Exception | None = None
        try:
            dispatch = dispatch_operator_run(
                run_id=int(run["id"]),
                action=action,
                repo_root=service.root,
            )
        except Exception as exc:
            dispatch_error = exc
            dispatch = None
        if dispatch is not None and dispatch.get("task_id"):
            updated = service.storage.update_operator_run(
                int(run["id"]),
                queue_task_id=str(dispatch["task_id"]),
            )
            if updated is not None:
                run = updated
        if dispatch is None and queue_settings.mode == "celery":
            failed = service.storage.update_operator_run(
                int(run["id"]),
                status="failed",
                stage="failed",
                error_code="queue_dispatch_failed",
                error_message=(
                    str(dispatch_error)
                    if dispatch_error is not None
                    else "Operator dispatch failed because no Celery worker is available."
                ),
                hint=(
                    "Start `var-project operator-worker` (or the celery-worker container) and retry."
                ),
                finished_at=utcnow(),
            )
            if failed is not None:
                run = failed
            raise HTTPException(
                status_code=503,
                detail={
                    "detail": "Failed to dispatch operator run to Celery worker.",
                    "error_code": "queue_dispatch_failed",
                    "hint": "Ensure celery-worker is running and Redis is reachable, then retry.",
                    "run_id": run.get("id"),
                    "request_id": run.get("request_id"),
                },
            ) from dispatch_error
        if dispatch is None:
            background_tasks.add_task(service.process_operator_run, int(run["id"]))
    return OperatorRunResponse.model_validate(run)


@router.post(
    "/operator/actions/sync",
    response_model=OperatorRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def enqueue_sync(
    payload: MarketDataSyncRequest,
    background_tasks: BackgroundTasks,
    service: DeskApiService = Depends(get_service),
) -> OperatorRunResponse:
    return _enqueue_operator_run(
        action="sync",
        payload=payload.model_dump(),
        background_tasks=background_tasks,
        service=service,
    )


@router.post(
    "/operator/actions/snapshot",
    response_model=OperatorRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def enqueue_snapshot(
    payload: RunSnapshotRequest,
    background_tasks: BackgroundTasks,
    service: DeskApiService = Depends(get_service),
) -> OperatorRunResponse:
    return _enqueue_operator_run(
        action="snapshot",
        payload=payload.model_dump(),
        background_tasks=background_tasks,
        service=service,
    )


@router.post(
    "/operator/actions/backtest",
    response_model=OperatorRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def enqueue_backtest(
    payload: RunBacktestRequest,
    background_tasks: BackgroundTasks,
    service: DeskApiService = Depends(get_service),
) -> OperatorRunResponse:
    return _enqueue_operator_run(
        action="backtest",
        payload=payload.model_dump(),
        background_tasks=background_tasks,
        service=service,
    )


@router.post(
    "/operator/actions/report",
    response_model=OperatorRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def enqueue_report(
    payload: RunReportRequest,
    background_tasks: BackgroundTasks,
    service: DeskApiService = Depends(get_service),
) -> OperatorRunResponse:
    return _enqueue_operator_run(
        action="report",
        payload=payload.model_dump(),
        background_tasks=background_tasks,
        service=service,
    )


@router.get("/operator/runs/{run_id}", response_model=OperatorRunResponse)
def operator_run(run_id: int, service: DeskApiService = Depends(get_service)) -> OperatorRunResponse:
    try:
        run = service.operator_run(run_id)
    except (OperationalError, ProgrammingError) as exc:
        _raise_storage_not_ready(exc)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": f"Unknown operator run '{run_id}'.",
                "error_code": "operator_run_not_found",
                "run_id": run_id,
            },
        )
    return OperatorRunResponse.model_validate(run)


@router.get("/operator/runs", response_model=list[OperatorRunResponse])
def operator_runs(
    portfolio_slug: str | None = Query(default=None),
    action: str | None = Query(default=None),
    status_filter: list[str] | None = Query(default=None, alias="status"),
    limit: int = Query(default=10, ge=1, le=100),
    service: DeskApiService = Depends(get_service),
) -> list[OperatorRunResponse]:
    try:
        runs = service.operator_runs(
            portfolio_slug=portfolio_slug,
            action=action,
            statuses=status_filter,
            limit=limit,
        )
    except (OperationalError, ProgrammingError) as exc:
        _raise_storage_not_ready(exc)
    return [OperatorRunResponse.model_validate(item) for item in runs]
