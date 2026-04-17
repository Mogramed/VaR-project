from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import Request

from var_project.api.service import DeskApiService


_service_lock = Lock()


def get_service(request: Request) -> DeskApiService:
    service = getattr(request.app.state, "_desk_api_service", None)
    if service is not None:
        return service
    with _service_lock:
        service = getattr(request.app.state, "_desk_api_service", None)
        if service is None:
            service = DeskApiService(
                Path(request.app.state.repo_root),
                mt5_connector_factory=getattr(request.app.state, "mt5_connector_factory", None),
                bootstrap_storage=bool(getattr(request.app.state, "bootstrap_storage", False)),
            )
            request.app.state._desk_api_service = service
    return service
