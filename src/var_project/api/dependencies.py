from __future__ import annotations

from pathlib import Path

from fastapi import Request

from var_project.api.service import DeskApiService


def get_service(request: Request) -> DeskApiService:
    return DeskApiService(
        Path(request.app.state.repo_root),
        mt5_connector_factory=getattr(request.app.state, "mt5_connector_factory", None),
        bootstrap_storage=bool(getattr(request.app.state, "bootstrap_storage", False)),
    )
