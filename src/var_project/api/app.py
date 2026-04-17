from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.routing import APIRoute

from var_project.api.routers import analytics, capital, desk, health, mt5, operator, reports, trading
from var_project.core.settings import find_repo_root


def _generate_operation_id(route: APIRoute) -> str:
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}_{route.name}"


def create_app(repo_root: Path | None = None, mt5_connector_factory=None, *, bootstrap_storage: bool = False) -> FastAPI:
    root = (repo_root or find_repo_root()).resolve()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield
        service = getattr(app.state, "_desk_api_service", None)
        if service is None:
            return
        try:
            close = getattr(service, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        try:
            service.storage.engine.dispose()
        except Exception:
            pass
        app.state._desk_api_service = None

    app = FastAPI(
        title="VaR Risk Desk Platform API",
        version="0.1.0",
        description="Canonical API facade for the FX VaR Risk Desk Platform.",
        generate_unique_id_function=_generate_operation_id,
        lifespan=_lifespan,
    )
    app.state.repo_root = root
    app.state.mt5_connector_factory = mt5_connector_factory
    app.state.bootstrap_storage = bool(bootstrap_storage)
    app.state._desk_api_service = None

    @app.get("/", tags=["health"], summary="API entrypoint")
    def root() -> dict[str, object]:
        return {
            "name": app.title,
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "readiness": "/health/readiness",
        }

    for module in (health, desk, analytics, capital, mt5, trading, reports, operator):
        app.include_router(module.router)

    return app
