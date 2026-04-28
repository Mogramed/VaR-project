from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.routing import APIRoute

from var_project.api.routers import analytics, capital, desk, health, mt5, operator, reports, trading
from var_project.core.settings import find_repo_root
from var_project.observability.context import bind_correlation_context
from var_project.observability.metrics import observe_http_request


LOGGER = logging.getLogger("var_project.api.http")


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

    @app.middleware("http")
    async def _observability_middleware(request: Request, call_next):
        started_at = time.perf_counter()
        request_id = str(request.headers.get("X-Request-ID") or uuid4().hex)
        response = None
        status_code = 500
        route_label = "unmatched"
        with bind_correlation_context(request_id=request_id, action="http_request"):
            try:
                response = await call_next(request)
                status_code = int(response.status_code)
                return response
            except Exception:
                LOGGER.exception(
                    "http_request_failed method=%s path=%s",
                    request.method,
                    request.url.path,
                )
                raise
            finally:
                route = request.scope.get("route")
                if route is not None:
                    route_label = str(getattr(route, "path", "") or request.url.path)
                else:
                    route_label = str(request.url.path)
                elapsed_seconds = max(time.perf_counter() - started_at, 0.0)
                observe_http_request(
                    method=request.method,
                    route=route_label,
                    status_code=status_code,
                    duration_seconds=elapsed_seconds,
                )
                if response is not None:
                    response.headers["X-Request-ID"] = request_id
                LOGGER.info(
                    "http_request method=%s route=%s status=%s latency_ms=%.2f",
                    request.method,
                    route_label,
                    status_code,
                    elapsed_seconds * 1000.0,
                )

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
