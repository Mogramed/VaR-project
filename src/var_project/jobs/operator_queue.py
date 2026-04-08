from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from var_project.core.settings import find_repo_root

try:
    from celery import Celery
except Exception:  # pragma: no cover - optional dependency at runtime
    Celery = None  # type: ignore[assignment]


log = logging.getLogger("var_project.operator_queue")

DEFAULT_QUEUE_NAME = "operator_runs"
DEFAULT_BROKER_URL = "redis://redis:6379/0"
DEFAULT_RESULT_BACKEND = "redis://redis:6379/1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class OperatorQueueSettings:
    mode: str
    broker_url: str
    result_backend: str
    queue_name: str
    retry_max_retries: int
    retry_backoff_seconds: int
    task_expires_seconds: int

    def action_soft_time_limit(self, action: str) -> int:
        key = f"VAR_PROJECT_OPERATOR_SOFT_TIMEOUT_{str(action).upper()}"
        value = _env_int(key, _env_int("VAR_PROJECT_OPERATOR_SOFT_TIMEOUT_DEFAULT", 120))
        return max(5, int(value))

    def action_hard_time_limit(self, action: str) -> int:
        key = f"VAR_PROJECT_OPERATOR_HARD_TIMEOUT_{str(action).upper()}"
        value = _env_int(key, _env_int("VAR_PROJECT_OPERATOR_HARD_TIMEOUT_DEFAULT", 180))
        return max(self.action_soft_time_limit(action) + 5, int(value))


def load_operator_queue_settings() -> OperatorQueueSettings:
    mode = str(os.getenv("VAR_PROJECT_OPERATOR_QUEUE_MODE", "auto")).strip().lower()
    if mode not in {"auto", "celery", "background"}:
        mode = "auto"
    return OperatorQueueSettings(
        mode=mode,
        broker_url=str(os.getenv("VAR_PROJECT_CELERY_BROKER_URL", DEFAULT_BROKER_URL)).strip(),
        result_backend=str(os.getenv("VAR_PROJECT_CELERY_RESULT_BACKEND", DEFAULT_RESULT_BACKEND)).strip(),
        queue_name=str(os.getenv("VAR_PROJECT_CELERY_QUEUE", DEFAULT_QUEUE_NAME)).strip() or DEFAULT_QUEUE_NAME,
        retry_max_retries=max(0, _env_int("VAR_PROJECT_OPERATOR_RETRY_MAX_RETRIES", 2)),
        retry_backoff_seconds=max(1, _env_int("VAR_PROJECT_OPERATOR_RETRY_BACKOFF_SECONDS", 4)),
        task_expires_seconds=max(30, _env_int("VAR_PROJECT_OPERATOR_TASK_EXPIRES_SECONDS", 600)),
    )


def _build_celery_app() -> Celery | None:
    if Celery is None:
        return None
    settings = load_operator_queue_settings()
    app = Celery("var_project_operator")
    app.conf.update(
        broker_url=settings.broker_url,
        result_backend=settings.result_backend,
        task_default_queue=settings.queue_name,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = _build_celery_app()


def _has_live_celery_worker(timeout_seconds: float = 0.35) -> bool:
    if celery_app is None:
        return False
    try:
        inspector = celery_app.control.inspect(timeout=max(0.1, float(timeout_seconds)))
        if inspector is None:
            return False
        response = inspector.ping()
    except Exception:
        return False
    return bool(response)


def _is_transient_operator_exception(exc: Exception) -> bool:
    transient_names = {
        "MT5ConnectionError",
        "ConnectError",
        "ReadError",
        "ReadTimeout",
        "TimeoutException",
        "OperationalError",
    }
    if exc.__class__.__name__ in transient_names:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "network",
        )
    )


if celery_app is not None:

    @celery_app.task(bind=True, name="var_project.jobs.process_operator_run")
    def process_operator_run_task(self: Any, run_id: int, repo_root: str | None = None) -> dict[str, Any]:
        from var_project.api.service import DeskApiService

        settings = load_operator_queue_settings()
        root = Path(repo_root).resolve() if repo_root else find_repo_root()
        service = DeskApiService(root, bootstrap_storage=False)
        try:
            return service.process_operator_run(int(run_id))
        except Exception as exc:
            if _is_transient_operator_exception(exc) and int(self.request.retries) < int(settings.retry_max_retries):
                raise self.retry(exc=exc, countdown=settings.retry_backoff_seconds) from exc
            raise

else:

    def process_operator_run_task(run_id: int, repo_root: str | None = None) -> dict[str, Any]:  # pragma: no cover
        raise RuntimeError("Celery is not available. Install celery[redis] to use operator worker queue.")


def dispatch_operator_run(
    *,
    run_id: int,
    action: str,
    repo_root: Path,
) -> dict[str, Any] | None:
    settings = load_operator_queue_settings()
    if settings.mode == "background":
        return None

    if celery_app is None:
        if settings.mode == "celery":
            raise RuntimeError("Operator queue mode is celery but Celery is not installed.")
        log.warning("Celery is unavailable, falling back to in-process execution for run %s.", run_id)
        return None

    probe_timeout_seconds = _env_float("VAR_PROJECT_CELERY_WORKER_PROBE_TIMEOUT_SECONDS", 0.35)
    if not _has_live_celery_worker(timeout_seconds=probe_timeout_seconds):
        if settings.mode == "celery":
            raise RuntimeError(
                "No active operator worker responded to Celery ping. "
                "Start `var-project operator-worker` and retry."
            )
        log.warning(
            "No active celery worker detected for run %s; switching to in-process execution.",
            run_id,
        )
        return None

    soft_limit = settings.action_soft_time_limit(action)
    hard_limit = settings.action_hard_time_limit(action)
    try:
        async_result = process_operator_run_task.apply_async(
            args=[int(run_id), str(repo_root.resolve())],
            queue=settings.queue_name,
            soft_time_limit=soft_limit,
            time_limit=hard_limit,
            expires=max(settings.task_expires_seconds, hard_limit * 2),
        )
        return {
            "mode": "celery",
            "task_id": str(async_result.id),
            "queue": settings.queue_name,
            "soft_time_limit": soft_limit,
            "hard_time_limit": hard_limit,
        }
    except Exception:
        if settings.mode == "celery":
            raise
        log.exception("Celery dispatch failed for run %s; switching to fallback execution.", run_id)
        return None


def run_operator_worker() -> None:
    settings = load_operator_queue_settings()
    if celery_app is None:
        raise RuntimeError("Celery is not available. Install celery[redis] to run operator-worker.")
    celery_app.worker_main(
        [
            "worker",
            "--loglevel=INFO",
            "--queues",
            settings.queue_name,
            "--concurrency",
            str(max(1, _env_int("VAR_PROJECT_CELERY_WORKER_CONCURRENCY", 1))),
        ]
    )
