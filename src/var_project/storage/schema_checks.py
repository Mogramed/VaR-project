from __future__ import annotations

from typing import Iterable

from sqlalchemy import inspect
from sqlalchemy.engine import Connection, Engine


OPERATOR_RUNS_TABLE = "operator_runs"

REQUIRED_OPERATOR_RUNS_COLUMNS: tuple[str, ...] = (
    "id",
    "portfolio_id",
    "portfolio_slug",
    "action",
    "request_id",
    "status",
    "stage",
    "request_payload_json",
    "artifact_refs_json",
    "result_json",
    "error_code",
    "error_message",
    "hint",
    "queue_task_id",
    "reused_run_id",
    "started_at",
    "finished_at",
    "created_at",
    "updated_at",
)

REQUIRED_OPERATOR_RUNS_INDEXES: tuple[str, ...] = (
    "ix_operator_runs_portfolio_id",
    "ix_operator_runs_portfolio_slug",
    "ix_operator_runs_action",
    "ix_operator_runs_request_id",
    "ix_operator_runs_status",
    "ix_operator_runs_stage",
    "ix_operator_runs_error_code",
    "ix_operator_runs_reused_run_id",
    "ix_operator_runs_started_at",
    "ix_operator_runs_finished_at",
    "ix_operator_runs_created_at",
    "ix_operator_runs_updated_at",
    "ix_operator_runs_queue_task_id",
)


def _is_single_request_id(column_names: Iterable[str] | None) -> bool:
    return list(column_names or []) == ["request_id"]


def validate_operator_runs_schema(bind: Engine | Connection) -> list[str]:
    inspector = inspect(bind)
    if OPERATOR_RUNS_TABLE not in inspector.get_table_names():
        return [f"Missing table '{OPERATOR_RUNS_TABLE}'."]

    issues: list[str] = []
    columns = {column.get("name") for column in inspector.get_columns(OPERATOR_RUNS_TABLE)}
    for column_name in REQUIRED_OPERATOR_RUNS_COLUMNS:
        if column_name not in columns:
            issues.append(f"Missing column '{OPERATOR_RUNS_TABLE}.{column_name}'.")

    indexes = {index.get("name"): index for index in inspector.get_indexes(OPERATOR_RUNS_TABLE)}
    for index_name in REQUIRED_OPERATOR_RUNS_INDEXES:
        if index_name not in indexes:
            issues.append(f"Missing index '{index_name}' on '{OPERATOR_RUNS_TABLE}'.")

    has_unique_request_id = any(
        bool(index.get("unique")) and _is_single_request_id(index.get("column_names"))
        for index in indexes.values()
    )
    if not has_unique_request_id:
        unique_constraints = inspector.get_unique_constraints(OPERATOR_RUNS_TABLE)
        has_unique_request_id = any(
            _is_single_request_id(constraint.get("column_names"))
            for constraint in unique_constraints
        )
    if not has_unique_request_id:
        issues.append("Missing unique constraint/index for 'operator_runs.request_id'.")

    return issues
