from __future__ import annotations

from typing import Iterable

from sqlalchemy import inspect
from sqlalchemy.engine import Connection, Engine


CRITICAL_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "portfolios": (
        "id",
        "slug",
        "name",
        "base_currency",
        "symbols_json",
        "positions_json",
        "created_at",
        "updated_at",
    ),
    "artifacts": (
        "id",
        "artifact_type",
        "format",
        "path",
        "size_bytes",
        "details_json",
        "created_at",
        "updated_at",
    ),
    "operator_runs": (
        "id",
        "portfolio_id",
        "portfolio_slug",
        "action",
        "request_id",
        "status",
        "stage",
        "status_reason",
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
    ),
}

OPERATOR_RUNS_TABLE = "operator_runs"

REQUIRED_OPERATOR_RUNS_INDEXES: tuple[str, ...] = (
    "ix_operator_runs_portfolio_id",
    "ix_operator_runs_portfolio_slug",
    "ix_operator_runs_action",
    "ix_operator_runs_request_id",
    "ix_operator_runs_status",
    "ix_operator_runs_stage",
    "ix_operator_runs_status_reason",
    "ix_operator_runs_error_code",
    "ix_operator_runs_reused_run_id",
    "ix_operator_runs_started_at",
    "ix_operator_runs_finished_at",
    "ix_operator_runs_created_at",
    "ix_operator_runs_updated_at",
    "ix_operator_runs_queue_task_id",
)


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item).strip()))


def _is_single_request_id(column_names: Iterable[str] | None) -> bool:
    return list(column_names or []) == ["request_id"]


def validate_critical_tables_and_columns(bind: Engine | Connection) -> list[str]:
    inspector = inspect(bind)
    table_names = set(inspector.get_table_names())
    issues: list[str] = []
    for table_name, required_columns in CRITICAL_TABLE_COLUMNS.items():
        if table_name not in table_names:
            issues.append(f"Missing table '{table_name}'.")
            continue
        present_columns = {column.get("name") for column in inspector.get_columns(table_name)}
        for column_name in required_columns:
            if column_name not in present_columns:
                issues.append(f"Missing column '{table_name}.{column_name}'.")
    return _dedupe(issues)


def validate_operator_runs_schema(bind: Engine | Connection) -> list[str]:
    inspector = inspect(bind)
    if OPERATOR_RUNS_TABLE not in inspector.get_table_names():
        return [f"Missing table '{OPERATOR_RUNS_TABLE}'."]

    issues: list[str] = []
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

    return _dedupe(issues)


def validate_storage_schema(bind: Engine | Connection) -> list[str]:
    issues = validate_critical_tables_and_columns(bind)
    if "Missing table 'operator_runs'." not in issues:
        issues.extend(validate_operator_runs_schema(bind))
    return _dedupe(issues)
