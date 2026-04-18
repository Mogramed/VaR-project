"""heal operator_runs schema drift and enforce read-path indexes

Revision ID: 20260418_0009
Revises: 20260405_0008
Create Date: 2026-04-18 09:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260418_0009"
down_revision = "20260405_0008"
branch_labels = None
depends_on = None


INDEX_SPECS: tuple[tuple[str, list[str], bool], ...] = (
    ("ix_operator_runs_portfolio_id", ["portfolio_id"], False),
    ("ix_operator_runs_portfolio_slug", ["portfolio_slug"], False),
    ("ix_operator_runs_action", ["action"], False),
    ("ix_operator_runs_request_id", ["request_id"], True),
    ("ix_operator_runs_status", ["status"], False),
    ("ix_operator_runs_stage", ["stage"], False),
    ("ix_operator_runs_error_code", ["error_code"], False),
    ("ix_operator_runs_reused_run_id", ["reused_run_id"], False),
    ("ix_operator_runs_started_at", ["started_at"], False),
    ("ix_operator_runs_finished_at", ["finished_at"], False),
    ("ix_operator_runs_created_at", ["created_at"], False),
    ("ix_operator_runs_updated_at", ["updated_at"], False),
    ("ix_operator_runs_queue_task_id", ["queue_task_id"], False),
)


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(column.get("name") == column_name for column in inspector.get_columns(table_name))


def _index_by_name(table_name: str, index_name: str) -> dict[str, object] | None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for index in inspector.get_indexes(table_name):
        if index.get("name") == index_name:
            return index
    return None


def _has_unique_request_id(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for index in inspector.get_indexes(table_name):
        if bool(index.get("unique")) and list(index.get("column_names") or []) == ["request_id"]:
            return True
    for constraint in inspector.get_unique_constraints(table_name):
        if list(constraint.get("column_names") or []) == ["request_id"]:
            return True
    return False


def _duplicate_request_ids(table_name: str, *, limit: int = 5) -> list[str]:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            f"""
            SELECT request_id
            FROM {table_name}
            WHERE request_id IS NOT NULL
            GROUP BY request_id
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC, request_id ASC
            LIMIT :limit
            """
        ),
        {"limit": int(limit)},
    ).all()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def upgrade() -> None:
    table_name = "operator_runs"
    if not _has_table(table_name):
        return

    if not _has_column(table_name, "queue_task_id"):
        op.add_column(table_name, sa.Column("queue_task_id", sa.String(length=128), nullable=True))

    for index_name, columns, unique in INDEX_SPECS:
        if _index_by_name(table_name, index_name) is None:
            op.create_index(index_name, table_name, columns, unique=unique)

    if not _has_unique_request_id(table_name):
        duplicates = _duplicate_request_ids(table_name)
        if duplicates:
            raise RuntimeError(
                "Cannot enforce unique operator_runs.request_id: duplicate values detected "
                f"(examples: {', '.join(duplicates)}). "
                "Remove or merge duplicate rows, then rerun `alembic upgrade head`."
            )
        # Some drifted local DBs keep request_id indexed but not unique.
        # Add a dedicated unique index so idempotent queue deduplication stays safe.
        fallback_unique_index = "ux_operator_runs_request_id"
        if _index_by_name(table_name, fallback_unique_index) is None:
            op.create_index(fallback_unique_index, table_name, ["request_id"], unique=True)


def downgrade() -> None:
    # This migration is a schema-healing guard. Keep downgrade non-blocking by leaving
    # the repaired state intact when moving back to 20260405_0008.
    return
