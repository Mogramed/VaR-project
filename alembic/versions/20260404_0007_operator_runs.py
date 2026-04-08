"""add operator run queue storage

Revision ID: 20260404_0007
Revises: 20260401_0006
Create Date: 2026-04-04 17:20:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260404_0007"
down_revision = "20260401_0006"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    table_name = "operator_runs"
    if not _has_table(table_name):
        op.create_table(
            table_name,
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("portfolio_id", sa.Integer(), nullable=True),
            sa.Column("portfolio_slug", sa.String(length=160), nullable=True),
            sa.Column("action", sa.String(length=32), nullable=False),
            sa.Column("request_id", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("stage", sa.String(length=64), nullable=False),
            sa.Column("request_payload_json", sa.JSON(), nullable=True),
            sa.Column("artifact_refs_json", sa.JSON(), nullable=True),
            sa.Column("result_json", sa.JSON(), nullable=True),
            sa.Column("error_code", sa.String(length=64), nullable=True),
            sa.Column("error_message", sa.String(length=1024), nullable=True),
            sa.Column("hint", sa.String(length=1024), nullable=True),
            sa.Column("reused_run_id", sa.Integer(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("request_id", name="uq_operator_runs_request_id"),
        )

    index_specs = (
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
    )
    for index_name, columns, unique in index_specs:
        if not _has_index(table_name, index_name):
            op.create_index(index_name, table_name, columns, unique=unique)


def downgrade() -> None:
    table_name = "operator_runs"
    if not _has_table(table_name):
        return

    for index_name in (
        "ix_operator_runs_updated_at",
        "ix_operator_runs_created_at",
        "ix_operator_runs_finished_at",
        "ix_operator_runs_started_at",
        "ix_operator_runs_reused_run_id",
        "ix_operator_runs_error_code",
        "ix_operator_runs_stage",
        "ix_operator_runs_status",
        "ix_operator_runs_request_id",
        "ix_operator_runs_action",
        "ix_operator_runs_portfolio_slug",
        "ix_operator_runs_portfolio_id",
    ):
        if _has_index(table_name, index_name):
            op.drop_index(index_name, table_name=table_name)

    op.drop_table(table_name)
