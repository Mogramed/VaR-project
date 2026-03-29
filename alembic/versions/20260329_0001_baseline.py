"""baseline schema

Revision ID: 20260329_0001
Revises:
Create Date: 2026-03-29 00:00:00
"""

from __future__ import annotations

from alembic import op

from var_project.storage.models import Base


revision = "20260329_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
