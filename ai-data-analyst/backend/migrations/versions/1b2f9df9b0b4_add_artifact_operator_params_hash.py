"""add artifact operator_params_hash

Revision ID: 1b2f9df9b0b4
Revises: 6a0d8f0a4d53
Create Date: 2026-02-10 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1b2f9df9b0b4"
down_revision = "6a0d8f0a4d53"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("artifacts", sa.Column("operator_params_hash", sa.String(length=64), nullable=True))
    op.create_index(
        "ix_artifacts_owner_op_cache",
        "artifacts",
        ["owner_id", "dataset_id", "dataset_version", "operator_name", "operator_params_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_artifacts_owner_op_cache", table_name="artifacts")
    op.drop_column("artifacts", "operator_params_hash")

