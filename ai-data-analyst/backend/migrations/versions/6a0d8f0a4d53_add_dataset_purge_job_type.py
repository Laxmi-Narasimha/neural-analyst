"""add dataset purge job type

Revision ID: 6a0d8f0a4d53
Revises: 022998a69b80
Create Date: 2026-02-10 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6a0d8f0a4d53"
down_revision = "022998a69b80"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        try:
            op.execute("ALTER TYPE jobtype ADD VALUE 'DATASET_PURGE'")
        except Exception:
            # Keep migration best-effort/idempotent for dev environments.
            pass
    elif bind.dialect.name == "sqlite":
        old = sa.Enum("DATASET_PROCESSING", "COMPUTE_PLAN", "DATASET_TRANSFORM", name="jobtype")
        new = sa.Enum("DATASET_PROCESSING", "COMPUTE_PLAN", "DATASET_TRANSFORM", "DATASET_PURGE", name="jobtype")
        with op.batch_alter_table("jobs") as batch_op:
            batch_op.alter_column(
                "job_type",
                existing_type=old,
                type_=new,
                existing_nullable=False,
            )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        old = sa.Enum("DATASET_PROCESSING", "COMPUTE_PLAN", "DATASET_TRANSFORM", "DATASET_PURGE", name="jobtype")
        new = sa.Enum("DATASET_PROCESSING", "COMPUTE_PLAN", "DATASET_TRANSFORM", name="jobtype")
        with op.batch_alter_table("jobs") as batch_op:
            batch_op.alter_column(
                "job_type",
                existing_type=old,
                type_=new,
                existing_nullable=False,
            )

