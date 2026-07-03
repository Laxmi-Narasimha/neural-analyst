"""add artifact_shares table

Revision ID: 8c6fd0a40d1a
Revises: 4c1e9e67e3d7
Create Date: 2026-02-10 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "8c6fd0a40d1a"
down_revision = "4c1e9e67e3d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifact_shares",
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("artifact_id", sa.Uuid().with_variant(sa.UUID(), "postgresql"), nullable=False),
        sa.Column("owner_id", sa.Uuid().with_variant(sa.UUID(), "postgresql"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid().with_variant(sa.UUID(), "postgresql"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_artifact_shares_token_hash"),
    )
    op.create_index("ix_artifact_shares_token_hash", "artifact_shares", ["token_hash"], unique=True)
    op.create_index("ix_artifact_shares_owner_created", "artifact_shares", ["owner_id", "created_at"], unique=False)
    op.create_index("ix_artifact_shares_artifact_id", "artifact_shares", ["artifact_id"], unique=False)
    op.create_index(op.f("ix_artifact_shares_owner_id"), "artifact_shares", ["owner_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_artifact_shares_owner_id"), table_name="artifact_shares")
    op.drop_index("ix_artifact_shares_artifact_id", table_name="artifact_shares")
    op.drop_index("ix_artifact_shares_owner_created", table_name="artifact_shares")
    op.drop_index("ix_artifact_shares_token_hash", table_name="artifact_shares")
    op.drop_table("artifact_shares")

