"""add connections and dataset versions

Revision ID: 022998a69b80
Revises: 27de10180dec
Create Date: 2026-02-05 01:22:18.756434
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '022998a69b80'
down_revision = '27de10180dec'
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid_type = sa.Uuid().with_variant(sa.UUID(), 'postgresql')
    json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')

    op.create_table(
        'dataset_versions',
        sa.Column('version_hash', sa.String(length=128), nullable=False),
        sa.Column('label', sa.String(length=255), nullable=True),
        sa.Column('parent_version_hash', sa.String(length=128), nullable=True),
        sa.Column('transform_spec', json_type, nullable=False),
        sa.Column('file_path', sa.String(length=1024), nullable=False),
        sa.Column('file_format', sa.String(length=50), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('column_count', sa.Integer(), nullable=True),
        sa.Column('schema_info', json_type, nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('quality_report', json_type, nullable=False),
        sa.Column('profile_report', json_type, nullable=False),
        sa.Column('created_by', uuid_type, nullable=True),
        sa.Column('updated_by', uuid_type, nullable=True),
        sa.Column('owner_id', uuid_type, nullable=False),
        sa.Column('dataset_id', uuid_type, nullable=False),
        sa.Column('id', uuid_type, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dataset_id', 'version_hash', name='uq_dataset_versions_dataset_hash'),
    )
    op.create_index('ix_dataset_versions_owner_created', 'dataset_versions', ['owner_id', 'created_at'], unique=False)
    op.create_index('ix_dataset_versions_dataset_created', 'dataset_versions', ['dataset_id', 'created_at'], unique=False)
    op.create_index('ix_dataset_versions_dataset_hash', 'dataset_versions', ['dataset_id', 'version_hash'], unique=False)
    op.create_index(op.f('ix_dataset_versions_version_hash'), 'dataset_versions', ['version_hash'], unique=False)

    op.create_table(
        'external_connections',
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('connector_type', sa.String(length=50), nullable=False),
        sa.Column('host', sa.String(length=255), nullable=False),
        sa.Column('port', sa.Integer(), nullable=False),
        sa.Column('database', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=255), nullable=False),
        sa.Column('encrypted_password', sa.Text(), nullable=True),
        sa.Column('ssl', sa.Boolean(), nullable=False),
        sa.Column('extra', json_type, nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('last_tested', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', uuid_type, nullable=True),
        sa.Column('updated_by', uuid_type, nullable=True),
        sa.Column('owner_id', uuid_type, nullable=False),
        sa.Column('id', uuid_type, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('owner_id', 'name', name='uq_external_connections_owner_name'),
    )
    op.create_index('ix_external_connections_owner_created', 'external_connections', ['owner_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_external_connections_connector_type'), 'external_connections', ['connector_type'], unique=False)
    op.create_index(op.f('ix_external_connections_status'), 'external_connections', ['status'], unique=False)

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # Add enum value used by new dataset transform jobs.
        try:
            op.execute("ALTER TYPE jobtype ADD VALUE 'DATASET_TRANSFORM'")
        except Exception:
            # If already present, keep migration idempotent-ish for dev environments.
            pass
    elif bind.dialect.name == "sqlite":
        # SQLite stores Enum as a CHECK constraint; recreate column with the expanded enum set.
        old = sa.Enum('DATASET_PROCESSING', 'COMPUTE_PLAN', name='jobtype')
        new = sa.Enum('DATASET_PROCESSING', 'COMPUTE_PLAN', 'DATASET_TRANSFORM', name='jobtype')
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
        # Best-effort: remove DATASET_TRANSFORM from the check constraint by recreating the column.
        old = sa.Enum('DATASET_PROCESSING', 'COMPUTE_PLAN', 'DATASET_TRANSFORM', name='jobtype')
        new = sa.Enum('DATASET_PROCESSING', 'COMPUTE_PLAN', name='jobtype')
        with op.batch_alter_table("jobs") as batch_op:
            batch_op.alter_column(
                "job_type",
                existing_type=old,
                type_=new,
                existing_nullable=False,
            )

    op.drop_index(op.f('ix_external_connections_status'), table_name='external_connections')
    op.drop_index(op.f('ix_external_connections_connector_type'), table_name='external_connections')
    op.drop_index('ix_external_connections_owner_created', table_name='external_connections')
    op.drop_table('external_connections')

    op.drop_index(op.f('ix_dataset_versions_version_hash'), table_name='dataset_versions')
    op.drop_index('ix_dataset_versions_dataset_hash', table_name='dataset_versions')
    op.drop_index('ix_dataset_versions_dataset_created', table_name='dataset_versions')
    op.drop_index('ix_dataset_versions_owner_created', table_name='dataset_versions')
    op.drop_table('dataset_versions')
