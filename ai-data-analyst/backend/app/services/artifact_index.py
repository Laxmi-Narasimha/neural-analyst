from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.compute.artifacts import ArtifactRef
from app.core.logging import LogContext, get_logger
from app.core.serialization import to_jsonable
from app.models import Artifact, ArtifactType

logger = get_logger(__name__)


class ArtifactIndexService:
    """Persist a searchable index of file-backed artifacts in the DB."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def index_many(self, *, owner_id: UUID, refs: list[ArtifactRef]) -> None:
        context = LogContext(component="ArtifactIndexService", operation="index_many")
        if not refs:
            return

        try:
            for ref in refs:
                created_at = None
                try:
                    created_at = datetime.fromisoformat(ref.created_at)
                except Exception:
                    created_at = None

                artifact = Artifact(
                    id=ref.artifact_id,
                    owner_id=owner_id,
                    dataset_id=ref.dataset_id,
                    artifact_type=ArtifactType(ref.artifact_type.value),
                    name=str(ref.name),
                    manifest_path=str(ref.storage_path),
                    data_path=None,
                    preview=to_jsonable(ref.preview or {}),
                    dataset_version=ref.dataset_version,
                    operator_name=ref.operator_name,
                    operator_params=to_jsonable(ref.operator_params or {}),
                    created_by=owner_id,
                    updated_by=owner_id,
                )
                if created_at is not None:
                    artifact.created_at = created_at

                # merge() is idempotent for repeated runs with same artifact_id.
                await self._session.merge(artifact)

            await self._session.commit()
        except Exception as e:
            await self._session.rollback()
            logger.warning("Artifact indexing failed", context=context, error=str(e), exc_info=True)

