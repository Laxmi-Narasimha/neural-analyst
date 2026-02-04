from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select

from app.core.logging import LogContext, get_logger
from app.core.serialization import to_jsonable
from app.models import Dataset, DatasetColumn, DatasetStatus
from app.services.data_ingestion import get_ingestion_service
from app.services.database import db_manager

logger = get_logger(__name__)


class DatasetProcessingService:
    """
    Background-safe dataset processing.

    This is intentionally DB-first:
    - fetch dataset record
    - set status=processing
    - ingest + profile
    - persist dataset-level metadata + per-column rows
    - set status=ready (or error)
    """

    async def process_dataset(self, dataset_id: UUID) -> None:
        context = LogContext(component="DatasetProcessingService", operation="process_dataset")

        async with db_manager.session() as session:
            result = await session.execute(
                select(Dataset).where(Dataset.id == dataset_id, Dataset.is_deleted == False)  # noqa: E712
            )
            dataset = result.scalars().first()
            if dataset is None:
                logger.warning("Dataset not found for processing", context=context, dataset_id=str(dataset_id))
                return

            if dataset.status == DatasetStatus.READY:
                logger.info("Dataset already ready; skipping processing", context=context, dataset_id=str(dataset_id))
                return

            dataset.status = DatasetStatus.PROCESSING
            dataset.error_message = None
            await session.commit()

        # Do IO + pandas work outside the session to avoid holding connections.
        ingestion_service = get_ingestion_service()
        file_path = Path(dataset.file_path)

        try:
            with file_path.open("rb") as f:
                df, profile = ingestion_service.ingest_file(f, dataset.original_filename)

            update_data: dict[str, Any] = {
                "status": DatasetStatus.READY,
                "row_count": int(profile.row_count),
                "column_count": int(profile.column_count),
                "schema_info": to_jsonable({"columns": [c.to_dict() for c in profile.columns]}),
                "quality_score": float(profile.overall_quality_score),
                "quality_report": to_jsonable(
                    {
                        "completeness": profile.completeness_score,
                        "uniqueness": profile.uniqueness_score,
                        "consistency": profile.consistency_score,
                        "warnings": profile.warnings,
                    }
                ),
                "profile_report": to_jsonable(profile.to_dict()),
            }

            async with db_manager.session() as session:
                result = await session.execute(
                    select(Dataset).where(Dataset.id == dataset_id, Dataset.is_deleted == False)  # noqa: E712
                )
                dataset = result.scalars().first()
                if dataset is None:
                    logger.warning(
                        "Dataset disappeared during processing", context=context, dataset_id=str(dataset_id)
                    )
                    return

                # Replace per-column rows on each processing run.
                await session.execute(delete(DatasetColumn).where(DatasetColumn.dataset_id == dataset_id))

                for col in profile.columns:
                    session.add(
                        DatasetColumn(
                            dataset_id=dataset_id,
                            name=str(col.name),
                            original_name=str(col.original_name),
                            position=int(col.position),
                            inferred_type=str(col.inferred_type.value),
                            semantic_type=None,
                            null_count=int(col.null_count),
                            null_percentage=float(col.null_percentage),
                            unique_count=int(col.unique_count),
                            min_value=col.min_value,
                            max_value=col.max_value,
                            mean_value=col.mean_value,
                            median_value=col.median_value,
                            std_value=col.std_value,
                            distribution_type=None,
                            value_distribution=to_jsonable(col.value_distribution),
                            has_outliers=bool(col.has_outliers),
                            is_sensitive=bool(col.is_potential_pii),
                            statistics=to_jsonable(col.to_dict()),
                        )
                    )

                for k, v in update_data.items():
                    setattr(dataset, k, v)

                await session.commit()

            logger.info(
                "Dataset processed",
                context=context,
                dataset_id=str(dataset_id),
                rows=int(profile.row_count),
                columns=int(profile.column_count),
                quality=float(profile.overall_quality_score),
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                "Dataset processing failed",
                context=context,
                dataset_id=str(dataset_id),
                error=str(e),
                traceback=tb,
            )
            async with db_manager.session() as session:
                result = await session.execute(
                    select(Dataset).where(Dataset.id == dataset_id, Dataset.is_deleted == False)  # noqa: E712
                )
                dataset = result.scalars().first()
                if dataset is None:
                    return
                dataset.status = DatasetStatus.ERROR
                dataset.error_message = str(e)
                await session.commit()

