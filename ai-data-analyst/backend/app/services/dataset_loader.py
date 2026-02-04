from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DataNotFoundException,
    DataProcessingException,
    FileNotFoundException,
    FileParseException,
)
from app.core.logging import LogContext, get_logger
from app.models import Dataset, DatasetStatus
from app.services.data_ingestion import DataIngestionService

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedDataset:
    dataset_id: UUID
    dataset_name: str
    version_hash: str
    file_path: str
    file_format: str
    df: pd.DataFrame
    profile_report: dict[str, Any]
    schema_info: dict[str, Any]


def _sha256_fileobj(fileobj) -> str:
    hasher = hashlib.sha256()
    while True:
        chunk = fileobj.read(1024 * 1024)
        if not chunk:
            break
        hasher.update(chunk)
    fileobj.seek(0)
    return hasher.hexdigest()


class DatasetLoaderService:
    """
    Single source of truth for loading datasets by dataset_id.

    This service is intentionally strict and is the foundation for grounding:
    every compute path should flow through this loader to ensure:
    - correct dataset resolution and access control
    - consistent parsing behavior
    - stable dataset fingerprinting
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._ingestion = DataIngestionService()

    async def get_dataset_record(
        self,
        dataset_id: UUID,
        *,
        owner_id: Optional[UUID] = None,
        require_ready: bool = True,
    ) -> Dataset:
        query = select(Dataset).where(Dataset.id == dataset_id, Dataset.is_deleted == False)  # noqa: E712
        if owner_id is not None:
            query = query.where(Dataset.owner_id == owner_id)

        result = await self._session.execute(query)
        dataset = result.scalars().first()
        if dataset is None:
            raise DataNotFoundException("Dataset", dataset_id)

        if require_ready and dataset.status != DatasetStatus.READY:
            raise FileParseException(
                filename=dataset.original_filename,
                parse_errors=[f"Dataset status is '{dataset.status.value}', expected 'ready'"],
            )

        return dataset

    async def load_dataset(
        self,
        dataset_id: UUID,
        *,
        owner_id: Optional[UUID] = None,
        require_ready: bool = True,
        max_rows: Optional[int] = None,
        sample_rows: Optional[int] = None,
    ) -> LoadedDataset:
        dataset = await self.get_dataset_record(
            dataset_id,
            owner_id=owner_id,
            require_ready=require_ready,
        )

        return await self._load_from_dataset_record(dataset, max_rows=max_rows, sample_rows=sample_rows)

    async def _load_from_dataset_record(
        self,
        dataset: Dataset,
        *,
        max_rows: Optional[int] = None,
        sample_rows: Optional[int] = None,
    ) -> LoadedDataset:
        context = LogContext(component="DatasetLoaderService", operation="load_dataset")

        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise FileNotFoundException(str(file_path))

        if max_rows is not None and sample_rows is None and dataset.row_count is not None:
            if int(dataset.row_count) > int(max_rows):
                raise DataProcessingException(
                    f"Dataset too large for interactive load (rows={dataset.row_count}, max_rows={max_rows}). "
                    "Use sampling or run as an async job."
                )

        def _load_sync() -> tuple[pd.DataFrame, str]:
            with file_path.open("rb") as f:
                version_hash = _sha256_fileobj(f)
                parse_opts: dict[str, Any] = {}
                if sample_rows is not None:
                    # Best-effort sampling at read time for formats that support it.
                    parse_opts["nrows"] = int(sample_rows)
                df = self._ingestion.parse_file(f, dataset.original_filename, **parse_opts)
                return df, version_hash

        try:
            df, version_hash = await asyncio.to_thread(_load_sync)
        except FileParseException:
            raise
        except Exception as e:
            raise FileParseException(
                filename=dataset.original_filename,
                parse_errors=[str(e)],
            ) from e

        logger.info(
            "Dataset loaded",
            context=context,
            dataset_id=str(dataset.id),
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            version_hash=version_hash,
        )

        return LoadedDataset(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            version_hash=version_hash,
            file_path=str(file_path),
            file_format=dataset.file_format,
            df=df,
            profile_report=dataset.profile_report or {},
            schema_info=dataset.schema_info or {},
        )
