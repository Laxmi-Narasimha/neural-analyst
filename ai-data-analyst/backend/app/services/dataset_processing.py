from __future__ import annotations

import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import pandas as pd

from sqlalchemy import delete, select

from app.core.logging import LogContext, clear_request_context, get_logger, set_request_context
from app.core.serialization import to_jsonable
from app.models import (
    Dataset,
    DatasetColumn,
    DatasetStatus,
    DatasetVersion,
    Job,
    JobStatus,
)
from app.services.data_ingestion import FileFormat, get_ingestion_service
from app.services.database import db_manager
from app.services.object_store import get_object_store

logger = get_logger(__name__)


class DatasetProcessingService:
    """Process an uploaded/imported dataset into derived metadata (schema, profile, columns, version)."""

    async def process_dataset(self, dataset_id: UUID, job_id: Optional[UUID] = None) -> None:
        set_request_context(request_id=str(job_id or dataset_id))
        context = LogContext(component="DatasetProcessingService", operation="process_dataset", request_id=str(dataset_id))

        owner_id: UUID | None = None
        original_filename: str = ""
        file_path: str = ""
        file_format_raw: str = ""

        try:
            async with db_manager.session() as session:
                ds_res = await session.execute(
                    select(Dataset).where(
                        Dataset.id == dataset_id,
                        Dataset.is_deleted == False,  # noqa: E712
                    )
                )
                dataset = ds_res.scalars().first()
                if dataset is None:
                    logger.warning("Dataset not found for processing", context=context, dataset_id=str(dataset_id))
                    return

                owner_id = dataset.owner_id
                set_request_context(user_id=str(owner_id))

                original_filename = str(dataset.original_filename or "")
                file_path = str(dataset.file_path or "")
                file_format_raw = str(dataset.file_format or "").lower().strip()

                job: Job | None = None
                if job_id is not None:
                    job_res = await session.execute(
                        select(Job).where(
                            Job.id == job_id,
                            Job.is_deleted == False,  # noqa: E712
                            Job.owner_id == owner_id,
                        )
                    )
                    job = job_res.scalars().first()
                    if job is not None and job.status == JobStatus.CANCELLED:
                        dataset.status = DatasetStatus.PENDING
                        dataset.error_message = "Processing cancelled"
                        dataset.updated_by = owner_id
                        await session.commit()
                        logger.info(
                            "Dataset processing cancelled before start",
                            context=context,
                            dataset_id=str(dataset_id),
                            job_id=str(job_id),
                        )
                        return

                    if job is not None:
                        job.status = JobStatus.RUNNING
                        job.started_at = job.started_at or datetime.utcnow()
                        job.completed_at = None
                        job.progress = max(float(job.progress or 0.0), 0.02)
                        job.status_message = "Profiling dataset"

                dataset.status = DatasetStatus.PROCESSING
                dataset.error_message = None
                dataset.updated_by = owner_id

                await session.commit()

            ingestion = get_ingestion_service()

            # For large datasets, avoid full in-memory loads. We compute a sample-based profile
            # and (when possible) compute exact row counts via an out-of-core scan.
            file_size_bytes = 0
            try:
                file_size_bytes = int(dataset.file_size_bytes or 0)  # type: ignore[name-defined]
            except Exception:
                file_size_bytes = 0

            use_sample = file_size_bytes >= 250 * 1024 * 1024
            sample_rows = 200_000 if use_sample else None

            # All dataset paths may live on local disk or remote object storage (S3). For compute and
            # parsing we require a local, seekable file.
            obj = get_object_store()
            local_path = obj.ensure_local_path(
                file_path,
                expected_size_bytes=int(file_size_bytes) if file_size_bytes else None,
                filename_hint=original_filename,
            )

            def _ingest_sync() -> tuple[Any, Any]:
                path = Path(local_path)
                if not path.exists():
                    raise FileNotFoundError(str(local_path))

                ff: FileFormat | None
                try:
                    ff = FileFormat(file_format_raw)
                except Exception:
                    ff = None

                with path.open("rb") as f:
                    parse_opts: dict[str, Any] = {}
                    if sample_rows is not None:
                        parse_opts["nrows"] = int(sample_rows)
                    return ingestion.ingest_file(f, original_filename, file_format=ff, **parse_opts)

            df, profile = await asyncio.to_thread(_ingest_sync)

            profiled_rows = int(getattr(df, "shape", [0, 0])[0] or 0) if df is not None else 0

            full_row_count: int | None = None
            if use_sample:
                try:
                    ff = FileFormat(file_format_raw)
                except Exception:
                    ff = None

                # Try to compute an exact row count without loading the whole dataset.
                # - Parquet: metadata contains total rows (fast).
                # - CSV/TSV: chunked pandas read (out-of-core; accurate for quoting/multiline).
                try:
                    if ff == FileFormat.PARQUET:
                        import pyarrow.parquet as pq

                        pf = pq.ParquetFile(str(local_path))
                        full_row_count = int(pf.metadata.num_rows) if pf.metadata is not None else None
                    elif ff in {FileFormat.CSV, FileFormat.TSV}:
                        from app.services.data_ingestion import CSVParser

                        parser = CSVParser()
                        with Path(local_path).open("rb") as f:
                            encoding = parser._detect_encoding(f)
                            delimiter = "\t" if ff == FileFormat.TSV else parser._detect_delimiter(f, encoding)

                        rows = 0
                        for chunk in pd.read_csv(
                            local_path,
                            encoding=encoding,
                            delimiter=delimiter,
                            low_memory=False,
                            on_bad_lines="warn",
                            chunksize=200_000,
                        ):
                            rows += int(chunk.shape[0])
                        full_row_count = int(rows)
                except Exception:
                    full_row_count = None

            # Patch the profile with full row_count when available and include sampling metadata.
            if full_row_count is not None:
                try:
                    profile.row_count = int(full_row_count)
                except Exception:
                    pass

            schema_info = to_jsonable({"columns": [c.to_dict() for c in getattr(profile, "columns", [])]})
            quality_report = to_jsonable(
                {
                    "completeness": float(getattr(profile, "completeness_score", 0.0) or 0.0),
                    "uniqueness": float(getattr(profile, "uniqueness_score", 0.0) or 0.0),
                    "consistency": float(getattr(profile, "consistency_score", 0.0) or 0.0),
                    "warnings": list(getattr(profile, "warnings", []) or []),
                }
            )
            profile_report = to_jsonable(getattr(profile, "to_dict")())
            if isinstance(profile_report, dict):
                profile_report["profile_rows"] = int(profiled_rows)
                profile_report["row_count_source"] = "full" if full_row_count is not None else ("sample" if use_sample else "full")
                if use_sample:
                    profile_report["sample_rows"] = int(sample_rows or profiled_rows)

            version_hash = str(getattr(profile, "file_hash", "") or "").strip()
            if not version_hash:
                version_hash = str(profile_report.get("file_hash") or "").strip()

            async with db_manager.session() as session:
                ds_res = await session.execute(
                    select(Dataset).where(
                        Dataset.id == dataset_id,
                        Dataset.is_deleted == False,  # noqa: E712
                        Dataset.owner_id == owner_id,
                    )
                )
                dataset = ds_res.scalars().first()
                if dataset is None:
                    raise RuntimeError("Dataset not found during processing commit")

                job: Job | None = None
                if job_id is not None:
                    job_res = await session.execute(
                        select(Job).where(
                            Job.id == job_id,
                            Job.is_deleted == False,  # noqa: E712
                            Job.owner_id == owner_id,
                        )
                    )
                    job = job_res.scalars().first()

                if job is not None and job.status == JobStatus.CANCELLED:
                    dataset.status = DatasetStatus.PENDING
                    dataset.error_message = "Processing cancelled"
                    dataset.updated_by = owner_id
                    job.completed_at = datetime.utcnow()
                    job.status_message = job.status_message or "Cancelled"
                    await session.commit()
                    logger.info(
                        "Dataset processing cancelled after compute",
                        context=context,
                        dataset_id=str(dataset_id),
                        job_id=str(job_id),
                    )
                    return

                dataset.status = DatasetStatus.READY
                dataset.error_message = None
                dataset.file_size_bytes = int(getattr(profile, "file_size_bytes", dataset.file_size_bytes) or dataset.file_size_bytes)
                dataset.row_count = int(getattr(profile, "row_count", 0) or 0)
                dataset.column_count = int(getattr(profile, "column_count", 0) or 0)
                dataset.schema_info = schema_info
                dataset.quality_score = float(getattr(profile, "overall_quality_score", 0.0) or 0.0)
                dataset.quality_report = quality_report
                dataset.profile_report = profile_report
                dataset.updated_by = owner_id

                await session.execute(delete(DatasetColumn).where(DatasetColumn.dataset_id == dataset_id))
                for col in getattr(profile, "columns", []) or []:
                    try:
                        inferred = getattr(getattr(col, "inferred_type", None), "value", None)
                        if inferred is None:
                            inferred = str(getattr(col, "inferred_type", "") or "")
                    except Exception:
                        inferred = str(getattr(col, "inferred_type", "") or "")

                    session.add(
                        DatasetColumn(
                            dataset_id=dataset_id,
                            name=str(getattr(col, "name", "")),
                            original_name=str(getattr(col, "original_name", getattr(col, "name", ""))),
                            position=int(getattr(col, "position", 0) or 0),
                            inferred_type=str(inferred),
                            semantic_type=None,
                            null_count=int(getattr(col, "null_count", 0) or 0),
                            null_percentage=float(getattr(col, "null_percentage", 0.0) or 0.0),
                            unique_count=int(getattr(col, "unique_count", 0) or 0),
                            min_value=getattr(col, "min_value", None),
                            max_value=getattr(col, "max_value", None),
                            mean_value=getattr(col, "mean_value", None),
                            median_value=getattr(col, "median_value", None),
                            std_value=getattr(col, "std_value", None),
                            distribution_type=None,
                            value_distribution=to_jsonable(getattr(col, "value_distribution", {}) or {}),
                            has_outliers=bool(getattr(col, "has_outliers", False)),
                            is_sensitive=bool(getattr(col, "is_potential_pii", False)),
                            statistics=to_jsonable(getattr(col, "to_dict")()),
                        )
                    )

                if version_hash:
                    v_res = await session.execute(
                        select(DatasetVersion).where(
                            DatasetVersion.dataset_id == dataset_id,
                            DatasetVersion.owner_id == owner_id,
                            DatasetVersion.version_hash == version_hash,
                            DatasetVersion.is_deleted == False,  # noqa: E712
                        )
                    )
                    version = v_res.scalars().first()
                    created = False
                    if version is None:
                        created = True
                        version = DatasetVersion(
                            owner_id=owner_id,
                            dataset_id=dataset_id,
                            version_hash=version_hash,
                            file_path=str(dataset.file_path),
                            file_format=str(dataset.file_format),
                            file_size_bytes=int(dataset.file_size_bytes or 0),
                            created_by=owner_id,
                            updated_by=owner_id,
                        )
                        session.add(version)

                    version.file_path = str(dataset.file_path)
                    version.file_format = str(dataset.file_format)
                    version.file_size_bytes = int(dataset.file_size_bytes or 0)
                    version.row_count = int(dataset.row_count or 0)
                    version.column_count = int(dataset.column_count or 0)
                    version.schema_info = schema_info
                    version.quality_score = float(dataset.quality_score or 0.0)
                    version.quality_report = quality_report
                    version.profile_report = profile_report
                    version.updated_by = owner_id

                    if created:
                        version.transform_spec = to_jsonable(
                            {
                                "source": "upload",
                                "original_filename": original_filename,
                                "created_at": datetime.utcnow().isoformat(),
                            }
                        )

                if job is not None:
                    job.status = JobStatus.COMPLETED
                    job.progress = 1.0
                    job.completed_at = datetime.utcnow()
                    job.status_message = "Dataset processed"
                    job.result = to_jsonable(
                        {
                            "dataset_id": str(dataset_id),
                            "row_count": int(dataset.row_count or 0),
                            "column_count": int(dataset.column_count or 0),
                            "version_hash": version_hash or None,
                        }
                    )

                await session.commit()

            logger.info(
                "Dataset processed",
                context=context,
                dataset_id=str(dataset_id),
                job_id=str(job_id) if job_id else None,
                version_hash=version_hash or None,
            )

        except Exception as e:
            tb = traceback.format_exc()
            await self._fail_job(
                dataset_id=dataset_id,
                job_id=job_id,
                owner_id=owner_id,
                error=str(e),
                traceback_text=tb,
            )
        finally:
            clear_request_context()

    async def _fail_job(
        self,
        *,
        dataset_id: UUID,
        job_id: Optional[UUID],
        owner_id: Optional[UUID],
        error: str,
        traceback_text: Optional[str],
    ) -> None:
        context = LogContext(component="DatasetProcessingService", operation="fail_job")
        logger.error(
            "Dataset processing failed",
            context=context,
            dataset_id=str(dataset_id),
            job_id=str(job_id) if job_id else None,
            error=str(error),
            exc_info=True,
        )

        async with db_manager.session() as session:
            ds_q = select(Dataset).where(
                Dataset.id == dataset_id,
                Dataset.is_deleted == False,  # noqa: E712
            )
            if owner_id is not None:
                ds_q = ds_q.where(Dataset.owner_id == owner_id)
            ds_res = await session.execute(ds_q)
            dataset = ds_res.scalars().first()
            if dataset is not None:
                owner_id = owner_id or dataset.owner_id
                dataset.status = DatasetStatus.ERROR
                dataset.error_message = str(error)[:10_000]
                dataset.updated_by = owner_id

            if job_id is not None and owner_id is not None:
                job_res = await session.execute(
                    select(Job).where(
                        Job.id == job_id,
                        Job.is_deleted == False,  # noqa: E712
                        Job.owner_id == owner_id,
                    )
                )
                job = job_res.scalars().first()
                if job is not None:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    job.status_message = "Failed"
                    job.error_message = str(error)[:10_000]
                    job.error_traceback = str(traceback_text or "")[:100_000]

            await session.commit()
