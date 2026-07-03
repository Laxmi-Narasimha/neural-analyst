from __future__ import annotations

import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import delete, select

from app.core.config import ObjectStoreBackend, settings
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
from app.services.dataset_transformations import DatasetTransformError, apply_transform_steps
from app.services.object_store import get_object_store

logger = get_logger(__name__)


def _sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_slug(value: str) -> str:
    v = str(value or "").strip().lower()
    v = re.sub(r"[^a-z0-9._-]+", "_", v)
    v = v.strip("._-")
    return v or "transform"


class DatasetTransformService:
    """Apply a validated transformation pipeline to create a new dataset version (job-friendly)."""

    async def run_transform(self, dataset_id: UUID, job_id: Optional[UUID] = None) -> None:
        set_request_context(request_id=str(job_id or dataset_id))
        try:
            await self._run_transform_inner(dataset_id, job_id)
        finally:
            clear_request_context()

    async def _run_transform_inner(self, dataset_id: UUID, job_id: Optional[UUID] = None) -> None:
        context = LogContext(component="DatasetTransformService", operation="run_transform")

        # Fetch dataset + job payload, and mark job running. Keep this DB session short.
        async with db_manager.session() as session:
            ds_res = await session.execute(
                select(Dataset).where(Dataset.id == dataset_id, Dataset.is_deleted == False)  # noqa: E712
            )
            dataset = ds_res.scalars().first()
            if dataset is None:
                logger.warning("Dataset not found for transform", context=context, dataset_id=str(dataset_id))
                return

            set_request_context(user_id=str(dataset.owner_id))

            job: Job | None = None
            payload: dict[str, Any] = {}
            if job_id is not None:
                job_res = await session.execute(
                    select(Job).where(
                        Job.id == job_id,
                        Job.is_deleted == False,  # noqa: E712
                        Job.owner_id == dataset.owner_id,
                    )
                )
                job = job_res.scalars().first()
                if job is not None:
                    if job.status == JobStatus.CANCELLED:
                        logger.info(
                            "Dataset transform cancelled before start",
                            context=context,
                            dataset_id=str(dataset_id),
                            job_id=str(job_id),
                        )
                        return
                    job.status = JobStatus.RUNNING
                    job.started_at = job.started_at or datetime.utcnow()
                    job.progress = max(float(job.progress or 0.0), 0.02)
                    job.status_message = "Loading dataset for transform"
                    payload = job.payload or {}

            await session.commit()

            owner_id = dataset.owner_id
            original_filename = str(dataset.original_filename)
            file_path = str(dataset.file_path)
            file_format_raw = str(dataset.file_format or "").lower()
            profile_report = dataset.profile_report if isinstance(dataset.profile_report, dict) else {}

        # Parse + transform outside DB session.
        try:
            steps = payload.get("steps") if isinstance(payload, dict) else None
            if not isinstance(steps, list) or not steps:
                raise ValueError("Job payload missing steps[]")

            label = payload.get("label") if isinstance(payload, dict) else None
            label = str(label)[:255] if label else None
            set_as_current = bool(payload.get("set_as_current", True)) if isinstance(payload, dict) else True

            # Best-effort parent version hash (for provenance).
            parent_hash = None
            if isinstance(profile_report, dict):
                ph = profile_report.get("file_hash")
                if isinstance(ph, str) and ph:
                    parent_hash = ph

            obj = get_object_store()
            src_local_path = obj.ensure_local_path(file_path, filename_hint=original_filename)
            if parent_hash is None:
                try:
                    parent_hash = _sha256_file(Path(src_local_path))
                except Exception:
                    parent_hash = None

            ingestion = get_ingestion_service()
            ff_in: FileFormat | None
            try:
                ff_in = FileFormat(file_format_raw)
            except Exception:
                ff_in = None

            src_path = Path(src_local_path)
            with src_path.open("rb") as f:
                df = ingestion.parse_file(f, original_filename, file_format=ff_in)

            transform_out = apply_transform_steps(df, steps)
            transformed = transform_out.df

            out_dir = settings.upload_directory / "versions" / str(dataset_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            slug = _safe_slug(label or "transform")
            out_path = out_dir / f"{ts}_{slug}.parquet"
            transformed.to_parquet(out_path, index=False)

            file_size_bytes = int(out_path.stat().st_size)
            version_hash = _sha256_file(out_path)

            ff_out = FileFormat.PARQUET
            profile = ingestion._generate_profile(  # noqa: SLF001 (internal reuse; profile code lives in ingestion service)
                df=transformed,
                filename=original_filename,
                file_format=ff_out,
                file_size=file_size_bytes,
                file_hash=version_hash,
            )
            profile.processing_time_ms = 0.0

            transform_spec = to_jsonable(
                {
                    "label": label,
                    "parent_version_hash": parent_hash,
                    "steps": steps,
                    "metrics": transform_out.metrics,
                    "warnings": transform_out.warnings,
                    "engine": "pandas",
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

        except DatasetTransformError as e:
            await self._fail_job(job_id=job_id, owner_id=owner_id, dataset_id=dataset_id, error=str(e), traceback_text=None)
            return
        except Exception as e:
            tb = traceback.format_exc()
            await self._fail_job(job_id=job_id, owner_id=owner_id, dataset_id=dataset_id, error=str(e), traceback_text=tb)
            return

        # Persist version + optionally activate it.
        try:
            async with db_manager.session() as session:
                # Cancellation check (best-effort).
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
                        try:
                            out_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                        except Exception:
                            pass
                        logger.info(
                            "Dataset transform cancelled after compute; dropped output file",
                            context=context,
                            dataset_id=str(dataset_id),
                            job_id=str(job_id),
                        )
                        return

                # Persist the new dataset-version file to the configured backend (local disk or S3).
                obj = get_object_store()
                out_storage_path = obj.put_upload_file(
                    owner_id=owner_id,
                    original_filename=str(out_path.name),
                    local_path=out_path,
                    content_type="application/octet-stream",
                )
                if obj.backend == ObjectStoreBackend.S3:
                    try:
                        out_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception:
                        pass

                ds_res = await session.execute(
                    select(Dataset).where(
                        Dataset.id == dataset_id,
                        Dataset.is_deleted == False,  # noqa: E712
                        Dataset.owner_id == owner_id,
                    )
                )
                dataset = ds_res.scalars().first()
                if dataset is None:
                    raise RuntimeError("Dataset not found during transform commit")

                v_res = await session.execute(
                    select(DatasetVersion).where(
                        DatasetVersion.dataset_id == dataset_id,
                        DatasetVersion.owner_id == owner_id,
                        DatasetVersion.version_hash == version_hash,
                        DatasetVersion.is_deleted == False,  # noqa: E712
                    )
                )
                version = v_res.scalars().first()
                if version is None:
                    version = DatasetVersion(
                        owner_id=owner_id,
                        dataset_id=dataset_id,
                        version_hash=version_hash,
                        file_path=str(out_storage_path),
                        file_format=ff_out.value,
                        file_size_bytes=file_size_bytes,
                        created_by=owner_id,
                        updated_by=owner_id,
                    )
                    session.add(version)

                version.label = label
                version.parent_version_hash = parent_hash
                version.transform_spec = transform_spec
                version.file_path = str(out_storage_path)
                version.file_format = ff_out.value
                version.file_size_bytes = file_size_bytes
                version.row_count = int(profile.row_count)
                version.column_count = int(profile.column_count)
                version.schema_info = to_jsonable({"columns": [c.to_dict() for c in profile.columns]})
                version.quality_score = float(profile.overall_quality_score)
                version.quality_report = to_jsonable(
                    {
                        "completeness": profile.completeness_score,
                        "uniqueness": profile.uniqueness_score,
                        "consistency": profile.consistency_score,
                        "warnings": profile.warnings,
                    }
                )
                version.profile_report = to_jsonable(profile.to_dict())
                version.updated_by = owner_id

                if set_as_current:
                    # Switch dataset to point at the new version and replace its derived metadata.
                    dataset.file_path = str(out_storage_path)
                    dataset.file_format = ff_out.value
                    dataset.file_size_bytes = file_size_bytes
                    dataset.status = DatasetStatus.READY
                    dataset.error_message = None
                    dataset.row_count = int(profile.row_count)
                    dataset.column_count = int(profile.column_count)
                    dataset.schema_info = to_jsonable({"columns": [c.to_dict() for c in profile.columns]})
                    dataset.quality_score = float(profile.overall_quality_score)
                    dataset.quality_report = to_jsonable(
                        {
                            "completeness": profile.completeness_score,
                            "uniqueness": profile.uniqueness_score,
                            "consistency": profile.consistency_score,
                            "warnings": profile.warnings,
                        }
                    )
                    dataset.profile_report = to_jsonable(profile.to_dict())
                    dataset.updated_by = owner_id

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

                if job is not None:
                    job.status = JobStatus.COMPLETED
                    job.progress = 1.0
                    job.completed_at = datetime.utcnow()
                    job.status_message = "Dataset transformed"
                    job.result = to_jsonable(
                        {
                            "dataset_id": str(dataset_id),
                            "version_id": str(version.id),
                            "version_hash": version_hash,
                            "label": label,
                            "set_as_current": bool(set_as_current),
                            "file_format": ff_out.value,
                        }
                    )

                await session.commit()

            logger.info(
                "Dataset transformed",
                context=context,
                dataset_id=str(dataset_id),
                job_id=str(job_id) if job_id else None,
                version_hash=version_hash,
                set_as_current=bool(set_as_current),
            )

        except Exception as e:
            tb = traceback.format_exc()
            await self._fail_job(job_id=job_id, owner_id=owner_id, dataset_id=dataset_id, error=str(e), traceback_text=tb)

    async def _fail_job(
        self,
        *,
        job_id: Optional[UUID],
        owner_id: UUID,
        dataset_id: UUID,
        error: str,
        traceback_text: Optional[str],
    ) -> None:
        context = LogContext(component="DatasetTransformService", operation="fail_job")
        logger.error(
            "Dataset transform failed",
            context=context,
            dataset_id=str(dataset_id),
            job_id=str(job_id) if job_id else None,
            error=str(error),
            exc_info=True,
        )
        if job_id is None:
            return
        async with db_manager.session() as session:
            job_res = await session.execute(
                select(Job).where(
                    Job.id == job_id,
                    Job.is_deleted == False,  # noqa: E712
                    Job.owner_id == owner_id,
                )
            )
            job = job_res.scalars().first()
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.status_message = "Failed"
            job.error_message = str(error)
            job.error_traceback = str(traceback_text or "")
            await session.commit()
