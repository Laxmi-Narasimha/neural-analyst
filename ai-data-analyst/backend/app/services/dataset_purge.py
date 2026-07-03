from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import delete, select

from app.core.logging import LogContext, clear_request_context, get_logger, set_request_context
from app.models import (
    Analysis,
    Artifact,
    DataAdequacySession,
    Dataset,
    DatasetColumn,
    DatasetVersion,
    Job,
    JobStatus,
    MLModel,
)
from app.services.database import db_manager
from app.services.job_service import JobService
from app.services.object_store import get_object_store

logger = get_logger(__name__)


class DatasetPurgeService:
    """
    Purge a dataset and all derived blobs/artifacts.

    This is a destructive operation meant for:
    - user-requested deletion (privacy/cleanup)
    - cleanup of large artifacts in self-hosted deployments

    Semantics:
    - delete storage objects first (uploads/artifacts/models)
    - only if storage deletion succeeds, hard-delete metadata rows
    - keep the purge Job record by detaching it from dataset_id before deleting the dataset (FK is CASCADE)
    """

    async def purge_dataset(self, dataset_id: UUID, job_id: Optional[UUID] = None) -> None:
        set_request_context(request_id=str(job_id or dataset_id))
        context = LogContext(component="DatasetPurgeService", operation="purge_dataset", request_id=str(dataset_id))

        owner_id: UUID | None = None

        try:
            async with db_manager.session() as session:
                ds = (await session.execute(select(Dataset).where(Dataset.id == dataset_id))).scalars().first()
                if ds is None:
                    logger.warning("Dataset not found for purge", context=context, dataset_id=str(dataset_id))
                    if job_id is not None:
                        job = (await session.execute(
                            select(Job).where(Job.id == job_id, Job.is_deleted == False)  # noqa: E712
                        )).scalars().first()
                        if job is not None and job.status not in {JobStatus.COMPLETED, JobStatus.CANCELLED}:
                            job.status = JobStatus.COMPLETED
                            job.progress = 1.0
                            job.completed_at = datetime.utcnow()
                            job.status_message = "Dataset already purged"
                            job.result = {
                                "purged_dataset_id": str(dataset_id),
                                "deleted_count": 0,
                                "note": "dataset metadata missing at purge time",
                            }
                            await session.commit()
                    return

                owner_id = ds.owner_id
                set_request_context(user_id=str(owner_id))

                job: Job | None = None
                if job_id is not None:
                    job = await JobService(session).get(job_id, owner_id=owner_id)
                    if job is None:
                        logger.warning("Purge job not found", context=context, dataset_id=str(dataset_id), job_id=str(job_id))
                    else:
                        if job.status == JobStatus.CANCELLED:
                            logger.info("Dataset purge cancelled before start", context=context, dataset_id=str(dataset_id), job_id=str(job_id))
                            return
                        await JobService(session).set_running(job_id, owner_id=owner_id, message="Purging dataset blobs")

                # Collect all storage paths while metadata still exists.
                obj = get_object_store()

                paths: set[str] = set()

                if str(ds.file_path or "").strip():
                    paths.add(str(ds.file_path))

                ver_rows = (await session.execute(
                    select(DatasetVersion.file_path).where(DatasetVersion.dataset_id == dataset_id)
                )).scalars().all()
                for p in ver_rows:
                    if str(p or "").strip():
                        paths.add(str(p))

                art_rows = (await session.execute(
                    select(Artifact.manifest_path, Artifact.data_path).where(Artifact.dataset_id == dataset_id)
                )).all()
                for mp, dp in art_rows:
                    if str(mp or "").strip():
                        paths.add(str(mp))
                    if str(dp or "").strip():
                        paths.add(str(dp))

                model_rows = (await session.execute(
                    select(MLModel.model_path)
                    .join(Analysis, MLModel.analysis_id == Analysis.id)
                    .where(Analysis.dataset_id == dataset_id)
                )).scalars().all()
                for p in model_rows:
                    if str(p or "").strip():
                        paths.add(str(p))

                all_paths = sorted(paths)
                deleted: list[str] = []
                failed: list[dict[str, Any]] = []

                total = max(1, len(all_paths))
                for i, sp in enumerate(all_paths):
                    # Cancellation check (best-effort).
                    if job is not None:
                        await session.refresh(job)
                        if job.status == JobStatus.CANCELLED:
                            logger.info("Dataset purge cancelled", context=context, dataset_id=str(dataset_id), job_id=str(job_id))
                            return

                    try:
                        ok = obj.delete(sp)
                        if ok:
                            deleted.append(sp)
                        else:
                            failed.append({"path": sp, "error": "refused to delete path (outside managed roots)"})
                    except Exception as e:
                        failed.append({"path": sp, "error": str(e)})

                    if job is not None:
                        prog = 0.05 + 0.75 * ((i + 1) / total)
                        await JobService(session).update_progress(
                            job.id, owner_id=owner_id, progress=prog, message="Deleting stored objects"
                        )

                # Best-effort: prune cache so purged datasets do not leave local residue.
                try:
                    obj.prune_cache()
                except Exception:
                    pass

                if failed:
                    if job is not None:
                        err = RuntimeError("One or more blobs failed to delete")
                        await JobService(session).set_failed(
                            job.id,
                            owner_id=owner_id,
                            error=err,
                            message=f"Failed to delete {len(failed)} objects; retry purge",
                        )
                        job.result = {
                            "deleted_count": len(deleted),
                            "failed_count": len(failed),
                            "failed": failed[:200],
                        }
                        await session.commit()
                    logger.error(
                        "Dataset purge failed to delete some objects",
                        context=context,
                        dataset_id=str(dataset_id),
                        failed_count=len(failed),
                    )
                    return

                # Detach the purge job from the dataset to avoid FK cascade deletion.
                if job is not None:
                    payload = job.payload if isinstance(job.payload, dict) else {}
                    payload["purged_dataset_id"] = str(dataset_id)
                    job.payload = payload
                    job.dataset_id = None
                    await session.commit()

                # Hard-delete rows that do not cascade via the dataset FK.
                await session.execute(delete(Artifact).where(Artifact.dataset_id == dataset_id))
                await session.execute(delete(DataAdequacySession).where(DataAdequacySession.dataset_id == dataset_id))

                # Hard-delete dependent metadata explicitly. This avoids ORM attempting to NULL-out
                # non-nullable foreign keys and is robust even when SQLite foreign key cascades
                # are not enabled.
                analysis_ids = select(Analysis.id).where(Analysis.dataset_id == dataset_id)
                await session.execute(delete(MLModel).where(MLModel.analysis_id.in_(analysis_ids)))
                await session.execute(delete(Analysis).where(Analysis.dataset_id == dataset_id))
                await session.execute(delete(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id))
                await session.execute(delete(DatasetColumn).where(DatasetColumn.dataset_id == dataset_id))
                await session.execute(delete(Job).where(Job.dataset_id == dataset_id))
                await session.execute(delete(Dataset).where(Dataset.id == dataset_id))
                await session.commit()

                if job is not None:
                    await JobService(session).set_completed(
                        job.id,
                        owner_id=owner_id,
                        result={
                            "purged_dataset_id": str(dataset_id),
                            "deleted_count": len(deleted),
                            "completed_at": datetime.utcnow().isoformat(),
                        },
                        message="Dataset purged",
                    )

                logger.info("Dataset purged", context=context, dataset_id=str(dataset_id), deleted_count=len(deleted))

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Dataset purge failed", context=context, dataset_id=str(dataset_id), error=str(e))
            try:
                if owner_id is not None and job_id is not None:
                    async with db_manager.session() as session:
                        job = await JobService(session).get(job_id, owner_id=owner_id)
                        if job is not None and job.status not in {JobStatus.COMPLETED, JobStatus.CANCELLED}:
                            job.status = JobStatus.FAILED
                            job.completed_at = datetime.utcnow()
                            job.error_message = str(e)
                            job.error_traceback = tb
                            await session.commit()
            except Exception:
                pass
        finally:
            clear_request_context()
