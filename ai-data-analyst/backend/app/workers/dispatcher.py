from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import BackgroundTasks

from app.core.config import JobExecutor, settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def enqueue_dataset_processing(
    *,
    background_tasks: BackgroundTasks,
    dataset_id: UUID,
    job_id: Optional[UUID],
) -> dict[str, Any]:
    """
    Enqueue dataset processing.

    - LOCAL: FastAPI BackgroundTasks (single instance, dev).
    - CELERY: Celery distributed task queue (multi instance, prod).
    """

    if settings.job_executor == JobExecutor.CELERY:
        try:
            from app.workers.celery_app import process_dataset_task

            process_dataset_task.delay(str(dataset_id), str(job_id) if job_id else None)
            return {"executor": "celery"}
        except Exception as e:
            logger.warning(f"Celery enqueue failed; falling back to local background task: {e}")

    from app.services.dataset_processing import DatasetProcessingService

    background_tasks.add_task(DatasetProcessingService().process_dataset, dataset_id, job_id)
    return {"executor": "local"}


def enqueue_compute_plan(
    *,
    background_tasks: BackgroundTasks,
    analysis_id: UUID,
    job_id: Optional[UUID],
) -> dict[str, Any]:
    """
    Enqueue an analysis/compute-plan execution.

    - LOCAL: FastAPI BackgroundTasks (dev)
    - CELERY: Celery distributed workers (prod)
    """

    if settings.job_executor == JobExecutor.CELERY:
        try:
            from app.workers.celery_app import run_analysis_task

            run_analysis_task.delay(str(analysis_id), str(job_id) if job_id else None)
            return {"executor": "celery"}
        except Exception as e:
            logger.warning(f"Celery enqueue failed; falling back to local background task: {e}")

    from app.services.analysis_execution import AnalysisExecutionService

    background_tasks.add_task(AnalysisExecutionService().run_analysis, analysis_id, job_id)
    return {"executor": "local"}


def enqueue_dataset_transform(
    *,
    background_tasks: BackgroundTasks,
    dataset_id: UUID,
    job_id: Optional[UUID],
) -> dict[str, Any]:
    """
    Enqueue a dataset transformation (create a new dataset version).

    - LOCAL: FastAPI BackgroundTasks
    - CELERY: Celery distributed workers
    """

    if settings.job_executor == JobExecutor.CELERY:
        try:
            from app.workers.celery_app import transform_dataset_task

            transform_dataset_task.delay(str(dataset_id), str(job_id) if job_id else None)
            return {"executor": "celery"}
        except Exception as e:
            logger.warning(f"Celery enqueue failed; falling back to local background task: {e}")

    from app.services.dataset_transform import DatasetTransformService

    background_tasks.add_task(DatasetTransformService().run_transform, dataset_id, job_id)
    return {"executor": "local"}


def enqueue_dataset_purge(
    *,
    background_tasks: BackgroundTasks,
    dataset_id: UUID,
    job_id: Optional[UUID],
) -> dict[str, Any]:
    """
    Enqueue a destructive dataset purge (delete blobs + hard-delete metadata).

    - LOCAL: FastAPI BackgroundTasks
    - CELERY: Celery distributed workers
    """

    if settings.job_executor == JobExecutor.CELERY:
        try:
            from app.workers.celery_app import purge_dataset_task

            purge_dataset_task.delay(str(dataset_id), str(job_id) if job_id else None)
            return {"executor": "celery"}
        except Exception as e:
            logger.warning(f"Celery enqueue failed; falling back to local background task: {e}")

    from app.services.dataset_purge import DatasetPurgeService

    background_tasks.add_task(DatasetPurgeService().purge_dataset, dataset_id, job_id)
    return {"executor": "local"}
