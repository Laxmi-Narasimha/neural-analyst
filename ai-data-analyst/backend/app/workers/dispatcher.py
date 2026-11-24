from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import BackgroundTasks

from app.core.config import DeploymentMode, JobExecutor, settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _handle_enqueue_failure(kind: str, error: Exception) -> None:
    if settings.deployment_mode == DeploymentMode.SAAS:
        raise RuntimeError(f"{kind} queue unavailable; hosted jobs cannot run in the API process") from error
    logger.warning(f"Celery {kind} enqueue failed; falling back to local background task: {error}")


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
            _handle_enqueue_failure("dataset", e)

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
            _handle_enqueue_failure("analysis", e)

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
            _handle_enqueue_failure("transform", e)

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
            _handle_enqueue_failure("purge", e)

    from app.services.dataset_purge import DatasetPurgeService

    background_tasks.add_task(DatasetPurgeService().purge_dataset, dataset_id, job_id)
    return {"executor": "local"}


def enqueue_ml_training(
    *,
    background_tasks: BackgroundTasks,
    job_id: UUID,
    owner_id: UUID,
    dataset_id: UUID,
    target_column: str,
    task: Optional[str] = None,
    sample_rows: Optional[int] = None,
    max_models: Optional[int] = None,
    optimize: bool = True,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Enqueue AutoML / full ML training on a representative sample.

    ``options`` carries the evaluation choices (time_column, split, test_start,
    balance_class_weights) so a queued job evaluates exactly like a direct run.

    - LOCAL: FastAPI BackgroundTasks
    - CELERY: Celery distributed workers
    """
    payload = {
        "job_id": str(job_id),
        "owner_id": str(owner_id),
        "dataset_id": str(dataset_id),
        "target_column": target_column,
        "task": task,
        "sample_rows": sample_rows,
        "max_models": max_models,
        "optimize": optimize,
        "options": dict(options or {}),
    }

    if settings.job_executor == JobExecutor.CELERY:
        try:
            from app.workers.celery_app import run_ml_training_task

            run_ml_training_task.delay(payload)
            return {"executor": "celery"}
        except Exception as e:
            _handle_enqueue_failure("ML", e)

    from app.services.ml_training_service import MLTrainingService
    from app.services.database import db_manager

    async def _run() -> None:
        async with db_manager.session() as session:
            service = MLTrainingService(session)
            await service.run_automl_job(
                job_id=job_id,
                owner_id=owner_id,
                dataset_id=dataset_id,
                target_column=target_column,
                task=task,
                sample_rows=sample_rows,
                max_models=max_models,
                optimize=optimize,
                **dict(options or {}),
            )

    background_tasks.add_task(lambda: __import__("asyncio").run(_run()))
    return {"executor": "local"}
