from __future__ import annotations

import asyncio
from uuid import UUID

from celery import Celery

from app.core.config import settings


celery_app = Celery(
    "ai_data_analyst",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
)

celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    worker_concurrency=settings.celery.worker_concurrency,
    task_soft_time_limit=settings.celery.task_soft_time_limit,
    task_time_limit=settings.celery.task_time_limit,
)


@celery_app.task(name="jobs.process_dataset")
def process_dataset_task(dataset_id: str, job_id: str | None = None) -> None:
    from app.services.dataset_processing import DatasetProcessingService

    ds_id = UUID(dataset_id)
    j_id = UUID(job_id) if job_id else None
    asyncio.run(DatasetProcessingService().process_dataset(ds_id, j_id))


@celery_app.task(name="jobs.run_analysis")
def run_analysis_task(analysis_id: str, job_id: str | None = None) -> None:
    """Run a persisted analysis (compute plan) out-of-process."""
    from app.services.analysis_execution import AnalysisExecutionService

    a_id = UUID(analysis_id)
    j_id = UUID(job_id) if job_id else None
    asyncio.run(AnalysisExecutionService().run_analysis(a_id, j_id))


@celery_app.task(name="jobs.transform_dataset")
def transform_dataset_task(dataset_id: str, job_id: str | None = None) -> None:
    """Apply a dataset transformation pipeline out-of-process."""
    from app.services.dataset_transform import DatasetTransformService

    ds_id = UUID(dataset_id)
    j_id = UUID(job_id) if job_id else None
    asyncio.run(DatasetTransformService().run_transform(ds_id, j_id))


_ML_OPTIONS = {"time_column", "split", "test_start", "balance_class_weights"}


def _ml_options(raw: object) -> dict:
    return {k: v for k, v in (raw or {}).items() if k in _ML_OPTIONS} if isinstance(raw, dict) else {}


@celery_app.task(name="jobs.run_ml_training")
def run_ml_training_task(payload: dict) -> None:
    """Run AutoML training out-of-process on a representative dataset sample."""
    from app.services.ml_training_service import MLTrainingService
    from app.services.database import db_manager

    job_id = UUID(str(payload["job_id"]))
    owner_id = UUID(str(payload["owner_id"]))
    dataset_id = UUID(str(payload["dataset_id"]))
    target_column = str(payload["target_column"])
    task = payload.get("task")
    sample_rows = payload.get("sample_rows")
    max_models = payload.get("max_models")
    optimize = bool(payload.get("optimize", True))

    async def _run() -> None:
        async with db_manager.session() as session:
            service = MLTrainingService(session)
            await service.run_automl_job(
                job_id=job_id,
                owner_id=owner_id,
                dataset_id=dataset_id,
                target_column=target_column,
                task=task,
                sample_rows=int(sample_rows) if sample_rows is not None else None,
                max_models=int(max_models) if max_models is not None else None,
                optimize=optimize,
                **_ml_options(payload.get("options")),
            )

    asyncio.run(_run())


@celery_app.task(name="jobs.purge_dataset")
def purge_dataset_task(dataset_id: str, job_id: str | None = None) -> None:
    """Purge a dataset (delete blobs + hard-delete metadata) out-of-process."""
    from app.services.dataset_purge import DatasetPurgeService

    ds_id = UUID(dataset_id)
    j_id = UUID(job_id) if job_id else None
    asyncio.run(DatasetPurgeService().purge_dataset(ds_id, j_id))
