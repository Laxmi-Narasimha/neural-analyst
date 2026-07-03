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


@celery_app.task(name="jobs.purge_dataset")
def purge_dataset_task(dataset_id: str, job_id: str | None = None) -> None:
    """Purge a dataset (delete blobs + hard-delete metadata) out-of-process."""
    from app.services.dataset_purge import DatasetPurgeService

    ds_id = UUID(dataset_id)
    j_id = UUID(job_id) if job_id else None
    asyncio.run(DatasetPurgeService().purge_dataset(ds_id, j_id))
