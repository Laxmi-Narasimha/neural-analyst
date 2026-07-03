from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.metrics import metrics_collector
from app.models import Job, JobStatus, JobType


class JobService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        owner_id: UUID,
        job_type: JobType,
        dataset_id: Optional[UUID] = None,
        payload: Optional[dict[str, Any]] = None,
        status_message: Optional[str] = None,
    ) -> Job:
        job = Job(
            owner_id=owner_id,
            dataset_id=dataset_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            progress=0.0,
            status_message=status_message,
            payload=payload or {},
            result={},
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get(self, job_id: UUID, *, owner_id: UUID) -> Job | None:
        q = select(Job).where(
            Job.id == job_id,
            Job.is_deleted == False,  # noqa: E712
            Job.owner_id == owner_id,
        )
        result = await self._session.execute(q)
        return result.scalars().first()

    async def set_running(self, job_id: UUID, *, owner_id: UUID, message: Optional[str] = None) -> None:
        job = await self.get(job_id, owner_id=owner_id)
        if job is None:
            return
        job.status = JobStatus.RUNNING
        job.started_at = job.started_at or datetime.utcnow()
        job.status_message = message or job.status_message
        await self._session.commit()

    async def update_progress(
        self,
        job_id: UUID,
        *,
        owner_id: UUID,
        progress: float,
        message: Optional[str] = None,
    ) -> None:
        job = await self.get(job_id, owner_id=owner_id)
        if job is None:
            return
        job.progress = max(0.0, min(float(progress), 1.0))
        if message is not None:
            job.status_message = message
        await self._session.commit()

    async def set_completed(
        self,
        job_id: UUID,
        *,
        owner_id: UUID,
        result: Optional[dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> None:
        job = await self.get(job_id, owner_id=owner_id)
        if job is None:
            return
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.completed_at = datetime.utcnow()
        job.status_message = message or job.status_message
        if result is not None:
            job.result = result
        duration_seconds = None
        try:
            if job.started_at is not None and job.completed_at is not None:
                duration_seconds = max(0.0, float((job.completed_at - job.started_at).total_seconds()))
        except Exception:
            duration_seconds = None
        if duration_seconds is not None:
            metrics_collector.record_job_run(
                job_type=getattr(job.job_type, "value", str(job.job_type)),
                duration_seconds=duration_seconds,
                success=True,
            )
        await self._session.commit()

    async def set_failed(
        self,
        job_id: UUID,
        *,
        owner_id: UUID,
        error: Exception,
        message: Optional[str] = None,
    ) -> None:
        job = await self.get(job_id, owner_id=owner_id)
        if job is None:
            return
        job.status = JobStatus.FAILED
        job.completed_at = datetime.utcnow()
        job.error_message = message or str(error)
        job.error_traceback = traceback.format_exc()
        duration_seconds = None
        try:
            if job.started_at is not None and job.completed_at is not None:
                duration_seconds = max(0.0, float((job.completed_at - job.started_at).total_seconds()))
        except Exception:
            duration_seconds = None
        if duration_seconds is not None:
            metrics_collector.record_job_run(
                job_type=getattr(job.job_type, "value", str(job.job_type)),
                duration_seconds=duration_seconds,
                success=False,
            )
        await self._session.commit()
