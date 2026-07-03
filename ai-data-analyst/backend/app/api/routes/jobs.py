# AI Enterprise Data Analyst - Job API Routes
# Persisted job status for background work (dataset processing, async compute).

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import APIResponse, JobResponse, JobStatus, JobType, PaginatedResponse, PaginationMeta
from app.core.exceptions import DataNotFoundException
from app.models import (
    Analysis as AnalysisModel,
    AnalysisStatus as ModelAnalysisStatus,
    Job as JobModel,
    JobStatus as ModelJobStatus,
    JobType as ModelJobType,
)
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session

router = APIRouter()


def _schema_job_type(value: Any) -> JobType:
    if isinstance(value, JobType):
        return value
    v = getattr(value, "value", value)
    return JobType(str(v))


def _schema_job_status(value: Any) -> JobStatus:
    if isinstance(value, JobStatus):
        return value
    v = getattr(value, "value", value)
    return JobStatus(str(v))


def _job_to_response(j: JobModel) -> JobResponse:
    return JobResponse(
        id=j.id,
        job_type=_schema_job_type(j.job_type),
        status=_schema_job_status(j.status),
        progress=float(j.progress or 0.0),
        status_message=j.status_message,
        dataset_id=j.dataset_id,
        payload=j.payload or {},
        result=j.result or {},
        error_message=j.error_message,
        error_traceback=j.error_traceback,
        started_at=j.started_at,
        completed_at=j.completed_at,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )


@router.get(
    "",
    response_model=PaginatedResponse[JobResponse],
    summary="List jobs",
    description="List background jobs for the current user",
)
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[JobStatus] = Query(None),
    job_type: Optional[JobType] = Query(None),
    dataset_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    skip = (page - 1) * page_size

    q = select(JobModel).where(
        JobModel.owner_id == user.user_id,
        JobModel.is_deleted == False,  # noqa: E712
    )
    if status is not None:
        q = q.where(JobModel.status == ModelJobStatus(status.value))
    if job_type is not None:
        q = q.where(JobModel.job_type == ModelJobType(job_type.value))
    if dataset_id is not None:
        q = q.where(JobModel.dataset_id == dataset_id)

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    q = q.order_by(JobModel.created_at.desc()).offset(skip).limit(page_size)
    jobs = (await db.execute(q)).scalars().all()

    total_pages = (total + page_size - 1) // page_size
    return PaginatedResponse(
        status="success",
        data=[_job_to_response(j) for j in jobs],
        pagination=PaginationMeta(
            total=int(total),
            page=page,
            page_size=page_size,
            total_pages=int(total_pages),
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
    )


@router.get(
    "/{job_id}",
    response_model=APIResponse[JobResponse],
    summary="Get job",
    description="Fetch a single job record (owner-only).",
)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    q = select(JobModel).where(
        JobModel.id == job_id,
        JobModel.is_deleted == False,  # noqa: E712
        JobModel.owner_id == user.user_id,
    )
    job = (await db.execute(q)).scalars().first()
    if job is None:
        raise DataNotFoundException("Job", job_id)
    return APIResponse.success(data=_job_to_response(job))


@router.post(
    "/{job_id}/cancel",
    response_model=APIResponse[JobResponse],
    summary="Cancel job",
    description="Best-effort job cancellation. Workers must poll job status to stop early.",
)
async def cancel_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    q = select(JobModel).where(
        JobModel.id == job_id,
        JobModel.is_deleted == False,  # noqa: E712
        JobModel.owner_id == user.user_id,
    )
    job = (await db.execute(q)).scalars().first()
    if job is None:
        raise DataNotFoundException("Job", job_id)

    if job.status not in {ModelJobStatus.QUEUED, ModelJobStatus.RUNNING}:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    job.status = ModelJobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    job.status_message = "Cancelled"
    await db.commit()
    await db.refresh(job)

    # Best-effort: if this job maps to an Analysis, cancel the analysis too.
    try:
        analysis_id = None
        payload = job.payload or {}
        if isinstance(payload, dict):
            analysis_id = payload.get("analysis_id")
        if analysis_id:
            aid = UUID(str(analysis_id))
            aq = select(AnalysisModel).where(
                AnalysisModel.id == aid,
                AnalysisModel.is_deleted == False,  # noqa: E712
                AnalysisModel.owner_id == user.user_id,
            )
            analysis = (await db.execute(aq)).scalars().first()
            if analysis is not None and analysis.status in {ModelAnalysisStatus.QUEUED, ModelAnalysisStatus.RUNNING}:
                analysis.status = ModelAnalysisStatus.CANCELLED
                analysis.completed_at = datetime.utcnow()
                analysis.status_message = "Cancelled"
                analysis.updated_by = user.user_id
                await db.commit()
    except Exception:
        # Cancellation is best-effort; ignore propagation errors.
        pass

    return APIResponse.success(data=_job_to_response(job), message="Job cancelled")
