# AI Enterprise Data Analyst - Dashboard API Routes
# Aggregated, computed dashboard metrics (no mock numbers).

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import APIResponse, DashboardSummary
from app.models import Analysis, AnalysisStatus, Dataset, DatasetStatus, Job, JobStatus
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session

router = APIRouter()


@router.get("/summary", response_model=APIResponse[DashboardSummary])
async def get_dashboard_summary(
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    datasets_total_q = select(func.count()).select_from(Dataset).where(
        Dataset.owner_id == user.user_id,
        Dataset.is_deleted == False,  # noqa: E712
    )
    datasets_ready_q = select(func.count()).select_from(Dataset).where(
        Dataset.owner_id == user.user_id,
        Dataset.is_deleted == False,  # noqa: E712
        Dataset.status == DatasetStatus.READY,
    )
    rows_processed_q = select(func.coalesce(func.sum(Dataset.row_count), 0)).select_from(Dataset).where(
        Dataset.owner_id == user.user_id,
        Dataset.is_deleted == False,  # noqa: E712
        Dataset.status == DatasetStatus.READY,
    )

    analyses_total_q = select(func.count()).select_from(Analysis).where(
        Analysis.owner_id == user.user_id,
        Analysis.is_deleted == False,  # noqa: E712
    )
    analyses_running_q = select(func.count()).select_from(Analysis).where(
        Analysis.owner_id == user.user_id,
        Analysis.is_deleted == False,  # noqa: E712
        Analysis.status.in_([AnalysisStatus.QUEUED, AnalysisStatus.RUNNING]),
    )
    analyses_this_month_q = select(func.count()).select_from(Analysis).where(
        Analysis.owner_id == user.user_id,
        Analysis.is_deleted == False,  # noqa: E712
        Analysis.created_at >= month_start,
    )

    compute_seconds_q = select(func.coalesce(func.sum(Analysis.duration_seconds), 0.0)).select_from(Analysis).where(
        Analysis.owner_id == user.user_id,
        Analysis.is_deleted == False,  # noqa: E712
        Analysis.status == AnalysisStatus.COMPLETED,
        Analysis.duration_seconds.is_not(None),
    )

    jobs_running_q = select(func.count()).select_from(Job).where(
        Job.owner_id == user.user_id,
        Job.is_deleted == False,  # noqa: E712
        Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
    )

    datasets_total = (await db.execute(datasets_total_q)).scalar() or 0
    datasets_ready = (await db.execute(datasets_ready_q)).scalar() or 0
    rows_processed = (await db.execute(rows_processed_q)).scalar() or 0

    analyses_total = (await db.execute(analyses_total_q)).scalar() or 0
    analyses_running = (await db.execute(analyses_running_q)).scalar() or 0
    analyses_this_month = (await db.execute(analyses_this_month_q)).scalar() or 0

    compute_seconds = float((await db.execute(compute_seconds_q)).scalar() or 0.0)
    jobs_running = (await db.execute(jobs_running_q)).scalar() or 0

    data = DashboardSummary(
        datasets_total=int(datasets_total),
        datasets_ready=int(datasets_ready),
        analyses_total=int(analyses_total),
        analyses_running=int(analyses_running),
        analyses_this_month=int(analyses_this_month),
        rows_processed=int(rows_processed),
        compute_seconds=float(compute_seconds),
        jobs_running=int(jobs_running),
        month_start=month_start,
    )
    return APIResponse.success(data=data)
