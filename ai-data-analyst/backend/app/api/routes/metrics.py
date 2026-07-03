# AI Enterprise Data Analyst - Metrics API
# Lightweight in-process metrics snapshot for debugging/operability.

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import APIResponse
from app.core.metrics import metrics_collector
from app.models import Analysis, Dataset, Job
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session

router = APIRouter()


@router.get(
    "",
    response_model=APIResponse[dict[str, Any]],
    summary="Metrics snapshot",
    description="Return a lightweight in-process metrics snapshot (dev/ops).",
)
async def get_metrics(
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    def _pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        if p <= 0:
            return float(min(values))
        if p >= 1:
            return float(max(values))
        xs = sorted(values)
        idx = int(round(p * (len(xs) - 1)))
        idx = max(0, min(idx, len(xs) - 1))
        return float(xs[idx])

    def _summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "avg": None, "p50": None, "p95": None, "min": None, "max": None}
        return {
            "n": int(len(values)),
            "avg": float(sum(values) / len(values)),
            "p50": _pct(values, 0.50),
            "p95": _pct(values, 0.95),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    # Keep this endpoint intentionally lightweight and safe: no PII, no raw data.
    snap = metrics_collector.snapshot()

    datasets_total = (await db.execute(
        select(func.count()).select_from(Dataset).where(Dataset.owner_id == user.user_id, Dataset.is_deleted == False)  # noqa: E712
    )).scalar() or 0
    analyses_total = (await db.execute(
        select(func.count()).select_from(Analysis).where(Analysis.owner_id == user.user_id, Analysis.is_deleted == False)  # noqa: E712
    )).scalar() or 0
    jobs_total = (await db.execute(
        select(func.count()).select_from(Job).where(Job.owner_id == user.user_id, Job.is_deleted == False)  # noqa: E712
    )).scalar() or 0

    snap["db_counts"] = {
        "datasets_total": int(datasets_total),
        "analyses_total": int(analyses_total),
        "jobs_total": int(jobs_total),
    }

    # Best-effort job timing stats (bounded window).
    rows = (await db.execute(
        select(Job.job_type, Job.status, Job.started_at, Job.completed_at)
        .where(
            Job.owner_id == user.user_id,
            Job.is_deleted == False,  # noqa: E712
            Job.completed_at.is_not(None),
            Job.started_at.is_not(None),
        )
        .order_by(Job.completed_at.desc())
        .limit(500)
    )).all()

    durations_all: list[float] = []
    durations_by_type: dict[str, list[float]] = defaultdict(list)
    durations_by_status: dict[str, list[float]] = defaultdict(list)
    for jt, st, started_at, completed_at in rows:
        try:
            if started_at is None or completed_at is None:
                continue
            dur = float((completed_at - started_at).total_seconds())
        except Exception:
            continue
        if dur < 0:
            continue
        jt_key = getattr(jt, "value", None) or str(jt)
        st_key = getattr(st, "value", None) or str(st)
        durations_all.append(dur)
        durations_by_type[str(jt_key)].append(dur)
        durations_by_status[str(st_key)].append(dur)

    snap["job_timings_seconds"] = {
        "all": _summary(durations_all),
        "by_type": {k: _summary(v) for k, v in sorted(durations_by_type.items(), key=lambda x: x[0])},
        "by_status": {k: _summary(v) for k, v in sorted(durations_by_status.items(), key=lambda x: x[0])},
        "window_jobs": int(len(rows)),
        "captured_at": datetime.utcnow().isoformat(),
    }

    return APIResponse.success(data=snap)
