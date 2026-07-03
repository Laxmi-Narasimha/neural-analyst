# AI Enterprise Data Analyst - Analysis API Routes
# Analysis jobs are persisted and executed via safe compute operators + artifacts.

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import (
    APIResponse,
    AnalysisCreate,
    AnalysisActionFeedItem,
    AnalysisActionKind,
    AnalysisActionRunRequest,
    AnalysisDetailResponse,
    AnalysisResponse,
    AnalysisStatus,
    AnalysisType,
    PaginatedResponse,
    PaginationMeta,
)
from app.core.exceptions import BaseApplicationException, DataNotFoundException
from app.core.logging import get_logger
from app.models import (
    Analysis as AnalysisModel,
    AnalysisStatus as ModelAnalysisStatus,
    AnalysisType as ModelAnalysisType,
)
from app.compute.artifacts import ArtifactStore
from app.services.auth_service import AuthUser, Permission
from app.services.database import db_manager, get_db_session
from app.services.dataset_loader import DatasetLoaderService
from app.services.artifact_index import ArtifactIndexService
from app.services.analysis_report import build_report_from_analysis, export_report
from app.services.report_generator import ReportFormat

logger = get_logger(__name__)

router = APIRouter()


def _sse(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, default=str)
    return f"event: {event}\ndata: {data}\n\n"


def _artifact_ref_to_dict(a: Any) -> dict[str, Any]:
    atype = getattr(getattr(a, "artifact_type", None), "value", None)
    return {
        "artifact_id": str(getattr(a, "artifact_id", "")),
        "artifact_type": atype,
        "name": getattr(a, "name", None),
        "created_at": getattr(a, "created_at", None),
        "storage_path": getattr(a, "storage_path", None),
        "preview": getattr(a, "preview", None),
        "dataset_id": str(getattr(a, "dataset_id", None)) if getattr(a, "dataset_id", None) else None,
        "dataset_version": getattr(a, "dataset_version", None),
        "operator_name": getattr(a, "operator_name", None),
        "operator_params": getattr(a, "operator_params", None) or {},
    }


def _schema_analysis_type(value: Any) -> AnalysisType:
    if isinstance(value, AnalysisType):
        return value
    v = getattr(value, "value", value)
    return AnalysisType(str(v))


def _schema_analysis_status(value: Any) -> AnalysisStatus:
    if isinstance(value, AnalysisStatus):
        return value
    v = getattr(value, "value", value)
    return AnalysisStatus(str(v))


def _analysis_to_response(a: AnalysisModel) -> AnalysisResponse:
    return AnalysisResponse(
        id=a.id,
        name=a.name,
        description=a.description,
        dataset_id=a.dataset_id,
        analysis_type=_schema_analysis_type(a.analysis_type),
        status=_schema_analysis_status(a.status),
        progress=float(a.progress or 0.0),
        status_message=a.status_message,
        started_at=a.started_at,
        completed_at=a.completed_at,
        duration_seconds=a.duration_seconds,
        created_at=a.created_at,
    )


def _collect_step_artifacts(steps: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(steps, list):
        return out
    for s in steps:
        if not isinstance(s, dict):
            continue
        arts = s.get("artifacts")
        if not isinstance(arts, list):
            continue
        for a in arts:
            if not isinstance(a, dict):
                continue
            aid = a.get("artifact_id")
            if not aid:
                continue
            out.append(
                {
                    "artifact_id": str(aid),
                    "artifact_type": a.get("artifact_type"),
                    "name": a.get("name"),
                    "operator_name": a.get("operator_name") or s.get("operator"),
                }
            )
    # Keep responses bounded.
    return out[:50]


def _action_defaults(action_id: str) -> tuple[str, str, list[dict[str, Any]]]:
    aid = str(action_id or "").strip().lower()
    if aid == "missingness_patterns":
        return (
            "Explain missingness patterns",
            "Break down missingness by category/time and surface likely causes.",
            [{"operator": "missingness_patterns", "params": {}}],
        )
    if aid == "outlier_explain":
        return (
            "Explain outliers",
            "Quantiles + bounds + (optional) outlier rate over time.",
            [{"operator": "outlier_explain", "params": {}}],
        )
    if aid == "segment_deep_dive":
        return (
            "Segment deep dive",
            "Compare top segments and how numeric features shift.",
            [{"operator": "segment_deep_dive", "params": {}}],
        )
    if aid == "privacy_risk_scan":
        return (
            "Review privacy & risk",
            "PII flags, identifiers, constants, and basic risk signals (schema-backed).",
            [{"operator": "privacy_risk_scan", "params": {}}],
        )
    if aid == "trend":
        return (
            "Deep dive time trend",
            "Resample and aggregate with bounded output.",
            [{"operator": "resample_aggregate", "params": {"freq": "M", "max_points": 200}}],
        )
    if aid == "relationships_scan":
        return (
            "Relationship scan",
            "Recompute correlation/association summaries (bounded).",
            [
                {"operator": "correlation_matrix", "params": {"max_columns": 25}},
                {"operator": "association_scan", "params": {"max_categorical_columns": 20, "max_numeric_columns": 20, "max_pairs": 200}},
            ],
        )
    if aid == "relationship_explain":
        return (
            "Explain relationship",
            "Drill down into a specific pair with bounded evidence tables (no raw rows).",
            [{"operator": "relationship_explain", "params": {}}],
        )
    if aid == "time_anomaly_scan":
        return (
            "Scan anomalies & change points",
            "Detect spikes/drops and biggest changes over time (bounded; auto metric).",
            [{"operator": "time_anomaly_scan", "params": {"freq": "M", "max_points": 200}}],
        )
    raise KeyError(aid)


class AnalysisRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, analysis_id: UUID, *, owner_id: UUID) -> AnalysisModel | None:
        q = select(AnalysisModel).where(
            AnalysisModel.id == analysis_id,
            AnalysisModel.is_deleted == False,  # noqa: E712
            AnalysisModel.owner_id == owner_id,
        )
        result = await self._session.execute(q)
        return result.scalars().first()

    async def list(
        self,
        *,
        owner_id: UUID,
        dataset_id: Optional[UUID],
        status_filter: Optional[AnalysisStatus],
        skip: int,
        limit: int,
    ) -> tuple[list[AnalysisModel], int]:
        q = select(AnalysisModel).where(
            AnalysisModel.owner_id == owner_id,
            AnalysisModel.is_deleted == False,  # noqa: E712
        )

        if dataset_id is not None:
            q = q.where(AnalysisModel.dataset_id == dataset_id)

        if status_filter is not None:
            q = q.where(AnalysisModel.status == ModelAnalysisStatus(status_filter.value))

        count_q = select(func.count()).select_from(q.subquery())
        total = (await self._session.execute(count_q)).scalar() or 0

        q = q.order_by(AnalysisModel.created_at.desc()).offset(skip).limit(limit)
        items = (await self._session.execute(q)).scalars().all()
        return items, int(total)

    async def create(
        self,
        *,
        owner_id: UUID,
        name: str,
        description: Optional[str],
        dataset_id: UUID,
        analysis_type: AnalysisType,
        config: dict[str, Any],
    ) -> AnalysisModel:
        a = AnalysisModel(
            name=name,
            description=description,
            dataset_id=dataset_id,
            analysis_type=ModelAnalysisType(analysis_type.value),
            status=ModelAnalysisStatus.QUEUED,
            progress=0.0,
            status_message="Queued",
            config=config,
            results={},
            insights=[],
            visualizations=[],
            agent_trace=[],
            owner_id=owner_id,
            created_by=owner_id,
            updated_by=owner_id,
        )

        self._session.add(a)
        await self._session.commit()
        await self._session.refresh(a)
        return a

    async def update(self, analysis: AnalysisModel) -> None:
        self._session.add(analysis)
        await self._session.commit()
        await self._session.refresh(analysis)

    async def soft_delete(self, analysis: AnalysisModel) -> None:
        analysis.soft_delete()
        await self._session.commit()


@router.post(
    "",
    response_model=APIResponse[AnalysisResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create analysis",
    description="Create a new analysis job and execute it asynchronously",
)
async def create_analysis(
    request: AnalysisCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    try:
        # Validate dataset ownership early.
        loader = DatasetLoaderService(db)
        await loader.get_dataset_record(request.dataset_id, owner_id=user.user_id, require_ready=False)

        if request.analysis_type != AnalysisType.EDA:
            raise HTTPException(status_code=400, detail="Only EDA analyses are implemented yet.")

        repo = AnalysisRepository(db)
        analysis = await repo.create(
            owner_id=user.user_id,
            name=request.name,
            description=request.description,
            dataset_id=request.dataset_id,
            analysis_type=request.analysis_type,
            config=request.config or {},
        )

        # Create a persisted job record for execution.
        from app.models import Job, JobStatus, JobType
        from app.workers.dispatcher import enqueue_compute_plan

        job = Job(
            owner_id=user.user_id,
            dataset_id=request.dataset_id,
            job_type=JobType.COMPUTE_PLAN,
            status=JobStatus.QUEUED,
            progress=0.0,
            status_message="Queued analysis",
            payload={
                "analysis_id": str(analysis.id),
                "dataset_id": str(request.dataset_id),
                "analysis_type": request.analysis_type.value,
            },
            result={},
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        # Store job_id for traceability + cancellation (avoid in-place JSON mutation;
        # SQLAlchemy won't reliably persist it unless we assign a fresh dict).
        cfg_raw = analysis.config if isinstance(analysis.config, dict) else {}
        cfg = dict(cfg_raw)
        cfg["job_id"] = str(job.id)
        analysis.config = cfg
        analysis.updated_by = user.user_id
        await repo.update(analysis)

        enqueue_compute_plan(background_tasks=background_tasks, analysis_id=analysis.id, job_id=job.id)

        return APIResponse.success(data=_analysis_to_response(analysis), message="Analysis created and queued")
    except BaseApplicationException as e:
        raise HTTPException(status_code=e.http_status_code, detail=e.message)


@router.get(
    "",
    response_model=PaginatedResponse[AnalysisResponse],
    summary="List analyses",
    description="Get a paginated list of analyses for the current user",
)
async def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    dataset_id: Optional[UUID] = Query(None),
    status_filter: Optional[AnalysisStatus] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    repo = AnalysisRepository(db)
    skip = (page - 1) * page_size
    items, total = await repo.list(
        owner_id=user.user_id,
        dataset_id=dataset_id,
        status_filter=status_filter,
        skip=skip,
        limit=page_size,
    )

    total_pages = (total + page_size - 1) // page_size
    return PaginatedResponse(
        status="success",
        data=[_analysis_to_response(a) for a in items],
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
    )


@router.get(
    "/{analysis_id}",
    response_model=APIResponse[AnalysisDetailResponse],
    summary="Get analysis",
    description="Get an analysis and its results",
)
async def get_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id, owner_id=user.user_id)
    if analysis is None:
        raise DataNotFoundException("Analysis", analysis_id)

    response = AnalysisDetailResponse(
        **_analysis_to_response(analysis).model_dump(),
        config=analysis.config or {},
        results=analysis.results or {},
        insights=analysis.insights or [],
        visualizations=analysis.visualizations or [],
        error_message=analysis.error_message,
        agent_trace=analysis.agent_trace or [],
    )
    return APIResponse.success(data=response)


@router.get(
    "/{analysis_id}/events",
    summary="Stream analysis events",
    description="Server-sent events stream for progress + step completion (use fetch streaming; sends no raw data)",
)
async def stream_analysis_events(
    analysis_id: UUID,
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    async def _event_gen():
        sent_steps = 0
        last_status = None
        last_progress = None
        last_message = None
        last_updated = None
        last_keepalive = datetime.utcnow()

        try:
            while True:
                async with db_manager.session() as session:
                    repo = AnalysisRepository(session)
                    analysis = await repo.get_by_id(analysis_id, owner_id=user.user_id)
                    if analysis is None:
                        yield _sse("error", {"message": "Analysis not found", "analysis_id": str(analysis_id)})
                        return

                    results = analysis.results or {}
                    steps = results.get("steps") if isinstance(results, dict) else None
                    steps_list = steps if isinstance(steps, list) else []

                    # Emit any newly completed steps as discrete events.
                    if sent_steps < len(steps_list):
                        for idx in range(sent_steps, len(steps_list)):
                            step = steps_list[idx]
                            if isinstance(step, dict):
                                yield _sse(
                                    "analysis_step",
                                    {
                                        "analysis_id": str(analysis.id),
                                        "step_index": idx + 1,
                                        "step": step,
                                    },
                                )
                        sent_steps = len(steps_list)

                    status_val = getattr(analysis.status, "value", str(analysis.status))
                    progress_val = float(analysis.progress or 0.0)
                    message_val = analysis.status_message
                    updated_val = analysis.updated_at

                    run_meta = results.get("run_meta") if isinstance(results, dict) else None
                    takeaways = results.get("takeaways") if isinstance(results, dict) else None
                    suggested_prompts = results.get("suggested_prompts") if isinstance(results, dict) else None

                    # Emit meta updates when something changes (or first iteration).
                    if (
                        last_status != status_val
                        or last_progress != progress_val
                        or last_message != message_val
                        or last_updated != updated_val
                    ):
                        last_status = status_val
                        last_progress = progress_val
                        last_message = message_val
                        last_updated = updated_val

                        yield _sse(
                            "analysis_meta",
                            {
                                "analysis_id": str(analysis.id),
                                "status": status_val,
                                "progress": progress_val,
                                "status_message": message_val,
                                "started_at": analysis.started_at,
                                "completed_at": analysis.completed_at,
                                "duration_seconds": analysis.duration_seconds,
                                "error_message": analysis.error_message,
                                "results": {
                                    "run_meta": run_meta,
                                    "takeaways": takeaways,
                                    "suggested_prompts": suggested_prompts,
                                },
                                "updated_at": updated_val,
                            },
                        )

                    if status_val in {"completed", "failed", "cancelled"}:
                        yield _sse("analysis_done", {"analysis_id": str(analysis.id), "status": status_val})
                        return

                # Keep-alive ping every ~15s to keep proxies from closing idle connections.
                now = datetime.utcnow()
                if (now - last_keepalive).total_seconds() >= 15:
                    last_keepalive = now
                    yield ": ping\n\n"

                await asyncio.sleep(0.75)
        except asyncio.CancelledError:
            return

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


@router.post(
    "/{analysis_id}/export",
    response_model=APIResponse[dict[str, Any]],
    summary="Export analysis report",
    description="Generate a grounded report artifact (markdown/html) from computed analysis artifacts.",
)
async def export_analysis_report(
    analysis_id: UUID,
    format: str = Query("markdown", pattern="^(markdown|html)$"),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id, owner_id=user.user_id)
    if analysis is None:
        raise DataNotFoundException("Analysis", analysis_id)

    results = analysis.results or {}
    if not isinstance(results, dict):
        results = {}
    steps = results.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        raise HTTPException(status_code=400, detail="Analysis has no results to export yet")

    report = build_report_from_analysis(analysis)
    report_format = ReportFormat.HTML if str(format).lower() == "html" else ReportFormat.MARKDOWN
    content = export_report(report, report_format=report_format)

    run_meta = results.get("run_meta") if isinstance(results.get("run_meta"), dict) else {}
    dataset_version = run_meta.get("dataset_version") if isinstance(run_meta, dict) else None

    store = ArtifactStore()
    ref = store.write_report(
        name=f"analysis_report:{analysis_id}:{report_format.value}",
        content=content,
        report_format=report_format.value,
        dataset_id=analysis.dataset_id,
        dataset_version=str(dataset_version) if dataset_version else None,
        operator_name="analysis_report",
        operator_params={"analysis_id": str(analysis_id), "format": report_format.value},
    )

    await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=[ref])

    report_artifacts = results.get("report_artifacts")
    if not isinstance(report_artifacts, list):
        report_artifacts = []
    report_artifacts.append(
        {
            "artifact_id": str(ref.artifact_id),
            "artifact_type": getattr(ref.artifact_type, "value", "report"),
            "name": ref.name,
            "format": report_format.value,
        }
    )
    results["report_artifacts"] = report_artifacts
    analysis.results = results
    analysis.updated_by = user.user_id
    await repo.update(analysis)

    return APIResponse.success(data=_artifact_ref_to_dict(ref), message="Report artifact created")


@router.get(
    "/{analysis_id}/actions",
    response_model=APIResponse[list[AnalysisActionFeedItem]],
    summary="List analysis action feed",
    description="List one-click actions executed from this analysis session.",
)
async def list_analysis_actions(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id, owner_id=user.user_id)
    if analysis is None:
        raise DataNotFoundException("Analysis", analysis_id)

    results = analysis.results if isinstance(analysis.results, dict) else {}
    feed_raw = results.get("action_feed")
    feed = feed_raw if isinstance(feed_raw, list) else []

    items: list[AnalysisActionFeedItem] = []
    for it in feed:
        if not isinstance(it, dict):
            continue
        try:
            items.append(
                AnalysisActionFeedItem(
                    action_id=str(it.get("action_id") or ""),
                    kind=AnalysisActionKind(str(it.get("kind") or "analysis")),
                    title=str(it.get("title") or it.get("label") or ""),
                    detail=(str(it.get("detail")) if it.get("detail") is not None else None),
                    params=it.get("params") if isinstance(it.get("params"), dict) else {},
                    created_at=it.get("created_at") or datetime.utcnow(),
                    analysis_id=UUID(str(it.get("analysis_id"))),
                    status=_schema_analysis_status(it.get("status") or "queued"),
                    status_message=(str(it.get("status_message")) if it.get("status_message") is not None else None),
                    error_message=(str(it.get("error_message")) if it.get("error_message") is not None else None),
                    takeaways=it.get("takeaways") if isinstance(it.get("takeaways"), list) else [],
                    artifacts=it.get("artifacts") if isinstance(it.get("artifacts"), list) else [],
                )
            )
        except Exception:
            continue

    return APIResponse.success(data=items)


@router.post(
    "/{analysis_id}/actions/run",
    response_model=APIResponse[AnalysisActionFeedItem],
    summary="Run a one-click action",
    description="Run a predefined action as a child analysis and append it to the action feed.",
)
async def run_analysis_action(
    analysis_id: UUID,
    request: AnalysisActionRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    repo = AnalysisRepository(db)
    parent = await repo.get_by_id(analysis_id, owner_id=user.user_id)
    if parent is None:
        raise DataNotFoundException("Analysis", analysis_id)

    if parent.status not in {ModelAnalysisStatus.COMPLETED}:
        raise HTTPException(status_code=400, detail="Actions can be run only after the parent analysis completes")

    action_id = str(request.action_id or "").strip().lower()
    params = request.params if isinstance(request.params, dict) else {}

    try:
        default_title, default_detail, plan = _action_defaults(action_id)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unsupported action_id: {action_id}")

    display_title = default_title
    try:
        if action_id in {"missingness_patterns", "outlier_explain"}:
            cols = params.get("columns")
            if isinstance(cols, list) and cols and cols[0]:
                display_title = f"{default_title} ({str(cols[0])})"
        elif action_id == "segment_deep_dive":
            gb = params.get("group_by")
            if gb:
                display_title = f"{default_title} ({str(gb)})"
        elif action_id == "trend":
            tc = params.get("time_column")
            if tc:
                display_title = f"{default_title} ({str(tc)})"
        elif action_id == "relationship_explain":
            a = params.get("column_a") or params.get("a")
            b = params.get("column_b") or params.get("b")
            if a and b:
                display_title = f"{default_title} ({str(a)} vs {str(b)})"
        elif action_id == "time_anomaly_scan":
            tc = params.get("time_column")
            if tc:
                display_title = f"{default_title} ({str(tc)})"
    except Exception:
        display_title = default_title

    # Merge caller-provided params into the first operator params when applicable.
    merged_plan: list[dict[str, Any]] = []
    for idx, step in enumerate(plan):
        op = str(step.get("operator") or "").strip()
        base_params = step.get("params") if isinstance(step.get("params"), dict) else {}
        if idx == 0 and op in {"missingness_patterns", "outlier_explain", "segment_deep_dive", "resample_aggregate", "relationship_explain", "time_anomaly_scan"} and params:
            merged = dict(base_params)
            merged.update(params)
            merged_plan.append({"operator": op, "params": merged})
        else:
            merged_plan.append({"operator": op, "params": dict(base_params)})

    # Try to inherit sampling from the parent run for consistency.
    sample_rows = 200_000
    if isinstance(parent.config, dict) and parent.config.get("sample_rows") is not None:
        try:
            sample_rows = int(parent.config.get("sample_rows"))
        except Exception:
            sample_rows = 200_000

    ds_id = parent.dataset_id
    child_name = f"{display_title}: {str(parent.name or 'Session')}"
    child_cfg: dict[str, Any] = {
        "sample_rows": int(sample_rows),
        "plan": merged_plan,
        "parent_analysis_id": str(parent.id),
        "action_id": action_id,
        "action_params": params,
    }

    child = await repo.create(
        owner_id=user.user_id,
        name=child_name,
        description=default_detail,
        dataset_id=ds_id,
        analysis_type=AnalysisType.EDA,
        config=child_cfg,
    )

    from app.models import Job, JobStatus, JobType
    from app.workers.dispatcher import enqueue_compute_plan

    job = Job(
        owner_id=user.user_id,
        dataset_id=ds_id,
        job_type=JobType.COMPUTE_PLAN,
        status=JobStatus.QUEUED,
        progress=0.0,
        status_message="Queued action",
        payload={
            "analysis_id": str(child.id),
            "dataset_id": str(ds_id),
            "analysis_type": AnalysisType.EDA.value,
            "parent_analysis_id": str(parent.id),
            "action_id": action_id,
        },
        result={},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    cfg_raw = child.config if isinstance(child.config, dict) else {}
    cfg = dict(cfg_raw)
    cfg["job_id"] = str(job.id)
    child.config = cfg
    child.updated_by = user.user_id
    await repo.update(child)

    enqueue_compute_plan(background_tasks=background_tasks, analysis_id=child.id, job_id=job.id)

    # Append to parent action feed (persisted in results JSON for OSS simplicity).
    parent_results_raw = parent.results if isinstance(parent.results, dict) else {}
    parent_results = dict(parent_results_raw)
    feed_raw = parent_results.get("action_feed")
    feed_list = list(feed_raw) if isinstance(feed_raw, list) else []

    created_at = datetime.utcnow().isoformat()
    feed_item = {
        "action_id": action_id,
        "kind": AnalysisActionKind.ANALYSIS.value,
        "title": display_title,
        "detail": default_detail,
        "params": params,
        "created_at": created_at,
        "analysis_id": str(child.id),
        "status": str(getattr(child.status, "value", child.status)),
        "status_message": child.status_message,
        "error_message": None,
        "takeaways": [],
        "artifacts": [],
    }
    feed_list.append(feed_item)
    parent_results["action_feed"] = feed_list
    parent.results = parent_results
    parent.updated_by = user.user_id
    await repo.update(parent)

    resp = AnalysisActionFeedItem(
        action_id=action_id,
        kind=AnalysisActionKind.ANALYSIS,
        title=display_title,
        detail=default_detail,
        params=params,
        created_at=created_at,
        analysis_id=child.id,
        status=_schema_analysis_status(child.status),
        status_message=child.status_message,
        error_message=None,
        takeaways=[],
        artifacts=[],
    )

    return APIResponse.success(data=resp, message="Action queued")


@router.delete(
    "/{analysis_id}",
    response_model=APIResponse[None],
    summary="Delete analysis",
    description="Soft-delete an analysis",
)
async def delete_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.DELETE_DATA)),
):
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id, owner_id=user.user_id)
    if analysis is None:
        raise DataNotFoundException("Analysis", analysis_id)
    await repo.soft_delete(analysis)
    return APIResponse.success(data=None, message="Analysis deleted")


@router.post(
    "/{analysis_id}/cancel",
    response_model=APIResponse[AnalysisResponse],
    summary="Cancel analysis",
    description="Mark an analysis as cancelled (best-effort)",
)
async def cancel_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id, owner_id=user.user_id)
    if analysis is None:
        raise DataNotFoundException("Analysis", analysis_id)

    if analysis.status not in {ModelAnalysisStatus.QUEUED, ModelAnalysisStatus.RUNNING}:
        raise HTTPException(status_code=400, detail="Analysis cannot be cancelled")

    analysis.status = ModelAnalysisStatus.CANCELLED
    analysis.completed_at = datetime.utcnow()
    analysis.status_message = "Cancelled"
    analysis.updated_by = user.user_id
    await repo.update(analysis)

    # Best-effort: also cancel the persisted job if we know it.
    job_id = None
    cfg = analysis.config or {}
    if isinstance(cfg, dict):
        job_id = cfg.get("job_id")
    try:
        job_uuid = UUID(str(job_id)) if job_id else None
    except Exception:
        job_uuid = None

    if job_uuid is not None:
        from app.models import Job as JobModel, JobStatus as ModelJobStatus

        res = await db.execute(
            select(JobModel).where(
                JobModel.id == job_uuid,
                JobModel.owner_id == user.user_id,
                JobModel.is_deleted == False,  # noqa: E712
            )
        )
        job = res.scalars().first()
        if job is not None and job.status in {ModelJobStatus.QUEUED, ModelJobStatus.RUNNING}:
            job.status = ModelJobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            job.status_message = "Cancelled"
            await db.commit()

    return APIResponse.success(data=_analysis_to_response(analysis), message="Analysis cancelled")
