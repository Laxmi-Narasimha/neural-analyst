# AI Enterprise Data Analyst - Analysis API Routes
# Analysis jobs are persisted and executed via safe compute operators + artifacts.

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import (
    APIResponse,
    AnalysisCreate,
    AnalysisDetailResponse,
    AnalysisResponse,
    AnalysisStatus,
    AnalysisType,
    PaginatedResponse,
    PaginationMeta,
)
from app.compute.executor import ComputeExecutor
from app.compute.plans import eda_p0_plan
from app.core.exceptions import BaseApplicationException, DataNotFoundException
from app.core.logging import LogContext, get_logger
from app.core.serialization import to_jsonable
from app.models import (
    Analysis as AnalysisModel,
    AnalysisStatus as ModelAnalysisStatus,
    AnalysisType as ModelAnalysisType,
)
from app.services.auth_service import AuthUser, Permission
from app.services.database import db_manager, get_db_session
from app.services.dataset_loader import DatasetLoaderService

logger = get_logger(__name__)

router = APIRouter()


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


def _artifact_to_dict(a: Any) -> dict[str, Any]:
    return {
        "artifact_id": str(a.artifact_id),
        "artifact_type": a.artifact_type.value,
        "name": a.name,
        "created_at": a.created_at,
        "storage_path": a.storage_path,
        "preview": a.preview,
        "dataset_id": str(a.dataset_id) if a.dataset_id else None,
        "dataset_version": a.dataset_version,
        "operator_name": a.operator_name,
        "operator_params": a.operator_params or {},
    }


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


async def run_analysis_task(*, analysis_id: UUID, owner_id: UUID) -> None:
    context = LogContext(component="AnalysisTask", operation="run_analysis", request_id=str(analysis_id))

    async with db_manager.session() as db:
        repo = AnalysisRepository(db)
        analysis = await repo.get_by_id(analysis_id, owner_id=owner_id)
        if analysis is None:
            logger.warning("Analysis not found for execution", context=context, analysis_id=str(analysis_id))
            return

        analysis.status = ModelAnalysisStatus.RUNNING
        analysis.started_at = datetime.utcnow()
        analysis.completed_at = None
        analysis.error_message = None
        analysis.error_traceback = None
        analysis.status_message = "Running"
        analysis.progress = 0.0
        analysis.updated_by = owner_id
        await repo.update(analysis)

        try:
            # Ensure the dataset exists and is accessible before doing compute.
            loader = DatasetLoaderService(db)
            await loader.get_dataset_record(analysis.dataset_id, owner_id=owner_id, require_ready=True)

            if analysis.analysis_type != ModelAnalysisType.EDA:
                raise NotImplementedError(f"Analysis type '{analysis.analysis_type.value}' not implemented yet")

            config = analysis.config or {}
            plan = config.get("plan")
            if not isinstance(plan, list):
                plan = eda_p0_plan()

            sample_rows = config.get("sample_rows")
            try:
                sample_rows_int = int(sample_rows) if sample_rows is not None else 200_000
            except Exception:
                sample_rows_int = 200_000

            executor = ComputeExecutor(db)

            steps_out: list[dict[str, Any]] = []
            total_steps = max(len(plan), 1)

            for i, step in enumerate(plan):
                operator_name = str(step.get("operator") or "").strip()
                if not operator_name:
                    continue
                params = step.get("params") or {}
                if not isinstance(params, dict):
                    params = {}

                result = await executor.run_operator(
                    dataset_id=analysis.dataset_id,
                    operator_name=operator_name,
                    params=params,
                    owner_id=owner_id,
                    sample_rows=sample_rows_int,
                )

                steps_out.append(
                    {
                        "operator": result.operator_name,
                        "summary": to_jsonable(result.summary),
                        "artifacts": to_jsonable([_artifact_to_dict(a) for a in result.artifacts]),
                    }
                )

                analysis.progress = float(min(1.0, (i + 1) / float(total_steps)))
                analysis.status_message = f"Completed {i + 1}/{total_steps}: {operator_name}"
                analysis.results = {"steps": steps_out}
                analysis.updated_by = owner_id
                await repo.update(analysis)

            analysis.status = ModelAnalysisStatus.COMPLETED
            analysis.completed_at = datetime.utcnow()
            analysis.progress = 1.0
            analysis.status_message = "Completed"
            if analysis.started_at:
                analysis.duration_seconds = (analysis.completed_at - analysis.started_at).total_seconds()
            analysis.updated_by = owner_id
            await repo.update(analysis)

            logger.info("Analysis completed", context=context, analysis_id=str(analysis_id))
        except Exception as e:
            analysis.status = ModelAnalysisStatus.FAILED
            analysis.completed_at = datetime.utcnow()
            analysis.status_message = "Failed"
            analysis.error_message = str(e)
            analysis.error_traceback = traceback.format_exc()
            analysis.updated_by = owner_id
            await repo.update(analysis)
            logger.error("Analysis failed", context=context, analysis_id=str(analysis_id), error=str(e), exc_info=True)


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

        repo = AnalysisRepository(db)
        analysis = await repo.create(
            owner_id=user.user_id,
            name=request.name,
            description=request.description,
            dataset_id=request.dataset_id,
            analysis_type=request.analysis_type,
            config=request.config or {},
        )

        background_tasks.add_task(run_analysis_task, analysis_id=analysis.id, owner_id=user.user_id)

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

    return APIResponse.success(data=_analysis_to_response(analysis), message="Analysis cancelled")
