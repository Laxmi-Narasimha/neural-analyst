# AI Enterprise Data Analyst - Data Speaks API
# Safe compute + artifact generation for one-click "Make the data speak"

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import APIResponse
from app.compute.artifacts import ArtifactStore
from app.compute.executor import ComputeExecutor
from app.compute.plans import eda_p0_plan
from app.compute.registry import default_registry
from app.core.exceptions import BaseApplicationException
from app.models import Artifact
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session

router = APIRouter()


class PlanStep(BaseModel):
    operator: str = Field(..., min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class DataSpeaksRunRequest(BaseModel):
    dataset_id: UUID
    plan: Optional[list[PlanStep]] = None


def _artifact_to_dict(a) -> dict[str, Any]:
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


@router.get("/operators", response_model=APIResponse[list[str]])
async def list_operators(user: AuthUser = Depends(require_permission(Permission.READ_DATA))):
    reg = default_registry()
    return APIResponse.success(data=reg.list())


@router.post("/run", response_model=APIResponse[dict[str, Any]])
async def run_data_speaks(
    request: DataSpeaksRunRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    try:
        plan = [s.model_dump() for s in request.plan] if request.plan else eda_p0_plan()
        executor = ComputeExecutor(db)
        results = await executor.run_plan(
            dataset_id=request.dataset_id,
            plan=plan,
            owner_id=user.user_id,
            sample_rows=200_000,
        )

        response_steps = []
        for r in results:
            response_steps.append(
                {
                    "operator": r.operator_name,
                    "summary": r.summary,
                    "artifacts": [_artifact_to_dict(a) for a in r.artifacts],
                }
            )

        return APIResponse.success(
            data={
                "dataset_id": str(request.dataset_id),
                "steps": response_steps,
            },
            message="Data Speaks run completed",
        )
    except BaseApplicationException as e:
        raise HTTPException(status_code=e.http_status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/artifacts/{artifact_id}", response_model=APIResponse[dict[str, Any]])
async def get_artifact_manifest(
    artifact_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    result = await db.execute(
        select(Artifact).where(
            Artifact.id == artifact_id,
            Artifact.is_deleted == False,  # noqa: E712
            Artifact.owner_id == user.user_id,
        )
    )
    if result.scalars().first() is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    store = ArtifactStore()
    manifest = store.read_manifest(artifact_id)
    return APIResponse.success(data=manifest)


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact_data(
    artifact_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    result = await db.execute(
        select(Artifact).where(
            Artifact.id == artifact_id,
            Artifact.is_deleted == False,  # noqa: E712
            Artifact.owner_id == user.user_id,
        )
    )
    if result.scalars().first() is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    store = ArtifactStore()
    manifest = store.read_manifest(artifact_id)
    data_path = manifest.get("data_path")
    if not data_path:
        raise HTTPException(status_code=400, detail="Artifact has no downloadable data file")
    return FileResponse(path=data_path, filename=str(artifact_id))
