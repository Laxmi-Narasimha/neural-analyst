from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import APIResponse, CachePruneResponse, StorageGcRequest, StorageGcResponse
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.services.object_store import get_object_store
from app.services.storage_gc import StorageGCService

router = APIRouter()


@router.post(
    "/storage-gc",
    response_model=APIResponse[StorageGcResponse],
    summary="Garbage-collect orphaned storage objects (local/S3)",
    description="Delete orphaned upload/artifact objects not referenced by DB records. Safe, age-gated.",
)
async def storage_gc(
    request: StorageGcRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.DELETE_DATA)),
):
    service = StorageGCService(db)
    result = await service.run(
        dry_run=bool(request.dry_run),
        min_age_days=int(request.min_age_days),
        include_cache=bool(request.include_cache),
        include_s3=bool(request.include_s3),
        s3_max_scan=int(request.s3_max_scan),
    )
    return APIResponse.success(data=StorageGcResponse(**result))


@router.post(
    "/cache-prune",
    response_model=APIResponse[CachePruneResponse],
    summary="Prune local object-store cache",
    description="Prune the local cache directory for remote object store downloads.",
)
async def cache_prune(
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.DELETE_DATA)),
):
    obj = get_object_store()
    res = obj.prune_cache()
    return APIResponse.success(data=CachePruneResponse(**res))
