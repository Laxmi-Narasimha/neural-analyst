# Share links for artifacts (initially: report artifacts only).
#
# Design notes:
# - Token is returned only on creation; DB stores only a sha256 hash.
# - Public read is read-only and does not require auth (token is the auth).
# - We intentionally restrict public sharing to REPORT artifacts in OSS mode.

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import APIResponse, PublicReportResponse, ReportShareCreateRequest, ReportShareResponse
from app.compute.artifacts import ArtifactStore
from app.core.exceptions import DataNotFoundException
from app.models import Artifact as ArtifactModel
from app.models import ArtifactShare as ArtifactShareModel
from app.models import ArtifactType as ModelArtifactType
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.services.object_store import get_object_store
from app.utils.hashing import sha256_hexdigest

router = APIRouter()
public_router = APIRouter()


def _utcnow() -> datetime:
    return datetime.utcnow()


@router.post(
    "/reports/{artifact_id}",
    response_model=APIResponse[ReportShareResponse],
    summary="Create a share link for a report artifact",
    description="Creates a token-based share link for a REPORT artifact (token returned once).",
)
async def create_report_share(
    artifact_id: UUID,
    request: ReportShareCreateRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    q = select(ArtifactModel).where(
        ArtifactModel.id == artifact_id,
        ArtifactModel.owner_id == user.user_id,
        ArtifactModel.is_deleted == False,  # noqa: E712
    )
    artifact = (await db.execute(q)).scalars().first()
    if artifact is None:
        raise DataNotFoundException("Artifact", artifact_id)
    if artifact.artifact_type != ModelArtifactType.REPORT:
        raise HTTPException(status_code=400, detail="Only report artifacts can be shared")

    expires_at: Optional[datetime] = None
    if request and request.expires_days is not None:
        try:
            days = int(request.expires_days)
            expires_at = _utcnow() + timedelta(days=days)  # type: ignore[name-defined]
        except Exception:
            raise HTTPException(status_code=400, detail="expires_days must be an integer number of days")

    # Token is returned once; store only a hash.
    for _ in range(5):
        token = secrets.token_urlsafe(32)
        token_hash = sha256_hexdigest(token)
        share = ArtifactShareModel(
            owner_id=user.user_id,
            artifact_id=artifact_id,
            token_hash=token_hash,
            expires_at=expires_at,
            access_count=0,
            last_accessed_at=None,
            revoked_at=None,
        )
        db.add(share)
        try:
            await db.commit()
            await db.refresh(share)
            share_path = f"/share/reports/{token}"
            return APIResponse.success(
                data=ReportShareResponse(share_id=share.id, share_token=token, share_path=share_path, expires_at=expires_at),
                message="Share link created",
            )
        except IntegrityError:
            await db.rollback()
            continue
        except Exception as e:
            await db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=500, detail="Failed to create share token (retry)")


@router.post(
    "/reports/{share_id}/revoke",
    response_model=APIResponse[dict[str, Any]],
    summary="Revoke a report share link",
    description="Revokes an existing share link by share id.",
)
async def revoke_report_share(
    share_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    q = select(ArtifactShareModel).where(
        ArtifactShareModel.id == share_id,
        ArtifactShareModel.owner_id == user.user_id,
    )
    share = (await db.execute(q)).scalars().first()
    if share is None:
        raise DataNotFoundException("Share", share_id)
    if share.revoked_at is None:
        share.revoked_at = _utcnow()
        await db.commit()
    return APIResponse.success(data={"share_id": str(share.id), "revoked_at": share.revoked_at}, message="Share revoked")


@public_router.get(
    "/reports/{share_token}",
    response_model=APIResponse[PublicReportResponse],
    summary="Get a shared report (public)",
    description="Resolve a report share token to the report content (read-only).",
)
async def get_shared_report(
    share_token: str,
    db: AsyncSession = Depends(get_db_session),
):
    token = str(share_token or "").strip()
    if not token:
        raise HTTPException(status_code=404, detail="Share not found")

    h = sha256_hexdigest(token)
    now = _utcnow()

    q = (
        select(ArtifactShareModel, ArtifactModel)
        .join(ArtifactModel, ArtifactModel.id == ArtifactShareModel.artifact_id)
        .where(
            ArtifactShareModel.token_hash == h,
            ArtifactShareModel.revoked_at.is_(None),
            or_(ArtifactShareModel.expires_at.is_(None), ArtifactShareModel.expires_at > now),
            ArtifactModel.is_deleted == False,  # noqa: E712
            ArtifactModel.artifact_type == ModelArtifactType.REPORT,
        )
    )

    row = (await db.execute(q)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Share not found")

    share, artifact = row[0], row[1]

    # Best-effort accounting; avoid failing the request if this commit fails.
    try:
        share.access_count = int(getattr(share, "access_count", 0) or 0) + 1
        share.last_accessed_at = now
        await db.commit()
    except Exception:
        await db.rollback()

    store = ArtifactStore()
    manifest = store.read_manifest(artifact.id)
    data_path = manifest.get("data_path")
    fmt = str(manifest.get("data_format") or "markdown").lower().strip()
    if not data_path:
        raise HTTPException(status_code=404, detail="Report data missing")

    obj = get_object_store()
    try:
        raw = obj.read_bytes(str(data_path))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Report data missing: {e}")

    # Safety: cap content returned to the browser to avoid accidental huge transfers.
    max_bytes = 10 * 1024 * 1024
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]

    try:
        content = raw.decode("utf-8")
    except Exception:
        content = raw.decode("utf-8", errors="replace")

    payload = PublicReportResponse(
        artifact_id=artifact.id,
        name=str(artifact.name),
        format=fmt,
        created_at=artifact.created_at,
        content=content,
    )
    return APIResponse.success(data=payload)
