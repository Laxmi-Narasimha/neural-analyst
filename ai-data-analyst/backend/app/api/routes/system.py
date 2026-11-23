from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import APIResponse
from app.services.system_info_service import build_system_info

router = APIRouter()


@router.get("/info", response_model=APIResponse[dict])
async def get_system_info():
    """Deployment and BYOK status — safe to expose (no secret values)."""
    return APIResponse.success(data=build_system_info())