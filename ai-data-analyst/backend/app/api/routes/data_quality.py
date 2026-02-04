"""API endpoints for Data Quality and Adequacy analysis."""

from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.data_adequacy.manager import DataAdequacyManager
from app.api.routes.auth import require_permission
from app.models import DataAdequacySession, AdequacySessionStatus
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.services.dataset_loader import DatasetLoaderService

router = APIRouter()

class ValidationRequest(BaseModel):
    goal: str
    domain: str = "general"
    dataset_id: Optional[UUID] = None
    files: List[str] = []  # Deprecated: use dataset_id

class AnswerRequest(BaseModel):
    session_id: str
    answers: Dict[str, str]

class StatusResponse(BaseModel):
    session_id: str
    status: str
    step: str
    processing_time: str
    llm_calls: int

@router.post("/validate", response_model=Dict[str, Any])
async def start_validation_session(
    request: ValidationRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    """Start a new data adequacy validation session."""
    try:
        files: list[str] = []
        if request.dataset_id is not None:
            loader = DatasetLoaderService(db)
            dataset = await loader.get_dataset_record(request.dataset_id, owner_id=user.user_id, require_ready=False)
            files = [dataset.file_path]
        elif request.files:
            raise HTTPException(status_code=400, detail="Direct file paths are not allowed. Use dataset_id.")

        manager = DataAdequacyManager()
        result = await manager.run_validation(
            user_goal=request.goal,
            domain=request.domain,
            files=files
        )

        session_id = result.get("session_id") or manager.session_state.get("session_id")
        if session_id:
            status_val: AdequacySessionStatus
            if not result.get("success"):
                status_val = AdequacySessionStatus.FAILED
            elif result.get("step") == "clarifying_questions":
                status_val = AdequacySessionStatus.QUESTIONS
            elif result.get("readiness_level") is not None:
                status_val = AdequacySessionStatus.COMPLETED
            else:
                status_val = AdequacySessionStatus.RUNNING

            db.add(
                DataAdequacySession(
                    session_id=session_id,
                    owner_id=user.user_id,
                    dataset_id=request.dataset_id,
                    status=status_val,
                    state=manager.session_state or {},
                    llm_calls=manager.llm_call_count,
                )
            )
            await db.commit()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/continue", response_model=Dict[str, Any])
async def continue_validation_session(
    request: AnswerRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    """Continue validation with user answers."""
    session_id = request.session_id
    res = await db.execute(
        select(DataAdequacySession).where(
            DataAdequacySession.session_id == session_id,
            DataAdequacySession.owner_id == user.user_id,
            DataAdequacySession.is_deleted == False,  # noqa: E712
        )
    )
    session = res.scalars().first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    manager = DataAdequacyManager()
    manager.session_state = session.state or {}
    manager.llm_call_count = int(session.llm_calls or 0)
    try:
        result = await manager.continue_validation(request.answers)

        if not result.get("success"):
            session.status = AdequacySessionStatus.FAILED
        elif result.get("readiness_level") is not None:
            session.status = AdequacySessionStatus.COMPLETED
        else:
            session.status = AdequacySessionStatus.RUNNING

        session.state = manager.session_state or {}
        session.llm_calls = manager.llm_call_count
        await db.commit()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}", response_model=Dict[str, Any])
async def get_session_status(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    """Get status of an active session."""
    res = await db.execute(
        select(DataAdequacySession).where(
            DataAdequacySession.session_id == session_id,
            DataAdequacySession.owner_id == user.user_id,
            DataAdequacySession.is_deleted == False,  # noqa: E712
        )
    )
    session = res.scalars().first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    manager = DataAdequacyManager()
    manager.session_state = session.state or {}
    manager.llm_call_count = int(session.llm_calls or 0)
    status = manager.get_session_status()
    status["status"] = session.status.value
    return status

@router.get("/domains", response_model=List[str])
async def get_supported_domains(user: AuthUser = Depends(require_permission(Permission.READ_DATA))):
    """Get list of supported domains."""
    from app.agents.data_adequacy.config import config
    return list(config.DOMAIN_CONFIGS.keys())
