"""API endpoints for Data Quality and Adequacy analysis."""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from app.agents.data_adequacy.manager import DataAdequacyManager, start_validation, continue_session_validation

router = APIRouter()

# In-memory session storage (In production, use Redis/Database)
active_sessions: Dict[str, DataAdequacyManager] = {}

class ValidationRequest(BaseModel):
    goal: str
    domain: str = "general"
    files: List[str] = []

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
async def start_validation_session(request: ValidationRequest):
    """Start a new data adequacy validation session."""
    try:
        manager = DataAdequacyManager()
        result = await manager.run_validation(
            user_goal=request.goal,
            domain=request.domain,
            files=request.files
        )
        
        if result.get("success"):
            session_id = result.get("session_id") or manager.session_state.get("session_id")
            if session_id:
                active_sessions[session_id] = manager
                
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/continue", response_model=Dict[str, Any])
async def continue_validation_session(request: AnswerRequest):
    """Continue validation with user answers."""
    session_id = request.session_id
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    manager = active_sessions[session_id]
    try:
        result = await manager.continue_validation(request.answers)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}", response_model=Dict[str, Any])
async def get_session_status(session_id: str):
    """Get status of an active session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    manager = active_sessions[session_id]
    return manager.get_session_status()

@router.get("/domains", response_model=List[str])
async def get_supported_domains():
    """Get list of supported domains."""
    from app.agents.data_adequacy.config import config
    return list(config.DOMAIN_CONFIGS.keys())
