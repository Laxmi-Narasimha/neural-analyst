# AI Enterprise Data Analyst - Analysis API Routes
# Production-grade REST API for analysis management

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.api.schemas import (
    APIResponse,
    PaginatedResponse,
    PaginationMeta,
    AnalysisCreate,
    AnalysisResponse,
    AnalysisDetailResponse,
    AnalysisType,
    AnalysisStatus,
)
from app.core.logging import get_logger, LogContext
from app.services.database import get_db_session
from app.agents import (
    get_orchestrator,
    get_eda_agent,
    get_statistical_agent,
    get_ml_agent,
    AgentContext,
)

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# Analysis Repository
# ============================================================================

class AnalysisRepository:
    """Repository for Analysis CRUD operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def create(
        self,
        name: str,
        description: Optional[str],
        dataset_id: UUID,
        user_id: UUID,
        analysis_type: AnalysisType,
        config: dict[str, Any],
    ):
        """Create new analysis record."""
        from app.models import Analysis
        
        analysis = Analysis(
            name=name,
            description=description,
            dataset_id=dataset_id,
            created_by=user_id,
            analysis_type=analysis_type.value,
            status=AnalysisStatus.QUEUED.value,
            config=config
        )
        
        self.session.add(analysis)
        await self.session.commit()
        await self.session.refresh(analysis)
        
        return analysis
    
    async def get_by_id(self, analysis_id: UUID):
        """Get analysis by ID."""
        from app.models import Analysis
        
        query = select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.is_deleted == False
        )
        result = await self.session.execute(query)
        return result.scalars().first()
    
    async def list_analyses(
        self,
        user_id: UUID,
        dataset_id: Optional[UUID] = None,
        skip: int = 0,
        limit: int = 20,
        status_filter: Optional[AnalysisStatus] = None,
    ):
        """List analyses for user."""
        from app.models import Analysis
        
        query = select(Analysis).where(
            Analysis.created_by == user_id,
            Analysis.is_deleted == False
        )
        
        if dataset_id:
            query = query.where(Analysis.dataset_id == dataset_id)
        
        if status_filter:
            query = query.where(Analysis.status == status_filter.value)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0
        
        # Apply pagination
        query = query.order_by(Analysis.created_at.desc()).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        analyses = result.scalars().all()
        
        return analyses, total
    
    async def update(self, analysis_id: UUID, data: dict[str, Any]):
        """Update analysis."""
        analysis = await self.get_by_id(analysis_id)
        if not analysis:
            return None
        
        for key, value in data.items():
            if hasattr(analysis, key):
                setattr(analysis, key, value)
        
        await self.session.commit()
        await self.session.refresh(analysis)
        
        return analysis
    
    async def delete(self, analysis_id: UUID):
        """Soft delete analysis."""
        analysis = await self.get_by_id(analysis_id)
        if analysis:
            analysis.soft_delete()
            await self.session.commit()
            return True
        return False


# ============================================================================
# Background Task for Running Analysis
# ============================================================================

async def run_analysis_task(
    analysis_id: UUID,
    dataset_id: UUID,
    analysis_type: str,
    config: dict[str, Any]
):
    """Background task to run analysis."""
    context = LogContext(
        component="AnalysisTask",
        operation="run_analysis"
    )
    
    logger.info(
        f"Starting analysis {analysis_id}",
        context=context,
        analysis_type=analysis_type
    )
    
    try:
        # Create agent context
        agent_context = AgentContext(
            request_id=analysis_id,
            dataset_id=dataset_id,
            task_description=f"Perform {analysis_type} analysis on dataset",
            metadata=config
        )
        
        # Select appropriate agent
        if analysis_type == AnalysisType.EDA.value:
            agent = get_eda_agent()
        elif analysis_type == AnalysisType.STATISTICAL.value:
            agent = get_statistical_agent()
        elif analysis_type in [AnalysisType.ML_CLASSIFICATION.value, AnalysisType.ML_REGRESSION.value]:
            agent = get_ml_agent()
        else:
            agent = get_orchestrator()
        
        # Run analysis
        results = await agent.execute(agent_context)
        
        # Update analysis record (would need session)
        logger.info(
            f"Analysis {analysis_id} completed",
            context=context
        )
        
        return results
        
    except Exception as e:
        logger.error(
            f"Analysis {analysis_id} failed: {e}",
            context=context,
            exc_info=True
        )
        raise


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "",
    response_model=APIResponse[AnalysisResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create analysis",
    description="Create a new data analysis job"
)
async def create_analysis(
    request: AnalysisCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create and queue a new analysis.
    
    The analysis will be processed in the background.
    """
    user_id = uuid4()  # TODO: Get from auth
    
    repo = AnalysisRepository(db)
    
    analysis = await repo.create(
        name=request.name,
        description=request.description,
        dataset_id=request.dataset_id,
        user_id=user_id,
        analysis_type=request.analysis_type,
        config=request.config
    )
    
    # Queue background task
    background_tasks.add_task(
        run_analysis_task,
        analysis.id,
        request.dataset_id,
        request.analysis_type.value,
        request.config
    )
    
    response = AnalysisResponse(
        id=analysis.id,
        name=analysis.name,
        description=analysis.description,
        dataset_id=analysis.dataset_id,
        analysis_type=request.analysis_type,
        status=AnalysisStatus.QUEUED,
        progress=0.0,
        status_message="Analysis queued",
        started_at=None,
        completed_at=None,
        duration_seconds=None,
        created_at=analysis.created_at
    )
    
    return APIResponse.success(
        data=response,
        message="Analysis created and queued"
    )


@router.get(
    "",
    response_model=PaginatedResponse[AnalysisResponse],
    summary="List analyses",
    description="Get paginated list of analyses"
)
async def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    dataset_id: Optional[UUID] = Query(None),
    status_filter: Optional[AnalysisStatus] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db_session),
):
    """List all analyses with pagination."""
    user_id = uuid4()  # TODO: Get from auth
    
    repo = AnalysisRepository(db)
    skip = (page - 1) * page_size
    
    analyses, total = await repo.list_analyses(
        user_id=user_id,
        dataset_id=dataset_id,
        skip=skip,
        limit=page_size,
        status_filter=status_filter
    )
    
    total_pages = (total + page_size - 1) // page_size
    
    response_data = [
        AnalysisResponse(
            id=a.id,
            name=a.name,
            description=a.description,
            dataset_id=a.dataset_id,
            analysis_type=AnalysisType(a.analysis_type),
            status=AnalysisStatus(a.status),
            progress=a.progress or 0.0,
            status_message=a.status_message,
            started_at=a.started_at,
            completed_at=a.completed_at,
            duration_seconds=a.duration_seconds,
            created_at=a.created_at
        )
        for a in analyses
    ]
    
    return PaginatedResponse(
        status="success",
        data=response_data,
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    )


@router.get(
    "/{analysis_id}",
    response_model=APIResponse[AnalysisDetailResponse],
    summary="Get analysis",
    description="Get detailed analysis results"
)
async def get_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get analysis with full results."""
    repo = AnalysisRepository(db)
    
    analysis = await repo.get_by_id(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    response = AnalysisDetailResponse(
        id=analysis.id,
        name=analysis.name,
        description=analysis.description,
        dataset_id=analysis.dataset_id,
        analysis_type=AnalysisType(analysis.analysis_type),
        status=AnalysisStatus(analysis.status),
        progress=analysis.progress or 0.0,
        status_message=analysis.status_message,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        duration_seconds=analysis.duration_seconds,
        created_at=analysis.created_at,
        config=analysis.config or {},
        results=analysis.results or {},
        insights=analysis.insights or [],
        visualizations=analysis.visualizations or [],
        error_message=analysis.error_message,
        agent_trace=analysis.agent_trace or []
    )
    
    return APIResponse.success(data=response)


@router.delete(
    "/{analysis_id}",
    response_model=APIResponse[None],
    summary="Delete analysis",
    description="Delete an analysis"
)
async def delete_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete analysis."""
    repo = AnalysisRepository(db)
    
    success = await repo.delete(analysis_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    return APIResponse.success(data=None, message="Analysis deleted")


@router.post(
    "/{analysis_id}/cancel",
    response_model=APIResponse[AnalysisResponse],
    summary="Cancel analysis",
    description="Cancel a running analysis"
)
async def cancel_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Cancel a running analysis."""
    repo = AnalysisRepository(db)
    
    analysis = await repo.get_by_id(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    if analysis.status not in [AnalysisStatus.QUEUED.value, AnalysisStatus.RUNNING.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis cannot be cancelled"
        )
    
    analysis = await repo.update(analysis_id, {
        "status": AnalysisStatus.CANCELLED.value,
        "completed_at": datetime.utcnow()
    })
    
    response = AnalysisResponse(
        id=analysis.id,
        name=analysis.name,
        description=analysis.description,
        dataset_id=analysis.dataset_id,
        analysis_type=AnalysisType(analysis.analysis_type),
        status=AnalysisStatus.CANCELLED,
        progress=analysis.progress or 0.0,
        status_message="Analysis cancelled",
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        duration_seconds=analysis.duration_seconds,
        created_at=analysis.created_at
    )
    
    return APIResponse.success(data=response, message="Analysis cancelled")
