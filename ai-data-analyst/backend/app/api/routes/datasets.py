# AI Enterprise Data Analyst - Dataset API Routes
# Production-grade REST API for dataset management

from __future__ import annotations

import io
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    APIResponse,
    PaginatedResponse,
    PaginationMeta,
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetDetailResponse,
    DatasetUploadResponse,
    DatasetStatus,
    ColumnInfo,
)
from app.core.config import settings
from app.core.exceptions import (
    FileFormatException,
    FileUploadException,
    DataNotFoundException,
)
from app.core.logging import get_logger, LogContext
from app.services.database import get_db_session
from app.services.data_ingestion import get_ingestion_service, FileFormat

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# Dataset Repository
# ============================================================================

class DatasetRepository:
    """Repository for Dataset CRUD operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def create(
        self,
        name: str,
        description: Optional[str],
        original_filename: str,
        file_path: str,
        file_size: int,
        file_format: str,
        owner_id: UUID,
        tags: list[str] = None,
    ):
        """Create new dataset record."""
        from app.models import Dataset
        
        dataset = Dataset(
            name=name,
            description=description,
            original_filename=original_filename,
            file_path=file_path,
            file_size_bytes=file_size,
            file_format=file_format,
            owner_id=owner_id,
            tags=tags or [],
            status=DatasetStatus.PENDING
        )
        
        self.session.add(dataset)
        await self.session.commit()
        await self.session.refresh(dataset)
        
        return dataset
    
    async def get_by_id(self, dataset_id: UUID):
        """Get dataset by ID."""
        from app.models import Dataset
        from sqlalchemy import select
        
        query = select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.is_deleted == False
        )
        result = await self.session.execute(query)
        return result.scalars().first()
    
    async def get_all(
        self,
        owner_id: UUID,
        skip: int = 0,
        limit: int = 20,
        status_filter: Optional[DatasetStatus] = None,
        search: Optional[str] = None,
    ):
        """Get all datasets for owner with filtering."""
        from app.models import Dataset
        from sqlalchemy import select, func, or_
        
        query = select(Dataset).where(
            Dataset.owner_id == owner_id,
            Dataset.is_deleted == False
        )
        
        if status_filter:
            query = query.where(Dataset.status == status_filter)
        
        if search:
            query = query.where(
                or_(
                    Dataset.name.ilike(f"%{search}%"),
                    Dataset.description.ilike(f"%{search}%")
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply pagination
        query = query.order_by(Dataset.created_at.desc()).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        datasets = result.scalars().all()
        
        return datasets, total
    
    async def update(self, dataset_id: UUID, data: dict[str, Any]):
        """Update dataset."""
        dataset = await self.get_by_id(dataset_id)
        if not dataset:
            raise DataNotFoundException("Dataset", dataset_id)
        
        for key, value in data.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
        
        await self.session.commit()
        await self.session.refresh(dataset)
        
        return dataset
    
    async def delete(self, dataset_id: UUID):
        """Soft delete dataset."""
        dataset = await self.get_by_id(dataset_id)
        if not dataset:
            raise DataNotFoundException("Dataset", dataset_id)
        
        dataset.soft_delete()
        await self.session.commit()
        
        return True


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/upload",
    response_model=APIResponse[DatasetUploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload dataset file",
    description="Upload a data file (CSV, Excel, JSON, Parquet) for analysis"
)
async def upload_dataset(
    file: UploadFile = File(..., description="Data file to upload"),
    name: str = Form(..., min_length=1, max_length=255, description="Dataset name"),
    description: Optional[str] = Form(None, max_length=2000, description="Dataset description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    db: AsyncSession = Depends(get_db_session),
    # user_id: UUID = Depends(get_current_user_id),  # TODO: Add auth
):
    """
    Upload and process a dataset file.
    
    Supported formats:
    - CSV (.csv)
    - TSV (.tsv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Parquet (.parquet)
    """
    context = LogContext(component="DatasetAPI", operation="upload")
    
    # For demo, using a fixed user ID (replace with auth)
    from uuid import uuid4
    user_id = uuid4()
    
    # Validate file extension
    filename = file.filename or "unknown"
    extension = Path(filename).suffix.lower().lstrip('.')
    
    if extension not in settings.allowed_extensions:
        raise FileFormatException(
            filename=filename,
            actual_format=extension,
            supported_formats=settings.allowed_extensions
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise FileUploadException(
            filename=filename,
            reason=f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum ({settings.max_upload_size_mb}MB)"
        )
    
    try:
        # Create upload directory
        upload_dir = Path(settings.upload_directory)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique file path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{user_id}_{filename}"
        file_path = upload_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(
            f"File uploaded: {filename}",
            context=context,
            file_size=file_size,
            format=extension
        )
        
        # Create database record
        repo = DatasetRepository(db)
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        
        dataset = await repo.create(
            name=name,
            description=description,
            original_filename=filename,
            file_path=str(file_path),
            file_size=file_size,
            file_format=extension,
            owner_id=user_id,
            tags=tag_list
        )
        
        # TODO: Queue background processing task
        # celery_app.send_task("process_dataset", args=[str(dataset.id)])
        
        response = DatasetUploadResponse(
            dataset_id=dataset.id,
            filename=filename,
            file_size_bytes=file_size,
            status=DatasetStatus.PENDING,
            message="Dataset uploaded successfully. Processing will begin shortly."
        )
        
        return APIResponse.success(
            data=response,
            message="Dataset uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}", context=context, exc_info=True)
        raise FileUploadException(filename=filename, reason=str(e))


@router.get(
    "",
    response_model=PaginatedResponse[DatasetResponse],
    summary="List datasets",
    description="Get paginated list of datasets for the current user"
)
async def list_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[DatasetStatus] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, max_length=100, description="Search in name/description"),
    db: AsyncSession = Depends(get_db_session),
):
    """List all datasets with pagination and filtering."""
    # For demo, using a fixed user ID (replace with auth)
    from uuid import uuid4
    user_id = uuid4()
    
    repo = DatasetRepository(db)
    skip = (page - 1) * page_size
    
    datasets, total = await repo.get_all(
        owner_id=user_id,
        skip=skip,
        limit=page_size,
        status_filter=status,
        search=search
    )
    
    total_pages = (total + page_size - 1) // page_size
    
    dataset_responses = [
        DatasetResponse(
            id=d.id,
            name=d.name,
            description=d.description,
            original_filename=d.original_filename,
            file_size_bytes=d.file_size_bytes,
            file_format=d.file_format,
            status=d.status,
            error_message=d.error_message,
            row_count=d.row_count,
            column_count=d.column_count,
            quality_score=d.quality_score,
            tags=d.tags,
            created_at=d.created_at,
            updated_at=d.updated_at
        )
        for d in datasets
    ]
    
    return PaginatedResponse(
        status="success",
        data=dataset_responses,
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
    "/{dataset_id}",
    response_model=APIResponse[DatasetDetailResponse],
    summary="Get dataset details",
    description="Get detailed information about a specific dataset"
)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get dataset by ID with full details."""
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    
    if not dataset:
        raise DataNotFoundException("Dataset", dataset_id)
    
    # Convert columns to ColumnInfo
    columns = [
        ColumnInfo(
            name=c.name,
            original_name=c.original_name,
            position=c.position,
            inferred_type=c.inferred_type,
            semantic_type=c.semantic_type,
            null_count=c.null_count,
            null_percentage=c.null_percentage,
            unique_count=c.unique_count,
            statistics=c.statistics
        )
        for c in dataset.columns
    ]
    
    response = DatasetDetailResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        original_filename=dataset.original_filename,
        file_size_bytes=dataset.file_size_bytes,
        file_format=dataset.file_format,
        status=dataset.status,
        error_message=dataset.error_message,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        quality_score=dataset.quality_score,
        tags=dataset.tags,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        columns=columns,
        schema_info=dataset.schema_info,
        quality_report=dataset.quality_report,
        profile_report=dataset.profile_report
    )
    
    return APIResponse.success(data=response)


@router.patch(
    "/{dataset_id}",
    response_model=APIResponse[DatasetResponse],
    summary="Update dataset",
    description="Update dataset metadata"
)
async def update_dataset(
    dataset_id: UUID,
    update_data: DatasetUpdate,
    db: AsyncSession = Depends(get_db_session),
):
    """Update dataset metadata."""
    repo = DatasetRepository(db)
    
    update_dict = update_data.model_dump(exclude_unset=True)
    dataset = await repo.update(dataset_id, update_dict)
    
    response = DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        original_filename=dataset.original_filename,
        file_size_bytes=dataset.file_size_bytes,
        file_format=dataset.file_format,
        status=dataset.status,
        error_message=dataset.error_message,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        quality_score=dataset.quality_score,
        tags=dataset.tags,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )
    
    return APIResponse.success(
        data=response,
        message="Dataset updated successfully"
    )


@router.delete(
    "/{dataset_id}",
    response_model=APIResponse[None],
    summary="Delete dataset",
    description="Delete a dataset (soft delete)"
)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete dataset."""
    repo = DatasetRepository(db)
    await repo.delete(dataset_id)
    
    return APIResponse.success(
        data=None,
        message="Dataset deleted successfully"
    )


@router.post(
    "/{dataset_id}/process",
    response_model=APIResponse[DatasetResponse],
    summary="Process dataset",
    description="Manually trigger dataset processing"
)
async def process_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Process dataset file to extract schema and profile."""
    context = LogContext(component="DatasetAPI", operation="process")
    
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    
    if not dataset:
        raise DataNotFoundException("Dataset", dataset_id)
    
    if dataset.status not in [DatasetStatus.PENDING, DatasetStatus.ERROR]:
        return APIResponse.success(
            data=DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                original_filename=dataset.original_filename,
                file_size_bytes=dataset.file_size_bytes,
                file_format=dataset.file_format,
                status=dataset.status,
                error_message=dataset.error_message,
                row_count=dataset.row_count,
                column_count=dataset.column_count,
                quality_score=dataset.quality_score,
                tags=dataset.tags,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at
            ),
            message="Dataset already processed"
        )
    
    try:
        # Update status
        await repo.update(dataset_id, {"status": DatasetStatus.PROCESSING})
        
        # Process file
        ingestion_service = get_ingestion_service()
        
        with open(dataset.file_path, "rb") as f:
            df, profile = ingestion_service.ingest_file(
                f,
                dataset.original_filename
            )
        
        # Update dataset with profile
        from app.models import DatasetColumn
        
        update_data = {
            "status": DatasetStatus.READY,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "schema_info": {"columns": [c.to_dict() for c in profile.columns]},
            "quality_score": profile.overall_quality_score,
            "quality_report": {
                "completeness": profile.completeness_score,
                "uniqueness": profile.uniqueness_score,
                "consistency": profile.consistency_score,
                "warnings": profile.warnings
            },
            "profile_report": profile.to_dict()
        }
        
        dataset = await repo.update(dataset_id, update_data)
        
        logger.info(
            f"Dataset processed: {dataset.name}",
            context=context,
            rows=profile.row_count,
            columns=profile.column_count,
            quality=profile.overall_quality_score
        )
        
        response = DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            original_filename=dataset.original_filename,
            file_size_bytes=dataset.file_size_bytes,
            file_format=dataset.file_format,
            status=dataset.status,
            error_message=dataset.error_message,
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            quality_score=dataset.quality_score,
            tags=dataset.tags,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at
        )
        
        return APIResponse.success(
            data=response,
            message="Dataset processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", context=context, exc_info=True)
        
        await repo.update(dataset_id, {
            "status": DatasetStatus.ERROR,
            "error_message": str(e)
        })
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process dataset: {str(e)}"
        )
