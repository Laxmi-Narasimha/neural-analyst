# AI Enterprise Data Analyst - Dataset API Routes
# Production-grade REST API for dataset management

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
    DatasetVersionResponse,
    DatasetTransformStep,
    DatasetTransformPreviewRequest,
    DatasetTransformPreviewResponse,
    DatasetTransformApplyRequest,
    DatasetTransformApplyResponse,
    DatasetTransformSuggestRequest,
    DatasetTransformSuggestion,
    DatasetTransformSuggestResponse,
    DatasetQueryRequest,
    DatasetQueryResponse,
    ArtifactResponse,
)
from app.core.config import settings
from app.core.exceptions import (
    FileFormatException,
    FileUploadException,
    DataNotFoundException,
)
from app.core.logging import get_logger, LogContext
from app.core.serialization import to_jsonable
from app.compute.artifacts import ArtifactStore, TableStorageFormat
from app.models import Artifact as ArtifactModel
from app.services.artifact_index import ArtifactIndexService
from app.services.database import get_db_session
from app.services.dataset_loader import DatasetLoaderService
from app.api.routes.auth import require_permission
from app.services.auth_service import AuthUser, Permission

logger = get_logger(__name__)

router = APIRouter()

def _schema_dataset_status(value: Any) -> DatasetStatus:
    if isinstance(value, DatasetStatus):
        return value
    v = getattr(value, "value", value)
    return DatasetStatus(str(v))


def _sanitize_filename(name: str) -> str:
    # Drop any path components and keep a conservative character set to avoid
    # weird filesystem edge cases (Windows reserved chars, path traversal, etc).
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "upload"


def _run_duckdb_query_sync(*, file_path: str, file_format: str, sql: str) -> tuple["pd.DataFrame", float]:
    import duckdb
    import pandas as pd

    def _quote(value: str) -> str:
        v = str(value or "")
        v = v.replace("\\", "/")
        v = v.replace("'", "''")
        return f"'{v}'"

    con = duckdb.connect(database=":memory:")
    try:
        fmt = str(file_format or "").lower().strip()
        if fmt == "parquet":
            con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet({_quote(file_path)})")
        elif fmt in {"csv", "tsv"}:
            con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_csv_auto({_quote(file_path)})")
        else:
            raise ValueError("Dataset SQL is supported for CSV/TSV/Parquet only")

        t0 = time.perf_counter()
        df: pd.DataFrame = con.execute(str(sql)).df()
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return df, float(dur_ms)
    finally:
        try:
            con.close()
        except Exception:
            pass


def _load_query_source_df(*, file_path: str, file_format: str) -> "pd.DataFrame":
    import pandas as pd

    fmt = str(file_format or "").lower().strip()
    if fmt == "parquet":
        return pd.read_parquet(file_path)
    if fmt == "csv":
        return pd.read_csv(file_path)
    if fmt == "tsv":
        return pd.read_csv(file_path, sep="\t")
    raise ValueError("Dataset SQL is supported for CSV/TSV/Parquet only")


def _run_sqlite_query_sync(*, file_path: str, file_format: str, sql: str) -> tuple["pd.DataFrame", float]:
    import sqlite3
    import pandas as pd

    src_df = _load_query_source_df(file_path=file_path, file_format=file_format)
    con = sqlite3.connect(":memory:")
    try:
        src_df.to_sql("dataset", con, if_exists="replace", index=False)
        t0 = time.perf_counter()
        out_df: pd.DataFrame = pd.read_sql_query(str(sql), con)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return out_df, float(dur_ms)
    finally:
        try:
            con.close()
        except Exception:
            pass


def _run_dataset_query_sync(*, file_path: str, file_format: str, sql: str) -> tuple["pd.DataFrame", float]:
    try:
        return _run_duckdb_query_sync(file_path=file_path, file_format=file_format, sql=sql)
    except ModuleNotFoundError as e:
        # Keep queries working in lightweight OSS/dev setups where DuckDB is optional.
        name = str(getattr(e, "name", "") or "").lower()
        if name not in {"duckdb", "_duckdb"} and "duckdb" not in str(e).lower():
            raise
        return _run_sqlite_query_sync(file_path=file_path, file_format=file_format, sql=sql)
    except ImportError as e:
        if "duckdb" not in str(e).lower():
            raise
        return _run_sqlite_query_sync(file_path=file_path, file_format=file_format, sql=sql)


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
        from app.models import Dataset, DatasetStatus as ModelDatasetStatus
        
        dataset = Dataset(
            name=name,
            description=description,
            original_filename=original_filename,
            file_path=file_path,
            file_size_bytes=file_size,
            file_format=file_format,
            owner_id=owner_id,
            tags=tags or [],
            status=ModelDatasetStatus.PENDING
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
        from app.models import Dataset, DatasetStatus as ModelDatasetStatus
        from sqlalchemy import select, func, or_
        
        query = select(Dataset).where(
            Dataset.owner_id == owner_id,
            Dataset.is_deleted == False
        )
        
        if status_filter:
            query = query.where(Dataset.status == ModelDatasetStatus(status_filter.value))
        
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Data file to upload"),
    name: str = Form(..., min_length=1, max_length=255, description="Dataset name"),
    description: Optional[str] = Form(None, max_length=2000, description="Dataset description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
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
    
    user_id = user.user_id
    
    # Validate file extension
    raw_filename = file.filename or "unknown"
    original_filename = Path(raw_filename).name or "unknown"
    safe_original_filename = _sanitize_filename(original_filename)
    extension = Path(original_filename).suffix.lower().lstrip(".")
    
    if extension not in settings.allowed_extensions:
        raise FileFormatException(
            filename=original_filename,
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
            filename=original_filename,
            reason=f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum ({settings.max_upload_size_mb}MB)"
        )
    
    try:
        # Store the file via the configured object store (local disk or S3).
        from app.services.object_store import get_object_store

        obj = get_object_store()
        try:
            file.file.seek(0)
        except Exception:
            pass

        storage_path = await asyncio.to_thread(
            obj.put_upload,
            owner_id=user_id,
            original_filename=safe_original_filename,
            src=file.file,
            content_type=getattr(file, "content_type", None),
        )
        
        logger.info(
            f"File uploaded: {original_filename}",
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
            original_filename=original_filename,
            file_path=str(storage_path),
            file_size=file_size,
            file_format=extension,
            owner_id=user_id,
            tags=tag_list
        )

        # Create a persisted job record for processing.
        from app.models import Job, JobType, JobStatus
        from app.workers.dispatcher import enqueue_dataset_processing

        job = Job(
            owner_id=user_id,
            dataset_id=dataset.id,
            job_type=JobType.DATASET_PROCESSING,
            status=JobStatus.QUEUED,
            progress=0.0,
            status_message="Queued dataset processing",
            payload={"dataset_id": str(dataset.id)},
            result={},
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        enqueue_dataset_processing(background_tasks=background_tasks, dataset_id=dataset.id, job_id=job.id)
        
        response = DatasetUploadResponse(
            dataset_id=dataset.id,
            job_id=job.id,
            filename=original_filename,
            file_size_bytes=file_size,
            status=_schema_dataset_status(dataset.status),
            message="Dataset uploaded successfully. Processing will begin shortly."
        )
        
        return APIResponse.success(
            data=response,
            message="Dataset uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}", context=context, exc_info=True)
        raise FileUploadException(filename=original_filename, reason=str(e))


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
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    """List all datasets with pagination and filtering."""
    user_id = user.user_id
    
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
            status=_schema_dataset_status(d.status),
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
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    """Get dataset by ID with full details."""
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    
    if not dataset:
        raise DataNotFoundException("Dataset", dataset_id)

    if dataset.owner_id != user.user_id:
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
        status=_schema_dataset_status(dataset.status),
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
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    """Update dataset metadata."""
    repo = DatasetRepository(db)
    existing = await repo.get_by_id(dataset_id)
    if not existing or existing.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)
    
    update_dict = update_data.model_dump(exclude_unset=True)
    dataset = await repo.update(dataset_id, update_dict)
    
    response = DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        original_filename=dataset.original_filename,
        file_size_bytes=dataset.file_size_bytes,
        file_format=dataset.file_format,
        status=_schema_dataset_status(dataset.status),
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
    user: AuthUser = Depends(require_permission(Permission.DELETE_DATA)),
):
    """Delete dataset."""
    repo = DatasetRepository(db)
    existing = await repo.get_by_id(dataset_id)
    if not existing or existing.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)
    await repo.delete(dataset_id)
    
    return APIResponse.success(
        data=None,
        message="Dataset deleted successfully"
    )


@router.post(
    "/{dataset_id}/purge",
    response_model=APIResponse[dict[str, Any]],
    summary="Purge dataset (delete blobs + metadata)",
    description="Destructive purge: delete stored dataset files, derived artifacts, and hard-delete metadata rows.",
)
async def purge_dataset(
    dataset_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.DELETE_DATA)),
):
    from app.models import Dataset as DatasetModel
    from app.models import Job as JobModel, JobStatus as ModelJobStatus, JobType as ModelJobType
    from app.workers.dispatcher import enqueue_dataset_purge

    ds = (await db.execute(
        select(DatasetModel).where(DatasetModel.id == dataset_id, DatasetModel.owner_id == user.user_id)
    )).scalars().first()
    if ds is None:
        raise DataNotFoundException("Dataset", dataset_id)

    # Hide from UI immediately; purge job will remove storage + metadata fully.
    if not bool(getattr(ds, "is_deleted", False)):
        ds.soft_delete()
        ds.updated_by = user.user_id
        await db.commit()

    # Create a durable job record so users can track progress.
    job = JobModel(
        owner_id=user.user_id,
        dataset_id=dataset_id,
        job_type=ModelJobType.DATASET_PURGE,
        status=ModelJobStatus.QUEUED,
        progress=0.0,
        status_message="Queued dataset purge",
        payload={"dataset_id": str(dataset_id)},
        result={},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    enqueue_dataset_purge(background_tasks=background_tasks, dataset_id=dataset_id, job_id=job.id)

    return APIResponse.success(
        data={"dataset_id": str(dataset_id), "job_id": str(job.id)},
        message="Dataset purge started",
    )


@router.post(
    "/{dataset_id}/process",
    response_model=APIResponse[DatasetResponse],
    summary="Process dataset",
    description="Manually trigger dataset processing"
)
async def process_dataset(
    dataset_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    """Process dataset file to extract schema and profile."""
    context = LogContext(component="DatasetAPI", operation="process")
    
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    
    if not dataset:
        raise DataNotFoundException("Dataset", dataset_id)

    if dataset.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)
    
    from app.models import DatasetStatus as ModelDatasetStatus

    if dataset.status == ModelDatasetStatus.READY:
        return APIResponse.success(
            data=DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                original_filename=dataset.original_filename,
                file_size_bytes=dataset.file_size_bytes,
                file_format=dataset.file_format,
                status=_schema_dataset_status(dataset.status),
                error_message=dataset.error_message,
                row_count=dataset.row_count,
                column_count=dataset.column_count,
                quality_score=dataset.quality_score,
                tags=dataset.tags,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
            ),
            message="Dataset already processed",
        )

    # Mark as processing and enqueue background work.
    dataset = await repo.update(dataset_id, {"status": ModelDatasetStatus.PROCESSING, "error_message": None})

    from app.models import Job, JobType, JobStatus
    from app.workers.dispatcher import enqueue_dataset_processing

    job = Job(
        owner_id=user.user_id,
        dataset_id=dataset.id,
        job_type=JobType.DATASET_PROCESSING,
        status=JobStatus.QUEUED,
        progress=0.0,
        status_message="Queued dataset processing",
        payload={"dataset_id": str(dataset.id)},
        result={},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    enqueue_dataset_processing(background_tasks=background_tasks, dataset_id=dataset.id, job_id=job.id)

    logger.info("Dataset processing enqueued", context=context, dataset_id=str(dataset.id))

    response = DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        original_filename=dataset.original_filename,
        file_size_bytes=dataset.file_size_bytes,
        file_format=dataset.file_format,
        status=_schema_dataset_status(dataset.status),
        error_message=dataset.error_message,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        quality_score=dataset.quality_score,
        tags=dataset.tags,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )

    return APIResponse.success(
        data=response,
        message="Dataset processing started",
        meta={"job_id": str(job.id)},
    )


@router.post(
    "/{dataset_id}/query",
    response_model=APIResponse[DatasetQueryResponse],
    summary="Query uploaded dataset (SQL)",
    description="Run a guarded, read-only SQL query against an uploaded dataset using DuckDB; produces a table artifact.",
)
async def query_uploaded_dataset(
    dataset_id: UUID,
    request: DatasetQueryRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    if not dataset or dataset.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)

    from app.models import DatasetStatus as ModelDatasetStatus

    if dataset.status != ModelDatasetStatus.READY:
        raise HTTPException(status_code=400, detail="Dataset must be processed (ready) before querying")

    from app.services.sql_safety import UnsafeSQLError, enforce_row_limit, validate_dataset_sql

    try:
        normalized = validate_dataset_sql(request.query)
    except UnsafeSQLError as e:
        raise HTTPException(status_code=400, detail=str(e))

    max_rows = int(request.max_rows or settings.connections.max_query_rows)
    max_rows = max(1, min(max_rows, int(settings.connections.max_query_rows)))
    limited = enforce_row_limit(normalized, max_rows)

    timeout = int(settings.connections.query_timeout_seconds)
    if request.timeout_seconds is not None:
        timeout = max(1, min(int(request.timeout_seconds), int(settings.connections.query_timeout_seconds)))

    file_path = str(getattr(dataset, "file_path", "") or "").strip()
    file_format = str(getattr(dataset, "file_format", "") or "").strip().lower()
    if not file_path:
        raise HTTPException(status_code=400, detail="Dataset is missing a file_path")

    # file_path may be on local disk or remote object storage (S3). DuckDB requires a local file path.
    try:
        from app.services.object_store import get_object_store

        obj = get_object_store()
        local_path = obj.ensure_local_path(
            file_path,
            expected_size_bytes=int(getattr(dataset, "file_size_bytes", 0) or 0) or None,
            filename_hint=str(getattr(dataset, "original_filename", "") or ""),
        )
        file_path = str(local_path)
    except Exception:
        # Best-effort: keep existing behavior for local paths.
        pass

    try:
        df, dur_ms = await asyncio.wait_for(
            asyncio.to_thread(_run_dataset_query_sync, file_path=file_path, file_format=file_format, sql=limited),
            timeout=float(timeout),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"Query timed out after {timeout}s")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    dataset_version = None
    if isinstance(getattr(dataset, "profile_report", None), dict):
        dataset_version = str(dataset.profile_report.get("file_hash") or "").strip() or None

    store = ArtifactStore()
    ref = store.write_table(
        name=f"dataset_sql:{dataset_id}",
        df=df,
        storage_format=TableStorageFormat.PARQUET,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        operator_name="dataset_sql",
        operator_params={"query": normalized, "max_rows": int(max_rows)},
    )
    await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=[ref])

    art = (
        (
            await db.execute(
                select(ArtifactModel).where(
                    ArtifactModel.id == ref.artifact_id,
                    ArtifactModel.owner_id == user.user_id,
                    ArtifactModel.is_deleted == False,  # noqa: E712
                )
            )
        )
        .scalars()
        .first()
    )
    if art is None:
        raise HTTPException(status_code=500, detail="Artifact indexing failed")

    resp = DatasetQueryResponse(
        columns=[str(c) for c in df.columns.tolist()],
        row_count=int(df.shape[0]),
        execution_time_ms=float(dur_ms),
        artifact=ArtifactResponse.model_validate(art),
    )

    # PII-safe default for UI previews: redact preview rows when the dataset schema flags PII.
    try:
        from app.core.redaction import mask_preview

        schema = getattr(dataset, "schema_info", None) or {}
        cols = schema.get("columns") if isinstance(schema, dict) else None
        pii_cols: set[str] = set()
        if isinstance(cols, list):
            for c in cols:
                if not isinstance(c, dict):
                    continue
                name = c.get("name")
                if not name:
                    continue
                stats = c.get("statistics") if isinstance(c.get("statistics"), dict) else {}
                is_pii = bool(c.get("is_potential_pii") or stats.get("is_potential_pii"))
                if is_pii:
                    pii_cols.add(str(name))
        if pii_cols and isinstance(resp.artifact.preview, dict):
            resp.artifact.preview = mask_preview(resp.artifact.preview, pii_cols)
    except Exception:
        pass
    return APIResponse.success(data=resp, message="Query executed")


@router.get(
    "/{dataset_id}/versions",
    response_model=PaginatedResponse[DatasetVersionResponse],
    summary="List dataset versions",
    description="List immutable dataset versions (uploads + transformations) for this dataset.",
)
async def list_dataset_versions(
    dataset_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    from sqlalchemy import func, select

    from app.models import DatasetVersion

    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    if not dataset or dataset.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)

    skip = (page - 1) * page_size
    q = select(DatasetVersion).where(
        DatasetVersion.dataset_id == dataset_id,
        DatasetVersion.owner_id == user.user_id,
        DatasetVersion.is_deleted == False,  # noqa: E712
    )

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    versions = (await db.execute(q.order_by(DatasetVersion.created_at.desc()).offset(skip).limit(page_size))).scalars().all()

    active_path = str(dataset.file_path)
    items: list[DatasetVersionResponse] = []
    for v in versions:
        spec = v.transform_spec if isinstance(getattr(v, "transform_spec", None), dict) else {}
        items.append(
            DatasetVersionResponse(
                id=v.id,
                dataset_id=v.dataset_id,
                version_hash=v.version_hash,
                label=v.label,
                parent_version_hash=v.parent_version_hash,
                transform_spec=spec,
                file_format=v.file_format,
                file_size_bytes=int(v.file_size_bytes or 0),
                row_count=v.row_count,
                column_count=v.column_count,
                quality_score=v.quality_score,
                created_at=v.created_at,
                updated_at=v.updated_at,
                is_active=str(getattr(v, "file_path", "")) == active_path,
            )
        )

    total_pages = (int(total) + page_size - 1) // page_size
    return PaginatedResponse(
        status="success",
        data=items,
        pagination=PaginationMeta(
            total=int(total),
            page=page,
            page_size=page_size,
            total_pages=int(total_pages),
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
    )


@router.post(
    "/{dataset_id}/versions/{version_id}/activate",
    response_model=APIResponse[DatasetResponse],
    summary="Activate dataset version",
    description="Switch the dataset to a previous version (rollback) without reprocessing (uses stored snapshots).",
)
async def activate_dataset_version(
    dataset_id: UUID,
    version_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    from sqlalchemy import delete, select

    from app.models import Dataset as DatasetModel
    from app.models import DatasetColumn as DatasetColumnModel
    from app.models import DatasetStatus as ModelDatasetStatus
    from app.models import DatasetVersion as DatasetVersionModel

    ds = (
        (await db.execute(
            select(DatasetModel).where(
                DatasetModel.id == dataset_id,
                DatasetModel.is_deleted == False,  # noqa: E712
                DatasetModel.owner_id == user.user_id,
            )
        ))
        .scalars()
        .first()
    )
    if ds is None:
        raise DataNotFoundException("Dataset", dataset_id)

    ver = (
        (await db.execute(
            select(DatasetVersionModel).where(
                DatasetVersionModel.id == version_id,
                DatasetVersionModel.dataset_id == dataset_id,
                DatasetVersionModel.owner_id == user.user_id,
                DatasetVersionModel.is_deleted == False,  # noqa: E712
            )
        ))
        .scalars()
        .first()
    )
    if ver is None:
        raise DataNotFoundException("DatasetVersion", version_id)

    try:
        from app.services.object_store import get_object_store

        obj = get_object_store()
        if not obj.exists(str(getattr(ver, "file_path", ""))):
            raise HTTPException(status_code=400, detail="Version file is missing on server storage")
    except HTTPException:
        raise
    except Exception:
        # Fall back to local path check.
        p = Path(str(getattr(ver, "file_path", "")))
        if not p.exists():
            raise HTTPException(status_code=400, detail="Version file is missing on server storage")

    # Apply the version snapshot to the dataset record.
    ds.file_path = str(ver.file_path)
    ds.file_format = str(ver.file_format)
    ds.file_size_bytes = int(ver.file_size_bytes or 0)
    ds.status = ModelDatasetStatus.READY
    ds.error_message = None
    ds.row_count = ver.row_count
    ds.column_count = ver.column_count
    ds.schema_info = to_jsonable(ver.schema_info or {})
    ds.quality_score = ver.quality_score
    ds.quality_report = to_jsonable(ver.quality_report or {})
    ds.profile_report = to_jsonable(ver.profile_report or {})
    ds.updated_by = user.user_id

    # Rebuild dataset_columns from the stored schema snapshot.
    cols = []
    schema = ver.schema_info if isinstance(ver.schema_info, dict) else {}
    schema_cols = schema.get("columns")
    if isinstance(schema_cols, list):
        cols = schema_cols
    elif isinstance(ver.profile_report, dict) and isinstance(ver.profile_report.get("columns"), list):
        cols = ver.profile_report.get("columns")

    await db.execute(delete(DatasetColumnModel).where(DatasetColumnModel.dataset_id == dataset_id))
    for col in cols:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name") or "")
        if not name:
            continue
        db.add(
            DatasetColumnModel(
                dataset_id=dataset_id,
                name=name,
                original_name=str(col.get("original_name") or name),
                position=int(col.get("position") or 0),
                inferred_type=str(col.get("inferred_type") or "unknown"),
                semantic_type=None,
                null_count=int(col.get("null_count") or 0),
                null_percentage=float(col.get("null_percentage") or 0.0),
                unique_count=int(col.get("unique_count") or 0),
                min_value=col.get("min_value"),
                max_value=col.get("max_value"),
                mean_value=col.get("mean_value"),
                median_value=col.get("median_value"),
                std_value=col.get("std_value"),
                distribution_type=None,
                value_distribution=to_jsonable(col.get("value_distribution") or {}),
                has_outliers=bool(col.get("has_outliers") or False),
                is_sensitive=bool(col.get("is_potential_pii") or False),
                statistics=to_jsonable(col),
            )
        )

    await db.commit()
    await db.refresh(ds)

    resp = DatasetResponse(
        id=ds.id,
        name=ds.name,
        description=ds.description,
        original_filename=ds.original_filename,
        file_size_bytes=ds.file_size_bytes,
        file_format=ds.file_format,
        status=_schema_dataset_status(ds.status),
        error_message=ds.error_message,
        row_count=ds.row_count,
        column_count=ds.column_count,
        quality_score=ds.quality_score,
        tags=ds.tags,
        created_at=ds.created_at,
        updated_at=ds.updated_at,
    )

    return APIResponse.success(data=resp, message="Dataset version activated")


@router.post(
    "/{dataset_id}/transform/suggest",
    response_model=APIResponse[DatasetTransformSuggestResponse],
    summary="Suggest dataset transformation plan",
    description="Generate a deterministic no-code cleaning plan from dataset metadata.",
)
async def suggest_dataset_transform(
    dataset_id: UUID,
    request: DatasetTransformSuggestRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    if not dataset or dataset.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)

    from app.models import DatasetStatus as ModelDatasetStatus

    if dataset.status != ModelDatasetStatus.READY:
        raise HTTPException(status_code=400, detail="Dataset must be processed (ready) before generating transform suggestions")

    from app.services.dataset_transform_suggestions import DatasetTransformSuggestionService

    suggested = DatasetTransformSuggestionService().suggest(
        dataset,
        max_steps=int(request.max_steps),
        include_drop_columns=bool(request.include_drop_columns),
        include_string_normalization=bool(request.include_string_normalization),
    )

    items: list[DatasetTransformSuggestion] = []
    for item in suggested.suggestions:
        step_raw = item.get("step") if isinstance(item, dict) else None
        if not isinstance(step_raw, dict):
            continue
        try:
            step = DatasetTransformStep.model_validate(step_raw)
        except Exception:
            continue
        items.append(
            DatasetTransformSuggestion(
                step=step,
                reason=str(item.get("reason") or "Suggested by deterministic rule"),
                impact=str(item.get("impact")) if item.get("impact") is not None else None,
            )
        )

    resp = DatasetTransformSuggestResponse(
        dataset_id=dataset_id,
        suggestions=items,
        warnings=list(suggested.warnings or []),
        summary=to_jsonable(suggested.summary or {}),
    )
    return APIResponse.success(data=resp, message="Transformation suggestions generated")


@router.post(
    "/{dataset_id}/transform/preview",
    response_model=APIResponse[DatasetTransformPreviewResponse],
    summary="Preview dataset transformation",
    description="Apply a transformation pipeline on a bounded sample and return a diff + preview.",
)
async def preview_dataset_transform(
    dataset_id: UUID,
    request: DatasetTransformPreviewRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    if not dataset or dataset.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)

    from app.services.dataset_transformations import DatasetTransformError, apply_transform_steps, build_preview_diff

    from app.models import DatasetStatus as ModelDatasetStatus

    if dataset.status != ModelDatasetStatus.READY:
        raise HTTPException(status_code=400, detail="Dataset must be processed (ready) before transforming")

    loader = DatasetLoaderService(db)
    loaded = await loader.load_dataset(
        dataset_id,
        owner_id=user.user_id,
        require_ready=True,
        sample_rows=int(request.sample_rows),
    )

    try:
        out = apply_transform_steps(loaded.df, request.steps)
        diff = build_preview_diff(before=loaded.df, after=out.df, preview_rows=int(request.preview_rows))
    except DatasetTransformError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resp = DatasetTransformPreviewResponse(
        **diff,
        warnings=list(out.warnings or []),
        metrics=to_jsonable(out.metrics or {}),
    )
    return APIResponse.success(data=resp, message="Preview computed")


@router.post(
    "/{dataset_id}/transform/apply",
    response_model=APIResponse[DatasetTransformApplyResponse],
    summary="Apply dataset transformation",
    description="Create a new dataset version by applying a validated transformation pipeline (async).",
)
async def apply_dataset_transform(
    dataset_id: UUID,
    request: DatasetTransformApplyRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(dataset_id)
    if not dataset or dataset.owner_id != user.user_id:
        raise DataNotFoundException("Dataset", dataset_id)

    from app.models import DatasetStatus as ModelDatasetStatus

    if dataset.status != ModelDatasetStatus.READY:
        raise HTTPException(status_code=400, detail="Dataset must be processed (ready) before transforming")

    from app.models import Job, JobStatus as ModelJobStatus, JobType as ModelJobType
    from app.workers.dispatcher import enqueue_dataset_transform

    payload = {
        "dataset_id": str(dataset_id),
        "steps": to_jsonable([s.model_dump() for s in request.steps]),
        "label": request.label,
        "set_as_current": bool(request.set_as_current),
    }

    job = Job(
        owner_id=user.user_id,
        dataset_id=dataset_id,
        job_type=ModelJobType.DATASET_TRANSFORM,
        status=ModelJobStatus.QUEUED,
        progress=0.0,
        status_message="Queued dataset transform",
        payload=payload,
        result={},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    enqueue_dataset_transform(background_tasks=background_tasks, dataset_id=dataset_id, job_id=job.id)

    resp = DatasetTransformApplyResponse(
        dataset_id=dataset_id,
        job_id=job.id,
        message="Dataset transformation queued",
    )
    return APIResponse.success(data=resp, meta={"job_id": str(job.id)})
