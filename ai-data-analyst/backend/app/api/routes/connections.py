# AI Enterprise Data Analyst - Database Connections API
# Persistent external connections with encrypted secrets + read-only query guardrails.

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.core.config import settings
from app.core.logging import get_logger
from app.core.serialization import to_jsonable
from app.models import Dataset, DatasetStatus, ExternalConnection, Job, JobStatus, JobType
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.services.external_connection_runtime import (
    ConnectionRuntimeConfig,
    ConnectionRuntimeError,
    list_tables as runtime_list_tables,
    run_query as runtime_run_query,
    test_connection as runtime_test_connection,
)
from app.services.secret_crypto import decrypt_secret, encrypt_secret
from app.services.sql_safety import (
    UnsafeSQLError,
    enforce_row_limit,
    quote_identifier,
    validate_readonly_sql,
)
from app.workers.dispatcher import enqueue_dataset_processing

logger = get_logger(__name__)
router = APIRouter()


class ConnectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    connector_type: str = Field(..., description="postgresql, mysql, sqlite")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(..., min_length=1)
    username: str = Field(default="")
    password: str = Field(default="")
    ssl: bool = Field(default=False)
    extra: dict[str, Any] = Field(default_factory=dict)


class ConnectionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    host: Optional[str] = None
    port: Optional[int] = Field(default=None, ge=0, le=65535)
    database: Optional[str] = Field(default=None, min_length=1)
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: Optional[bool] = None
    extra: Optional[dict[str, Any]] = None


class ConnectionResponse(BaseModel):
    id: UUID
    name: str
    connector_type: str
    host: str
    port: int
    database: str
    username: str
    ssl: bool
    status: str
    created_at: datetime
    last_tested: Optional[datetime] = None


class ConnectionListResponse(BaseModel):
    items: list[ConnectionResponse]
    total: int


class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    latency_ms: Optional[float] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_rows: Optional[int] = Field(default=None, ge=1, le=50000)


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float


class ImportRequest(BaseModel):
    table_name: str = Field(..., min_length=1, max_length=255)
    dataset_name: str = Field(..., min_length=1, max_length=255)
    max_rows: Optional[int] = Field(default=None, ge=1, le=5000000)
    description: Optional[str] = Field(default=None, max_length=2000)


def _runtime_cfg(conn: ExternalConnection) -> ConnectionRuntimeConfig:
    password = decrypt_secret(conn.encrypted_password) if conn.encrypted_password else None
    return ConnectionRuntimeConfig(
        connector_type=conn.connector_type,
        host=conn.host or "",
        port=int(conn.port or 0),
        database=conn.database or "",
        username=conn.username or "",
        password=password,
        ssl=bool(conn.ssl),
        timeout_seconds=int(settings.connections.query_timeout_seconds),
    )


async def _get_connection(db: AsyncSession, *, owner_id: UUID, connection_id: UUID) -> ExternalConnection:
    result = await db.execute(
        select(ExternalConnection).where(
            ExternalConnection.id == connection_id,
            ExternalConnection.owner_id == owner_id,
            ExternalConnection.is_deleted == False,  # noqa: E712
        )
    )
    conn = result.scalars().first()
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn


def _sanitize_basename(value: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._")
    return base or "import"


@router.post("", response_model=ConnectionResponse, status_code=status.HTTP_201_CREATED)
async def create_connection(
    request: ConnectionCreate,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    connector_type = (request.connector_type or "").strip().lower()
    if connector_type not in ("postgresql", "postgres", "sqlite", "mysql", "mariadb"):
        raise HTTPException(status_code=400, detail="Unsupported connector type")

    encrypted_password = encrypt_secret(request.password) if request.password else None
    conn = ExternalConnection(
        name=request.name,
        connector_type="postgresql" if connector_type == "postgres" else connector_type,
        host=request.host or "",
        port=int(request.port or 0),
        database=request.database,
        username=request.username or "",
        encrypted_password=encrypted_password,
        ssl=bool(request.ssl),
        extra=to_jsonable(request.extra or {}),
        status="created",
        last_tested=None,
        owner_id=user.user_id,
        created_by=user.user_id,
        updated_by=user.user_id,
    )
    db.add(conn)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Connection name already exists")
    await db.refresh(conn)

    return ConnectionResponse(
        id=conn.id,
        name=conn.name,
        connector_type=conn.connector_type,
        host=conn.host,
        port=int(conn.port or 0),
        database=conn.database,
        username=conn.username,
        ssl=bool(conn.ssl),
        status=conn.status,
        created_at=conn.created_at,
        last_tested=conn.last_tested,
    )


@router.get("", response_model=ConnectionListResponse)
async def list_connections(
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    result = await db.execute(
        select(ExternalConnection)
        .where(
            ExternalConnection.owner_id == user.user_id,
            ExternalConnection.is_deleted == False,  # noqa: E712
        )
        .order_by(ExternalConnection.created_at.desc())
    )
    conns = result.scalars().all()
    items = [
        ConnectionResponse(
            id=c.id,
            name=c.name,
            connector_type=c.connector_type,
            host=c.host,
            port=int(c.port or 0),
            database=c.database,
            username=c.username,
            ssl=bool(c.ssl),
            status=c.status,
            created_at=c.created_at,
            last_tested=c.last_tested,
        )
        for c in conns
    ]
    return ConnectionListResponse(items=items, total=len(items))


@router.get("/{connection_id}", response_model=ConnectionResponse)
async def get_connection(
    connection_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    conn = await _get_connection(db, owner_id=user.user_id, connection_id=connection_id)
    return ConnectionResponse(
        id=conn.id,
        name=conn.name,
        connector_type=conn.connector_type,
        host=conn.host,
        port=int(conn.port or 0),
        database=conn.database,
        username=conn.username,
        ssl=bool(conn.ssl),
        status=conn.status,
        created_at=conn.created_at,
        last_tested=conn.last_tested,
    )


@router.patch("/{connection_id}", response_model=ConnectionResponse)
async def update_connection(
    connection_id: UUID,
    request: ConnectionUpdate,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    conn = await _get_connection(db, owner_id=user.user_id, connection_id=connection_id)

    patch = request.model_dump(exclude_unset=True)
    if "name" in patch and patch["name"]:
        conn.name = patch["name"]
    if "host" in patch and patch["host"] is not None:
        conn.host = patch["host"]
    if "port" in patch and patch["port"] is not None:
        conn.port = int(patch["port"])
    if "database" in patch and patch["database"] is not None:
        conn.database = patch["database"]
    if "username" in patch and patch["username"] is not None:
        conn.username = patch["username"]
    if "password" in patch:
        pwd = patch["password"]
        conn.encrypted_password = encrypt_secret(pwd) if pwd else None
    if "ssl" in patch and patch["ssl"] is not None:
        conn.ssl = bool(patch["ssl"])
    if "extra" in patch and patch["extra"] is not None:
        conn.extra = to_jsonable(patch["extra"] or {})

    conn.updated_by = user.user_id
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Connection name already exists")
    await db.refresh(conn)

    return ConnectionResponse(
        id=conn.id,
        name=conn.name,
        connector_type=conn.connector_type,
        host=conn.host,
        port=int(conn.port or 0),
        database=conn.database,
        username=conn.username,
        ssl=bool(conn.ssl),
        status=conn.status,
        created_at=conn.created_at,
        last_tested=conn.last_tested,
    )


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
    connection_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.DELETE_DATA)),
):
    conn = await _get_connection(db, owner_id=user.user_id, connection_id=connection_id)
    conn.soft_delete()
    conn.updated_by = user.user_id
    await db.commit()


@router.post("/{connection_id}/test", response_model=TestConnectionResponse)
async def test_connection(
    connection_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    conn = await _get_connection(db, owner_id=user.user_id, connection_id=connection_id)
    cfg = _runtime_cfg(conn)

    try:
        latency = await runtime_test_connection(cfg)
        conn.status = "active"
        conn.last_tested = datetime.utcnow()
        conn.updated_by = user.user_id
        await db.commit()
        return TestConnectionResponse(success=True, message="Connection successful", latency_ms=float(latency))
    except Exception as e:
        conn.status = "failed"
        conn.last_tested = datetime.utcnow()
        conn.updated_by = user.user_id
        await db.commit()
        return TestConnectionResponse(success=False, message=str(e))


@router.post("/{connection_id}/query", response_model=QueryResponse)
async def query_database(
    connection_id: UUID,
    request: QueryRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    conn = await _get_connection(db, owner_id=user.user_id, connection_id=connection_id)

    try:
        normalized = validate_readonly_sql(request.query)
    except UnsafeSQLError as e:
        raise HTTPException(status_code=400, detail=str(e))

    max_rows = int(request.max_rows or settings.connections.max_query_rows)
    max_rows = max(1, min(max_rows, int(settings.connections.max_query_rows)))
    limited = enforce_row_limit(normalized, max_rows)

    try:
        cfg = _runtime_cfg(conn)
        result = await runtime_run_query(cfg, limited, jsonable=True)
        return QueryResponse(**result)
    except ConnectionRuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{connection_id}/tables")
async def get_tables(
    connection_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    conn = await _get_connection(db, owner_id=user.user_id, connection_id=connection_id)
    try:
        cfg = _runtime_cfg(conn)
        tables = await runtime_list_tables(cfg)
        return {"tables": tables}
    except ConnectionRuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{connection_id}/import")
async def import_table_as_dataset(
    connection_id: UUID,
    request: ImportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    conn = await _get_connection(db, owner_id=user.user_id, connection_id=connection_id)

    dialect = (conn.connector_type or "").strip().lower()
    try:
        table_ref = quote_identifier(request.table_name, dialect=dialect)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    max_rows = int(request.max_rows or settings.connections.max_import_rows)
    max_rows = max(1, min(max_rows, int(settings.connections.max_import_rows)))
    sql = enforce_row_limit(f"SELECT * FROM {table_ref}", max_rows)

    cfg = _runtime_cfg(conn)
    try:
        query_res = await runtime_run_query(cfg, sql, jsonable=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = query_res.get("rows") or []
    df = pd.DataFrame.from_records(rows)

    upload_dir = Path(settings.upload_directory) / "imports"
    upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_basename(request.dataset_name)
    file_path = upload_dir / f"{timestamp}_{user.user_id}_{safe_name}.parquet"

    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write parquet: {e}")

    size_bytes = int(os.path.getsize(file_path))
    storage_path = str(file_path)
    try:
        from app.core.config import ObjectStoreBackend
        from app.services.object_store import get_object_store

        obj = get_object_store()
        storage_path = obj.put_upload_file(
            owner_id=user.user_id,
            original_filename=f"{safe_name}.parquet",
            local_path=file_path,
            content_type="application/octet-stream",
        )
        if obj.backend == ObjectStoreBackend.S3:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        storage_path = str(file_path)

    ds = Dataset(
        name=request.dataset_name,
        description=request.description
        or f"Imported from connection '{conn.name}' table '{request.table_name}'",
        original_filename=f"{safe_name}.parquet",
        file_path=str(storage_path),
        file_size_bytes=size_bytes,
        file_format="parquet",
        status=DatasetStatus.PENDING,
        owner_id=user.user_id,
        tags=["imported"],
        extra_data=to_jsonable(
            {
                "source": {
                    "type": "connection_import",
                    "connection_id": str(conn.id),
                    "connector_type": conn.connector_type,
                    "table_name": request.table_name,
                    "max_rows": max_rows,
                }
            }
        ),
        created_by=user.user_id,
        updated_by=user.user_id,
    )
    db.add(ds)
    await db.commit()
    await db.refresh(ds)

    job = Job(
        owner_id=user.user_id,
        dataset_id=ds.id,
        job_type=JobType.DATASET_PROCESSING,
        status=JobStatus.QUEUED,
        progress=0.0,
        status_message="Queued dataset processing (import)",
        payload={"dataset_id": str(ds.id)},
        result={},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    enqueue_dataset_processing(background_tasks=background_tasks, dataset_id=ds.id, job_id=job.id)

    return {
        "success": True,
        "dataset_id": str(ds.id),
        "job_id": str(job.id),
        "row_count": int(query_res.get("row_count") or 0),
        "columns": list(query_res.get("columns") or []),
        "file_format": "parquet",
    }
