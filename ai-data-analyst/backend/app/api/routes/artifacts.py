# AI Enterprise Data Analyst - Artifact API Routes
# List/search indexed artifacts (reports, tables, metrics) created by safe compute.

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import require_permission
from app.api.schemas import (
    APIResponse,
    ArtifactResponse,
    ArtifactRowsResponse,
    ArtifactType,
    PaginatedResponse,
    PaginationMeta,
)
from app.core.exceptions import DataNotFoundException
from app.core.redaction import mask_pii_rows, mask_preview
from app.models import Artifact as ArtifactModel, ArtifactType as ModelArtifactType
from app.models import Dataset as DatasetModel
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.compute.artifacts import ArtifactStore

router = APIRouter()


def _schema_artifact_type(value: Any) -> ArtifactType:
    if isinstance(value, ArtifactType):
        return value
    v = getattr(value, "value", value)
    return ArtifactType(str(v))


def _extract_pii_columns(schema_info: Any) -> set[str]:
    if not isinstance(schema_info, dict):
        return set()
    cols = schema_info.get("columns")
    if not isinstance(cols, list):
        return set()
    out: set[str] = set()
    for c in cols:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name:
            continue
        stats = c.get("statistics") if isinstance(c.get("statistics"), dict) else {}
        is_pii = bool(c.get("is_potential_pii") or stats.get("is_potential_pii"))
        if is_pii:
            out.add(str(name))
    return out


async def _pii_columns_for_dataset(db: AsyncSession, *, dataset_id: UUID, owner_id: UUID) -> set[str]:
    q = select(DatasetModel).where(
        DatasetModel.id == dataset_id,
        DatasetModel.owner_id == owner_id,
        DatasetModel.is_deleted == False,  # noqa: E712
    )
    ds = (await db.execute(q)).scalars().first()
    if ds is None:
        return set()
    return _extract_pii_columns(getattr(ds, "schema_info", None) or {})


def _artifact_to_response(a: ArtifactModel, *, pii_columns: set[str] | None = None) -> ArtifactResponse:
    pii_columns = pii_columns or set()
    preview = a.preview or {}
    if pii_columns and a.artifact_type == ModelArtifactType.TABLE and isinstance(preview, dict):
        preview = mask_preview(preview, pii_columns)
    return ArtifactResponse(
        id=a.id,
        artifact_type=_schema_artifact_type(a.artifact_type),
        name=a.name,
        dataset_id=a.dataset_id,
        dataset_version=a.dataset_version,
        operator_name=a.operator_name,
        operator_params=a.operator_params or {},
        preview=preview,
        created_at=a.created_at,
    )


@router.get(
    "",
    response_model=PaginatedResponse[ArtifactResponse],
    summary="List artifacts",
    description="List indexed artifacts for the current user (filterable by type/dataset).",
)
async def list_artifacts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    artifact_type: Optional[ArtifactType] = Query(None, alias="type"),
    dataset_id: Optional[UUID] = Query(None),
    operator_name: Optional[str] = Query(None, max_length=128),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    skip = (page - 1) * page_size

    q = select(ArtifactModel).where(
        ArtifactModel.owner_id == user.user_id,
        ArtifactModel.is_deleted == False,  # noqa: E712
    )

    if artifact_type is not None:
        q = q.where(ArtifactModel.artifact_type == ModelArtifactType(artifact_type.value))
    if dataset_id is not None:
        q = q.where(ArtifactModel.dataset_id == dataset_id)
    if operator_name is not None:
        q = q.where(ArtifactModel.operator_name == operator_name)

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    q = q.order_by(ArtifactModel.created_at.desc()).offset(skip).limit(page_size)
    artifacts = (await db.execute(q)).scalars().all()

    # Best-effort PII masking for preview rows. Cache per dataset_id to avoid N+1.
    pii_cache: dict[UUID, set[str]] = {}
    for a in artifacts:
        if a.dataset_id is None:
            continue
        did = a.dataset_id
        if did in pii_cache:
            continue
        try:
            pii_cache[did] = await _pii_columns_for_dataset(db, dataset_id=did, owner_id=user.user_id)
        except Exception:
            pii_cache[did] = set()

    total_pages = (int(total) + page_size - 1) // page_size
    return PaginatedResponse(
        status="success",
        data=[
            _artifact_to_response(a, pii_columns=pii_cache.get(a.dataset_id) or set())
            for a in artifacts
        ],
        pagination=PaginationMeta(
            total=int(total),
            page=page,
            page_size=page_size,
            total_pages=int(total_pages),
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
    )


@router.get(
    "/{artifact_id}",
    response_model=APIResponse[ArtifactResponse],
    summary="Get artifact",
    description="Fetch a single artifact record (owner-only).",
)
async def get_artifact(
    artifact_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    q = select(ArtifactModel).where(
        ArtifactModel.id == artifact_id,
        ArtifactModel.owner_id == user.user_id,
        ArtifactModel.is_deleted == False,  # noqa: E712
    )
    artifact = (await db.execute(q)).scalars().first()
    if artifact is None:
        raise DataNotFoundException("Artifact", artifact_id)
    pii_cols: set[str] = set()
    try:
        if artifact.dataset_id is not None:
            pii_cols = await _pii_columns_for_dataset(db, dataset_id=artifact.dataset_id, owner_id=user.user_id)
    except Exception:
        pii_cols = set()
    return APIResponse.success(data=_artifact_to_response(artifact, pii_columns=pii_cols))


def _run_duckdb_slice_sync(*, data_path: str, data_format: str, offset: int, limit: int) -> tuple[list[str], list[dict[str, Any]], float]:
    import duckdb

    def _quote(value: str) -> str:
        v = str(value or "")
        v = v.replace("\\", "/")
        v = v.replace("'", "''")
        return f"'{v}'"

    fmt = str(data_format or "").lower().strip()
    if fmt in {"parquet"}:
        rel = f"read_parquet({_quote(data_path)})"
    elif fmt in {"csv"}:
        rel = f"read_csv_auto({_quote(data_path)})"
    else:
        raise ValueError(f"unsupported table data_format: {fmt}")

    sql = f"SELECT * FROM {rel} LIMIT {int(limit)} OFFSET {int(offset)}"
    con = duckdb.connect(database=":memory:")
    try:
        t0 = time.perf_counter()
        df = con.execute(sql).df()
        dur_ms = (time.perf_counter() - t0) * 1000.0
        cols = [str(c) for c in df.columns.tolist()]
        rows = df.to_dict(orient="records")
        return cols, rows, float(dur_ms)
    finally:
        try:
            con.close()
        except Exception:
            pass


def _load_table_df(*, data_path: str, data_format: str) -> "pd.DataFrame":
    import pandas as pd

    fmt = str(data_format or "").lower().strip()
    if fmt == "parquet":
        return pd.read_parquet(data_path)
    if fmt == "csv":
        return pd.read_csv(data_path)
    raise ValueError(f"unsupported table data_format: {fmt}")


def _run_sqlite_slice_sync(*, data_path: str, data_format: str, offset: int, limit: int) -> tuple[list[str], list[dict[str, Any]], float]:
    import sqlite3
    import pandas as pd

    src_df = _load_table_df(data_path=data_path, data_format=data_format)
    con = sqlite3.connect(":memory:")
    try:
        src_df.to_sql("dataset", con, if_exists="replace", index=False)
        t0 = time.perf_counter()
        out_df: pd.DataFrame = pd.read_sql_query(
            "SELECT * FROM dataset LIMIT ? OFFSET ?",
            con,
            params=[int(limit), int(offset)],
        )
        dur_ms = (time.perf_counter() - t0) * 1000.0
        cols = [str(c) for c in out_df.columns.tolist()]
        rows = out_df.to_dict(orient="records")
        return cols, rows, float(dur_ms)
    finally:
        try:
            con.close()
        except Exception:
            pass


def _run_table_slice_sync(*, data_path: str, data_format: str, offset: int, limit: int) -> tuple[list[str], list[dict[str, Any]], float]:
    try:
        return _run_duckdb_slice_sync(data_path=data_path, data_format=data_format, offset=offset, limit=limit)
    except ModuleNotFoundError as e:
        name = str(getattr(e, "name", "") or "").lower()
        if name not in {"duckdb", "_duckdb"} and "duckdb" not in str(e).lower():
            raise
        return _run_sqlite_slice_sync(data_path=data_path, data_format=data_format, offset=offset, limit=limit)
    except ImportError as e:
        if "duckdb" not in str(e).lower():
            raise
        return _run_sqlite_slice_sync(data_path=data_path, data_format=data_format, offset=offset, limit=limit)


@router.get(
    "/{artifact_id}/rows",
    response_model=APIResponse[ArtifactRowsResponse],
    summary="Get artifact rows (paged)",
    description="Fetch a bounded page of rows for a TABLE artifact (no unbounded downloads).",
)
async def get_artifact_rows(
    artifact_id: UUID,
    offset: int = Query(0, ge=0, le=5_000_000),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    q = select(ArtifactModel).where(
        ArtifactModel.id == artifact_id,
        ArtifactModel.owner_id == user.user_id,
        ArtifactModel.is_deleted == False,  # noqa: E712
    )
    artifact = (await db.execute(q)).scalars().first()
    if artifact is None:
        raise DataNotFoundException("Artifact", artifact_id)

    if artifact.artifact_type != ModelArtifactType.TABLE:
        raise HTTPException(status_code=400, detail="Artifact is not a table")

    store = ArtifactStore()
    manifest = store.read_manifest(artifact_id)
    data_path = str(manifest.get("data_path") or "").strip()
    data_format = str(manifest.get("data_format") or "").strip()
    if not data_path or not data_format:
        raise HTTPException(status_code=400, detail="Artifact has no table data_path")

    # Artifact table data may live on local disk or remote object storage. DuckDB requires a local file.
    try:
        from app.services.object_store import get_object_store

        obj = get_object_store()
        data_local = obj.ensure_local_path(data_path, filename_hint=f"{artifact_id}.{data_format}")
        data_path = str(data_local)
    except Exception:
        pass

    # Derive total rows from preview metadata when available (cheap).
    total_rows = None
    preview = manifest.get("preview") if isinstance(manifest.get("preview"), dict) else {}
    if isinstance(preview, dict) and preview.get("rows") is not None:
        try:
            total_rows = int(preview.get("rows"))
        except Exception:
            total_rows = None

    # Execute slice in a thread + timeout.
    try:
        cols, rows, dur_ms = await asyncio.wait_for(
            asyncio.to_thread(
                _run_table_slice_sync,
                data_path=data_path,
                data_format=data_format,
                offset=int(offset),
                limit=int(limit),
            ),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Artifact read timed out")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    pii_cols: set[str] = set()
    try:
        if artifact.dataset_id is not None:
            pii_cols = await _pii_columns_for_dataset(db, dataset_id=artifact.dataset_id, owner_id=user.user_id)
    except Exception:
        pii_cols = set()

    if pii_cols:
        rows = mask_pii_rows([r for r in rows if isinstance(r, dict)], pii_cols)

    payload = ArtifactRowsResponse(
        artifact_id=artifact_id,
        columns=list(cols),
        rows=list(rows),
        offset=int(offset),
        limit=int(limit),
        total_rows=total_rows,
        execution_time_ms=float(dur_ms),
    )
    return APIResponse.success(data=payload)
