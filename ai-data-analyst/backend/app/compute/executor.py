from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.compute.artifacts import ArtifactRef, ArtifactStore, TableStorageFormat, ArtifactType as ComputeArtifactType
from app.compute.operators.base import OperatorContext
from app.compute.registry import OperatorRegistry, default_registry
from app.core.config import settings
from app.core.exceptions import DataProcessingException
from app.core.logging import LogContext, get_logger
from app.core.metrics import metrics_collector
from app.core.serialization import to_jsonable
from app.models import Artifact as ArtifactModel
from app.services.artifact_index import ArtifactIndexService
from app.services.dataset_loader import DatasetLoaderService, LoadedDataset
from app.services.object_store import get_object_store, ObjectStoreBackend
from app.utils.hashing import hash_json

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    operator_name: str
    artifacts: list[ArtifactRef]
    summary: dict[str, Any]


class ComputeExecutor:
    def __init__(
        self,
        session: AsyncSession,
        *,
        registry: Optional[OperatorRegistry] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> None:
        self._session = session
        self._registry = registry or default_registry()
        self._artifacts = artifact_store or ArtifactStore()
        self._artifact_index = ArtifactIndexService(session)

    def _params_with_runtime_meta(
        self,
        params: dict[str, Any],
        *,
        max_rows: Optional[int],
        sample_rows: Optional[int],
    ) -> dict[str, Any]:
        base = to_jsonable(params or {})
        out = dict(base) if isinstance(base, dict) else {}
        out["_na_meta"] = {
            "sample_rows": int(sample_rows) if sample_rows is not None else None,
            "max_rows": int(max_rows) if max_rows is not None else None,
            "app_version": str(settings.app_version),
        }
        return out

    def _artifact_row_to_ref(self, a: ArtifactModel) -> ArtifactRef:
        try:
            atype = getattr(a.artifact_type, "value", str(a.artifact_type))
        except Exception:
            atype = str(a.artifact_type)
        return ArtifactRef(
            artifact_id=a.id,
            artifact_type=ComputeArtifactType(str(atype)),
            name=str(a.name),
            created_at=a.created_at.isoformat() if getattr(a, "created_at", None) else "",
            storage_path=str(a.manifest_path),
            preview=to_jsonable(a.preview or {}),
            dataset_id=a.dataset_id,
            dataset_version=a.dataset_version,
            operator_name=a.operator_name,
            operator_params=to_jsonable(a.operator_params or {}),
        )

    async def _load_cached(
        self,
        *,
        owner_id: UUID,
        dataset_id: UUID,
        dataset_version: str,
        operator_name: str,
        operator_params_hash: str,
    ) -> tuple[list[ArtifactRef], dict[str, Any]] | None:
        q = select(ArtifactModel).where(
            ArtifactModel.owner_id == owner_id,
            ArtifactModel.dataset_id == dataset_id,
            ArtifactModel.dataset_version == dataset_version,
            ArtifactModel.operator_name == operator_name,
            ArtifactModel.operator_params_hash == operator_params_hash,
            ArtifactModel.is_deleted == False,  # noqa: E712
        )
        rows = (await self._session.execute(q)).scalars().all()
        if not rows:
            return None

        # Choose the newest artifact per name to avoid returning multiple generations.
        latest_by_name: dict[str, ArtifactModel] = {}
        for a in rows:
            key = str(a.name or "")
            prev = latest_by_name.get(key)
            if prev is None:
                latest_by_name[key] = a
                continue
            try:
                if a.created_at and prev.created_at and a.created_at > prev.created_at:
                    latest_by_name[key] = a
            except Exception:
                # If timestamps are missing or unorderable, keep the existing row.
                continue

        # Best-effort: validate local manifests exist (avoid returning broken refs when disk was pruned).
        obj = get_object_store()
        if obj.backend == ObjectStoreBackend.LOCAL:
            existing: dict[str, ArtifactModel] = {}
            for name, a in latest_by_name.items():
                try:
                    if obj.exists(str(a.manifest_path)):
                        existing[name] = a
                except Exception:
                    continue
            latest_by_name = existing
            if not latest_by_name:
                return None

        artifacts = [self._artifact_row_to_ref(latest_by_name[k]) for k in sorted(latest_by_name.keys())]

        summary: dict[str, Any] = {}
        for a in latest_by_name.values():
            prev = a.preview if isinstance(a.preview, dict) else {}
            if isinstance(prev, dict) and isinstance(prev.get("operator_summary"), dict):
                summary = dict(prev.get("operator_summary") or {})
                break
        summary["_cache_hit"] = True
        return artifacts, to_jsonable(summary)

    async def run_operator(
        self,
        *,
        dataset_id: UUID,
        operator_name: str,
        params: Optional[dict[str, Any]] = None,
        owner_id: Optional[UUID] = None,
        max_rows: Optional[int] = None,
        sample_rows: Optional[int] = None,
        loaded: Optional[LoadedDataset] = None,
    ) -> ExecutionResult:
        context = LogContext(component="ComputeExecutor", operation="run_operator")
        params = params or {}
        stored_params = self._params_with_runtime_meta(params, max_rows=max_rows, sample_rows=sample_rows)

        # Cache lookup before expensive dataset loads when possible.
        if owner_id is not None:
            ds_version = None
            if loaded is not None:
                ds_version = str(loaded.version_hash or "").strip() or None
            else:
                try:
                    loader = DatasetLoaderService(self._session)
                    ds = await loader.get_dataset_record(dataset_id, owner_id=owner_id, require_ready=True)
                    pr = ds.profile_report if isinstance(getattr(ds, "profile_report", None), dict) else {}
                    vh = pr.get("file_hash") if isinstance(pr, dict) else None
                    if isinstance(vh, str) and vh.strip():
                        ds_version = vh.strip()
                except Exception:
                    ds_version = None

            if ds_version:
                cached = await self._load_cached(
                    owner_id=owner_id,
                    dataset_id=dataset_id,
                    dataset_version=ds_version,
                    operator_name=str(operator_name or "").strip(),
                    operator_params_hash=hash_json(stored_params),
                )
                if cached is not None:
                    artifacts, summary = cached
                    metrics_collector.record_operator(
                        operator_name=operator_name,
                        duration_ms=0.0,
                        scanned_rows=0,
                        success=True,
                    )
                    logger.info(
                        "Operator cache hit",
                        context=context,
                        operator=operator_name,
                        dataset_id=str(dataset_id),
                        artifacts=len(artifacts),
                    )
                    return ExecutionResult(operator_name=operator_name, artifacts=artifacts, summary=summary)

        if loaded is None:
            loader = DatasetLoaderService(self._session)
            loaded = await loader.load_dataset(
                dataset_id,
                owner_id=owner_id,
                max_rows=max_rows,
                sample_rows=sample_rows,
            )
        elif loaded.dataset_id != dataset_id:
            raise DataProcessingException(
                f"Loaded dataset context mismatch (expected={dataset_id}, got={loaded.dataset_id})"
            )

        op = self._registry.get(operator_name)
        op_ctx = OperatorContext(
            dataset_id=loaded.dataset_id,
            dataset_version=loaded.version_hash,
            df=loaded.df,
            profile_report=loaded.profile_report,
            schema_info=loaded.schema_info,
        )

        def _run_sync():
            return op.run(op_ctx, params)

        try:
            scanned_rows = int(getattr(loaded.df, "shape", [0, 0])[0] or 0)
            t0 = time.perf_counter()
            result = await asyncio.to_thread(_run_sync)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            metrics_collector.record_operator(
                operator_name=operator_name,
                duration_ms=dur_ms,
                scanned_rows=scanned_rows,
                success=True,
            )
        except Exception as e:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0  # type: ignore[name-defined]
            except Exception:
                dur_ms = 0.0
            try:
                scanned_rows = int(getattr(loaded.df, "shape", [0, 0])[0] or 0)
            except Exception:
                scanned_rows = 0
            metrics_collector.record_operator(
                operator_name=operator_name,
                duration_ms=dur_ms,
                scanned_rows=scanned_rows,
                success=False,
            )
            logger.error(
                "Operator failed",
                context=context,
                operator=operator_name,
                dataset_id=str(dataset_id),
                error=str(e),
            )
            raise DataProcessingException(f"Operator '{operator_name}' failed: {e}") from e

        artifacts: list[ArtifactRef] = []
        for name, df in result.tables.items():
            artifacts.append(
                self._artifacts.write_table(
                    name=f"{operator_name}:{name}",
                    df=df,
                    storage_format=TableStorageFormat.PARQUET,
                    dataset_id=loaded.dataset_id,
                    dataset_version=loaded.version_hash,
                    operator_name=operator_name,
                    operator_params=stored_params,
                    operator_summary=result.summary,
                )
            )

        for key, value in result.metrics.items():
            artifacts.append(
                self._artifacts.write_metric(
                    name=f"{operator_name}:{key}",
                    value=value,
                    dataset_id=loaded.dataset_id,
                    dataset_version=loaded.version_hash,
                    operator_name=operator_name,
                    operator_params=stored_params,
                    operator_summary=result.summary,
                )
            )

        for name, spec in result.charts.items():
            artifacts.append(
                self._artifacts.write_chart(
                    name=f"{operator_name}:{name}",
                    spec=spec,
                    dataset_id=loaded.dataset_id,
                    dataset_version=loaded.version_hash,
                    operator_name=operator_name,
                    operator_params=stored_params,
                    operator_summary=result.summary,
                )
            )

        if owner_id is not None:
            await self._artifact_index.index_many(owner_id=owner_id, refs=artifacts)

        logger.info(
            "Operator executed",
            context=context,
            operator=operator_name,
            dataset_id=str(dataset_id),
            artifacts=len(artifacts),
        )

        return ExecutionResult(operator_name=operator_name, artifacts=artifacts, summary=result.summary)

    async def run_plan(
        self,
        *,
        dataset_id: UUID,
        plan: list[dict[str, Any]],
        owner_id: Optional[UUID] = None,
        max_rows: Optional[int] = None,
        sample_rows: Optional[int] = None,
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        loader = DatasetLoaderService(self._session)
        loaded = await loader.load_dataset(
            dataset_id,
            owner_id=owner_id,
            max_rows=max_rows,
            sample_rows=sample_rows,
        )
        for step in plan:
            name = str(step.get("operator"))
            params = step.get("params") or {}
            results.append(
                await self.run_operator(
                    dataset_id=dataset_id,
                    operator_name=name,
                    params=params,
                    owner_id=owner_id,
                    max_rows=max_rows,
                    sample_rows=sample_rows,
                    loaded=loaded,
                )
            )
        return results
