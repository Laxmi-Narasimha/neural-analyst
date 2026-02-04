from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.compute.artifacts import ArtifactRef, ArtifactStore, TableStorageFormat
from app.compute.operators.base import OperatorContext
from app.compute.registry import OperatorRegistry, default_registry
from app.core.exceptions import DataProcessingException
from app.core.logging import LogContext, get_logger
from app.services.artifact_index import ArtifactIndexService
from app.services.dataset_loader import DatasetLoaderService

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

    async def run_operator(
        self,
        *,
        dataset_id: UUID,
        operator_name: str,
        params: Optional[dict[str, Any]] = None,
        owner_id: Optional[UUID] = None,
        max_rows: Optional[int] = None,
        sample_rows: Optional[int] = None,
    ) -> ExecutionResult:
        context = LogContext(component="ComputeExecutor", operation="run_operator")
        params = params or {}

        loader = DatasetLoaderService(self._session)
        loaded = await loader.load_dataset(
            dataset_id,
            owner_id=owner_id,
            max_rows=max_rows,
            sample_rows=sample_rows,
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
            result = await asyncio.to_thread(_run_sync)
        except Exception as e:
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
                    operator_params=params,
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
                    operator_params=params,
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
                    operator_params=params,
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
                )
            )
        return results
