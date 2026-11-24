from __future__ import annotations

import traceback
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import LogContext, get_logger
from app.core.serialization import to_jsonable
from app.ml.auto_ml_engine import OptimizationStrategy
from app.ml.training_run import run_training
from app.services.dataset_loader import DatasetLoaderService
from app.services.job_service import JobService
from app.services.ml_dataset_sampler import aload_representative_sample
from app.services.ml_task_inference import infer_target_candidates

logger = get_logger(__name__)


class MLTrainingService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._loader = DatasetLoaderService(session)
        self._jobs = JobService(session)

    async def _load_training_frame(
        self,
        dataset_id: UUID,
        *,
        owner_id: UUID,
        sample_rows: int,
    ):
        record = await self._loader.get_dataset_record(dataset_id, owner_id=owner_id, require_ready=True)
        pr = record.profile_report if isinstance(record.profile_report, dict) else {}
        total_rows = pr.get("row_count") or getattr(record, "row_count", None)
        try:
            total_rows = int(total_rows) if total_rows is not None else None
        except Exception:
            total_rows = None

        from app.services.object_store import get_object_store

        obj = get_object_store()
        local_path = obj.ensure_local_path(
            str(record.file_path or ""),
            expected_size_bytes=int(getattr(record, "file_size_bytes", 0) or 0) or None,
            filename_hint=str(getattr(record, "original_filename", "") or None),
        )

        df, sampling_meta = await aload_representative_sample(
            file_path=local_path,
            filename=str(record.original_filename or "dataset"),
            file_format=str(record.file_format or ""),
            sample_rows=int(sample_rows),
            total_rows=total_rows,
            random_state=int(settings.ml.default_random_state),
        )
        version_hash = ""
        try:
            vh = pr.get("file_hash")
            if isinstance(vh, str) and vh.strip():
                version_hash = vh.strip()
        except Exception:
            version_hash = ""

        return record, df, sampling_meta, version_hash

    async def run_baseline_training(
        self,
        *,
        dataset_id: UUID,
        owner_id: UUID,
        target_column: str,
        task: str | None = None,
        optimize: bool = True,
        max_models: int | None = None,
        sample_rows: int | None = None,
        time_column: str | None = None,
        split: str | None = None,
    ) -> dict[str, Any]:
        context = LogContext(component="MLTrainingService", operation="run_baseline_training")
        sample_rows = int(sample_rows or settings.ml.chat_training_sample_rows)
        max_models = int(max_models or settings.ml.baseline_max_models)

        record, df, sampling_meta, version_hash = await self._load_training_frame(
            dataset_id, owner_id=owner_id, sample_rows=sample_rows
        )

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset sample")

        trials = int(settings.ml.baseline_optimization_trials) if optimize else 0
        run = run_training(
            df,
            target_column=target_column,
            registered_name=f"ds_{str(dataset_id)[:8]}_{target_column}",
            owner_id=owner_id,
            task=task,
            optimize=optimize and trials > 0,
            max_models=max_models,
            optimization_trials=max(5, trials),
            optimization_timeout=min(600, int(settings.ml.automl_time_budget)),
            time_column=time_column,
            split=split,
            description=f"Baseline training on {target_column}",
            tags={"dataset_id": str(dataset_id), "dataset_version": version_hash, "mode": "baseline"},
            extra_params={"sampling": sampling_meta},
            random_state=int(settings.ml.default_random_state),
        )
        summary = run.summary()
        payload = to_jsonable({
            "dataset_id": str(dataset_id),
            "dataset_version": version_hash,
            "sampling": sampling_meta,
            **summary,
        })
        best_name = summary["best_model"]
        logger.info("Baseline ML training complete", context=context, dataset_id=str(dataset_id), model=best_name)
        return payload

    async def run_automl_job(
        self,
        *,
        job_id: UUID,
        owner_id: UUID,
        dataset_id: UUID,
        target_column: str,
        task: str | None = None,
        sample_rows: int | None = None,
        max_models: int | None = None,
        optimize: bool = True,
        time_column: str | None = None,
        split: str | None = None,
    ) -> None:
        context = LogContext(component="MLTrainingService", operation="run_automl_job")
        sample_rows = int(sample_rows or settings.ml.automl_training_sample_rows)
        max_models = int(max_models or settings.ml.automl_max_models)

        try:
            await self._jobs.set_running(job_id, owner_id=owner_id, message="Loading representative dataset sample")
            await self._jobs.update_progress(job_id, owner_id=owner_id, progress=0.05, message="Sampling data")

            record, df, sampling_meta, version_hash = await self._load_training_frame(
                dataset_id, owner_id=owner_id, sample_rows=sample_rows
            )

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")

            await self._jobs.update_progress(job_id, owner_id=owner_id, progress=0.15, message="Preprocessing features")

            trials = int(settings.ml.automl_optimization_trials) if optimize else 0
            strategy = OptimizationStrategy.BAYESIAN if optimize and trials > 0 else OptimizationStrategy.RANDOM_SEARCH

            await self._jobs.update_progress(job_id, owner_id=owner_id, progress=0.25, message="Training and tuning models")

            run = run_training(
                df,
                target_column=target_column,
                registered_name=f"ds_{str(dataset_id)[:8]}_{target_column}_automl",
                owner_id=owner_id,
                task=task,
                optimize=optimize,
                max_models=max_models,
                optimization_strategy=strategy,
                optimization_trials=max(10, trials),
                optimization_timeout=int(settings.ml.automl_time_budget),
                time_column=time_column,
                split=split,
                description=f"AutoML training on {target_column}",
                tags={"dataset_id": str(dataset_id), "dataset_version": version_hash, "mode": "automl"},
                extra_params={"sampling": sampling_meta},
                random_state=int(settings.ml.default_random_state),
            )

            await self._jobs.update_progress(job_id, owner_id=owner_id, progress=0.95, message="Model registered")
            summary = run.summary()
            best_name = summary["best_model"]
            result_payload = to_jsonable({
                "dataset_id": str(dataset_id),
                "dataset_version": version_hash,
                "sampling": sampling_meta,
                **summary,
            })

            await self._jobs.set_completed(
                job_id,
                owner_id=owner_id,
                result=result_payload,
                message=f"AutoML complete — best model: {best_name}",
            )
            logger.info("AutoML job complete", context=context, job_id=str(job_id), model=best_name)
        except Exception as e:
            logger.error(f"AutoML job failed: {e}", context=context, job_id=str(job_id))
            await self._jobs.set_failed(
                job_id,
                owner_id=owner_id,
                error=e,
                message=str(e),
            )

    async def infer_targets(
        self,
        *,
        dataset_id: UUID,
        owner_id: UUID,
        preferred_target: str | None = None,
        sample_rows: int | None = None,
    ) -> dict[str, Any]:
        sample_rows = int(sample_rows or settings.ml.chat_training_sample_rows)
        _, df, sampling_meta, version_hash = await self._load_training_frame(
            dataset_id, owner_id=owner_id, sample_rows=min(sample_rows, 100_000)
        )
        candidates, selected_target, selected_task, warnings = infer_target_candidates(
            df, preferred_target=preferred_target
        )
        candidate_payload = []
        for c in candidates:
            candidate_payload.append(
                {
                    "column": c.column,
                    "inferred_task": c.inferred_task,
                    "score": c.score,
                    "non_null_rate": c.non_null_rate,
                    "unique_ratio": c.unique_ratio,
                    "reasons": list(c.reasons),
                    "leakage_warnings": [
                        {
                            "column": w.column,
                            "warning_type": w.warning_type,
                            "detail": w.detail,
                            "severity": w.severity,
                        }
                        for w in c.leakage_warnings
                    ],
                }
            )
        return {
            "dataset_id": str(dataset_id),
            "dataset_version": version_hash,
            "sample_rows": int(len(df)),
            "sampling": sampling_meta,
            "selected_target": selected_target,
            "selected_task": selected_task,
            "candidates": candidate_payload,
            "warnings": warnings,
        }