from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select

from app.compute.executor import ComputeExecutor
from app.compute.plans import eda_p0_plan
from app.core.logging import LogContext, clear_request_context, get_logger, set_request_context
from app.core.serialization import to_jsonable
from app.models import (
    Analysis,
    AnalysisStatus,
    AnalysisType,
    Job,
    JobStatus,
)
from app.services.database import db_manager
from app.services.dataset_loader import DatasetLoaderService
from app.services.insight_extraction import extract_eda_insights
from app.services.narrator import NarratorService

logger = get_logger(__name__)


def _artifact_to_dict(a: Any) -> dict[str, Any]:
    return {
        "artifact_id": str(getattr(a, "artifact_id", "")),
        "artifact_type": getattr(getattr(a, "artifact_type", None), "value", None),
        "name": getattr(a, "name", None),
        "created_at": getattr(a, "created_at", None),
        "storage_path": getattr(a, "storage_path", None),
        "preview": getattr(a, "preview", None),
        "dataset_id": str(getattr(a, "dataset_id", None)) if getattr(a, "dataset_id", None) else None,
        "dataset_version": getattr(a, "dataset_version", None),
        "operator_name": getattr(a, "operator_name", None),
        "operator_params": getattr(a, "operator_params", None) or {},
    }


def _artifact_refs_from_results(results: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(results, dict):
        return out
    steps = results.get("steps")
    if not isinstance(steps, list):
        return out
    for s in steps:
        if not isinstance(s, dict):
            continue
        arts = s.get("artifacts")
        if not isinstance(arts, list):
            continue
        for a in arts:
            if not isinstance(a, dict):
                continue
            aid = a.get("artifact_id")
            if not aid:
                continue
            out.append(
                {
                    "artifact_id": str(aid),
                    "artifact_type": a.get("artifact_type"),
                    "name": a.get("name"),
                    "operator_name": a.get("operator_name") or s.get("operator"),
                }
            )
    return out[:50]


async def _update_parent_action_feed(
    session, *, owner_id: UUID, child: Analysis
) -> None:
    cfg = child.config if isinstance(child.config, dict) else {}
    parent_id = cfg.get("parent_analysis_id")
    if not parent_id:
        return
    try:
        parent_uuid = UUID(str(parent_id))
    except Exception:
        return

    res = await session.execute(
        select(Analysis).where(
            Analysis.id == parent_uuid,
            Analysis.is_deleted == False,  # noqa: E712
            Analysis.owner_id == owner_id,
        )
    )
    parent = res.scalars().first()
    if parent is None:
        return

    parent_results_raw = parent.results if isinstance(parent.results, dict) else {}
    parent_results: dict[str, Any] = dict(parent_results_raw)
    feed_raw = parent_results.get("action_feed")
    feed_list = list(feed_raw) if isinstance(feed_raw, list) else []

    child_id = str(child.id)
    status_val = getattr(child.status, "value", str(child.status))
    takeaways = []
    if isinstance(child.results, dict) and isinstance(child.results.get("takeaways"), list):
        takeaways = child.results.get("takeaways") or []
    artifacts = _artifact_refs_from_results(child.results)

    now_iso = datetime.utcnow().isoformat()
    completed_iso = child.completed_at.isoformat() if getattr(child, "completed_at", None) else None

    new_feed: list[dict[str, Any]] = []
    found = False
    for it in feed_list:
        if not isinstance(it, dict):
            continue
        if str(it.get("analysis_id") or "") != child_id:
            new_feed.append(it)
            continue
        found = True
        updated = dict(it)
        updated["status"] = status_val
        updated["status_message"] = child.status_message
        updated["error_message"] = child.error_message
        updated["takeaways"] = takeaways
        updated["artifacts"] = artifacts
        updated["updated_at"] = now_iso
        if completed_iso:
            updated["completed_at"] = completed_iso
        new_feed.append(updated)

    if not found:
        new_feed.append(
            {
                "action_id": str(cfg.get("action_id") or ""),
                "kind": "analysis",
                "title": str(child.name or "Action"),
                "detail": str(child.description or ""),
                "params": cfg.get("action_params") if isinstance(cfg.get("action_params"), dict) else {},
                "created_at": child.created_at.isoformat() if getattr(child, "created_at", None) else now_iso,
                "analysis_id": child_id,
                "status": status_val,
                "status_message": child.status_message,
                "error_message": child.error_message,
                "takeaways": takeaways,
                "artifacts": artifacts,
                "updated_at": now_iso,
                "completed_at": completed_iso,
            }
        )

    parent_results["action_feed"] = new_feed
    parent.results = parent_results
    parent.updated_by = owner_id


class AnalysisExecutionService:
    """Execute an Analysis record via the safe ComputeExecutor (job-friendly)."""

    async def run_analysis(self, analysis_id: UUID, job_id: Optional[UUID] = None) -> None:
        set_request_context(request_id=str(job_id or analysis_id))
        context = LogContext(component="AnalysisExecutionService", operation="run_analysis", request_id=str(analysis_id))

        try:
            async with db_manager.session() as session:
                res = await session.execute(
                    select(Analysis).where(
                        Analysis.id == analysis_id,
                        Analysis.is_deleted == False,  # noqa: E712
                    )
                )
                analysis = res.scalars().first()
                if analysis is None:
                    logger.warning("Analysis not found for execution", context=context, analysis_id=str(analysis_id))
                    return

                owner_id = analysis.owner_id
                set_request_context(user_id=str(owner_id))

                job: Job | None = None
                if job_id is not None:
                    job_res = await session.execute(
                        select(Job).where(
                            Job.id == job_id,
                            Job.is_deleted == False,  # noqa: E712
                            Job.owner_id == owner_id,
                        )
                    )
                    job = job_res.scalars().first()

                # Respect cancellation before starting.
                if analysis.status == AnalysisStatus.CANCELLED:
                    if job is not None and job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                        job.status = JobStatus.CANCELLED
                        job.completed_at = datetime.utcnow()
                        job.status_message = job.status_message or "Cancelled"
                    await session.commit()
                    logger.info("Analysis already cancelled; skipping", context=context, analysis_id=str(analysis_id))
                    return

                analysis.status = AnalysisStatus.RUNNING
                analysis.started_at = analysis.started_at or datetime.utcnow()
                analysis.completed_at = None
                analysis.duration_seconds = None
                analysis.progress = 0.0
                analysis.status_message = "Running"
                analysis.error_message = None
                analysis.error_traceback = None
                analysis.updated_by = owner_id

                if job is not None:
                    job.status = JobStatus.RUNNING
                    job.started_at = job.started_at or datetime.utcnow()
                    job.completed_at = None
                    job.progress = max(float(job.progress or 0.0), 0.01)
                    job.status_message = job.status_message or "Running analysis"

                await session.commit()

                try:
                    loader = DatasetLoaderService(session)
                    dataset_record = await loader.get_dataset_record(
                        analysis.dataset_id, owner_id=owner_id, require_ready=True
                    )

                    if analysis.analysis_type != AnalysisType.EDA:
                        raise NotImplementedError(f"Analysis type '{analysis.analysis_type.value}' not implemented yet")

                    config = analysis.config or {}
                    plan = config.get("plan")
                    if not isinstance(plan, list):
                        plan = eda_p0_plan()

                    sample_rows = config.get("sample_rows")
                    try:
                        sample_rows_int = int(sample_rows) if sample_rows is not None else 200_000
                    except Exception:
                        sample_rows_int = 200_000

                    # Load the dataset once, then reuse it across operators for performance.
                    loaded = await loader.load_dataset(
                        analysis.dataset_id,
                        owner_id=owner_id,
                        require_ready=True,
                        sample_rows=sample_rows_int,
                    )

                    executor = ComputeExecutor(session)
                    steps_out: list[dict[str, Any]] = []
                    total_steps = max(len(plan), 1)

                    for i, step in enumerate(plan):
                        # Refresh for best-effort cancellation.
                        try:
                            await session.refresh(analysis)
                        except Exception:
                            pass
                        if job is not None:
                            try:
                                await session.refresh(job)
                            except Exception:
                                pass

                        if analysis.status == AnalysisStatus.CANCELLED or (
                            job is not None and job.status == JobStatus.CANCELLED
                        ):
                            analysis.status = AnalysisStatus.CANCELLED
                            analysis.completed_at = analysis.completed_at or datetime.utcnow()
                            analysis.status_message = analysis.status_message or "Cancelled"
                            if analysis.started_at and analysis.completed_at:
                                analysis.duration_seconds = (
                                    analysis.completed_at - analysis.started_at
                                ).total_seconds()
                            analysis.updated_by = owner_id

                            if job is not None:
                                job.status = JobStatus.CANCELLED
                                job.completed_at = job.completed_at or datetime.utcnow()
                                job.status_message = job.status_message or "Cancelled"

                            await session.commit()
                            logger.info(
                                "Analysis cancelled; stopping execution",
                                context=context,
                                analysis_id=str(analysis_id),
                            )
                            return

                        operator_name = str((step or {}).get("operator") or "").strip()
                        if not operator_name:
                            continue
                        params = (step or {}).get("params") or {}
                        if not isinstance(params, dict):
                            params = {}

                        result = await executor.run_operator(
                            dataset_id=analysis.dataset_id,
                            operator_name=operator_name,
                            params=params,
                            owner_id=owner_id,
                            sample_rows=sample_rows_int,
                            loaded=loaded,
                        )

                        steps_out.append(
                            {
                                "operator": result.operator_name,
                                "summary": to_jsonable(result.summary),
                                "artifacts": to_jsonable([_artifact_to_dict(a) for a in result.artifacts]),
                            }
                        )

                        progress = float(min(1.0, (i + 1) / float(total_steps)))
                        analysis.progress = progress
                        analysis.status_message = f"Completed {i + 1}/{total_steps}: {operator_name}"
                        analysis.results = {"steps": steps_out}
                        analysis.updated_by = owner_id

                        if job is not None:
                            job.progress = progress
                            job.status_message = analysis.status_message

                        await session.commit()

                    # Deterministic insight extraction from computed artifacts (no LLM).
                    insight_pack = extract_eda_insights(steps=steps_out)
                    analysis.insights = to_jsonable(insight_pack.get("insights") or [])
                    scanned_rows = int(getattr(loaded.df, "shape", [0, 0])[0] or 0)
                    dataset_rows = (
                        int(dataset_record.row_count)
                        if getattr(dataset_record, "row_count", None) is not None
                        else scanned_rows
                    )
                    scan_ratio = float(scanned_rows) / float(max(dataset_rows, 1))
                    if scan_ratio >= 0.95:
                        confidence = "high"
                    elif scan_ratio >= 0.20:
                        confidence = "medium"
                    else:
                        confidence = "low"

                    analysis.results = {
                        "steps": steps_out,
                        "takeaways": insight_pack.get("takeaways") or [],
                        "suggested_prompts": insight_pack.get("suggested_prompts") or [],
                        "suggested_actions": insight_pack.get("suggested_actions") or [],
                        "run_meta": {
                            "dataset_version": loaded.version_hash,
                            "sample_rows": int(sample_rows_int),
                            "scanned_rows": int(scanned_rows),
                            "dataset_rows": int(dataset_rows),
                            "scan_ratio": float(scan_ratio),
                            "confidence": confidence,
                        },
                    }

                    # Always attach a narrative markdown (deterministic by default; optional LLM rewrite).
                    narrative = await NarratorService().narrate_eda(
                        analysis_name=str(analysis.name or "Data Speaks"),
                        run_meta=analysis.results.get("run_meta") if isinstance(analysis.results, dict) else {},
                        takeaways=analysis.results.get("takeaways") if isinstance(analysis.results, dict) else [],
                        suggested_prompts=analysis.results.get("suggested_prompts") if isinstance(analysis.results, dict) else [],
                        insights=analysis.insights if isinstance(getattr(analysis, "insights", None), list) else [],
                        steps=steps_out,
                    )
                    if isinstance(analysis.results, dict):
                        analysis.results["narrative_md"] = narrative.markdown
                        meta = narrative.to_dict()
                        meta.pop("markdown", None)
                        analysis.results["narrative_meta"] = meta

                    analysis.status = AnalysisStatus.COMPLETED
                    analysis.completed_at = datetime.utcnow()
                    analysis.progress = 1.0
                    analysis.status_message = "Completed"
                    if analysis.started_at:
                        analysis.duration_seconds = (analysis.completed_at - analysis.started_at).total_seconds()
                    analysis.updated_by = owner_id

                    if job is not None:
                        job.status = JobStatus.COMPLETED
                        job.progress = 1.0
                        job.completed_at = datetime.utcnow()
                        job.status_message = "Completed"
                        job.result = to_jsonable(
                            {"analysis_id": str(analysis.id), "dataset_id": str(analysis.dataset_id)}
                        )

                    # If this analysis is a child "action", update the parent's action feed.
                    try:
                        await _update_parent_action_feed(session, owner_id=owner_id, child=analysis)
                    except Exception:
                        pass

                    await session.commit()
                    logger.info("Analysis completed", context=context, analysis_id=str(analysis_id))

                except Exception as e:
                    tb = traceback.format_exc()

                    analysis.status = AnalysisStatus.FAILED
                    analysis.completed_at = datetime.utcnow()
                    analysis.status_message = "Failed"
                    analysis.error_message = str(e)
                    analysis.error_traceback = tb
                    analysis.updated_by = owner_id

                    if job is not None:
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.utcnow()
                        job.status_message = "Failed"
                        job.error_message = str(e)
                        job.error_traceback = tb

                    try:
                        await _update_parent_action_feed(session, owner_id=owner_id, child=analysis)
                    except Exception:
                        pass

                    await session.commit()
                    logger.error(
                        "Analysis failed",
                        context=context,
                        analysis_id=str(analysis_id),
                        error=str(e),
                        exc_info=True,
                    )
        finally:
            clear_request_context()
