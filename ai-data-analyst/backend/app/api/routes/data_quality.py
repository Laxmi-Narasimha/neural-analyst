"""API endpoints for Data Quality and Adequacy analysis."""

from datetime import datetime
import json
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


from app.api.schemas import APIResponse
from app.api.routes.auth import require_permission
from app.core.serialization import to_jsonable
from app.compute.artifacts import ArtifactStore
from app.models import DataAdequacySession, AdequacySessionStatus, Dataset, DatasetVersion
from app.services.artifact_index import ArtifactIndexService
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.services.dataset_loader import DatasetLoaderService

router = APIRouter()

class ValidationRequest(BaseModel):
    goal: str
    domain: str = "general"
    dataset_id: Optional[UUID] = None
    files: List[str] = []  # Deprecated: use dataset_id

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
async def start_validation_session(
    request: ValidationRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    """Start a new data adequacy validation session."""
    try:
        files: list[str] = []
        dataset_version_hash: str | None = None
        if request.dataset_id is not None:
            loader = DatasetLoaderService(db)
            dataset = await loader.get_dataset_record(request.dataset_id, owner_id=user.user_id, require_ready=False)
            try:
                from app.services.object_store import get_object_store

                obj = get_object_store()
                local_path = obj.ensure_local_path(
                    str(dataset.file_path),
                    expected_size_bytes=int(getattr(dataset, "file_size_bytes", 0) or 0) or None,
                    filename_hint=str(getattr(dataset, "original_filename", "") or ""),
                )
                files = [str(local_path)]
            except Exception:
                files = [dataset.file_path]
            if isinstance(getattr(dataset, "profile_report", None), dict):
                dataset_version_hash = str(dataset.profile_report.get("file_hash") or "").strip() or None
        elif request.files:
            raise HTTPException(status_code=400, detail="Direct file paths are not allowed. Use dataset_id.")

        from app.agents.data_adequacy.manager import DataAdequacyManager
        manager = DataAdequacyManager()
        result = await manager.run_validation(
            user_goal=request.goal,
            domain=request.domain,
            files=files
        )

        session_id = result.get("session_id") or manager.session_state.get("session_id")
        if session_id:
            state = dict(manager.session_state or {})
            if request.dataset_id is not None:
                state["dataset_id"] = str(request.dataset_id)
                if dataset_version_hash is not None:
                    state["dataset_version_hash"] = dataset_version_hash
            status_val: AdequacySessionStatus
            if not result.get("success"):
                status_val = AdequacySessionStatus.FAILED
            elif result.get("step") == "clarifying_questions":
                status_val = AdequacySessionStatus.QUESTIONS
            elif result.get("readiness_level") is not None:
                status_val = AdequacySessionStatus.COMPLETED
            else:
                status_val = AdequacySessionStatus.RUNNING

            db.add(
                DataAdequacySession(
                    session_id=session_id,
                    owner_id=user.user_id,
                    dataset_id=request.dataset_id,
                    status=status_val,
                    state=to_jsonable(state),
                    llm_calls=manager.llm_call_count,
                )
            )
            await db.commit()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/continue", response_model=Dict[str, Any])
async def continue_validation_session(
    request: AnswerRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    """Continue validation with user answers."""
    session_id = request.session_id
    res = await db.execute(
        select(DataAdequacySession).where(
            DataAdequacySession.session_id == session_id,
            DataAdequacySession.owner_id == user.user_id,
            DataAdequacySession.is_deleted == False,  # noqa: E712
        )
    )
    session = res.scalars().first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from app.agents.data_adequacy.manager import DataAdequacyManager
    manager = DataAdequacyManager()
    manager.session_state = session.state or {}
    manager.llm_call_count = int(session.llm_calls or 0)
    try:
        result = await manager.continue_validation(request.answers)

        if not result.get("success"):
            session.status = AdequacySessionStatus.FAILED
        elif result.get("readiness_level") is not None:
            session.status = AdequacySessionStatus.COMPLETED
        else:
            session.status = AdequacySessionStatus.RUNNING

        session.state = manager.session_state or {}
        session.llm_calls = manager.llm_call_count
        session.state = to_jsonable(session.state or {})
        await db.commit()

        # If completed and attached to a dataset, persist readiness to the dataset + active version,
        # and generate artifacts for reproducibility.
        if session.dataset_id is not None and result.get("readiness_level") is not None:
            ds_res = await db.execute(
                select(Dataset).where(
                    Dataset.id == session.dataset_id,
                    Dataset.owner_id == user.user_id,
                    Dataset.is_deleted == False,  # noqa: E712
                )
            )
            dataset = ds_res.scalars().first()

            score: float | None = None
            try:
                score = float(result.get("composite_score")) if result.get("composite_score") is not None else None
            except Exception:
                score = None

            readiness_payload = {
                "source": "data_adequacy",
                "readiness_level": result.get("readiness_level"),
                "composite_score": score,
                "executive_summary": result.get("executive_summary") or result.get("summary"),
                "top_recommendations": result.get("top_recommendations") or result.get("recommendations") or [],
                "next_steps": result.get("next_steps") or [],
                "updated_at": datetime.utcnow().isoformat(),
            }

            dataset_version_hash = None
            if isinstance(session.state, dict):
                dataset_version_hash = session.state.get("dataset_version_hash")
            if dataset_version_hash is None and dataset is not None and isinstance(getattr(dataset, "profile_report", None), dict):
                dataset_version_hash = str(dataset.profile_report.get("file_hash") or "").strip() or None

            if dataset is not None:
                if score is not None:
                    dataset.quality_score = score
                dataset.quality_report = to_jsonable(readiness_payload)
                dataset.updated_by = user.user_id

            if dataset_version_hash and dataset is not None:
                ver_res = await db.execute(
                    select(DatasetVersion).where(
                        DatasetVersion.dataset_id == dataset.id,
                        DatasetVersion.owner_id == user.user_id,
                        DatasetVersion.version_hash == dataset_version_hash,
                        DatasetVersion.is_deleted == False,  # noqa: E712
                    )
                )
                ver = ver_res.scalars().first()
                if ver is not None:
                    if score is not None:
                        ver.quality_score = score
                    ver.quality_report = to_jsonable(readiness_payload)
                    ver.updated_by = user.user_id

            await db.commit()

            try:
                store = ArtifactStore()
                refs = []
                if score is not None:
                    refs.append(
                        store.write_metric(
                            name="Data Readiness Score",
                            value=float(score),
                            unit="score_0_1",
                            details={
                                "readiness_level": result.get("readiness_level"),
                                "goal": str((session.state or {}).get("user_goal") or ""),
                                "domain": str((session.state or {}).get("domain") or ""),
                            },
                            dataset_id=session.dataset_id,
                            dataset_version=dataset_version_hash,
                            operator_name="data_adequacy",
                            operator_params={"session_id": session.session_id},
                        )
                    )

                md = result.get("markdown_report") or ""
                if isinstance(md, str) and md.strip():
                    refs.append(
                        store.write_report(
                            name="Data Readiness Report",
                            content=md,
                            report_format="markdown",
                            dataset_id=session.dataset_id,
                            dataset_version=dataset_version_hash,
                            operator_name="data_adequacy",
                            operator_params={"session_id": session.session_id},
                        )
                    )

                js = result.get("json_report")
                if js is not None:
                    try:
                        js_text = json.dumps(to_jsonable(js), ensure_ascii=True, default=str)
                    except Exception:
                        js_text = str(js)
                    refs.append(
                        store.write_report(
                            name="Data Readiness Report (JSON)",
                            content=js_text,
                            report_format="json",
                            dataset_id=session.dataset_id,
                            dataset_version=dataset_version_hash,
                            operator_name="data_adequacy",
                            operator_params={"session_id": session.session_id},
                        )
                    )

                if refs:
                    await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
                    result["artifacts"] = [
                        {
                            "artifact_id": str(r.artifact_id),
                            "artifact_type": r.artifact_type.value,
                            "name": r.name,
                            "created_at": r.created_at,
                            "storage_path": r.storage_path,
                            "preview": r.preview,
                            "dataset_id": str(r.dataset_id) if r.dataset_id else None,
                            "dataset_version": r.dataset_version,
                            "operator_name": r.operator_name,
                            "operator_params": r.operator_params or {},
                        }
                        for r in refs
                    ]
            except Exception:
                # Don't fail the user flow if artifact persistence isn't available.
                pass

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}", response_model=Dict[str, Any])
async def get_session_status(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    """Get status of an active session."""
    res = await db.execute(
        select(DataAdequacySession).where(
            DataAdequacySession.session_id == session_id,
            DataAdequacySession.owner_id == user.user_id,
            DataAdequacySession.is_deleted == False,  # noqa: E712
        )
    )
    session = res.scalars().first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from app.agents.data_adequacy.manager import DataAdequacyManager
    manager = DataAdequacyManager()
    manager.session_state = session.state or {}
    manager.llm_call_count = int(session.llm_calls or 0)
    status = manager.get_session_status()
    status["status"] = session.status.value
    return status

@router.get("/domains", response_model=List[str])
async def get_supported_domains(user: AuthUser = Depends(require_permission(Permission.READ_DATA))):
    """Get list of supported domains."""
    from app.agents.data_adequacy.config import config
    return list(config.DOMAIN_CONFIGS.keys())


@router.get("/sessions/latest", response_model=APIResponse[dict[str, Any]])
async def get_latest_quality_session(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    """Fetch the latest persisted adequacy session for a dataset (owner-only)."""
    res = await db.execute(
        select(DataAdequacySession)
        .where(
            DataAdequacySession.owner_id == user.user_id,
            DataAdequacySession.dataset_id == dataset_id,
            DataAdequacySession.is_deleted == False,  # noqa: E712
        )
        .order_by(DataAdequacySession.created_at.desc())
        .limit(1)
    )
    sess = res.scalars().first()
    if sess is None:
        return APIResponse.success(data=None)

    state = sess.state or {}
    agent_results = state.get("agent_results") if isinstance(state, dict) else None
    validation = (agent_results or {}).get("validation") if isinstance(agent_results, dict) else None
    if not isinstance(validation, dict):
        validation = {}

    payload = {
        "session_id": sess.session_id,
        "status": sess.status.value,
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
        "readiness_level": validation.get("readiness_level"),
        "composite_score": validation.get("composite_score"),
        "executive_summary": validation.get("summary"),
    }
    return APIResponse.success(data=payload)
