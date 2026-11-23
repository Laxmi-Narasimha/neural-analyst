# AI Enterprise Data Analyst - ML API Routes
# Machine learning endpoints for model training and prediction

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import APIResponse
from app.api.routes.auth import get_current_user, require_permission
from app.core.config import settings
from app.services.database import get_db_session
from app.services.auth_service import AuthUser, Permission
from app.services.job_service import JobService
from app.services.ml_training_service import MLTrainingService
from app.services.subscription_service import SubscriptionService, UsageAction
from app.models import JobType
from app.workers.dispatcher import enqueue_ml_training

from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class TrainModelRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: Optional[list[str]] = None
    task: Optional[str] = None
    model_name: Optional[str] = None
    hyperparameters: Optional[dict] = None
    optimize: bool = True
    sample_rows: Optional[int] = None
    max_models: Optional[int] = None
    mode: str = Field(default="baseline", description="baseline (sync) or automl (async job)")


class TrainModelResponse(BaseModel):
    job_id: Optional[str] = None
    status: str
    message: str
    result: Optional[dict[str, Any]] = None


class PredictRequest(BaseModel):
    model_name: str
    model_version: Optional[int] = None
    data: list[dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: list[Any]
    probabilities: Optional[list[list[float]]] = None


class AutoMLRequest(BaseModel):
    dataset_id: str
    target_column: str
    task: Optional[str] = None
    time_budget_minutes: int = 60
    max_models: int = 10
    sample_rows: Optional[int] = None
    optimize: bool = True


class ExplainRequest(BaseModel):
    model_name: str
    data: list[dict[str, Any]]
    method: str = "shap"


class ModelListResponse(BaseModel):
    models: list[dict[str, Any]]
    total: int


class LeakageWarningResponse(BaseModel):
    column: str
    warning_type: str
    detail: str
    severity: str


class TargetCandidateResponse(BaseModel):
    column: str
    inferred_task: str
    score: float
    non_null_rate: float
    unique_ratio: float
    reasons: list[str] = Field(default_factory=list)
    leakage_warnings: list[LeakageWarningResponse] = Field(default_factory=list)


class TaskInferenceRequest(BaseModel):
    dataset_id: UUID
    preferred_target: Optional[str] = None
    sample_rows: int = Field(default=100_000, ge=1_000, le=2_000_000)


class TaskInferenceResponse(BaseModel):
    dataset_id: UUID
    dataset_version: str
    sample_rows: int
    split_strategy: str
    split_time_column: Optional[str] = None
    selected_target: Optional[str] = None
    selected_task: Optional[str] = None
    candidates: list[TargetCandidateResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@router.post(
    "/train",
    response_model=TrainModelResponse,
    dependencies=[Depends(require_permission(Permission.MANAGE_MODELS))],
)
async def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Train models on a dataset sample. baseline=sync quick train; automl=async full pipeline."""
    try:
        dataset_id = UUID(str(request.dataset_id))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid dataset_id")

    mode = str(request.mode or "baseline").strip().lower()
    service = MLTrainingService(db)
    subscription = SubscriptionService(db)
    await subscription.assert_can_run(user.user_id, action=UsageAction.ML_TRAIN)

    if mode in {"automl", "full", "async"}:
        jobs = JobService(db)
        job = await jobs.create(
            owner_id=user.user_id,
            job_type=JobType.ML_TRAINING,
            dataset_id=dataset_id,
            payload={
                "target_column": request.target_column,
                "task": request.task,
                "mode": "automl",
                "sample_rows": request.sample_rows or settings.ml.automl_training_sample_rows,
                "max_models": request.max_models or settings.ml.automl_max_models,
                "optimize": request.optimize,
            },
            status_message="Queued AutoML training",
        )
        enqueue_ml_training(
            background_tasks=background_tasks,
            job_id=job.id,
            owner_id=user.user_id,
            dataset_id=dataset_id,
            target_column=request.target_column,
            task=request.task,
            sample_rows=request.sample_rows,
            max_models=request.max_models,
            optimize=request.optimize,
        )
        return TrainModelResponse(
            job_id=str(job.id),
            status="queued",
            message=(
                f"AutoML job queued on up to {(request.sample_rows or settings.ml.automl_training_sample_rows):,} "
                f"representative rows with hyperparameter tuning."
            ),
        )

    result = await service.run_baseline_training(
        dataset_id=dataset_id,
        owner_id=user.user_id,
        target_column=request.target_column,
        task=request.task,
        optimize=request.optimize,
        max_models=request.max_models,
        sample_rows=request.sample_rows,
    )
    return TrainModelResponse(
        status="completed",
        message=f"Training complete — best model: {result.get('best_model')}",
        result=result,
    )


@router.post(
    "/automl",
    response_model=TrainModelResponse,
    dependencies=[Depends(require_permission(Permission.MANAGE_MODELS))],
)
async def run_automl(
    request: AutoMLRequest,
    background_tasks: BackgroundTasks,
    user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Run full AutoML pipeline as a background job."""
    try:
        dataset_id = UUID(str(request.dataset_id))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid dataset_id")

    jobs = JobService(db)
    await SubscriptionService(db).assert_can_run(user.user_id, action=UsageAction.ML_TRAIN)
    job = await jobs.create(
        owner_id=user.user_id,
        job_type=JobType.ML_TRAINING,
        dataset_id=dataset_id,
        payload={
            "target_column": request.target_column,
            "task": request.task,
            "mode": "automl",
            "time_budget_minutes": request.time_budget_minutes,
            "sample_rows": request.sample_rows or settings.ml.automl_training_sample_rows,
            "max_models": request.max_models,
            "optimize": request.optimize,
        },
        status_message="Queued AutoML",
    )
    enqueue_ml_training(
        background_tasks=background_tasks,
        job_id=job.id,
        owner_id=user.user_id,
        dataset_id=dataset_id,
        target_column=request.target_column,
        task=request.task,
        sample_rows=request.sample_rows,
        max_models=request.max_models,
        optimize=request.optimize,
    )
    return TrainModelResponse(
        job_id=str(job.id),
        status="queued",
        message="AutoML job started",
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_permission(Permission.RUN_ANALYSIS))],
)
async def predict(
    request: PredictRequest,
    user: AuthUser = Depends(get_current_user),
):
    """Make predictions using a trained model."""
    try:
        import pandas as pd
        from app.ml.model_registry import get_model_registry

        registry = get_model_registry()
        model = registry.load_model(request.model_name, version=request.model_version)
        df = pd.DataFrame(request.data)
        predictions = model.predict(df).tolist()
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df).tolist()
        return PredictResponse(predictions=predictions, probabilities=probabilities)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/explain",
    dependencies=[Depends(require_permission(Permission.RUN_ANALYSIS))],
)
async def explain_model(
    request: ExplainRequest,
    user: AuthUser = Depends(get_current_user),
):
    """Get model explanations (SHAP values)."""
    try:
        import pandas as pd
        from app.ml.explainability import get_explainability_engine
        from app.ml.model_registry import get_model_registry

        registry = get_model_registry()
        model = registry.load_model(request.model_name)
        df = pd.DataFrame(request.data)
        explainer = get_explainability_engine()

        if request.method == "shap":
            result = explainer.explain_batch(model, df)
            return {"method": "shap", "explanations": [r.to_dict() for r in result]}
        global_exp = explainer.explain_model(model, df)
        return {"method": "global", "explanation": global_exp.to_dict()}
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/models", response_model=ModelListResponse)
async def list_models(user: AuthUser = Depends(get_current_user)):
    from app.ml.model_registry import get_model_registry

    registry = get_model_registry()
    models = registry.list_models()
    return ModelListResponse(models=models, total=len(models))


@router.get("/models/{model_name}")
async def get_model(model_name: str, user: AuthUser = Depends(get_current_user)):
    from app.ml.model_registry import get_model_registry

    registry = get_model_registry()
    versions = registry.get_model_versions(model_name)
    if not versions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return {"model_name": model_name, "versions": versions}


@router.post("/models/{model_name}/versions/{version}/promote")
async def promote_model(
    model_name: str,
    version: int,
    stage: str = "production",
    user: AuthUser = Depends(require_permission(Permission.MANAGE_MODELS)),
):
    try:
        from app.ml.model_registry import ModelStage, get_model_registry

        registry = get_model_registry()
        model_stage = ModelStage(stage)
        registry.transition_stage(model_name, version, model_stage)
        return {"status": "success", "model_name": model_name, "version": version, "stage": stage}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/models/{model_name}/compare")
async def compare_model_versions(
    model_name: str,
    version1: int,
    version2: int,
    user: AuthUser = Depends(get_current_user),
):
    from app.ml.model_registry import get_model_registry

    registry = get_model_registry()
    try:
        return registry.compare_versions(model_name, version1, version2)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get ML training job status."""
    try:
        jid = UUID(str(job_id))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job_id")

    jobs = JobService(db)
    job = await jobs.get(jid, owner_id=user.user_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return {
        "job_id": str(job.id),
        "status": getattr(job.status, "value", str(job.status)),
        "progress": float(job.progress or 0.0),
        "message": job.status_message,
        "result": job.result or {},
        "error_message": job.error_message,
    }


@router.post(
    "/task-inference",
    response_model=APIResponse[TaskInferenceResponse],
    dependencies=[Depends(require_permission(Permission.READ_DATA))],
)
async def infer_task_and_target(
    request: TaskInferenceRequest,
    user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    from app.services.ml_task_inference import MLTaskInferenceService

    service = MLTaskInferenceService(db)
    result = await service.infer(
        dataset_id=request.dataset_id,
        owner_id=user.user_id,
        preferred_target=request.preferred_target,
        sample_rows=request.sample_rows,
    )

    candidates = [
        TargetCandidateResponse(
            column=c.column,
            inferred_task=c.inferred_task,
            score=c.score,
            non_null_rate=c.non_null_rate,
            unique_ratio=c.unique_ratio,
            reasons=list(c.reasons),
            leakage_warnings=[
                LeakageWarningResponse(
                    column=w.column,
                    warning_type=w.warning_type,
                    detail=w.detail,
                    severity=w.severity,
                )
                for w in c.leakage_warnings
            ],
        )
        for c in result.candidates
    ]
    payload = TaskInferenceResponse(
        dataset_id=result.dataset_id,
        dataset_version=result.dataset_version,
        sample_rows=result.sample_rows,
        split_strategy=result.split_strategy,
        split_time_column=result.split_time_column,
        selected_target=result.selected_target,
        selected_task=result.selected_task,
        candidates=candidates,
        warnings=result.warnings,
    )
    return APIResponse.success(data=payload)
