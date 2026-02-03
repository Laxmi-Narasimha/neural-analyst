# AI Enterprise Data Analyst - ML API Routes
# Machine learning endpoints for model training and prediction

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from app.api.routes.auth import get_current_user, require_permission
from app.services.auth_service import AuthUser, Permission
from app.ml.ml_engine import get_ml_engine, MLTask
from app.ml.automl import get_automl_engine
from app.ml.explainability import get_explainability_engine
from app.ml.model_registry import get_model_registry, ModelStage
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class TrainModelRequest(BaseModel):
    """Model training request."""
    dataset_id: str
    target_column: str
    feature_columns: Optional[list[str]] = None
    task: Optional[str] = None  # classification, regression
    model_name: Optional[str] = None
    hyperparameters: Optional[dict] = None


class TrainModelResponse(BaseModel):
    """Training response."""
    job_id: str
    status: str
    message: str


class PredictRequest(BaseModel):
    """Prediction request."""
    model_name: str
    model_version: Optional[int] = None
    data: list[dict[str, Any]]


class PredictResponse(BaseModel):
    """Prediction response."""
    predictions: list[Any]
    probabilities: Optional[list[list[float]]] = None


class AutoMLRequest(BaseModel):
    """AutoML request."""
    dataset_id: str
    target_column: str
    task: Optional[str] = None
    time_budget_minutes: int = 5
    max_models: int = 10


class ExplainRequest(BaseModel):
    """Model explanation request."""
    model_name: str
    data: list[dict[str, Any]]
    method: str = "shap"


class ModelListResponse(BaseModel):
    """Model list response."""
    models: list[dict[str, Any]]
    total: int


# ============================================================================
# Background Tasks
# ============================================================================

async def run_training_job(
    job_id: str,
    dataset_id: str,
    target_column: str,
    feature_columns: list[str],
    task: str,
    model_name: str,
    hyperparameters: dict
):
    """Background training job."""
    try:
        logger.info(f"Starting training job {job_id}")
        
        # In production, load dataset from database
        # For now, this is a placeholder
        
        ml_engine = get_ml_engine()
        # result = ml_engine.train(...)
        
        logger.info(f"Training job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")


# ============================================================================
# Routes
# ============================================================================

@router.post(
    "/train",
    response_model=TrainModelResponse,
    dependencies=[Depends(require_permission(Permission.MANAGE_MODELS))]
)
async def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    user: AuthUser = Depends(get_current_user)
):
    """Start model training job."""
    import uuid
    job_id = str(uuid.uuid4())
    
    # Add to background tasks
    background_tasks.add_task(
        run_training_job,
        job_id=job_id,
        dataset_id=request.dataset_id,
        target_column=request.target_column,
        feature_columns=request.feature_columns or [],
        task=request.task or "auto",
        model_name=request.model_name or f"model_{job_id[:8]}",
        hyperparameters=request.hyperparameters or {}
    )
    
    return TrainModelResponse(
        job_id=job_id,
        status="started",
        message="Training job started successfully"
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_permission(Permission.RUN_ANALYSIS))]
)
async def predict(
    request: PredictRequest,
    user: AuthUser = Depends(get_current_user)
):
    """Make predictions using a trained model."""
    try:
        import pandas as pd
        
        # Load model from registry
        registry = get_model_registry()
        
        model = registry.load_model(
            request.model_name,
            version=request.model_version
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(df).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df).tolist()
        
        return PredictResponse(
            predictions=predictions,
            probabilities=probabilities
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post(
    "/automl",
    response_model=TrainModelResponse,
    dependencies=[Depends(require_permission(Permission.MANAGE_MODELS))]
)
async def run_automl(
    request: AutoMLRequest,
    background_tasks: BackgroundTasks,
    user: AuthUser = Depends(get_current_user)
):
    """Run AutoML pipeline."""
    import uuid
    job_id = str(uuid.uuid4())
    
    # In production, add AutoML task to queue
    return TrainModelResponse(
        job_id=job_id,
        status="started",
        message="AutoML job started"
    )


@router.post(
    "/explain",
    dependencies=[Depends(require_permission(Permission.RUN_ANALYSIS))]
)
async def explain_model(
    request: ExplainRequest,
    user: AuthUser = Depends(get_current_user)
):
    """Get model explanations (SHAP values)."""
    try:
        import pandas as pd
        
        # Load model
        registry = get_model_registry()
        model = registry.load_model(request.model_name)
        
        # Convert data
        df = pd.DataFrame(request.data)
        
        # Get explanations
        explainer = get_explainability_engine()
        
        if request.method == "shap":
            result = explainer.explain_batch(model, df)
            return {
                "method": "shap",
                "explanations": [r.to_dict() for r in result]
            }
        else:
            # Global importance
            global_exp = explainer.explain_model(model, df)
            return {
                "method": "global",
                "explanation": global_exp.to_dict()
            }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    user: AuthUser = Depends(get_current_user)
):
    """List all registered models."""
    registry = get_model_registry()
    models = registry.list_models()
    
    return ModelListResponse(
        models=models,
        total=len(models)
    )


@router.get("/models/{model_name}")
async def get_model(
    model_name: str,
    user: AuthUser = Depends(get_current_user)
):
    """Get model details."""
    registry = get_model_registry()
    versions = registry.get_model_versions(model_name)
    
    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    return {
        "model_name": model_name,
        "versions": versions
    }


@router.post("/models/{model_name}/versions/{version}/promote")
async def promote_model(
    model_name: str,
    version: int,
    stage: str = "production",
    user: AuthUser = Depends(require_permission(Permission.MANAGE_MODELS))
):
    """Promote model version to a stage."""
    try:
        registry = get_model_registry()
        
        model_stage = ModelStage(stage)
        result = registry.transition_stage(model_name, version, model_stage)
        
        return {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "stage": stage
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/models/{model_name}/compare")
async def compare_model_versions(
    model_name: str,
    version1: int,
    version2: int,
    user: AuthUser = Depends(get_current_user)
):
    """Compare two model versions."""
    registry = get_model_registry()
    
    try:
        comparison = registry.compare_versions(model_name, version1, version2)
        return comparison
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    user: AuthUser = Depends(get_current_user)
):
    """Get training job status."""
    # In production, check job status from database/queue
    return {
        "job_id": job_id,
        "status": "running",
        "progress": 50,
        "message": "Training in progress..."
    }
