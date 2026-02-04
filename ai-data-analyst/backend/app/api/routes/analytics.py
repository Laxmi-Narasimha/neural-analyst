# AI Enterprise Data Analyst - Analytics API Routes
# Advanced analytics endpoints

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user, require_permission
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.services.dataset_loader import DatasetLoaderService
from app.ml.statistical_tests import get_statistical_testing_engine
from app.ml.segmentation import get_segmentation_engine
from app.ml.customer_analytics import get_customer_analytics_engine
from app.ml.forecasting import get_forecast_engine
from app.ml.anomaly_detection import get_anomaly_detection_engine
from app.ml.ab_testing import get_ab_testing_engine
from app.ml.data_profiling import get_data_profiling_engine
from app.ml.bi_metrics import get_bi_metrics_engine
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class StatisticalTestRequest(BaseModel):
    """Statistical test request."""
    dataset_id: UUID
    test_type: str  # t_test, anova, chi_square, correlation
    group_column: Optional[str] = None
    value_column: str
    groups: Optional[list[str]] = None
    alpha: float = 0.05


class SegmentationRequest(BaseModel):
    """Segmentation request."""
    dataset_id: UUID
    method: str = "kmeans"
    n_segments: int = 5
    features: Optional[list[str]] = None


class CustomerAnalyticsRequest(BaseModel):
    """Customer analytics request."""
    dataset_id: UUID
    analysis_type: str  # rfm, cohort, clv, churn
    customer_column: str
    date_column: str
    value_column: Optional[str] = None


class ForecastRequest(BaseModel):
    """Forecasting request."""
    dataset_id: UUID
    date_column: str
    value_column: str
    periods: int = 30
    method: str = "ensemble"


class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request."""
    dataset_id: UUID
    columns: Optional[list[str]] = None
    method: str = "isolation_forest"
    contamination: float = 0.1


class ABTestRequest(BaseModel):
    """A/B test analysis request."""
    control_conversions: int
    control_total: int
    treatment_conversions: int
    treatment_total: int
    method: str = "frequentist"
    alpha: float = 0.05


class DataProfileRequest(BaseModel):
    """Data profiling request."""
    dataset_id: UUID


class BIMetricsRequest(BaseModel):
    """BI metrics request."""
    dataset_id: UUID
    revenue_column: Optional[str] = None
    user_column: Optional[str] = None
    date_column: Optional[str] = None


# ============================================================================
# Routes
# ============================================================================

@router.post("/statistical-test")
async def run_statistical_test(
    request: StatisticalTestRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS))
):
    """Run statistical test."""
    try:
        loader = DatasetLoaderService(db)
        df = (await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)).df
        engine = get_statistical_testing_engine(request.alpha)
        
        if request.test_type == "correlation":
            # Correlation matrix
            result = engine.correlation_matrix(df)
        else:
            # Group comparison
            result = engine.compare_groups(
                df,
                group_col=request.group_column,
                value_col=request.value_column,
                groups=request.groups
            )
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"Statistical test error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/segmentation")
async def run_segmentation(
    request: SegmentationRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS))
):
    """Run customer/data segmentation."""
    try:
        from app.ml.segmentation import SegmentationMethod
        
        loader = DatasetLoaderService(db)
        df = (await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)).df
        engine = get_segmentation_engine()
        
        method = SegmentationMethod(request.method)
        result = engine.segment(
            df,
            method=method,
            n_segments=request.n_segments,
            features=request.features
        )
        
        return {
            "status": "success",
            "result": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/customer-analytics")
async def run_customer_analytics(
    request: CustomerAnalyticsRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS))
):
    """Run customer analytics (RFM, Cohort, CLV, Churn)."""
    try:
        loader = DatasetLoaderService(db)
        df = (await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)).df
        engine = get_customer_analytics_engine()
        
        if request.analysis_type == "rfm":
            result = engine.rfm.analyze(
                df,
                customer_col=request.customer_column,
                date_col=request.date_column,
                value_col=request.value_column or 'revenue'
            )
        elif request.analysis_type == "cohort":
            result = engine.cohort.analyze(
                df,
                customer_col=request.customer_column,
                date_col=request.date_column,
                value_col=request.value_column
            )
        elif request.analysis_type == "clv":
            result = engine.clv.calculate(
                df,
                customer_col=request.customer_column,
                date_col=request.date_column,
                value_col=request.value_column or 'revenue'
            )
        else:
            raise ValueError(f"Unknown analysis type: {request.analysis_type}")
        
        return {
            "status": "success",
            "analysis_type": request.analysis_type,
            "result": result.to_dict() if hasattr(result, 'to_dict') else result
        }
        
    except Exception as e:
        logger.error(f"Customer analytics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/forecast")
async def run_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS))
):
    """Run time series forecasting."""
    try:
        from app.ml.forecasting import ForecastMethod
        
        loader = DatasetLoaderService(db)
        df = (await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)).df
        engine = get_forecast_engine()
        
        result = engine.forecast(
            df,
            date_col=request.date_column,
            value_col=request.value_column,
            periods=request.periods,
            method=ForecastMethod(request.method)
        )
        
        return {
            "status": "success",
            "result": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/anomaly-detection")
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS))
):
    """Detect anomalies in data."""
    try:
        from app.ml.anomaly_detection import AnomalyMethod
        
        loader = DatasetLoaderService(db)
        df = (await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)).df
        engine = get_anomaly_detection_engine()
        
        result = engine.detect(
            df,
            columns=request.columns,
            method=AnomalyMethod(request.method),
            contamination=request.contamination
        )
        
        return {
            "status": "success",
            "result": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/ab-test")
async def analyze_ab_test(
    request: ABTestRequest,
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS))
):
    """Analyze A/B test results."""
    try:
        engine = get_ab_testing_engine()
        
        if request.method == "bayesian":
            result = engine.bayesian_test(
                control_conversions=request.control_conversions,
                control_total=request.control_total,
                treatment_conversions=request.treatment_conversions,
                treatment_total=request.treatment_total
            )
        else:
            result = engine.frequentist_test(
                control_conversions=request.control_conversions,
                control_total=request.control_total,
                treatment_conversions=request.treatment_conversions,
                treatment_total=request.treatment_total,
                alpha=request.alpha
            )
        
        return {
            "status": "success",
            "method": request.method,
            "result": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"A/B test error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/profile")
async def profile_data(
    request: DataProfileRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA))
):
    """Generate data profile."""
    try:
        loader = DatasetLoaderService(db)
        df = (await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)).df
        engine = get_data_profiling_engine()
        
        profile = engine.profile(df, name=request.dataset_id)
        
        return {
            "status": "success",
            "profile": profile.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Data profiling error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/bi-metrics")
async def calculate_bi_metrics(
    request: BIMetricsRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA))
):
    """Calculate business intelligence metrics."""
    try:
        loader = DatasetLoaderService(db)
        df = (await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)).df
        engine = get_bi_metrics_engine()
        
        metrics = engine.calculate_core_metrics(
            df,
            revenue_col=request.revenue_column,
            user_col=request.user_column,
            date_col=request.date_column
        )
        
        result = {name: m.to_dict() for name, m in metrics.items()}
        
        return {
            "status": "success",
            "metrics": result
        }
        
    except Exception as e:
        logger.error(f"BI metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/sample-size")
async def calculate_sample_size(
    baseline_rate: float = 0.1,
    mde: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.8,
    user: AuthUser = Depends(get_current_user)
):
    """Calculate required sample size for A/B test."""
    engine = get_ab_testing_engine()
    
    sample_size = engine.sample_size_calculator.calculate(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=alpha,
        power=power
    )
    
    return {
        "required_sample_size_per_variant": sample_size,
        "total_sample_size": sample_size * 2,
        "parameters": {
            "baseline_rate": baseline_rate,
            "mde": mde,
            "alpha": alpha,
            "power": power
        }
    }
