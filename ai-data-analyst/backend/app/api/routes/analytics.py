# AI Enterprise Data Analyst - Analytics API Routes
# Advanced analytics endpoints

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user, require_permission
from app.compute.artifacts import ArtifactStore
from app.core.serialization import to_jsonable
from app.services.auth_service import AuthUser, Permission
from app.services.artifact_index import ArtifactIndexService
from app.services.database import get_db_session
from app.services.dataset_loader import DatasetLoaderService
# NOTE: ML imports (statistical_tests, segmentation, customer_analytics, etc.)
# are done lazily INSIDE each route function to avoid loading torch/sklearn/scipy
# at startup, which adds 2+ minutes of import time.
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _artifact_ref_to_dict(a: Any) -> dict[str, Any]:
    atype = getattr(getattr(a, "artifact_type", None), "value", None)
    return {
        "artifact_id": str(getattr(a, "artifact_id", "")),
        "artifact_type": atype,
        "name": getattr(a, "name", None),
        "created_at": getattr(a, "created_at", None),
        "storage_path": getattr(a, "storage_path", None),
        "preview": to_jsonable(getattr(a, "preview", None) or {}),
        "dataset_id": str(getattr(a, "dataset_id", None)) if getattr(a, "dataset_id", None) else None,
        "dataset_version": getattr(a, "dataset_version", None),
        "operator_name": getattr(a, "operator_name", None),
        "operator_params": to_jsonable(getattr(a, "operator_params", None) or {}),
    }


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
        loaded = await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)
        df = loaded.df
        dataset_version = loaded.version_hash
        from app.ml.statistical_tests import get_statistical_testing_engine
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

        result_json = to_jsonable(result)
        artifacts: list[dict[str, Any]] = []
        try:
            import pandas as pd

            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = "analytics.statistical_test"

            # Always persist the full result as a JSON report artifact.
            refs.append(
                store.write_report(
                    name=f"Statistical Test ({request.test_type})",
                    content=json.dumps(result_json, ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=request.dataset_id,
                    dataset_version=dataset_version,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            if request.test_type == "correlation":
                corrs = result_json.get("correlations") if isinstance(result_json, dict) else None
                if isinstance(corrs, dict):
                    rows = []
                    for k, v in corrs.items():
                        if not isinstance(v, dict):
                            continue
                        key = str(k)
                        a, b = (key.split("_vs_", 1) + [""])[:2] if "_vs_" in key else (key, "")
                        rows.append(
                            {
                                "column_a": a,
                                "column_b": b,
                                "corr": v.get("statistic"),
                                "p_value": v.get("p_value"),
                                "is_significant": v.get("is_significant"),
                                "method": ((v.get("details") or {}) if isinstance(v.get("details"), dict) else {}).get("method"),
                            }
                        )
                    if rows:
                        pairs = pd.DataFrame(rows)
                        try:
                            pairs["abs_corr"] = pairs["corr"].abs()
                            pairs = pairs.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])
                        except Exception:
                            pass
                        pairs = pairs.head(200).reset_index(drop=True)
                        refs.append(
                            store.write_table(
                                name="Correlation Pairs (top)",
                                df=pairs,
                                dataset_id=request.dataset_id,
                                dataset_version=dataset_version,
                                operator_name=op_name,
                                operator_params=params,
                            )
                        )

            else:
                main = result_json.get("main_test") if isinstance(result_json, dict) else None
                if isinstance(main, dict):
                    if main.get("p_value") is not None:
                        refs.append(
                            store.write_metric(
                                name="p_value",
                                value=float(main["p_value"]),
                                unit="p",
                                details={"test_type": request.test_type},
                                dataset_id=request.dataset_id,
                                dataset_version=dataset_version,
                                operator_name=op_name,
                                operator_params=params,
                            )
                        )
                    if main.get("effect_size") is not None:
                        try:
                            refs.append(
                                store.write_metric(
                                    name="effect_size",
                                    value=float(main["effect_size"]),
                                    unit="effect",
                                    details={"test_type": request.test_type},
                                    dataset_id=request.dataset_id,
                                    dataset_version=dataset_version,
                                    operator_name=op_name,
                                    operator_params=params,
                                )
                            )
                        except Exception:
                            pass

            if refs:
                await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
                artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {"status": "success", "result": result_json, "dataset_version": dataset_version, "artifacts": artifacts}
        
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
        loaded = await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)
        df = loaded.df
        dataset_version = loaded.version_hash
        from app.ml.segmentation import get_segmentation_engine
        engine = get_segmentation_engine()
        
        method = SegmentationMethod(request.method)
        result = engine.segment(
            df,
            method=method,
            n_segments=request.n_segments,
            features=request.features
        )
        result_json = to_jsonable(result.to_dict())
        artifacts: list[dict[str, Any]] = []
        try:
            import pandas as pd

            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = "analytics.segmentation"
            segs = result_json.get("segments") if isinstance(result_json, dict) else None
            if isinstance(segs, list):
                rows = []
                for s in segs:
                    if not isinstance(s, dict):
                        continue
                    rows.append(
                        {
                            "segment_id": s.get("segment_id"),
                            "name": s.get("name"),
                            "size": s.get("size"),
                            "percentage": s.get("percentage"),
                        }
                    )
                if rows:
                    refs.append(
                        store.write_table(
                            name="Segments Summary",
                            df=pd.DataFrame(rows),
                            dataset_id=request.dataset_id,
                            dataset_version=dataset_version,
                            operator_name=op_name,
                            operator_params=params,
                        )
                    )

            qm = result_json.get("quality_metrics") if isinstance(result_json, dict) else None
            if isinstance(qm, dict):
                for k in ["silhouette_score", "davies_bouldin", "calinski_harabasz"]:
                    if qm.get(k) is None:
                        continue
                    try:
                        refs.append(
                            store.write_metric(
                                name=f"Segmentation {k}",
                                value=float(qm[k]),
                                unit="metric",
                                details={"method": request.method, "n_segments": request.n_segments},
                                dataset_id=request.dataset_id,
                                dataset_version=dataset_version,
                                operator_name=op_name,
                                operator_params=params,
                            )
                        )
                    except Exception:
                        pass

            refs.append(
                store.write_report(
                    name="Segmentation Result (JSON)",
                    content=json.dumps(result_json, ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=request.dataset_id,
                    dataset_version=dataset_version,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
            artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {
            "status": "success",
            "result": result_json,
            "dataset_version": dataset_version,
            "artifacts": artifacts,
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
        loaded = await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)
        df = loaded.df
        dataset_version = loaded.version_hash
        from app.ml.customer_analytics import get_customer_analytics_engine
        engine = get_customer_analytics_engine()

        analysis_type = str(request.analysis_type or "").strip().lower()
        result_json: dict[str, Any]
        if analysis_type == "rfm":
            amount_col = request.value_column or "revenue"
            scores = engine.rfm.analyze(
                df,
                customer_col=request.customer_column,
                date_col=request.date_column,
                amount_col=amount_col,
            )
            segments = engine.rfm.get_segment_summary(scores)
            sample_scores = [s.to_dict() for s in scores[:100]]
            result_json = {
                "analysis_type": "rfm",
                "total_customers": int(len(scores)),
                "amount_column": amount_col,
                "segments": to_jsonable(segments),
                "sample_scores": to_jsonable(sample_scores),
            }
        elif analysis_type == "cohort":
            cohort_df = engine.cohort.analyze(
                df,
                customer_col=request.customer_column,
                date_col=request.date_column,
            )
            rows = []
            try:
                rows = cohort_df.head(2000).to_dict(orient="records")
            except Exception:
                rows = []
            result_json = {
                "analysis_type": "cohort",
                "rows": to_jsonable(rows),
                "row_count": int(getattr(cohort_df, "shape", [0, 0])[0] or 0),
            }
        elif analysis_type == "clv":
            amount_col = request.value_column or "revenue"
            clv_results = engine.clv.calculate_predictive(
                df,
                customer_col=request.customer_column,
                date_col=request.date_column,
                amount_col=amount_col,
            )
            clv_sorted = sorted(clv_results, key=lambda x: float(getattr(x, "predicted_clv", 0.0) or 0.0), reverse=True)
            top = [c.to_dict() for c in clv_sorted[:100]]
            total_pred = float(sum(float(getattr(c, "predicted_clv", 0.0) or 0.0) for c in clv_results))
            avg_pred = float(total_pred / float(max(len(clv_results), 1)))
            result_json = {
                "analysis_type": "clv",
                "amount_column": amount_col,
                "total_customers": int(len(clv_results)),
                "total_predicted_clv": float(total_pred),
                "avg_predicted_clv": float(avg_pred),
                "top_customers": to_jsonable(top),
            }
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        artifacts: list[dict[str, Any]] = []
        try:
            import pandas as pd

            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = f"analytics.customer_analytics.{analysis_type}"
            refs.append(
                store.write_report(
                    name=f"Customer Analytics ({analysis_type}) (JSON)",
                    content=json.dumps(result_json, ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=request.dataset_id,
                    dataset_version=dataset_version,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            if analysis_type == "rfm":
                segs = result_json.get("segments")
                if isinstance(segs, dict) and segs:
                    seg_rows = []
                    for seg, stats in segs.items():
                        if not isinstance(stats, dict):
                            continue
                        seg_rows.append(
                            {
                                "segment": seg,
                                "count": stats.get("count"),
                                "total_revenue": stats.get("total_revenue"),
                                "avg_frequency": stats.get("avg_frequency"),
                                "avg_recency": stats.get("avg_recency"),
                            }
                        )
                    if seg_rows:
                        refs.append(
                            store.write_table(
                                name="RFM Segment Summary",
                                df=pd.DataFrame(seg_rows).sort_values("count", ascending=False).reset_index(drop=True),
                                dataset_id=request.dataset_id,
                                dataset_version=dataset_version,
                                operator_name=op_name,
                                operator_params=params,
                            )
                        )
                sample_scores = result_json.get("sample_scores")
                if isinstance(sample_scores, list) and sample_scores:
                    refs.append(
                        store.write_table(
                            name="RFM Scores (sample)",
                            df=pd.DataFrame(sample_scores),
                            dataset_id=request.dataset_id,
                            dataset_version=dataset_version,
                            operator_name=op_name,
                            operator_params=params,
                        )
                    )

            if analysis_type == "cohort":
                rows = result_json.get("rows")
                if isinstance(rows, list) and rows:
                    refs.append(
                        store.write_table(
                            name="Cohort Analysis (sample)",
                            df=pd.DataFrame(rows),
                            dataset_id=request.dataset_id,
                            dataset_version=dataset_version,
                            operator_name=op_name,
                            operator_params=params,
                        )
                    )

            if analysis_type == "clv":
                top = result_json.get("top_customers")
                if isinstance(top, list) and top:
                    refs.append(
                        store.write_table(
                            name="Top Customers by Predicted CLV",
                            df=pd.DataFrame(top),
                            dataset_id=request.dataset_id,
                            dataset_version=dataset_version,
                            operator_name=op_name,
                            operator_params=params,
                        )
                    )
                for k in ["total_predicted_clv", "avg_predicted_clv", "total_customers"]:
                    if result_json.get(k) is None:
                        continue
                    try:
                        refs.append(
                            store.write_metric(
                                name=f"CLV {k}",
                                value=float(result_json[k]),
                                unit="metric",
                                details={},
                                dataset_id=request.dataset_id,
                                dataset_version=dataset_version,
                                operator_name=op_name,
                                operator_params=params,
                            )
                        )
                    except Exception:
                        pass

            await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
            artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {
            "status": "success",
            "analysis_type": analysis_type,
            "result": to_jsonable(result_json),
            "dataset_version": dataset_version,
            "artifacts": artifacts,
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
        loaded = await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)
        df = loaded.df
        dataset_version = loaded.version_hash
        from app.ml.forecasting import get_forecast_engine
        engine = get_forecast_engine()
        
        result = engine.forecast(
            df,
            date_col=request.date_column,
            value_col=request.value_column,
            periods=request.periods,
            method=ForecastMethod(request.method)
        )
        result_json = to_jsonable(result.to_dict())
        artifacts: list[dict[str, Any]] = []
        try:
            import pandas as pd

            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = "analytics.forecast"
            forecast_rows = result_json.get("forecast") if isinstance(result_json, dict) else None
            if isinstance(forecast_rows, list) and forecast_rows:
                refs.append(
                    store.write_table(
                        name="Forecast",
                        df=pd.DataFrame(forecast_rows),
                        dataset_id=request.dataset_id,
                        dataset_version=dataset_version,
                        operator_name=op_name,
                        operator_params=params,
                    )
                )

            metrics = result_json.get("metrics") if isinstance(result_json, dict) else None
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if v is None:
                        continue
                    try:
                        refs.append(
                            store.write_metric(
                                name=f"Forecast {k}",
                                value=float(v),
                                unit="metric",
                                details={"method": request.method},
                                dataset_id=request.dataset_id,
                                dataset_version=dataset_version,
                                operator_name=op_name,
                                operator_params=params,
                            )
                        )
                    except Exception:
                        pass

            refs.append(
                store.write_report(
                    name="Forecast Result (JSON)",
                    content=json.dumps(result_json, ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=request.dataset_id,
                    dataset_version=dataset_version,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
            artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {
            "status": "success",
            "result": result_json,
            "dataset_version": dataset_version,
            "artifacts": artifacts,
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
        loaded = await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)
        df = loaded.df
        dataset_version = loaded.version_hash
        from app.ml.anomaly_detection import get_anomaly_detection_engine
        engine = get_anomaly_detection_engine()
        
        result = engine.detect(
            df,
            columns=request.columns,
            method=AnomalyMethod(request.method),
            contamination=request.contamination
        )
        result_json = to_jsonable(result.to_dict())
        artifacts: list[dict[str, Any]] = []
        try:
            import pandas as pd

            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = "analytics.anomaly_detection"
            anomalies = result_json.get("anomalies") if isinstance(result_json, dict) else None
            if isinstance(anomalies, list) and anomalies:
                refs.append(
                    store.write_table(
                        name="Anomalies (top)",
                        df=pd.DataFrame(anomalies),
                        dataset_id=request.dataset_id,
                        dataset_version=dataset_version,
                        operator_name=op_name,
                        operator_params=params,
                    )
                )

            for k in ["n_anomalies", "anomaly_ratio", "threshold", "total_samples"]:
                if not isinstance(result_json, dict) or result_json.get(k) is None:
                    continue
                try:
                    refs.append(
                        store.write_metric(
                            name=f"Anomaly {k}",
                            value=float(result_json[k]),
                            unit="metric",
                            details={"method": request.method},
                            dataset_id=request.dataset_id,
                            dataset_version=dataset_version,
                            operator_name=op_name,
                            operator_params=params,
                        )
                    )
                except Exception:
                    pass

            refs.append(
                store.write_report(
                    name="Anomaly Detection Result (JSON)",
                    content=json.dumps(result_json, ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=request.dataset_id,
                    dataset_version=dataset_version,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
            artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {
            "status": "success",
            "result": result_json,
            "dataset_version": dataset_version,
            "artifacts": artifacts,
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
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS))
):
    """Analyze A/B test results."""
    try:
        from app.ml.ab_testing import get_ab_testing_engine
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

        result_json = to_jsonable(result.to_dict())
        artifacts: list[dict[str, Any]] = []
        try:
            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = "analytics.ab_test"
            refs.append(
                store.write_report(
                    name=f"A/B Test ({request.method}) (JSON)",
                    content=json.dumps(result_json, ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=None,
                    dataset_version=None,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            res = result_json.get("results") if isinstance(result_json, dict) else None
            if isinstance(res, dict):
                for k in ["p_value", "relative_lift", "absolute_lift", "achieved_power"]:
                    if res.get(k) is None:
                        continue
                    try:
                        refs.append(
                            store.write_metric(
                                name=f"A/B {k}",
                                value=float(res[k]),
                                unit="metric",
                                details={"method": request.method},
                                dataset_id=None,
                                dataset_version=None,
                                operator_name=op_name,
                                operator_params=params,
                            )
                        )
                    except Exception:
                        pass

            await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
            artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {
            "status": "success",
            "method": request.method,
            "result": result_json,
            "artifacts": artifacts,
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
        loaded = await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)
        df = loaded.df
        dataset_version = loaded.version_hash
        from app.ml.data_profiling import get_data_profiling_engine
        engine = get_data_profiling_engine()
        
        profile = engine.profile(df, name=request.dataset_id)
        profile_json = to_jsonable(profile.to_dict())
        artifacts: list[dict[str, Any]] = []
        try:
            import pandas as pd

            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = "analytics.profile"
            cols = profile_json.get("columns") if isinstance(profile_json, dict) else None
            if isinstance(cols, list) and cols:
                refs.append(
                    store.write_table(
                        name="Profile - Columns",
                        df=pd.DataFrame(cols),
                        dataset_id=request.dataset_id,
                        dataset_version=dataset_version,
                        operator_name=op_name,
                        operator_params=params,
                    )
                )

            refs.append(
                store.write_report(
                    name="Profile Result (JSON)",
                    content=json.dumps(profile_json, ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=request.dataset_id,
                    dataset_version=dataset_version,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
            artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {
            "status": "success",
            "profile": profile_json,
            "dataset_version": dataset_version,
            "artifacts": artifacts,
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
        loaded = await loader.load_dataset(request.dataset_id, owner_id=user.user_id, sample_rows=200_000)
        df = loaded.df
        dataset_version = loaded.version_hash
        from app.ml.bi_metrics import get_bi_metrics_engine
        engine = get_bi_metrics_engine()
        
        metrics = engine.calculate_core_metrics(
            df,
            revenue_col=request.revenue_column,
            user_col=request.user_column,
            date_col=request.date_column
        )
        
        result = {name: m.to_dict() for name, m in metrics.items()}

        artifacts: list[dict[str, Any]] = []
        try:
            store = ArtifactStore()
            refs = []
            try:
                params = to_jsonable(request.model_dump())
            except Exception:
                params = to_jsonable(dict(request))

            op_name = "analytics.bi_metrics"
            for name, metric in result.items():
                if not isinstance(metric, dict):
                    continue
                value = metric.get("value")
                if value is None:
                    continue
                try:
                    refs.append(
                        store.write_metric(
                            name=str(metric.get("name") or name),
                            value=float(value),
                            unit=str(metric.get("unit") or ""),
                            details={k: v for k, v in metric.items() if k not in {"name", "value", "unit"}},
                            dataset_id=request.dataset_id,
                            dataset_version=dataset_version,
                            operator_name=op_name,
                            operator_params=params,
                        )
                    )
                except Exception:
                    pass

            refs.append(
                store.write_report(
                    name="BI Metrics (JSON)",
                    content=json.dumps(to_jsonable(result), ensure_ascii=True, default=str),
                    report_format="json",
                    dataset_id=request.dataset_id,
                    dataset_version=dataset_version,
                    operator_name=op_name,
                    operator_params=params,
                )
            )

            await ArtifactIndexService(db).index_many(owner_id=user.user_id, refs=refs)
            artifacts = [_artifact_ref_to_dict(r) for r in refs]
        except Exception:
            artifacts = []

        return {
            "status": "success",
            "metrics": to_jsonable(result),
            "dataset_version": dataset_version,
            "artifacts": artifacts,
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
    from app.ml.ab_testing import get_ab_testing_engine
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
