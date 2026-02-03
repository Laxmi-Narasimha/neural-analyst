# AI Enterprise Data Analyst - Advanced Anomaly Detection
# Multi-method anomaly detection for univariate and multivariate data

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Anomaly Detection Types
# ============================================================================

class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    DBSCAN = "dbscan"
    MAHALANOBIS = "mahalanobis"


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Single anomaly detection."""
    
    index: int
    value: Any
    score: float  # Anomaly score (higher = more anomalous)
    severity: AnomalySeverity
    method: AnomalyMethod
    column: Optional[str] = None
    reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "value": self.value if not isinstance(self.value, np.ndarray) else self.value.tolist(),
            "score": round(self.score, 4),
            "severity": self.severity.value,
            "method": self.method.value,
            "column": self.column,
            "reason": self.reason
        }


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection."""
    
    method: AnomalyMethod
    total_samples: int
    n_anomalies: int
    anomaly_ratio: float
    
    anomalies: list[Anomaly] = field(default_factory=list)
    threshold: float = 0.0
    
    # Additional info
    statistics: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "total_samples": self.total_samples,
            "n_anomalies": self.n_anomalies,
            "anomaly_ratio": round(self.anomaly_ratio, 4),
            "threshold": round(self.threshold, 4),
            "anomalies": [a.to_dict() for a in self.anomalies[:100]],
            "statistics": self.statistics
        }


# ============================================================================
# Univariate Anomaly Detectors
# ============================================================================

class UnivariateAnomalyDetector:
    """
    Anomaly detection for single variables.
    
    Methods:
    - Z-Score (parametric)
    - Modified Z-Score (robust to outliers)
    - IQR (non-parametric)
    """
    
    @staticmethod
    def zscore(
        series: pd.Series,
        threshold: float = 3.0
    ) -> AnomalyDetectionResult:
        """Detect anomalies using Z-score."""
        values = series.dropna()
        
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return AnomalyDetectionResult(
                method=AnomalyMethod.ZSCORE,
                total_samples=len(values),
                n_anomalies=0,
                anomaly_ratio=0.0,
                threshold=threshold
            )
        
        z_scores = np.abs((values - mean) / std)
        
        anomalies = []
        for idx, (val, z) in enumerate(zip(values.values, z_scores)):
            if z > threshold:
                severity = UnivariateAnomalyDetector._get_severity(z, threshold)
                anomalies.append(Anomaly(
                    index=values.index[idx],
                    value=float(val),
                    score=float(z),
                    severity=severity,
                    method=AnomalyMethod.ZSCORE,
                    column=series.name,
                    reason=f"Z-score {z:.2f} exceeds threshold {threshold}"
                ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.ZSCORE,
            total_samples=len(values),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(values),
            anomalies=anomalies,
            threshold=threshold,
            statistics={"mean": float(mean), "std": float(std)}
        )
    
    @staticmethod
    def modified_zscore(
        series: pd.Series,
        threshold: float = 3.5
    ) -> AnomalyDetectionResult:
        """
        Modified Z-score using median and MAD.
        
        More robust to outliers than standard Z-score.
        """
        values = series.dropna()
        
        median = values.median()
        mad = np.median(np.abs(values - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.mean(np.abs(values - median))
        
        if mad == 0:
            return AnomalyDetectionResult(
                method=AnomalyMethod.MODIFIED_ZSCORE,
                total_samples=len(values),
                n_anomalies=0,
                anomaly_ratio=0.0,
                threshold=threshold
            )
        
        # Modified Z-score formula
        modified_z = 0.6745 * (values - median) / mad
        
        anomalies = []
        for idx, (val, mz) in enumerate(zip(values.values, np.abs(modified_z))):
            if mz > threshold:
                severity = UnivariateAnomalyDetector._get_severity(mz, threshold)
                anomalies.append(Anomaly(
                    index=values.index[idx],
                    value=float(val),
                    score=float(mz),
                    severity=severity,
                    method=AnomalyMethod.MODIFIED_ZSCORE,
                    column=series.name,
                    reason=f"Modified Z-score {mz:.2f} exceeds threshold"
                ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.MODIFIED_ZSCORE,
            total_samples=len(values),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(values),
            anomalies=anomalies,
            threshold=threshold,
            statistics={"median": float(median), "mad": float(mad)}
        )
    
    @staticmethod
    def iqr(
        series: pd.Series,
        multiplier: float = 1.5
    ) -> AnomalyDetectionResult:
        """Detect anomalies using Interquartile Range."""
        values = series.dropna()
        
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        anomalies = []
        for idx, val in enumerate(values.values):
            if val < lower_bound or val > upper_bound:
                distance = max(lower_bound - val, val - upper_bound)
                score = distance / iqr if iqr > 0 else abs(val)
                
                severity = UnivariateAnomalyDetector._get_severity(score + multiplier, multiplier)
                anomalies.append(Anomaly(
                    index=values.index[idx],
                    value=float(val),
                    score=float(score),
                    severity=severity,
                    method=AnomalyMethod.IQR,
                    column=series.name,
                    reason=f"Value outside [{lower_bound:.2f}, {upper_bound:.2f}]"
                ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.IQR,
            total_samples=len(values),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(values),
            anomalies=anomalies,
            threshold=multiplier,
            statistics={
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        )
    
    @staticmethod
    def _get_severity(score: float, threshold: float) -> AnomalySeverity:
        """Determine anomaly severity based on score."""
        ratio = score / threshold
        
        if ratio <= 1.5:
            return AnomalySeverity.LOW
        elif ratio <= 2.5:
            return AnomalySeverity.MEDIUM
        elif ratio <= 4.0:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL


# ============================================================================
# Multivariate Anomaly Detectors
# ============================================================================

class MultivariateAnomalyDetector:
    """
    Anomaly detection for multiple variables jointly.
    
    Methods:
    - Isolation Forest
    - Local Outlier Factor
    - One-Class SVM
    - Mahalanobis Distance
    """
    
    @staticmethod
    def isolation_forest(
        df: pd.DataFrame,
        columns: list[str] = None,
        contamination: float = 0.1
    ) -> AnomalyDetectionResult:
        """Detect anomalies using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ValidationException("sklearn required for Isolation Forest")
        
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[columns].fillna(df[columns].median()).values
        
        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = clf.fit_predict(X)
        scores = -clf.score_samples(X)  # Negative to make higher = more anomalous
        
        # Normalize scores to [0, 1]
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        anomalies = []
        for idx in np.where(predictions == -1)[0]:
            severity = MultivariateAnomalyDetector._score_to_severity(scores_normalized[idx])
            anomalies.append(Anomaly(
                index=df.index[idx],
                value=X[idx].tolist(),
                score=float(scores_normalized[idx]),
                severity=severity,
                method=AnomalyMethod.ISOLATION_FOREST,
                reason=f"Isolated in {clf.n_estimators} trees"
            ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.ISOLATION_FOREST,
            total_samples=len(df),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(df),
            anomalies=anomalies,
            threshold=contamination,
            statistics={
                "contamination": contamination,
                "n_features": len(columns),
                "score_range": [float(scores.min()), float(scores.max())]
            }
        )
    
    @staticmethod
    def local_outlier_factor(
        df: pd.DataFrame,
        columns: list[str] = None,
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> AnomalyDetectionResult:
        """Detect anomalies using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            raise ValidationException("sklearn required for LOF")
        
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[columns].fillna(df[columns].median()).values
        
        n_neighbors = min(n_neighbors, len(X) - 1)
        
        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        predictions = clf.fit_predict(X)
        scores = -clf.negative_outlier_factor_
        
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        anomalies = []
        for idx in np.where(predictions == -1)[0]:
            severity = MultivariateAnomalyDetector._score_to_severity(scores_normalized[idx])
            anomalies.append(Anomaly(
                index=df.index[idx],
                value=X[idx].tolist(),
                score=float(scores_normalized[idx]),
                severity=severity,
                method=AnomalyMethod.LOCAL_OUTLIER_FACTOR,
                reason=f"LOF score indicates local outlier"
            ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.LOCAL_OUTLIER_FACTOR,
            total_samples=len(df),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(df),
            anomalies=anomalies,
            threshold=contamination,
            statistics={
                "n_neighbors": n_neighbors,
                "n_features": len(columns)
            }
        )
    
    @staticmethod
    def mahalanobis_distance(
        df: pd.DataFrame,
        columns: list[str] = None,
        threshold_percentile: float = 97.5
    ) -> AnomalyDetectionResult:
        """Detect anomalies using Mahalanobis distance."""
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[columns].dropna()
        
        # Calculate mean and covariance
        mean = X.mean().values
        cov = X.cov().values
        
        # Regularize covariance for numerical stability
        cov = cov + np.eye(cov.shape[0]) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        
        # Calculate Mahalanobis distances
        distances = []
        for i, row in enumerate(X.values):
            diff = row - mean
            d = np.sqrt(diff @ cov_inv @ diff)
            distances.append(d)
        
        distances = np.array(distances)
        
        # Chi-squared threshold
        threshold = np.percentile(distances, threshold_percentile)
        
        anomalies = []
        for idx, d in enumerate(distances):
            if d > threshold:
                score = d / threshold
                severity = MultivariateAnomalyDetector._score_to_severity(min(score / 2, 1.0))
                anomalies.append(Anomaly(
                    index=X.index[idx],
                    value=X.iloc[idx].values.tolist(),
                    score=float(d),
                    severity=severity,
                    method=AnomalyMethod.MAHALANOBIS,
                    reason=f"Mahalanobis distance {d:.2f} exceeds threshold {threshold:.2f}"
                ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.MAHALANOBIS,
            total_samples=len(X),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(X),
            anomalies=anomalies,
            threshold=float(threshold),
            statistics={
                "mean_distance": float(distances.mean()),
                "std_distance": float(distances.std()),
                "threshold_percentile": threshold_percentile
            }
        )
    
    @staticmethod
    def _score_to_severity(normalized_score: float) -> AnomalySeverity:
        """Convert normalized score to severity."""
        if normalized_score <= 0.3:
            return AnomalySeverity.LOW
        elif normalized_score <= 0.6:
            return AnomalySeverity.MEDIUM
        elif normalized_score <= 0.85:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL


# ============================================================================
# Time Series Anomaly Detection
# ============================================================================

class TimeSeriesAnomalyDetector:
    """
    Anomaly detection specialized for time series.
    
    Detects:
    - Point anomalies
    - Contextual anomalies
    - Collective anomalies (changepoints)
    """
    
    @staticmethod
    def rolling_window(
        series: pd.Series,
        window_size: int = 20,
        n_std: float = 3.0
    ) -> AnomalyDetectionResult:
        """Detect anomalies using rolling statistics."""
        values = series.dropna()
        
        rolling_mean = values.rolling(window=window_size, center=True).mean()
        rolling_std = values.rolling(window=window_size, center=True).std()
        
        # Fill NaN edges
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        
        z_scores = np.abs((values - rolling_mean) / (rolling_std + 1e-10))
        
        anomalies = []
        for idx, (val, z, rm, rs) in enumerate(zip(
            values.values, z_scores.values, rolling_mean.values, rolling_std.values
        )):
            if z > n_std:
                severity = UnivariateAnomalyDetector._get_severity(z, n_std)
                anomalies.append(Anomaly(
                    index=values.index[idx],
                    value=float(val),
                    score=float(z),
                    severity=severity,
                    method=AnomalyMethod.ZSCORE,
                    column=series.name,
                    reason=f"Deviation from rolling mean ({rm:.2f} ± {rs:.2f})"
                ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.ZSCORE,
            total_samples=len(values),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(values),
            anomalies=anomalies,
            threshold=n_std,
            statistics={"window_size": window_size}
        )
    
    @staticmethod
    def seasonal_decomposition(
        series: pd.Series,
        period: int = 7,
        n_std: float = 3.0
    ) -> AnomalyDetectionResult:
        """Detect anomalies in residuals after seasonal decomposition."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            # Fallback to rolling window
            return TimeSeriesAnomalyDetector.rolling_window(series)
        
        values = series.dropna()
        
        if len(values) < 2 * period:
            return TimeSeriesAnomalyDetector.rolling_window(series)
        
        try:
            decomposition = seasonal_decompose(
                values, period=period, extrapolate_trend='freq'
            )
            residuals = decomposition.resid
        except:
            return TimeSeriesAnomalyDetector.rolling_window(series)
        
        # Detect anomalies in residuals
        residuals_clean = residuals.dropna()
        mean = residuals_clean.mean()
        std = residuals_clean.std()
        
        z_scores = np.abs((residuals_clean - mean) / (std + 1e-10))
        
        anomalies = []
        for idx, (val, z) in enumerate(zip(residuals_clean.values, z_scores)):
            if z > n_std:
                severity = UnivariateAnomalyDetector._get_severity(z, n_std)
                anomalies.append(Anomaly(
                    index=residuals_clean.index[idx],
                    value=float(values.loc[residuals_clean.index[idx]]),
                    score=float(z),
                    severity=severity,
                    method=AnomalyMethod.ZSCORE,
                    column=series.name,
                    reason=f"Anomaly in deseasonalized residuals (z={z:.2f})"
                ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.ZSCORE,
            total_samples=len(values),
            n_anomalies=len(anomalies),
            anomaly_ratio=len(anomalies) / len(values),
            anomalies=anomalies,
            threshold=n_std,
            statistics={
                "period": period,
                "residual_std": float(std)
            }
        )


# ============================================================================
# Anomaly Detection Engine
# ============================================================================

class AnomalyDetectionEngine:
    """
    Unified anomaly detection engine.
    
    Features:
    - Multiple detection methods
    - Automatic method selection
    - Ensemble detection
    - Severity classification
    """
    
    def __init__(self):
        self.univariate = UnivariateAnomalyDetector()
        self.multivariate = MultivariateAnomalyDetector()
        self.timeseries = TimeSeriesAnomalyDetector()
    
    def detect(
        self,
        df: pd.DataFrame,
        columns: list[str] = None,
        method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST,
        contamination: float = 0.1,
        is_timeseries: bool = False
    ) -> dict[str, AnomalyDetectionResult]:
        """
        Detect anomalies in dataframe.
        
        Args:
            df: Input data
            columns: Columns to analyze (None for all numeric)
            method: Detection method
            contamination: Expected proportion of anomalies
            is_timeseries: Whether data is time series
        """
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        results = {}
        
        if len(columns) == 1:
            # Univariate detection
            series = df[columns[0]]
            
            if is_timeseries:
                results[columns[0]] = self.timeseries.rolling_window(series)
            elif method == AnomalyMethod.ZSCORE:
                results[columns[0]] = self.univariate.zscore(series)
            elif method == AnomalyMethod.MODIFIED_ZSCORE:
                results[columns[0]] = self.univariate.modified_zscore(series)
            else:
                results[columns[0]] = self.univariate.iqr(series)
        
        else:
            # Multivariate detection
            if method == AnomalyMethod.ISOLATION_FOREST:
                results["multivariate"] = self.multivariate.isolation_forest(
                    df, columns, contamination
                )
            elif method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
                results["multivariate"] = self.multivariate.local_outlier_factor(
                    df, columns, contamination=contamination
                )
            elif method == AnomalyMethod.MAHALANOBIS:
                results["multivariate"] = self.multivariate.mahalanobis_distance(
                    df, columns
                )
            else:
                # Default to Isolation Forest for multivariate
                results["multivariate"] = self.multivariate.isolation_forest(
                    df, columns, contamination
                )
        
        return results
    
    def ensemble_detect(
        self,
        df: pd.DataFrame,
        columns: list[str] = None,
        min_votes: int = 2
    ) -> AnomalyDetectionResult:
        """
        Ensemble anomaly detection using multiple methods.
        
        An observation is flagged as anomaly if detected by >= min_votes methods.
        """
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Run multiple methods
        methods_results = []
        
        try:
            methods_results.append(self.multivariate.isolation_forest(df, columns))
        except:
            pass
        
        try:
            methods_results.append(self.multivariate.local_outlier_factor(df, columns))
        except:
            pass
        
        try:
            methods_results.append(self.multivariate.mahalanobis_distance(df, columns))
        except:
            pass
        
        if not methods_results:
            return AnomalyDetectionResult(
                method=AnomalyMethod.ISOLATION_FOREST,
                total_samples=len(df),
                n_anomalies=0,
                anomaly_ratio=0.0
            )
        
        # Vote counting
        votes = {}
        scores = {}
        
        for result in methods_results:
            for anomaly in result.anomalies:
                idx = anomaly.index
                votes[idx] = votes.get(idx, 0) + 1
                scores[idx] = max(scores.get(idx, 0), anomaly.score)
        
        # Create ensemble result
        ensemble_anomalies = []
        for idx, vote_count in votes.items():
            if vote_count >= min_votes:
                severity = MultivariateAnomalyDetector._score_to_severity(
                    vote_count / len(methods_results)
                )
                ensemble_anomalies.append(Anomaly(
                    index=idx,
                    value=df.iloc[idx][columns].values.tolist() if idx < len(df) else None,
                    score=float(scores[idx]),
                    severity=severity,
                    method=AnomalyMethod.ISOLATION_FOREST,  # Ensemble
                    reason=f"Detected by {vote_count}/{len(methods_results)} methods"
                ))
        
        return AnomalyDetectionResult(
            method=AnomalyMethod.ISOLATION_FOREST,
            total_samples=len(df),
            n_anomalies=len(ensemble_anomalies),
            anomaly_ratio=len(ensemble_anomalies) / len(df),
            anomalies=ensemble_anomalies,
            statistics={
                "methods_used": len(methods_results),
                "min_votes": min_votes
            }
        )


# Factory function
def get_anomaly_detection_engine() -> AnomalyDetectionEngine:
    """Get anomaly detection engine instance."""
    return AnomalyDetectionEngine()
