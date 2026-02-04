# AI Enterprise Data Analyst - Advanced Anomaly Detection Engine
# Production-grade anomaly detection for ANY data
# Handles: univariate, multivariate, time series, categorical anomalies

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import mahalanobis

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums and Types
# ============================================================================

class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""
    # Statistical methods
    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    GRUBBS = "grubbs"
    ESD = "esd"  # Extreme Studentized Deviate
    
    # Machine learning methods
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    
    # Clustering based
    DBSCAN = "dbscan"
    KMEANS = "kmeans"
    
    # Distance based
    MAHALANOBIS = "mahalanobis"
    
    # Time series specific
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    STL_RESIDUAL = "stl_residual"
    
    # Ensemble
    ENSEMBLE = "ensemble"
    AUTO = "auto"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Anomaly:
    """Single anomaly instance."""
    index: int
    value: Any
    column: str
    method: AnomalyMethod
    score: float  # Anomaly score (higher = more anomalous)
    severity: AnomalySeverity
    expected_range: Tuple[float, float] = (0, 0)
    deviation: float = 0.0  # How far from expected
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionResult:
    """Complete result of anomaly detection."""
    method: AnomalyMethod
    n_anomalies: int
    n_total: int
    anomaly_rate: float
    
    # Anomaly details
    anomalies: List[Anomaly] = field(default_factory=list)
    anomaly_indices: List[int] = field(default_factory=list)
    anomaly_scores: Dict[int, float] = field(default_factory=dict)
    
    # By column (for multivariate)
    by_column: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Thresholds used
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Data characteristics
    data_stats: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    detection_time_sec: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "summary": {
                "n_anomalies": self.n_anomalies,
                "n_total": self.n_total,
                "anomaly_rate": round(self.anomaly_rate * 100, 2)
            },
            "anomaly_indices": self.anomaly_indices[:100],  # Limit output
            "by_column": self.by_column,
            "thresholds": self.thresholds,
            "top_anomalies": [
                {
                    "index": a.index,
                    "column": a.column,
                    "value": a.value if not isinstance(a.value, (np.floating, np.integer)) else float(a.value),
                    "score": round(a.score, 4),
                    "severity": a.severity.value,
                    "deviation": round(a.deviation, 4)
                }
                for a in sorted(self.anomalies, key=lambda x: -x.score)[:20]
            ],
            "recommendations": self.recommendations,
            "warnings": self.warnings
        }


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    method: AnomalyMethod = AnomalyMethod.AUTO
    
    # Contamination (expected proportion of anomalies)
    contamination: float = 0.05
    
    # IQR/Z-score thresholds
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    
    # ML model parameters
    n_estimators: int = 100
    n_neighbors: int = 20
    
    # Time series parameters
    window_size: int = 7
    
    # Ensemble settings
    ensemble_methods: List[AnomalyMethod] = field(default_factory=lambda: [
        AnomalyMethod.IQR,
        AnomalyMethod.ISOLATION_FOREST,
        AnomalyMethod.LOCAL_OUTLIER_FACTOR
    ])
    ensemble_threshold: float = 0.5  # Proportion of methods that must agree
    
    # Output options
    return_scores: bool = True
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.6,
        "medium": 0.75,
        "high": 0.9,
        "critical": 0.95
    })


# ============================================================================
# Statistical Anomaly Detectors
# ============================================================================

class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection."""
    
    @staticmethod
    def iqr_detect(
        data: np.ndarray,
        multiplier: float = 1.5
    ) -> Tuple[np.ndarray, float, float]:
        """IQR-based outlier detection."""
        q1, q3 = np.nanpercentile(data, [25, 75])
        iqr = q3 - q1
        
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        
        is_anomaly = (data < lower) | (data > upper)
        return is_anomaly, lower, upper
    
    @staticmethod
    def zscore_detect(
        data: np.ndarray,
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Z-score based outlier detection."""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        if std == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        z_scores = np.abs((data - mean) / std)
        is_anomaly = z_scores > threshold
        
        return is_anomaly, z_scores
    
    @staticmethod
    def modified_zscore_detect(
        data: np.ndarray,
        threshold: float = 3.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Modified Z-score using MAD (robust to outliers)."""
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        # 0.6745 is the scaling factor for normal distribution
        modified_z = 0.6745 * (data - median) / mad
        is_anomaly = np.abs(modified_z) > threshold
        
        return is_anomaly, np.abs(modified_z)
    
    @staticmethod
    def grubbs_test(
        data: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, List[int]]:
        """Grubbs' test for outliers (iterative)."""
        data = data.copy()
        n = len(data)
        outlier_indices = []
        
        while len(data) > 2:
            mean = np.nanmean(data)
            std = np.nanstd(data)
            
            if std == 0:
                break
            
            # Find the value furthest from mean
            abs_dev = np.abs(data - mean)
            max_idx = np.nanargmax(abs_dev)
            max_dev = abs_dev[max_idx]
            
            # Calculate G statistic
            g = max_dev / std
            
            # Critical value
            t_crit = scipy_stats.t.ppf(1 - alpha / (2 * len(data)), len(data) - 2)
            g_crit = ((len(data) - 1) * t_crit) / np.sqrt(len(data) * (len(data) - 2 + t_crit**2))
            
            if g > g_crit:
                # Find original index
                original_idx = np.where(~np.isnan(data))[0][max_idx]
                outlier_indices.append(original_idx)
                data[max_idx] = np.nan
            else:
                break
        
        is_anomaly = np.zeros(n, dtype=bool)
        is_anomaly[outlier_indices] = True
        
        return is_anomaly, outlier_indices


# ============================================================================
# ML-based Anomaly Detectors
# ============================================================================

class MLAnomalyDetector:
    """Machine learning based anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self._scaler = RobustScaler()
    
    def isolation_forest_detect(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Isolation Forest anomaly detection."""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for Isolation Forest")
        
        # Handle 1D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Handle missing values
        mask = ~np.isnan(data).any(axis=1)
        clean_data = data[mask]
        
        if len(clean_data) < 10:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        # Fit model
        model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Predict
        predictions = model.fit_predict(clean_data)
        scores = -model.score_samples(clean_data)  # Higher = more anomalous
        
        # Map back to original indices
        is_anomaly = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))
        
        is_anomaly[mask] = (predictions == -1)
        anomaly_scores[mask] = scores
        
        # Normalize scores to 0-1
        if scores.max() > scores.min():
            anomaly_scores[mask] = (scores - scores.min()) / (scores.max() - scores.min())
        
        return is_anomaly, anomaly_scores
    
    def lof_detect(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Local Outlier Factor detection."""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for LOF")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        mask = ~np.isnan(data).any(axis=1)
        clean_data = data[mask]
        
        if len(clean_data) < self.config.n_neighbors + 1:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        # Adjust n_neighbors if needed
        n_neighbors = min(self.config.n_neighbors, len(clean_data) - 1)
        
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.config.contamination,
            n_jobs=-1
        )
        
        predictions = model.fit_predict(clean_data)
        scores = -model.negative_outlier_factor_
        
        is_anomaly = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))
        
        is_anomaly[mask] = (predictions == -1)
        anomaly_scores[mask] = scores
        
        # Normalize
        if scores.max() > scores.min():
            anomaly_scores[mask] = (scores - scores.min()) / (scores.max() - scores.min())
        
        return is_anomaly, anomaly_scores
    
    def one_class_svm_detect(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One-Class SVM detection."""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for One-Class SVM")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        mask = ~np.isnan(data).any(axis=1)
        clean_data = data[mask]
        
        if len(clean_data) < 10:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        # Scale data
        scaled = self._scaler.fit_transform(clean_data)
        
        model = OneClassSVM(nu=self.config.contamination, kernel='rbf', gamma='auto')
        predictions = model.fit_predict(scaled)
        scores = -model.decision_function(scaled)
        
        is_anomaly = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))
        
        is_anomaly[mask] = (predictions == -1)
        anomaly_scores[mask] = scores
        
        # Normalize
        if scores.max() > scores.min():
            anomaly_scores[mask] = (scores - scores.min()) / (scores.max() - scores.min())
        
        return is_anomaly, anomaly_scores
    
    def mahalanobis_detect(
        self,
        data: np.ndarray,
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mahalanobis distance based detection (multivariate)."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        mask = ~np.isnan(data).any(axis=1)
        clean_data = data[mask]
        
        if len(clean_data) < data.shape[1] + 1:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        # Calculate mean and covariance
        mean = np.mean(clean_data, axis=0)
        
        try:
            cov = np.cov(clean_data.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            cov_inv = np.linalg.pinv(np.cov(clean_data.T))
        
        # Calculate Mahalanobis distances
        distances = np.zeros(len(clean_data))
        for i, point in enumerate(clean_data):
            try:
                distances[i] = mahalanobis(point, mean, cov_inv)
            except:
                distances[i] = 0
        
        # Chi-squared critical value
        chi2_crit = scipy_stats.chi2.ppf(1 - self.config.contamination, data.shape[1])
        
        is_anomaly = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))
        
        is_anomaly[mask] = (distances > np.sqrt(chi2_crit * threshold))
        anomaly_scores[mask] = distances
        
        # Normalize
        if distances.max() > distances.min():
            anomaly_scores[mask] = (distances - distances.min()) / (distances.max() - distances.min())
        
        return is_anomaly, anomaly_scores


# ============================================================================
# Time Series Anomaly Detector
# ============================================================================

class TimeSeriesAnomalyDetector:
    """Time series specific anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
    
    def moving_average_detect(
        self,
        data: np.ndarray,
        window: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies as deviations from moving average."""
        window = window or self.config.window_size
        
        series = pd.Series(data)
        
        # Calculate moving average and std
        ma = series.rolling(window=window, center=True, min_periods=1).mean()
        ms = series.rolling(window=window, center=True, min_periods=1).std()
        
        # Calculate deviation
        deviation = np.abs(series - ma) / (ms + 1e-10)
        
        # Threshold
        is_anomaly = deviation > self.config.zscore_threshold
        
        # Normalize scores
        scores = deviation.values
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return is_anomaly.values, scores
    
    def exponential_smoothing_detect(
        self,
        data: np.ndarray,
        alpha: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using exponential smoothing residuals."""
        n = len(data)
        smoothed = np.zeros(n)
        smoothed[0] = data[0] if not np.isnan(data[0]) else np.nanmean(data)
        
        for i in range(1, n):
            if np.isnan(data[i]):
                smoothed[i] = smoothed[i-1]
            else:
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        
        # Calculate residuals
        residuals = np.abs(data - smoothed)
        
        # Use modified z-score on residuals
        median_res = np.nanmedian(residuals)
        mad = np.nanmedian(np.abs(residuals - median_res))
        
        if mad == 0:
            return np.zeros(n, dtype=bool), np.zeros(n)
        
        scores = 0.6745 * np.abs(residuals - median_res) / mad
        is_anomaly = scores > self.config.zscore_threshold
        
        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return is_anomaly, scores


# ============================================================================
# Advanced Anomaly Detection Engine
# ============================================================================

class AdvancedAnomalyEngine:
    """
    Advanced anomaly detection engine for ANY data.
    
    Features:
    - Auto-selects appropriate method based on data
    - Handles univariate and multivariate data
    - Time series aware detection
    - Ensemble methods for robustness
    - Provides severity scores and explanations
    """
    
    def __init__(self, config: AnomalyConfig = None, verbose: bool = True):
        self.config = config or AnomalyConfig()
        self.verbose = verbose
        self.stat_detector = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector(self.config)
        self.ts_detector = TimeSeriesAnomalyDetector(self.config)
    
    def detect(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: AnomalyMethod = None,
        is_time_series: bool = False
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in any data.
        
        Handles:
        - Single column (univariate)
        - Multiple columns (multivariate)
        - Time series data
        - Mixed data types
        """
        start_time = datetime.now()
        method = method or self.config.method
        
        # Select columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            raise DataProcessingException("No numeric columns found for anomaly detection")
        
        # Get data
        data = df[columns].values
        
        # Auto-select method if needed
        if method == AnomalyMethod.AUTO:
            method = self._select_method(data, is_time_series)
        
        if self.verbose:
            logger.info(f"Using method: {method.value} on {len(columns)} columns")
        
        # Run detection
        if method == AnomalyMethod.ENSEMBLE:
            is_anomaly, scores = self._ensemble_detect(data, is_time_series)
        elif method in [AnomalyMethod.IQR, AnomalyMethod.ZSCORE, 
                       AnomalyMethod.MODIFIED_ZSCORE, AnomalyMethod.GRUBBS]:
            is_anomaly, scores = self._statistical_detect(data, method)
        elif method in [AnomalyMethod.ISOLATION_FOREST, AnomalyMethod.LOCAL_OUTLIER_FACTOR,
                       AnomalyMethod.ONE_CLASS_SVM, AnomalyMethod.MAHALANOBIS]:
            is_anomaly, scores = self._ml_detect(data, method)
        elif method in [AnomalyMethod.MOVING_AVERAGE, AnomalyMethod.EXPONENTIAL_SMOOTHING]:
            is_anomaly, scores = self._time_series_detect(data, method)
        else:
            is_anomaly, scores = self._statistical_detect(data, AnomalyMethod.IQR)
        
        # Build result
        result = self._build_result(
            df, columns, is_anomaly, scores, method, start_time
        )
        
        return result
    
    def detect_univariate(
        self,
        series: pd.Series,
        method: AnomalyMethod = None
    ) -> AnomalyDetectionResult:
        """Detect anomalies in a single series."""
        df = pd.DataFrame({series.name or 'value': series})
        return self.detect(df, method=method)
    
    def _select_method(
        self,
        data: np.ndarray,
        is_time_series: bool
    ) -> AnomalyMethod:
        """Auto-select best method based on data characteristics."""
        n_samples, n_features = data.shape if data.ndim > 1 else (len(data), 1)
        
        # Check for missing values
        has_missing = np.isnan(data).any()
        
        # Time series
        if is_time_series:
            return AnomalyMethod.MOVING_AVERAGE
        
        # Small samples - use statistical
        if n_samples < 50:
            return AnomalyMethod.MODIFIED_ZSCORE
        
        # Large multivariate - use ML
        if n_features > 1 and n_samples > 100:
            return AnomalyMethod.ISOLATION_FOREST
        
        # Medium samples - ensemble
        if n_samples >= 100:
            return AnomalyMethod.ENSEMBLE
        
        # Default
        return AnomalyMethod.IQR
    
    def _statistical_detect(
        self,
        data: np.ndarray,
        method: AnomalyMethod
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run statistical detection."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Detect per column and combine
        n_samples = len(data)
        combined_anomaly = np.zeros(n_samples, dtype=bool)
        combined_scores = np.zeros(n_samples)
        
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            
            if method == AnomalyMethod.IQR:
                is_anom, lower, upper = self.stat_detector.iqr_detect(
                    col_data, self.config.iqr_multiplier
                )
                # Calculate scores as distance from bounds
                scores = np.zeros_like(col_data)
                mask_low = col_data < lower
                mask_high = col_data > upper
                scores[mask_low] = (lower - col_data[mask_low]) / (upper - lower + 1e-10)
                scores[mask_high] = (col_data[mask_high] - upper) / (upper - lower + 1e-10)
                
            elif method == AnomalyMethod.ZSCORE:
                is_anom, scores = self.stat_detector.zscore_detect(
                    col_data, self.config.zscore_threshold
                )
                
            elif method == AnomalyMethod.MODIFIED_ZSCORE:
                is_anom, scores = self.stat_detector.modified_zscore_detect(col_data)
                
            elif method == AnomalyMethod.GRUBBS:
                is_anom, _ = self.stat_detector.grubbs_test(col_data)
                scores = np.zeros_like(col_data)
                scores[is_anom] = 1.0
            
            else:
                is_anom, scores = self.stat_detector.modified_zscore_detect(col_data)
            
            combined_anomaly |= is_anom
            combined_scores = np.maximum(combined_scores, scores)
        
        # Normalize scores
        if combined_scores.max() > 0:
            combined_scores = combined_scores / combined_scores.max()
        
        return combined_anomaly, combined_scores
    
    def _ml_detect(
        self,
        data: np.ndarray,
        method: AnomalyMethod
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ML-based detection."""
        if method == AnomalyMethod.ISOLATION_FOREST:
            return self.ml_detector.isolation_forest_detect(data)
        elif method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
            return self.ml_detector.lof_detect(data)
        elif method == AnomalyMethod.ONE_CLASS_SVM:
            return self.ml_detector.one_class_svm_detect(data)
        elif method == AnomalyMethod.MAHALANOBIS:
            return self.ml_detector.mahalanobis_detect(data)
        else:
            return self.ml_detector.isolation_forest_detect(data)
    
    def _time_series_detect(
        self,
        data: np.ndarray,
        method: AnomalyMethod
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run time series detection."""
        if data.ndim > 1:
            data = data.flatten()
        
        if method == AnomalyMethod.MOVING_AVERAGE:
            return self.ts_detector.moving_average_detect(data)
        elif method == AnomalyMethod.EXPONENTIAL_SMOOTHING:
            return self.ts_detector.exponential_smoothing_detect(data)
        else:
            return self.ts_detector.moving_average_detect(data)
    
    def _ensemble_detect(
        self,
        data: np.ndarray,
        is_time_series: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ensemble detection."""
        n_samples = len(data) if data.ndim == 1 else data.shape[0]
        votes = np.zeros(n_samples)
        scores_sum = np.zeros(n_samples)
        n_methods = 0
        
        for method in self.config.ensemble_methods:
            try:
                if method in [AnomalyMethod.IQR, AnomalyMethod.ZSCORE, 
                             AnomalyMethod.MODIFIED_ZSCORE]:
                    is_anom, scores = self._statistical_detect(data, method)
                elif method in [AnomalyMethod.ISOLATION_FOREST, 
                               AnomalyMethod.LOCAL_OUTLIER_FACTOR]:
                    is_anom, scores = self._ml_detect(data, method)
                elif method in [AnomalyMethod.MOVING_AVERAGE] and is_time_series:
                    is_anom, scores = self._time_series_detect(data, method)
                else:
                    continue
                
                votes += is_anom.astype(int)
                scores_sum += scores
                n_methods += 1
                
            except Exception as e:
                logger.warning(f"Ensemble method {method} failed: {e}")
                continue
        
        if n_methods == 0:
            # Fallback to IQR
            return self._statistical_detect(data, AnomalyMethod.IQR)
        
        # Consensus voting
        is_anomaly = votes >= (n_methods * self.config.ensemble_threshold)
        avg_scores = scores_sum / n_methods
        
        return is_anomaly, avg_scores
    
    def _build_result(
        self,
        df: pd.DataFrame,
        columns: List[str],
        is_anomaly: np.ndarray,
        scores: np.ndarray,
        method: AnomalyMethod,
        start_time: datetime
    ) -> AnomalyDetectionResult:
        """Build detailed result object."""
        anomaly_indices = np.where(is_anomaly)[0].tolist()
        n_anomalies = len(anomaly_indices)
        n_total = len(df)
        
        # Create anomaly objects
        anomalies = []
        for idx in anomaly_indices[:100]:  # Limit for performance
            for col in columns:
                value = df[col].iloc[idx]
                if pd.isna(value):
                    continue
                
                # Determine severity
                score = scores[idx]
                severity = self._get_severity(score)
                
                # Calculate deviation
                col_data = df[col].dropna()
                median = col_data.median()
                mad = (col_data - median).abs().median()
                deviation = abs(value - median) / (mad + 1e-10)
                
                anomalies.append(Anomaly(
                    index=idx,
                    value=value,
                    column=col,
                    method=method,
                    score=score,
                    severity=severity,
                    deviation=deviation,
                    expected_range=(
                        float(col_data.quantile(0.25)),
                        float(col_data.quantile(0.75))
                    )
                ))
        
        # By column analysis
        by_column = {}
        for col in columns:
            col_data = df[col].dropna()
            by_column[col] = {
                "n_anomalies": sum(1 for a in anomalies if a.column == col),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max())
            }
        
        # Recommendations
        recommendations = []
        if n_anomalies / n_total > 0.1:
            recommendations.append("High anomaly rate detected. Consider data quality review.")
        if n_anomalies > 0:
            recommendations.append(f"Review {min(n_anomalies, 20)} flagged records for data validation.")
        
        return AnomalyDetectionResult(
            method=method,
            n_anomalies=n_anomalies,
            n_total=n_total,
            anomaly_rate=n_anomalies / n_total if n_total > 0 else 0,
            anomalies=anomalies,
            anomaly_indices=anomaly_indices,
            anomaly_scores={i: float(scores[i]) for i in anomaly_indices[:100]},
            by_column=by_column,
            thresholds={
                "iqr_multiplier": self.config.iqr_multiplier,
                "zscore_threshold": self.config.zscore_threshold,
                "contamination": self.config.contamination
            },
            detection_time_sec=(datetime.now() - start_time).total_seconds(),
            recommendations=recommendations
        )
    
    def _get_severity(self, score: float) -> AnomalySeverity:
        """Map score to severity level."""
        thresholds = self.config.severity_thresholds
        
        if score >= thresholds.get("critical", 0.95):
            return AnomalySeverity.CRITICAL
        elif score >= thresholds.get("high", 0.9):
            return AnomalySeverity.HIGH
        elif score >= thresholds.get("medium", 0.75):
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


# ============================================================================
# Factory Functions
# ============================================================================

def get_anomaly_engine(config: AnomalyConfig = None) -> AdvancedAnomalyEngine:
    """Get an anomaly detection engine."""
    return AdvancedAnomalyEngine(config=config)


def quick_detect_anomalies(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = "auto"
) -> Dict[str, Any]:
    """
    Quick anomaly detection on any data.
    
    Example:
        result = quick_detect_anomalies(df, columns=['price', 'quantity'])
        print(result['summary'])
    """
    engine = AdvancedAnomalyEngine(verbose=False)
    method_enum = AnomalyMethod(method) if method != "auto" else AnomalyMethod.AUTO
    result = engine.detect(df, columns=columns, method=method_enum)
    return result.to_dict()


# Backwards-compatible names expected by repo tests
DetectionMethod = AnomalyMethod
AdvancedAnomalyDetectionEngine = AdvancedAnomalyEngine
