# AI Enterprise Data Analyst - Advanced Feature Store
# Production-grade feature engineering with 2025 state-of-the-art techniques

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union
from uuid import UUID, uuid4
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import DataProcessingException, ValidationException
except ImportError:
    class DataProcessingException(Exception):
        pass

    class ValidationException(Exception):
        pass

logger = get_logger(__name__)


# ============================================================================
# Feature Types and Metadata
# ============================================================================

class FeatureType(str, Enum):
    """Feature data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    DATETIME = "datetime"
    TEXT = "text"
    EMBEDDING = "embedding"


class FeatureStatus(str, Enum):
    """Feature lifecycle status."""
    DRAFT = "draft"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class FeatureMetadata:
    """Complete feature metadata for governance and lineage."""
    
    name: str
    description: str
    feature_type: FeatureType
    status: FeatureStatus = FeatureStatus.DRAFT
    
    # Lineage
    source_columns: list[str] = field(default_factory=list)
    transformation_logic: str = ""
    version: str = "1.0.0"
    
    # Statistics (computed)
    null_ratio: float = 0.0
    unique_count: int = 0
    cardinality_ratio: float = 0.0
    distribution_type: Optional[str] = None
    
    # Quality metrics
    stability_score: float = 1.0  # Feature drift monitoring
    importance_score: float = 0.0  # Model importance
    
    # Governance
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    pii_flag: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.feature_type.value,
            "status": self.status.value,
            "source_columns": self.source_columns,
            "version": self.version,
            "null_ratio": round(self.null_ratio, 4),
            "unique_count": self.unique_count,
            "cardinality_ratio": round(self.cardinality_ratio, 4),
            "stability_score": round(self.stability_score, 4),
            "importance_score": round(self.importance_score, 4),
            "pii_flag": self.pii_flag
        }


# ============================================================================
# Feature Transformers (Strategy Pattern)
# ============================================================================

class BaseTransformer(ABC):
    """Abstract base for feature transformers."""
    
    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseTransformer":
        """Fit transformer on data."""
        pass
    
    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        """Transform data."""
        pass
    
    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Fit and transform."""
        return self.fit(series).transform(series)
    
    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters for serialization."""
        pass


class NumericScaler(BaseTransformer):
    """Advanced numeric scaler with multiple strategies."""
    
    def __init__(self, method: str = "robust"):
        """
        Initialize scaler.
        
        Args:
            method: 'standard', 'minmax', 'robust', 'yeo-johnson', 'quantile'
        """
        self.method = method
        self._params: dict[str, float] = {}
        self._fitted = False
    
    def fit(self, series: pd.Series) -> "NumericScaler":
        series_clean = series.dropna()
        
        if self.method == "standard":
            self._params = {
                "mean": float(series_clean.mean()),
                "std": float(series_clean.std())
            }
        elif self.method == "minmax":
            self._params = {
                "min": float(series_clean.min()),
                "max": float(series_clean.max())
            }
        elif self.method == "robust":
            self._params = {
                "median": float(series_clean.median()),
                "iqr": float(series_clean.quantile(0.75) - series_clean.quantile(0.25))
            }
        elif self.method == "quantile":
            # Fit quantile transformer
            self._params = {
                f"q_{i}": float(series_clean.quantile(i / 100))
                for i in range(0, 101, 5)
            }
        elif self.method == "yeo-johnson":
            # Find optimal lambda using scipy
            from scipy.stats import yeojohnson_normmax
            try:
                self._params["lambda"] = yeojohnson_normmax(series_clean.values)
            except:
                self._params["lambda"] = 0  # Fallback to log
        
        self._fitted = True
        return self
    
    def transform(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise ValidationException("Scaler not fitted. Call fit() first.")
        
        result = series.copy()
        
        if self.method == "standard":
            std = self._params["std"]
            if std > 0:
                result = (series - self._params["mean"]) / std
            else:
                result = series - self._params["mean"]
        
        elif self.method == "minmax":
            range_val = self._params["max"] - self._params["min"]
            if range_val > 0:
                result = (series - self._params["min"]) / range_val
            else:
                result = pd.Series(0.5, index=series.index)
        
        elif self.method == "robust":
            iqr = self._params["iqr"]
            if iqr > 0:
                result = (series - self._params["median"]) / iqr
            else:
                result = series - self._params["median"]
        
        elif self.method == "yeo-johnson":
            from scipy.stats import yeojohnson
            lmbda = self._params.get("lambda", 0)
            result = pd.Series(
                yeojohnson(series.dropna().values, lmbda=lmbda),
                index=series.dropna().index
            )
        
        return result
    
    def get_params(self) -> dict[str, Any]:
        return {"method": self.method, **self._params}


class CategoricalEncoder(BaseTransformer):
    """Advanced categorical encoder with multiple strategies."""
    
    def __init__(
        self,
        method: str = "target",
        min_frequency: float = 0.01,
        handle_unknown: str = "encode"
    ):
        """
        Initialize encoder.
        
        Args:
            method: 'onehot', 'target', 'ordinal', 'frequency', 'woe', 'hash'
            min_frequency: Minimum frequency for category (others grouped)
            handle_unknown: 'error', 'encode' (use default), 'ignore'
        """
        self.method = method
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self._mapping: dict[str, Any] = {}
        self._default_value: Any = None
        self._fitted = False
    
    def fit(
        self,
        series: pd.Series,
        target: Optional[pd.Series] = None
    ) -> "CategoricalEncoder":
        """Fit encoder, optionally with target for target encoding."""
        value_counts = series.value_counts(normalize=True)
        
        # Filter rare categories
        frequent_cats = value_counts[value_counts >= self.min_frequency].index.tolist()
        
        if self.method == "onehot":
            self._mapping = {cat: i for i, cat in enumerate(frequent_cats)}
            self._default_value = -1
        
        elif self.method == "frequency":
            self._mapping = value_counts.to_dict()
            self._default_value = 0.0
        
        elif self.method == "ordinal":
            self._mapping = {cat: i for i, cat in enumerate(frequent_cats)}
            self._default_value = len(frequent_cats)
        
        elif self.method == "target":
            if target is None:
                raise ValidationException("Target encoding requires target variable")
            
            # Calculate target mean for each category with smoothing
            global_mean = target.mean()
            smoothing = 10  # Regularization parameter
            
            for cat in frequent_cats:
                mask = series == cat
                n = mask.sum()
                cat_mean = target[mask].mean()
                # Smoothed target encoding
                smoothed = (n * cat_mean + smoothing * global_mean) / (n + smoothing)
                self._mapping[cat] = float(smoothed)
            
            self._default_value = float(global_mean)
        
        elif self.method == "woe":
            # Weight of Evidence encoding
            if target is None or not np.isin(target.unique(), [0, 1]).all():
                raise ValidationException("WoE requires binary target (0/1)")
            
            for cat in frequent_cats:
                mask = series == cat
                good = ((target == 1) & mask).sum()
                bad = ((target == 0) & mask).sum()
                
                total_good = (target == 1).sum()
                total_bad = (target == 0).sum()
                
                good_rate = (good + 0.5) / (total_good + 1)
                bad_rate = (bad + 0.5) / (total_bad + 1)
                
                woe = np.log(good_rate / bad_rate)
                self._mapping[cat] = float(woe)
            
            self._default_value = 0.0
        
        elif self.method == "hash":
            # Feature hashing for high-cardinality
            n_features = 32
            for cat in series.unique():
                if pd.notna(cat):
                    hash_val = int(hashlib.md5(str(cat).encode()).hexdigest(), 16)
                    self._mapping[cat] = hash_val % n_features
            self._default_value = 0
        
        self._fitted = True
        return self
    
    def transform(self, series: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        if not self._fitted:
            raise ValidationException("Encoder not fitted")
        
        if self.method == "onehot":
            # Return DataFrame with one-hot columns
            result = pd.DataFrame(index=series.index)
            for cat, idx in self._mapping.items():
                result[f"{series.name}_{cat}"] = (series == cat).astype(int)
            return result
        
        else:
            # Return single Series
            def map_value(x):
                if pd.isna(x):
                    return np.nan
                if x in self._mapping:
                    return self._mapping[x]
                if self.handle_unknown == "error":
                    raise ValidationException(f"Unknown category: {x}")
                return self._default_value
            
            return series.map(map_value)
    
    def get_params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "min_frequency": self.min_frequency,
            "mapping": self._mapping,
            "default_value": self._default_value
        }


class DatetimeFeaturizer(BaseTransformer):
    """Extract comprehensive datetime features."""
    
    def __init__(self, features: list[str] = None):
        """
        Initialize datetime featurizer.
        
        Args:
            features: List of features to extract. Default: all
                Options: year, month, day, dayofweek, hour, minute,
                        quarter, weekofyear, dayofyear, is_weekend,
                        is_month_start, is_month_end, sin_hour, cos_hour,
                        sin_dayofweek, cos_dayofweek
        """
        self.features = features or [
            "year", "month", "day", "dayofweek", "hour",
            "quarter", "is_weekend", "sin_hour", "cos_hour"
        ]
        self._fitted = True  # No fitting needed
    
    def fit(self, series: pd.Series) -> "DatetimeFeaturizer":
        return self
    
    def transform(self, series: pd.Series) -> pd.DataFrame:
        """Extract datetime features."""
        dt = pd.to_datetime(series)
        result = pd.DataFrame(index=series.index)
        
        for feat in self.features:
            if feat == "year":
                result[f"{series.name}_year"] = dt.dt.year
            elif feat == "month":
                result[f"{series.name}_month"] = dt.dt.month
            elif feat == "day":
                result[f"{series.name}_day"] = dt.dt.day
            elif feat == "dayofweek":
                result[f"{series.name}_dayofweek"] = dt.dt.dayofweek
            elif feat == "hour":
                result[f"{series.name}_hour"] = dt.dt.hour
            elif feat == "minute":
                result[f"{series.name}_minute"] = dt.dt.minute
            elif feat == "quarter":
                result[f"{series.name}_quarter"] = dt.dt.quarter
            elif feat == "weekofyear":
                result[f"{series.name}_weekofyear"] = dt.dt.isocalendar().week
            elif feat == "dayofyear":
                result[f"{series.name}_dayofyear"] = dt.dt.dayofyear
            elif feat == "is_weekend":
                result[f"{series.name}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            elif feat == "is_month_start":
                result[f"{series.name}_is_month_start"] = dt.dt.is_month_start.astype(int)
            elif feat == "is_month_end":
                result[f"{series.name}_is_month_end"] = dt.dt.is_month_end.astype(int)
            elif feat == "sin_hour":
                result[f"{series.name}_sin_hour"] = np.sin(2 * np.pi * dt.dt.hour / 24)
            elif feat == "cos_hour":
                result[f"{series.name}_cos_hour"] = np.cos(2 * np.pi * dt.dt.hour / 24)
            elif feat == "sin_dayofweek":
                result[f"{series.name}_sin_dow"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
            elif feat == "cos_dayofweek":
                result[f"{series.name}_cos_dow"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        
        return result
    
    def get_params(self) -> dict[str, Any]:
        return {"features": self.features}


# ============================================================================
# Advanced Feature Generator
# ============================================================================

class AdvancedFeatureGenerator:
    """
    2025 state-of-the-art feature engineering.
    
    Implements:
    - Automated feature synthesis (Deep Feature Synthesis pattern)
    - Interaction features with selection
    - Statistical aggregations
    - Lag/lead features for time series
    - Target-aware features
    """
    
    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self._generated_features: list[FeatureMetadata] = []
    
    def generate_interactions(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        max_interactions: int = 20
    ) -> pd.DataFrame:
        """Generate interaction features with importance-based selection."""
        if len(numeric_cols) < 2:
            return pd.DataFrame(index=df.index)
        
        interactions = pd.DataFrame(index=df.index)
        candidates = []
        
        # Generate all pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                # Multiplication
                mult_name = f"{col1}_x_{col2}"
                mult_vals = df[col1] * df[col2]
                
                # Division (with epsilon to avoid div by zero)
                div_name = f"{col1}_div_{col2}"
                div_vals = df[col1] / (df[col2].abs() + 1e-8)
                
                # Sum
                sum_name = f"{col1}_plus_{col2}"
                sum_vals = df[col1] + df[col2]
                
                # Diff
                diff_name = f"{col1}_minus_{col2}"
                diff_vals = df[col1] - df[col2]
                
                candidates.extend([
                    (mult_name, mult_vals, col1, col2, "multiply"),
                    (div_name, div_vals, col1, col2, "divide"),
                    (sum_name, sum_vals, col1, col2, "add"),
                    (diff_name, diff_vals, col1, col2, "subtract"),
                ])
        
        # Score and select top interactions by variance
        scored = []
        for name, vals, c1, c2, op in candidates:
            variance = vals.var()
            if np.isfinite(variance) and variance > 0:
                scored.append((name, vals, variance, [c1, c2], op))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        
        for name, vals, _, source_cols, op in scored[:max_interactions]:
            interactions[name] = vals
            self._generated_features.append(FeatureMetadata(
                name=name,
                description=f"Interaction feature: {source_cols[0]} {op} {source_cols[1]}",
                feature_type=FeatureType.NUMERIC,
                source_columns=source_cols,
                transformation_logic=op
            ))
        
        return interactions
    
    def generate_aggregations(
        self,
        df: pd.DataFrame,
        group_col: str,
        agg_cols: list[str],
        agg_funcs: list[str] = None
    ) -> pd.DataFrame:
        """Generate group-wise aggregation features."""
        agg_funcs = agg_funcs or ["mean", "std", "min", "max", "count"]
        result = pd.DataFrame(index=df.index)
        
        for col in agg_cols:
            for func in agg_funcs:
                feat_name = f"{col}_by_{group_col}_{func}"
                
                if func == "count":
                    agg_values = df.groupby(group_col)[col].transform("count")
                elif func == "mean":
                    agg_values = df.groupby(group_col)[col].transform("mean")
                elif func == "std":
                    agg_values = df.groupby(group_col)[col].transform("std").fillna(0)
                elif func == "min":
                    agg_values = df.groupby(group_col)[col].transform("min")
                elif func == "max":
                    agg_values = df.groupby(group_col)[col].transform("max")
                elif func == "median":
                    agg_values = df.groupby(group_col)[col].transform("median")
                else:
                    continue
                
                result[feat_name] = agg_values
                
                self._generated_features.append(FeatureMetadata(
                    name=feat_name,
                    description=f"{func} of {col} grouped by {group_col}",
                    feature_type=FeatureType.NUMERIC,
                    source_columns=[col, group_col],
                    transformation_logic=f"groupby.{func}"
                ))
        
        return result
    
    def generate_lag_features(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: list[str],
        lags: list[int] = None,
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate lag/lead features for time series."""
        lags = lags or [1, 2, 3, 7, 14, 30]
        result = pd.DataFrame(index=df.index)
        
        # Sort by time
        df_sorted = df.sort_values(time_col)
        
        for col in value_cols:
            for lag in lags:
                feat_name = f"{col}_lag_{lag}"
                
                if group_col:
                    lag_values = df_sorted.groupby(group_col)[col].shift(lag)
                else:
                    lag_values = df_sorted[col].shift(lag)
                
                result[feat_name] = lag_values.reindex(df.index)
                
                self._generated_features.append(FeatureMetadata(
                    name=feat_name,
                    description=f"Lag {lag} of {col}",
                    feature_type=FeatureType.NUMERIC,
                    source_columns=[col],
                    transformation_logic=f"lag({lag})"
                ))
                
                # Also compute diff from lag
                diff_name = f"{col}_diff_{lag}"
                diff_values = df_sorted[col] - lag_values
                result[diff_name] = diff_values.reindex(df.index)
        
        return result
    
    def generate_rolling_features(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: list[str],
        windows: list[int] = None,
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate rolling window statistics."""
        windows = windows or [7, 14, 30]
        result = pd.DataFrame(index=df.index)
        
        df_sorted = df.sort_values(time_col)
        
        for col in value_cols:
            for window in windows:
                # Rolling mean
                mean_name = f"{col}_rolling_mean_{window}"
                if group_col:
                    rolling = df_sorted.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                else:
                    rolling = df_sorted[col].rolling(window, min_periods=1).mean()
                result[mean_name] = rolling.reindex(df.index)
                
                # Rolling std
                std_name = f"{col}_rolling_std_{window}"
                if group_col:
                    rolling_std = df_sorted.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                else:
                    rolling_std = df_sorted[col].rolling(window, min_periods=1).std()
                result[std_name] = rolling_std.fillna(0).reindex(df.index)
                
                # Rolling min/max
                min_name = f"{col}_rolling_min_{window}"
                max_name = f"{col}_rolling_max_{window}"
                
                if group_col:
                    rolling_min = df_sorted.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                    rolling_max = df_sorted.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                else:
                    rolling_min = df_sorted[col].rolling(window, min_periods=1).min()
                    rolling_max = df_sorted[col].rolling(window, min_periods=1).max()
                
                result[min_name] = rolling_min.reindex(df.index)
                result[max_name] = rolling_max.reindex(df.index)
        
        return result
    
    def get_feature_metadata(self) -> list[dict[str, Any]]:
        """Get metadata for all generated features."""
        return [f.to_dict() for f in self._generated_features]


# ============================================================================
# Feature Store
# ============================================================================

class FeatureStore:
    """
    Production-grade feature store for ML pipelines.
    
    Features:
    - Feature registration and versioning
    - Transformation caching
    - Feature drift monitoring
    - Lineage tracking
    - Online/offline feature serving
    """
    
    def __init__(self):
        self._features: dict[str, FeatureMetadata] = {}
        self._transformers: dict[str, BaseTransformer] = {}
        self._cache: dict[str, pd.Series] = {}
        self._history: dict[str, list[dict]] = {}  # Feature statistics history
    
    def register_feature(
        self,
        name: str,
        transformer: BaseTransformer,
        metadata: FeatureMetadata
    ) -> None:
        """Register a feature with its transformer."""
        if name in self._features:
            logger.warning(f"Overwriting feature: {name}")
        
        self._features[name] = metadata
        self._transformers[name] = transformer
    
    def compute_feature(
        self,
        name: str,
        data: pd.Series,
        use_cache: bool = True
    ) -> pd.Series:
        """Compute a registered feature."""
        if name not in self._transformers:
            raise ValidationException(f"Feature not registered: {name}")
        
        # Check cache
        cache_key = f"{name}_{hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:8]}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Transform
        transformer = self._transformers[name]
        result = transformer.transform(data)
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def monitor_drift(
        self,
        feature_name: str,
        current_data: pd.Series,
        reference_data: pd.Series
    ) -> dict[str, float]:
        """
        Monitor feature drift between current and reference data.
        
        Returns PSI (Population Stability Index) and other metrics.
        """
        current = current_data.dropna()
        reference = reference_data.dropna()
        
        metrics = {}
        
        # KS Test
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(current, reference)
        metrics["ks_statistic"] = float(ks_stat)
        metrics["ks_pvalue"] = float(ks_pvalue)
        
        # PSI (Population Stability Index)
        # Bin the data
        bins = np.histogram_bin_edges(reference, bins=10)
        current_hist, _ = np.histogram(current, bins=bins, density=True)
        reference_hist, _ = np.histogram(reference, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        current_hist = current_hist + eps
        reference_hist = reference_hist + eps
        
        # Normalize
        current_hist = current_hist / current_hist.sum()
        reference_hist = reference_hist / reference_hist.sum()
        
        psi = np.sum((current_hist - reference_hist) * np.log(current_hist / reference_hist))
        metrics["psi"] = float(psi)
        
        # Interpretation
        if psi < 0.1:
            metrics["drift_status"] = "stable"
        elif psi < 0.25:
            metrics["drift_status"] = "moderate_drift"
        else:
            metrics["drift_status"] = "significant_drift"
        
        # Mean/std shift
        metrics["mean_shift"] = float(current.mean() - reference.mean())
        metrics["std_ratio"] = float(current.std() / (reference.std() + eps))
        
        return metrics
    
    def get_feature_statistics(
        self,
        feature_name: str,
        data: pd.Series
    ) -> dict[str, Any]:
        """Compute comprehensive statistics for a feature."""
        series = data.dropna()
        
        stats = {
            "count": len(data),
            "null_count": data.isnull().sum(),
            "null_ratio": float(data.isnull().sum() / len(data)),
            "unique_count": int(series.nunique()),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            })
            
            # Distribution test
            if len(series) >= 20:
                _, pvalue = scipy_stats.normaltest(series)
                stats["normality_pvalue"] = float(pvalue)
                stats["is_normal"] = pvalue > 0.05
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._cache.clear()
    
    def export_registry(self) -> dict[str, Any]:
        """Export feature registry for persistence."""
        return {
            "features": {
                name: meta.to_dict()
                for name, meta in self._features.items()
            },
            "transformers": {
                name: trans.get_params()
                for name, trans in self._transformers.items()
            }
        }


# Factory function
def get_feature_store() -> FeatureStore:
    """Get feature store instance."""
    return FeatureStore()


def get_feature_generator() -> AdvancedFeatureGenerator:
    """Get feature generator instance."""
    return AdvancedFeatureGenerator()
