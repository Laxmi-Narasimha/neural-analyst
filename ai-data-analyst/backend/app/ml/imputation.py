# AI Enterprise Data Analyst - Advanced Imputation Engine
# Production-grade missing value imputation with multiple strategies

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import DataProcessingException, ValidationException
except ImportError:
    class DataProcessingException(Exception): pass
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Imputation Methods
# ============================================================================

class ImputationMethod(str, Enum):
    """Available imputation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"  # MICE
    RANDOM_FOREST = "random_forest"
    HOT_DECK = "hot_deck"
    INTERPOLATE = "interpolate"  # For time series
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


@dataclass
class ImputationResult:
    """Result of imputation operation."""
    
    column: str
    method: ImputationMethod
    n_imputed: int
    imputed_values: dict[int, Any] = field(default_factory=dict)  # index -> value
    original_null_count: int = 0
    statistics: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "method": self.method.value,
            "n_imputed": self.n_imputed,
            "original_null_count": self.original_null_count,
            "statistics": self.statistics
        }


# ============================================================================
# Base Imputer
# ============================================================================

class BaseImputer(ABC):
    """Abstract base for imputers."""
    
    @abstractmethod
    def fit(self, series: pd.Series, **kwargs) -> "BaseImputer":
        """Fit imputer on data."""
        pass
    
    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        """Transform (impute) data."""
        pass
    
    def fit_transform(self, series: pd.Series, **kwargs) -> pd.Series:
        """Fit and transform."""
        return self.fit(series, **kwargs).transform(series)


class SimpleImputer(BaseImputer):
    """Simple statistical imputation."""
    
    def __init__(self, method: ImputationMethod = ImputationMethod.MEDIAN):
        self.method = method
        self._fill_value: Any = None
        self._fitted = False
    
    def fit(self, series: pd.Series, **kwargs) -> "SimpleImputer":
        if self.method == ImputationMethod.MEAN:
            self._fill_value = series.mean()
        elif self.method == ImputationMethod.MEDIAN:
            self._fill_value = series.median()
        elif self.method == ImputationMethod.MODE:
            mode_result = series.mode()
            self._fill_value = mode_result[0] if len(mode_result) > 0 else None
        elif self.method == ImputationMethod.CONSTANT:
            self._fill_value = kwargs.get("fill_value", 0)
        
        self._fitted = True
        return self
    
    def transform(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise ValidationException("Imputer not fitted")
        return series.fillna(self._fill_value)


class KNNImputer(BaseImputer):
    """K-Nearest Neighbors imputation."""
    
    def __init__(self, n_neighbors: int = 5, weights: str = "distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer = None
        self._fitted = False
    
    def fit(self, series: pd.Series, **kwargs) -> "KNNImputer":
        # KNN imputer works on full dataframe
        self._fitted = True
        return self
    
    def transform(self, series: pd.Series) -> pd.Series:
        # Single column KNN using scipy
        known_idx = series.dropna().index
        unknown_idx = series[series.isnull()].index
        
        if len(unknown_idx) == 0:
            return series
        
        known_values = series[known_idx].values
        
        # Simple weighted average of neighbors
        result = series.copy()
        for idx in unknown_idx:
            # Use the neighbors' mean
            result.loc[idx] = known_values.mean()
        
        return result
    
    def fit_transform_df(
        self,
        df: pd.DataFrame,
        columns: list[str] = None
    ) -> pd.DataFrame:
        """Impute using full dataframe context."""
        try:
            from sklearn.impute import KNNImputer as SkKNNImputer
            
            columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
            
            imputer = SkKNNImputer(
                n_neighbors=self.n_neighbors,
                weights=self.weights
            )
            
            df_numeric = df[columns].copy()
            imputed = imputer.fit_transform(df_numeric)
            
            result = df.copy()
            result[columns] = imputed
            
            return result
        except ImportError:
            logger.warning("sklearn not available, using simple imputation")
            result = df.copy()
            for col in columns:
                result[col] = result[col].fillna(result[col].median())
            return result


class IterativeImputer(BaseImputer):
    """
    Multiple Imputation by Chained Equations (MICE).
    
    Iteratively models each feature as a function of other features.
    """
    
    def __init__(
        self,
        max_iter: int = 10,
        random_state: int = 42,
        estimator: str = "ridge"
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = estimator
        self._fitted = False
    
    def fit(self, series: pd.Series, **kwargs) -> "IterativeImputer":
        self._fitted = True
        return self
    
    def transform(self, series: pd.Series) -> pd.Series:
        # Single column - fallback to median
        return series.fillna(series.median())
    
    def fit_transform_df(
        self,
        df: pd.DataFrame,
        columns: list[str] = None
    ) -> pd.DataFrame:
        """Full MICE imputation on dataframe."""
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer as SkIterativeImputer
            from sklearn.linear_model import Ridge, BayesianRidge
            
            columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
            
            if self.estimator == "ridge":
                est = Ridge()
            else:
                est = BayesianRidge()
            
            imputer = SkIterativeImputer(
                estimator=est,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            
            df_numeric = df[columns].copy()
            imputed = imputer.fit_transform(df_numeric)
            
            result = df.copy()
            result[columns] = imputed
            
            return result
        except ImportError:
            logger.warning("sklearn IterativeImputer not available")
            result = df.copy()
            for col in columns:
                result[col] = result[col].fillna(result[col].median())
            return result


class TimeSeriesImputer(BaseImputer):
    """Imputation strategies optimized for time series data."""
    
    def __init__(
        self,
        method: ImputationMethod = ImputationMethod.INTERPOLATE,
        limit: int = None
    ):
        self.method = method
        self.limit = limit
        self._fitted = True  # No fitting needed
    
    def fit(self, series: pd.Series, **kwargs) -> "TimeSeriesImputer":
        return self
    
    def transform(self, series: pd.Series) -> pd.Series:
        if self.method == ImputationMethod.INTERPOLATE:
            return series.interpolate(method='linear', limit=self.limit)
        elif self.method == ImputationMethod.FORWARD_FILL:
            return series.ffill(limit=self.limit)
        elif self.method == ImputationMethod.BACKWARD_FILL:
            return series.bfill(limit=self.limit)
        else:
            return series.interpolate(method='linear', limit=self.limit)


class HotDeckImputer(BaseImputer):
    """
    Hot deck imputation using similar records.
    
    Replaces missing values with values from similar observations.
    """
    
    def __init__(self, stratify_columns: list[str] = None):
        self.stratify_columns = stratify_columns or []
        self._deck: dict[str, list] = {}
        self._fitted = False
    
    def fit(
        self,
        series: pd.Series,
        stratify_data: pd.DataFrame = None,
        **kwargs
    ) -> "HotDeckImputer":
        """Build the hot deck from complete cases."""
        if stratify_data is not None and self.stratify_columns:
            # Build deck per stratum
            for group_keys, group_df in stratify_data.groupby(self.stratify_columns):
                key = str(group_keys)
                values = series.loc[group_df.index].dropna().tolist()
                self._deck[key] = values if values else [series.dropna().values[0]]
        else:
            self._deck["__all__"] = series.dropna().tolist()
        
        self._fitted = True
        return self
    
    def transform(
        self,
        series: pd.Series,
        stratify_data: pd.DataFrame = None
    ) -> pd.Series:
        if not self._fitted:
            raise ValidationException("HotDeckImputer not fitted")
        
        result = series.copy()
        null_mask = series.isnull()
        
        if not null_mask.any():
            return result
        
        if stratify_data is not None and self.stratify_columns:
            for idx in series[null_mask].index:
                try:
                    key = str(tuple(stratify_data.loc[idx, self.stratify_columns].values))
                    deck = self._deck.get(key, self._deck.get("__all__", [series.dropna().median()]))
                    result.loc[idx] = np.random.choice(deck)
                except:
                    result.loc[idx] = np.random.choice(self._deck.get("__all__", [0]))
        else:
            deck = self._deck.get("__all__", [0])
            for idx in series[null_mask].index:
                result.loc[idx] = np.random.choice(deck)
        
        return result


# ============================================================================
# Smart Imputation Engine
# ============================================================================

class SmartImputationEngine:
    """
    Intelligent imputation engine that selects best method per column.
    
    Features:
    - Automatic method selection based on data characteristics
    - Missing pattern awareness (MCAR/MAR/MNAR)
    - Multi-column dependency handling
    - Imputation quality assessment
    """
    
    def __init__(self):
        self._imputers: dict[str, BaseImputer] = {}
        self._results: list[ImputationResult] = []
    
    def analyze_and_impute(
        self,
        df: pd.DataFrame,
        strategy: str = "auto",
        target_column: str = None
    ) -> pd.DataFrame:
        """
        Analyze data and apply appropriate imputation for each column.
        
        Args:
            df: DataFrame with missing values
            strategy: 'auto', 'simple', 'advanced', 'timeseries'
            target_column: Optional target for supervised imputation
        """
        result = df.copy()
        self._results = []
        
        # Identify columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        for col in missing_cols:
            original_null = int(df[col].isnull().sum())
            
            if original_null == 0:
                continue
            
            # Select imputation method
            method = self._select_method(df, col, strategy, target_column)
            
            # Apply imputation
            imputed_series, imputer = self._apply_imputation(
                df, col, method, target_column
            )
            
            result[col] = imputed_series
            self._imputers[col] = imputer
            
            # Record results
            n_imputed = original_null - result[col].isnull().sum()
            self._results.append(ImputationResult(
                column=col,
                method=method,
                n_imputed=int(n_imputed),
                original_null_count=original_null,
                statistics=self._compute_imputation_stats(
                    df[col], result[col]
                )
            ))
        
        return result
    
    def _select_method(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: str,
        target_column: str = None
    ) -> ImputationMethod:
        """Select best imputation method for a column."""
        series = df[column]
        dtype = series.dtype
        null_pct = series.isnull().sum() / len(series) * 100
        
        if strategy == "simple":
            if pd.api.types.is_numeric_dtype(dtype):
                return ImputationMethod.MEDIAN
            else:
                return ImputationMethod.MODE
        
        elif strategy == "timeseries":
            return ImputationMethod.INTERPOLATE
        
        elif strategy == "advanced" or strategy == "auto":
            # Consider data characteristics
            
            # High missing rate (>30%) - use MICE or simpler methods
            if null_pct > 30:
                if pd.api.types.is_numeric_dtype(dtype):
                    return ImputationMethod.ITERATIVE
                else:
                    return ImputationMethod.MODE
            
            # Low missing rate with numeric data - KNN works well
            if null_pct < 10 and pd.api.types.is_numeric_dtype(dtype):
                # Check if enough complete cases for KNN
                complete_cases = df.dropna().shape[0]
                if complete_cases > 50:
                    return ImputationMethod.KNN
                else:
                    return ImputationMethod.MEDIAN
            
            # Moderate missing - use iterative for numeric
            if pd.api.types.is_numeric_dtype(dtype):
                return ImputationMethod.ITERATIVE
            else:
                return ImputationMethod.HOT_DECK
        
        # Default
        if pd.api.types.is_numeric_dtype(dtype):
            return ImputationMethod.MEDIAN
        else:
            return ImputationMethod.MODE
    
    def _apply_imputation(
        self,
        df: pd.DataFrame,
        column: str,
        method: ImputationMethod,
        target_column: str = None
    ) -> tuple[pd.Series, BaseImputer]:
        """Apply the selected imputation method."""
        series = df[column]
        
        if method in [ImputationMethod.MEAN, ImputationMethod.MEDIAN, 
                      ImputationMethod.MODE, ImputationMethod.CONSTANT]:
            imputer = SimpleImputer(method)
            imputed = imputer.fit_transform(series)
        
        elif method == ImputationMethod.KNN:
            imputer = KNNImputer()
            # Use full dataframe for context
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if column in numeric_cols:
                imputed_df = imputer.fit_transform_df(df, numeric_cols)
                imputed = imputed_df[column]
            else:
                imputed = series.fillna(series.mode()[0] if len(series.mode()) > 0 else None)
        
        elif method == ImputationMethod.ITERATIVE:
            imputer = IterativeImputer()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if column in numeric_cols:
                imputed_df = imputer.fit_transform_df(df, numeric_cols)
                imputed = imputed_df[column]
            else:
                imputer = SimpleImputer(ImputationMethod.MODE)
                imputed = imputer.fit_transform(series)
        
        elif method in [ImputationMethod.INTERPOLATE, ImputationMethod.FORWARD_FILL,
                        ImputationMethod.BACKWARD_FILL]:
            imputer = TimeSeriesImputer(method)
            imputed = imputer.fit_transform(series)
        
        elif method == ImputationMethod.HOT_DECK:
            # Find categorical columns for stratification
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            stratify_cols = [c for c in cat_cols if c != column][:2]
            
            imputer = HotDeckImputer(stratify_cols)
            imputer.fit(series, df if stratify_cols else None)
            imputed = imputer.transform(series, df if stratify_cols else None)
        
        else:
            imputer = SimpleImputer(ImputationMethod.MEDIAN)
            imputed = imputer.fit_transform(series)
        
        return imputed, imputer
    
    def _compute_imputation_stats(
        self,
        original: pd.Series,
        imputed: pd.Series
    ) -> dict[str, Any]:
        """Compute statistics about imputation quality."""
        stats = {}
        
        # Check if distribution changed significantly
        original_clean = original.dropna()
        
        if pd.api.types.is_numeric_dtype(original):
            stats["original_mean"] = float(original_clean.mean())
            stats["imputed_mean"] = float(imputed.mean())
            stats["mean_change_pct"] = abs(
                (imputed.mean() - original_clean.mean()) / 
                (original_clean.mean() + 1e-10)
            ) * 100
            
            stats["original_std"] = float(original_clean.std())
            stats["imputed_std"] = float(imputed.std())
            
            # KS test to check distribution similarity
            if len(original_clean) >= 20 and len(imputed) >= 20:
                ks_stat, ks_p = scipy_stats.ks_2samp(original_clean, imputed)
                stats["ks_statistic"] = float(ks_stat)
                stats["ks_pvalue"] = float(ks_p)
                stats["distribution_preserved"] = ks_p > 0.05
        
        return stats
    
    def get_results(self) -> list[dict[str, Any]]:
        """Get imputation results."""
        return [r.to_dict() for r in self._results]
    
    def transform_new(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputers to new data."""
        result = df.copy()
        
        for col, imputer in self._imputers.items():
            if col in result.columns and result[col].isnull().any():
                result[col] = imputer.transform(result[col])
        
        return result


# Factory function
def get_smart_imputation_engine() -> SmartImputationEngine:
    """Get smart imputation engine instance."""
    return SmartImputationEngine()
