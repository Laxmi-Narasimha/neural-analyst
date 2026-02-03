# AI Enterprise Data Analyst - Data Profiling Engine
# Comprehensive automated data profiling and EDA

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

class InferredType(str, Enum):
    """Inferred column types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"


@dataclass
class ColumnStats:
    """Statistics for a single column."""
    
    name: str
    dtype: str
    inferred_type: InferredType
    
    # Counts
    count: int
    missing: int
    unique: int
    
    # Distribution
    distribution: dict[str, Any] = field(default_factory=dict)
    
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Categorical stats
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    
    # Quality indicators
    completeness: float = 0.0
    uniqueness: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "inferred_type": self.inferred_type.value,
            "count": self.count,
            "missing": self.missing,
            "missing_pct": round(self.missing / (self.count + self.missing) * 100, 2) if (self.count + self.missing) > 0 else 0,
            "unique": self.unique,
            "completeness": round(self.completeness, 4),
            "uniqueness": round(self.uniqueness, 4)
        }
        
        if self.inferred_type == InferredType.NUMERIC:
            result.update({
                "mean": round(self.mean, 4) if self.mean is not None else None,
                "std": round(self.std, 4) if self.std is not None else None,
                "min": self.min,
                "max": self.max,
                "median": self.median,
                "q25": self.q25,
                "q75": self.q75,
                "skewness": round(self.skewness, 4) if self.skewness is not None else None,
                "kurtosis": round(self.kurtosis, 4) if self.kurtosis is not None else None
            })
        
        if self.top_values:
            result["top_values"] = [{"value": v, "count": c} for v, c in self.top_values[:10]]
        
        return result


@dataclass
class DataProfile:
    """Complete data profile."""
    
    name: str
    rows: int
    columns: int
    
    column_stats: list[ColumnStats]
    
    # Overall quality
    overall_completeness: float = 0.0
    duplicate_rows: int = 0
    memory_usage_mb: float = 0.0
    
    # Correlations
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Warnings
    warnings: list[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": {"rows": self.rows, "columns": self.columns},
            "overall_completeness": round(self.overall_completeness, 4),
            "duplicate_rows": self.duplicate_rows,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "columns": [c.to_dict() for c in self.column_stats],
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat()
        }


# ============================================================================
# Type Inference
# ============================================================================

class TypeInferer:
    """Infer semantic types from data."""
    
    def infer(self, series: pd.Series) -> InferredType:
        """Infer semantic type of a column."""
        # Check for boolean
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2:
            if set(map(str, unique_vals)).issubset({'0', '1', 'true', 'false', 'yes', 'no', 'True', 'False'}):
                return InferredType.BOOLEAN
        
        # Check dtype
        if pd.api.types.is_numeric_dtype(series):
            # Check if identifier (sequential integers, high cardinality)
            if series.nunique() / len(series) > 0.9 and pd.api.types.is_integer_dtype(series):
                return InferredType.IDENTIFIER
            return InferredType.NUMERIC
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return InferredType.DATETIME
        
        # String analysis
        sample = series.dropna().head(100).astype(str)
        
        # Email pattern
        if sample.str.contains(r'@.*\.[a-z]{2,}', case=False, regex=True).mean() > 0.8:
            return InferredType.EMAIL
        
        # URL pattern
        if sample.str.contains(r'^https?://', regex=True).mean() > 0.8:
            return InferredType.URL
        
        # Phone pattern
        if sample.str.contains(r'^\+?[\d\-\(\)\s]{7,}$', regex=True).mean() > 0.8:
            return InferredType.PHONE
        
        # Text vs categorical
        avg_length = sample.str.len().mean()
        cardinality = series.nunique() / len(series) if len(series) > 0 else 0
        
        if avg_length > 50 or cardinality > 0.5:
            return InferredType.TEXT
        
        return InferredType.CATEGORICAL


# ============================================================================
# Data Profiler
# ============================================================================

class DataProfiler:
    """
    Comprehensive data profiling.
    
    Features:
    - Type inference
    - Descriptive statistics
    - Distribution analysis
    - Quality metrics
    - Correlation detection
    - Warnings generation
    """
    
    def __init__(self):
        self.type_inferer = TypeInferer()
    
    def profile(
        self,
        df: pd.DataFrame,
        name: str = "dataset"
    ) -> DataProfile:
        """Generate comprehensive data profile."""
        column_stats = []
        
        for col in df.columns:
            stats = self._profile_column(df[col])
            column_stats.append(stats)
        
        # Overall metrics
        overall_completeness = 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_rows = df.duplicated().sum()
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            correlations = corr_matrix.to_dict()
        
        # Generate warnings
        warnings = self._generate_warnings(df, column_stats)
        
        return DataProfile(
            name=name,
            rows=len(df),
            columns=len(df.columns),
            column_stats=column_stats,
            overall_completeness=overall_completeness,
            duplicate_rows=duplicate_rows,
            memory_usage_mb=memory_mb,
            correlations=correlations,
            warnings=warnings
        )
    
    def _profile_column(self, series: pd.Series) -> ColumnStats:
        """Profile a single column."""
        inferred_type = self.type_inferer.infer(series)
        
        count = series.count()
        missing = series.isnull().sum()
        total = len(series)
        unique = series.nunique()
        
        stats = ColumnStats(
            name=series.name,
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            count=count,
            missing=missing,
            unique=unique,
            completeness=count / total if total > 0 else 0,
            uniqueness=unique / count if count > 0 else 0
        )
        
        if inferred_type == InferredType.NUMERIC:
            self._add_numeric_stats(series, stats)
        
        if inferred_type in [InferredType.CATEGORICAL, InferredType.TEXT]:
            self._add_categorical_stats(series, stats)
        
        return stats
    
    def _add_numeric_stats(self, series: pd.Series, stats: ColumnStats) -> None:
        """Add numeric statistics."""
        clean = series.dropna()
        
        if len(clean) > 0:
            stats.mean = float(clean.mean())
            stats.std = float(clean.std())
            stats.min = float(clean.min())
            stats.max = float(clean.max())
            stats.median = float(clean.median())
            stats.q25 = float(clean.quantile(0.25))
            stats.q75 = float(clean.quantile(0.75))
            
            try:
                from scipy.stats import skew, kurtosis
                stats.skewness = float(skew(clean))
                stats.kurtosis = float(kurtosis(clean))
            except:
                pass
        
        # Distribution (histogram)
        try:
            hist, bins = np.histogram(clean, bins=20)
            stats.distribution = {
                "histogram": hist.tolist(),
                "bin_edges": bins.tolist()
            }
        except:
            pass
    
    def _add_categorical_stats(self, series: pd.Series, stats: ColumnStats) -> None:
        """Add categorical statistics."""
        value_counts = series.value_counts().head(20)
        stats.top_values = list(value_counts.items())
        
        # Distribution
        stats.distribution = {
            "value_counts": value_counts.to_dict()
        }
    
    def _generate_warnings(
        self,
        df: pd.DataFrame,
        column_stats: list[ColumnStats]
    ) -> list[str]:
        """Generate data quality warnings."""
        warnings = []
        
        for stats in column_stats:
            # High missing values
            missing_pct = stats.missing / (stats.count + stats.missing) * 100 if (stats.count + stats.missing) > 0 else 0
            if missing_pct > 50:
                warnings.append(f"Column '{stats.name}' has {missing_pct:.1f}% missing values")
            
            # Low cardinality
            if stats.inferred_type == InferredType.NUMERIC and stats.unique < 5:
                warnings.append(f"Column '{stats.name}' may be categorical (only {stats.unique} unique values)")
            
            # High cardinality categorical
            if stats.inferred_type == InferredType.CATEGORICAL and stats.uniqueness > 0.9:
                warnings.append(f"Column '{stats.name}' has very high cardinality ({stats.unique} unique)")
            
            # Skewed distribution
            if stats.skewness and abs(stats.skewness) > 2:
                warnings.append(f"Column '{stats.name}' is highly skewed (skewness: {stats.skewness:.2f})")
            
            # Constant column
            if stats.unique == 1:
                warnings.append(f"Column '{stats.name}' is constant")
        
        # Duplicates
        dup = df.duplicated().sum()
        if dup > 0:
            warnings.append(f"Dataset has {dup} duplicate rows ({dup / len(df) * 100:.1f}%)")
        
        return warnings


# ============================================================================
# Quick Stats
# ============================================================================

class QuickStats:
    """Fast statistical summaries."""
    
    @staticmethod
    def summarize(df: pd.DataFrame) -> dict[str, Any]:
        """Quick summary statistics."""
        return {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing": df.isnull().sum().sum(),
            "missing_pct": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100,
            "duplicates": df.duplicated().sum(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns)
        }
    
    @staticmethod
    def describe_extended(df: pd.DataFrame) -> pd.DataFrame:
        """Extended describe with more statistics."""
        desc = df.describe().T
        
        # Add more stats
        desc['missing'] = df.isnull().sum()
        desc['missing_pct'] = df.isnull().mean() * 100
        desc['unique'] = df.nunique()
        
        # Skewness and kurtosis for numeric
        numeric = df.select_dtypes(include=[np.number])
        try:
            from scipy.stats import skew, kurtosis
            desc['skewness'] = numeric.apply(skew)
            desc['kurtosis'] = numeric.apply(kurtosis)
        except:
            pass
        
        return desc


# ============================================================================
# Data Profiling Engine
# ============================================================================

class DataProfilingEngine:
    """
    Unified data profiling engine.
    
    Features:
    - Comprehensive profiling
    - Type inference
    - Quality metrics
    - Quick statistics
    - Distribution analysis
    """
    
    def __init__(self):
        self.profiler = DataProfiler()
        self.quick = QuickStats()
    
    def profile(
        self,
        df: pd.DataFrame,
        name: str = "dataset"
    ) -> DataProfile:
        """Generate full data profile."""
        return self.profiler.profile(df, name)
    
    def quick_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Quick summary."""
        return self.quick.summarize(df)
    
    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extended describe."""
        return self.quick.describe_extended(df)
    
    def compare_datasets(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> dict[str, Any]:
        """Compare two datasets."""
        profile1 = self.quick.summarize(df1)
        profile2 = self.quick.summarize(df2)
        
        return {
            "dataset1": profile1,
            "dataset2": profile2,
            "differences": {
                "rows": profile2["shape"]["rows"] - profile1["shape"]["rows"],
                "columns": profile2["shape"]["columns"] - profile1["shape"]["columns"],
                "missing_pct_change": profile2["missing_pct"] - profile1["missing_pct"]
            }
        }


# Factory function
def get_data_profiling_engine() -> DataProfilingEngine:
    """Get data profiling engine instance."""
    return DataProfilingEngine()
</Parameter>
<parameter name="Complexity">8
