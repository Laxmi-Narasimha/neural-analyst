# AI Enterprise Data Analyst - Summary Statistics Engine
# Comprehensive statistical summaries for any data
# Handles: numeric, categorical, datetime, mixed data types

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NumericStats:
    """Statistics for numeric column."""
    count: int
    missing: int
    missing_pct: float
    mean: float
    std: float
    min: float
    max: float
    median: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float  # Coefficient of variation
    zeros: int
    negatives: int


@dataclass
class CategoricalStats:
    """Statistics for categorical column."""
    count: int
    missing: int
    missing_pct: float
    unique: int
    top_value: str
    top_freq: int
    top_pct: float
    entropy: float
    value_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatetimeStats:
    """Statistics for datetime column."""
    count: int
    missing: int
    missing_pct: float
    min_date: datetime
    max_date: datetime
    range_days: int
    most_common_day: str
    most_common_month: int
    weekday_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class ColumnSummary:
    """Summary for a single column."""
    name: str
    dtype: str
    semantic_type: str
    stats: Any  # NumericStats, CategoricalStats, or DatetimeStats


@dataclass
class SummaryStatsResult:
    """Complete summary statistics result."""
    n_rows: int = 0
    n_columns: int = 0
    memory_usage_mb: float = 0.0
    
    # Column summaries
    columns: List[ColumnSummary] = field(default_factory=list)
    
    # Overview
    numeric_columns: int = 0
    categorical_columns: int = 0
    datetime_columns: int = 0
    
    # Data quality
    total_missing: int = 0
    total_missing_pct: float = 0.0
    complete_rows: int = 0
    complete_rows_pct: float = 0.0
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overview": {
                "n_rows": self.n_rows,
                "n_columns": self.n_columns,
                "memory_mb": round(self.memory_usage_mb, 2),
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "datetime_columns": self.datetime_columns
            },
            "data_quality": {
                "total_missing": self.total_missing,
                "total_missing_pct": round(self.total_missing_pct, 2),
                "complete_rows": self.complete_rows,
                "complete_rows_pct": round(self.complete_rows_pct, 2)
            },
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "semantic_type": c.semantic_type,
                    "missing_pct": round(c.stats.missing_pct, 2) if hasattr(c.stats, 'missing_pct') else 0
                }
                for c in self.columns
            ]
        }


# ============================================================================
# Summary Statistics Engine
# ============================================================================

class SummaryStatisticsEngine:
    """
    Comprehensive Summary Statistics engine.
    
    Features:
    - Handles all data types
    - Semantic type detection
    - Data quality metrics
    - Detailed per-column statistics
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def summarize(self, df: pd.DataFrame) -> SummaryStatsResult:
        """Generate comprehensive summary statistics."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Generating summary for {len(df)} rows, {len(df.columns)} columns")
        
        columns = []
        numeric_count = 0
        categorical_count = 0
        datetime_count = 0
        
        for col in df.columns:
            summary = self._summarize_column(df[col])
            columns.append(summary)
            
            if summary.semantic_type == 'numeric':
                numeric_count += 1
            elif summary.semantic_type == 'datetime':
                datetime_count += 1
            else:
                categorical_count += 1
        
        # Data quality
        total_missing = df.isna().sum().sum()
        total_cells = df.size
        complete_rows = df.dropna().shape[0]
        
        # Memory
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SummaryStatsResult(
            n_rows=len(df),
            n_columns=len(df.columns),
            memory_usage_mb=memory_mb,
            columns=columns,
            numeric_columns=numeric_count,
            categorical_columns=categorical_count,
            datetime_columns=datetime_count,
            total_missing=total_missing,
            total_missing_pct=total_missing / total_cells * 100 if total_cells > 0 else 0,
            complete_rows=complete_rows,
            complete_rows_pct=complete_rows / len(df) * 100 if len(df) > 0 else 0,
            processing_time_sec=processing_time
        )
    
    def _summarize_column(self, series: pd.Series) -> ColumnSummary:
        """Summarize a single column."""
        dtype = str(series.dtype)
        
        # Detect semantic type
        if pd.api.types.is_numeric_dtype(series):
            semantic_type = 'numeric'
            stats = self._numeric_stats(series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            semantic_type = 'datetime'
            stats = self._datetime_stats(series)
        else:
            semantic_type = 'categorical'
            stats = self._categorical_stats(series)
        
        return ColumnSummary(
            name=series.name,
            dtype=dtype,
            semantic_type=semantic_type,
            stats=stats
        )
    
    def _numeric_stats(self, series: pd.Series) -> NumericStats:
        """Calculate numeric statistics."""
        clean = series.dropna()
        
        if len(clean) == 0:
            return NumericStats(
                count=0, missing=len(series), missing_pct=100.0,
                mean=0, std=0, min=0, max=0, median=0,
                q1=0, q3=0, iqr=0, skewness=0, kurtosis=0,
                cv=0, zeros=0, negatives=0
            )
        
        mean = clean.mean()
        std = clean.std()
        
        return NumericStats(
            count=len(clean),
            missing=series.isna().sum(),
            missing_pct=series.isna().sum() / len(series) * 100,
            mean=float(mean),
            std=float(std),
            min=float(clean.min()),
            max=float(clean.max()),
            median=float(clean.median()),
            q1=float(clean.quantile(0.25)),
            q3=float(clean.quantile(0.75)),
            iqr=float(clean.quantile(0.75) - clean.quantile(0.25)),
            skewness=float(scipy_stats.skew(clean)),
            kurtosis=float(scipy_stats.kurtosis(clean)),
            cv=float(std / mean) if mean != 0 else 0,
            zeros=int((clean == 0).sum()),
            negatives=int((clean < 0).sum())
        )
    
    def _categorical_stats(self, series: pd.Series) -> CategoricalStats:
        """Calculate categorical statistics."""
        clean = series.dropna()
        
        if len(clean) == 0:
            return CategoricalStats(
                count=0, missing=len(series), missing_pct=100.0,
                unique=0, top_value="", top_freq=0, top_pct=0,
                entropy=0, value_counts={}
            )
        
        value_counts = clean.value_counts()
        top_value = str(value_counts.index[0]) if len(value_counts) > 0 else ""
        top_freq = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        
        # Shannon entropy
        probs = value_counts / value_counts.sum()
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return CategoricalStats(
            count=len(clean),
            missing=series.isna().sum(),
            missing_pct=series.isna().sum() / len(series) * 100,
            unique=clean.nunique(),
            top_value=top_value,
            top_freq=top_freq,
            top_pct=top_freq / len(clean) * 100 if len(clean) > 0 else 0,
            entropy=float(entropy),
            value_counts=value_counts.head(10).to_dict()
        )
    
    def _datetime_stats(self, series: pd.Series) -> DatetimeStats:
        """Calculate datetime statistics."""
        clean = series.dropna()
        
        if len(clean) == 0:
            return DatetimeStats(
                count=0, missing=len(series), missing_pct=100.0,
                min_date=None, max_date=None, range_days=0,
                most_common_day="", most_common_month=0,
                weekday_distribution={}
            )
        
        min_date = clean.min()
        max_date = clean.max()
        
        # Weekday distribution
        weekdays = clean.dt.day_name().value_counts().to_dict()
        
        return DatetimeStats(
            count=len(clean),
            missing=series.isna().sum(),
            missing_pct=series.isna().sum() / len(series) * 100,
            min_date=min_date.to_pydatetime() if pd.notna(min_date) else None,
            max_date=max_date.to_pydatetime() if pd.notna(max_date) else None,
            range_days=(max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else 0,
            most_common_day=clean.dt.day_name().mode().iloc[0] if len(clean) > 0 else "",
            most_common_month=int(clean.dt.month.mode().iloc[0]) if len(clean) > 0 else 0,
            weekday_distribution=weekdays
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_summary_engine() -> SummaryStatisticsEngine:
    """Get summary statistics engine."""
    return SummaryStatisticsEngine()


def quick_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick summary statistics."""
    engine = SummaryStatisticsEngine(verbose=False)
    result = engine.summarize(df)
    return result.to_dict()
