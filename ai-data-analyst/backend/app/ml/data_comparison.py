# AI Enterprise Data Analyst - Data Comparison Engine
# Production-grade dataset comparison
# Handles: schema, statistics, value comparisons

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class ComparisonStatus(str, Enum):
    """Comparison status."""
    MATCH = "match"
    DIFFER = "differ"
    MISSING = "missing"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ColumnComparison:
    """Comparison result for a column."""
    column: str
    status: ComparisonStatus
    dtype_match: bool
    stats_match: bool
    value_diff_pct: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataComparisonResult:
    """Complete data comparison result."""
    # Schema comparison
    common_columns: List[str] = field(default_factory=list)
    only_in_first: List[str] = field(default_factory=list)
    only_in_second: List[str] = field(default_factory=list)
    
    # Row comparison
    rows_first: int = 0
    rows_second: int = 0
    row_diff: int = 0
    row_diff_pct: float = 0.0
    
    # Column comparisons
    column_comparisons: List[ColumnComparison] = field(default_factory=list)
    
    # Overall
    match_score: float = 0.0
    key_differences: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": {
                "common_columns": len(self.common_columns),
                "only_in_first": self.only_in_first[:10],
                "only_in_second": self.only_in_second[:10]
            },
            "rows": {
                "first": self.rows_first,
                "second": self.rows_second,
                "difference": self.row_diff,
                "diff_pct": round(self.row_diff_pct, 2)
            },
            "match_score": round(self.match_score * 100, 1),
            "column_comparisons": [
                {
                    "column": c.column,
                    "status": c.status.value,
                    "dtype_match": c.dtype_match,
                    "value_diff_pct": round(c.value_diff_pct, 2)
                }
                for c in self.column_comparisons[:20]
            ],
            "key_differences": self.key_differences[:10]
        }


# ============================================================================
# Data Comparison Engine
# ============================================================================

class DataComparisonEngine:
    """
    Data Comparison engine.
    
    Features:
    - Schema comparison
    - Statistical comparison
    - Value-level comparison
    - Difference highlighting
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def compare(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        key_columns: List[str] = None
    ) -> DataComparisonResult:
        """Compare two DataFrames."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Comparing datasets: {df1.shape} vs {df2.shape}")
        
        # Column comparison
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        common = list(cols1 & cols2)
        only1 = list(cols1 - cols2)
        only2 = list(cols2 - cols1)
        
        # Row comparison
        rows1 = len(df1)
        rows2 = len(df2)
        row_diff = abs(rows1 - rows2)
        row_diff_pct = row_diff / max(rows1, rows2) * 100 if max(rows1, rows2) > 0 else 0
        
        # Column-level comparison
        column_comps = []
        differences = []
        
        for col in common:
            comp = self._compare_column(df1[col], df2[col], col)
            column_comps.append(comp)
            
            if comp.status == ComparisonStatus.DIFFER:
                differences.append(f"{col}: {comp.value_diff_pct:.1f}% different")
        
        # Calculate match score
        if len(common) > 0:
            dtype_matches = sum(1 for c in column_comps if c.dtype_match)
            value_matches = sum(1 for c in column_comps if c.value_diff_pct < 5)
            
            schema_score = len(common) / max(len(cols1), len(cols2))
            dtype_score = dtype_matches / len(common)
            value_score = value_matches / len(common)
            row_score = 1 - min(row_diff_pct / 100, 1)
            
            match_score = (schema_score + dtype_score + value_score + row_score) / 4
        else:
            match_score = 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DataComparisonResult(
            common_columns=common,
            only_in_first=only1,
            only_in_second=only2,
            rows_first=rows1,
            rows_second=rows2,
            row_diff=row_diff,
            row_diff_pct=row_diff_pct,
            column_comparisons=column_comps,
            match_score=match_score,
            key_differences=differences,
            processing_time_sec=processing_time
        )
    
    def _compare_column(
        self,
        col1: pd.Series,
        col2: pd.Series,
        name: str
    ) -> ColumnComparison:
        """Compare a single column."""
        dtype_match = str(col1.dtype) == str(col2.dtype)
        
        details = {}
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
            mean1, mean2 = col1.mean(), col2.mean()
            std1, std2 = col1.std(), col2.std()
            
            mean_diff = abs(mean1 - mean2) / abs(mean1) * 100 if mean1 != 0 else 0
            std_diff = abs(std1 - std2) / abs(std1) * 100 if std1 != 0 else 0
            
            value_diff = (mean_diff + std_diff) / 2
            stats_match = value_diff < 10
            
            details = {
                "mean_1": mean1,
                "mean_2": mean2,
                "std_1": std1,
                "std_2": std2
            }
        else:
            # For categorical
            unique1 = set(col1.dropna().unique())
            unique2 = set(col2.dropna().unique())
            
            common_vals = len(unique1 & unique2)
            total_vals = len(unique1 | unique2)
            
            value_diff = (1 - common_vals / total_vals) * 100 if total_vals > 0 else 0
            stats_match = value_diff < 20
            
            details = {
                "unique_1": len(unique1),
                "unique_2": len(unique2),
                "common": common_vals
            }
        
        status = ComparisonStatus.MATCH if value_diff < 5 else ComparisonStatus.DIFFER
        
        return ColumnComparison(
            column=name,
            status=status,
            dtype_match=dtype_match,
            stats_match=stats_match,
            value_diff_pct=value_diff,
            details=details
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_comparison_engine() -> DataComparisonEngine:
    """Get data comparison engine."""
    return DataComparisonEngine()


def quick_compare(
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> Dict[str, Any]:
    """Quick data comparison."""
    engine = DataComparisonEngine(verbose=False)
    result = engine.compare(df1, df2)
    return result.to_dict()
