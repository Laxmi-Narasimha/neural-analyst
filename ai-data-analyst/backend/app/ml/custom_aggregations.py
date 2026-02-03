# AI Enterprise Data Analyst - Custom Aggregations Engine
# Production-grade flexible data aggregation
# Handles: group by, window functions, rollups

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

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

class AggFunction(str, Enum):
    """Aggregation functions."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"
    NUNIQUE = "nunique"
    MODE = "mode"
    PERCENTILE_25 = "p25"
    PERCENTILE_75 = "p75"
    PERCENTILE_90 = "p90"
    RANGE = "range"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AggregationSpec:
    """Specification for an aggregation."""
    column: str
    function: AggFunction
    alias: str = None
    
    def __post_init__(self):
        if self.alias is None:
            self.alias = f"{self.column}_{self.function.value}"


@dataclass
class AggregationResult:
    """Complete aggregation result."""
    group_columns: List[str] = field(default_factory=list)
    n_groups: int = 0
    
    aggregated_df: pd.DataFrame = None
    
    # Summary
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_columns": self.group_columns,
            "n_groups": self.n_groups,
            "summary": self.summary_stats,
            "preview": self.aggregated_df.head(10).to_dict() if self.aggregated_df is not None else {}
        }


# ============================================================================
# Custom Aggregations Engine
# ============================================================================

class CustomAggregationsEngine:
    """
    Production-grade Custom Aggregations engine.
    
    Features:
    - Flexible group by
    - Multiple aggregation functions
    - Window functions
    - Rollups and subtotals
    - Custom aggregations
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.custom_functions: Dict[str, Callable] = {}
    
    def register_function(self, name: str, func: Callable):
        """Register a custom aggregation function."""
        self.custom_functions[name] = func
        return self
    
    def aggregate(
        self,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        aggregations: List[AggregationSpec],
        include_totals: bool = False,
        sort_by: str = None,
        ascending: bool = True
    ) -> AggregationResult:
        """Perform custom aggregations."""
        start_time = datetime.now()
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        if self.verbose:
            logger.info(f"Aggregating by {group_by}: {len(aggregations)} aggregations")
        
        # Build aggregation dict
        agg_dict = {}
        for spec in aggregations:
            if spec.column not in df.columns:
                continue
            
            func = self._get_agg_function(spec.function)
            
            if spec.column not in agg_dict:
                agg_dict[spec.column] = []
            
            agg_dict[spec.column].append((spec.alias, func))
        
        # Perform aggregation
        try:
            # Use named aggregation
            named_agg = {}
            for col, funcs in agg_dict.items():
                for alias, func in funcs:
                    named_agg[alias] = pd.NamedAgg(column=col, aggfunc=func)
            
            result = df.groupby(group_by, as_index=False).agg(**named_agg)
        except Exception as e:
            # Fallback to simple aggregation
            simple_agg = {}
            for col, funcs in agg_dict.items():
                if funcs:
                    simple_agg[col] = funcs[0][1]
            
            result = df.groupby(group_by, as_index=False).agg(simple_agg)
        
        # Add totals row
        if include_totals and len(result) > 0:
            totals = pd.DataFrame([result.select_dtypes(include=[np.number]).sum()])
            for col in group_by:
                totals[col] = 'TOTAL'
            result = pd.concat([result, totals], ignore_index=True)
        
        # Sort
        if sort_by and sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=ascending)
        
        # Summary
        summary = {
            "total_rows": len(df),
            "grouped_rows": len(result),
            "compression_ratio": len(df) / len(result) if len(result) > 0 else 0
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AggregationResult(
            group_columns=group_by,
            n_groups=len(result),
            aggregated_df=result,
            summary_stats=summary,
            processing_time_sec=processing_time
        )
    
    def _get_agg_function(self, func: AggFunction) -> Callable:
        """Get aggregation function."""
        if func == AggFunction.SUM:
            return 'sum'
        elif func == AggFunction.MEAN:
            return 'mean'
        elif func == AggFunction.MEDIAN:
            return 'median'
        elif func == AggFunction.MIN:
            return 'min'
        elif func == AggFunction.MAX:
            return 'max'
        elif func == AggFunction.COUNT:
            return 'count'
        elif func == AggFunction.STD:
            return 'std'
        elif func == AggFunction.VAR:
            return 'var'
        elif func == AggFunction.FIRST:
            return 'first'
        elif func == AggFunction.LAST:
            return 'last'
        elif func == AggFunction.NUNIQUE:
            return 'nunique'
        elif func == AggFunction.MODE:
            return lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
        elif func == AggFunction.PERCENTILE_25:
            return lambda x: x.quantile(0.25)
        elif func == AggFunction.PERCENTILE_75:
            return lambda x: x.quantile(0.75)
        elif func == AggFunction.PERCENTILE_90:
            return lambda x: x.quantile(0.90)
        elif func == AggFunction.RANGE:
            return lambda x: x.max() - x.min()
        else:
            return 'sum'
    
    def window_aggregate(
        self,
        df: pd.DataFrame,
        value_col: str,
        window_size: int = 3,
        functions: List[str] = None
    ) -> pd.DataFrame:
        """Add window aggregations."""
        result = df.copy()
        
        if functions is None:
            functions = ['mean', 'sum', 'min', 'max']
        
        for func in functions:
            col_name = f'{value_col}_rolling_{func}_{window_size}'
            
            if func == 'mean':
                result[col_name] = result[value_col].rolling(window_size).mean()
            elif func == 'sum':
                result[col_name] = result[value_col].rolling(window_size).sum()
            elif func == 'min':
                result[col_name] = result[value_col].rolling(window_size).min()
            elif func == 'max':
                result[col_name] = result[value_col].rolling(window_size).max()
            elif func == 'std':
                result[col_name] = result[value_col].rolling(window_size).std()
        
        return result
    
    def cumulative_aggregate(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str = None
    ) -> pd.DataFrame:
        """Add cumulative aggregations."""
        result = df.copy()
        
        if group_col:
            result[f'{value_col}_cumsum'] = result.groupby(group_col)[value_col].cumsum()
            result[f'{value_col}_cumcount'] = result.groupby(group_col).cumcount() + 1
            result[f'{value_col}_cummean'] = result[f'{value_col}_cumsum'] / result[f'{value_col}_cumcount']
        else:
            result[f'{value_col}_cumsum'] = result[value_col].cumsum()
            result[f'{value_col}_cumcount'] = range(1, len(result) + 1)
            result[f'{value_col}_cummean'] = result[f'{value_col}_cumsum'] / result[f'{value_col}_cumcount']
        
        return result
    
    def pivot_aggregate(
        self,
        df: pd.DataFrame,
        index: str,
        columns: str,
        values: str,
        aggfunc: str = 'sum',
        fill_value: float = 0
    ) -> pd.DataFrame:
        """Create pivot table with aggregation."""
        return pd.pivot_table(
            df,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_aggregation_engine() -> CustomAggregationsEngine:
    """Get custom aggregations engine."""
    return CustomAggregationsEngine()


def quick_aggregate(
    df: pd.DataFrame,
    group_by: Union[str, List[str]],
    value_col: str,
    func: str = "sum"
) -> pd.DataFrame:
    """Quick aggregation."""
    engine = CustomAggregationsEngine(verbose=False)
    
    spec = AggregationSpec(column=value_col, function=AggFunction(func))
    result = engine.aggregate(df, group_by, [spec])
    
    return result.aggregated_df


def multi_aggregate(
    df: pd.DataFrame,
    group_by: str,
    specs: List[Dict[str, str]]
) -> pd.DataFrame:
    """Multiple aggregations."""
    engine = CustomAggregationsEngine(verbose=False)
    
    agg_specs = [
        AggregationSpec(
            column=s['column'],
            function=AggFunction(s.get('function', 'sum')),
            alias=s.get('alias')
        )
        for s in specs
    ]
    
    result = engine.aggregate(df, group_by, agg_specs)
    return result.aggregated_df
