# AI Enterprise Data Analyst - Pivot Analysis Engine
# Production-grade pivot table operations
# Handles: any tabular data, aggregations, cross-tabulations

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

class AggregationType(str, Enum):
    """Aggregation types."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PivotConfig:
    """Configuration for pivot analysis."""
    index: Optional[List[str]] = None  # Row labels
    columns: Optional[List[str]] = None  # Column labels
    values: Optional[List[str]] = None  # Values to aggregate
    aggfunc: Union[str, List[str]] = "sum"
    fill_value: Any = 0
    margins: bool = True  # Include totals


@dataclass
class PivotResult:
    """Complete pivot analysis result."""
    pivot_table: pd.DataFrame = None
    
    # Dimensions
    n_rows: int = 0
    n_columns: int = 0
    row_labels: List[str] = field(default_factory=list)
    column_labels: List[str] = field(default_factory=list)
    
    # Totals
    grand_total: float = 0.0
    row_totals: Dict[str, float] = field(default_factory=dict)
    column_totals: Dict[str, float] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": {
                "n_rows": self.n_rows,
                "n_columns": self.n_columns
            },
            "grand_total": round(self.grand_total, 2) if self.grand_total else 0,
            "pivot_data": self.pivot_table.to_dict() if self.pivot_table is not None else {},
            "row_totals": {k: round(v, 2) for k, v in list(self.row_totals.items())[:10]},
            "column_totals": {str(k): round(v, 2) for k, v in list(self.column_totals.items())[:10]}
        }


# ============================================================================
# Pivot Analysis Engine
# ============================================================================

class PivotAnalysisEngine:
    """
    Pivot Analysis engine.
    
    Features:
    - Flexible pivot table creation
    - Multiple aggregation functions
    - Cross-tabulations
    - Margins and subtotals
    """
    
    def __init__(self, config: PivotConfig = None, verbose: bool = True):
        self.config = config or PivotConfig()
        self.verbose = verbose
    
    def create_pivot(
        self,
        df: pd.DataFrame,
        index: Union[str, List[str]] = None,
        columns: Union[str, List[str]] = None,
        values: Union[str, List[str]] = None,
        aggfunc: Union[str, List[str]] = None
    ) -> PivotResult:
        """Create pivot table."""
        start_time = datetime.now()
        
        # Use config or parameters
        index = index or self.config.index
        columns = columns or self.config.columns
        values = values or self.config.values
        aggfunc = aggfunc or self.config.aggfunc
        
        # Auto-detect if not specified
        if index is None:
            index = self._detect_index(df)
        if values is None:
            values = self._detect_values(df)
        
        # Ensure lists
        if isinstance(index, str):
            index = [index]
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(values, str):
            values = [values]
        
        if self.verbose:
            logger.info(f"Creating pivot: index={index}, columns={columns}, values={values}")
        
        # Create pivot table
        pivot_table = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=self.config.fill_value,
            margins=self.config.margins
        )
        
        # Extract totals
        row_totals = {}
        column_totals = {}
        grand_total = 0
        
        if self.config.margins and 'All' in pivot_table.index:
            grand_total = float(pivot_table.loc['All'].sum()) if hasattr(pivot_table.loc['All'], 'sum') else float(pivot_table.loc['All'])
            
            for idx in pivot_table.index:
                if idx != 'All':
                    row_totals[str(idx)] = float(pivot_table.loc[idx].sum()) if hasattr(pivot_table.loc[idx], 'sum') else float(pivot_table.loc[idx])
            
            if isinstance(pivot_table.columns, pd.MultiIndex):
                for col in pivot_table.columns:
                    if 'All' not in str(col):
                        column_totals[str(col)] = float(pivot_table[col].sum())
            else:
                for col in pivot_table.columns:
                    if col != 'All':
                        column_totals[str(col)] = float(pivot_table[col].sum())
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PivotResult(
            pivot_table=pivot_table,
            n_rows=len(pivot_table),
            n_columns=len(pivot_table.columns),
            row_labels=[str(x) for x in pivot_table.index.tolist()],
            column_labels=[str(x) for x in pivot_table.columns.tolist()],
            grand_total=grand_total,
            row_totals=row_totals,
            column_totals=column_totals,
            processing_time_sec=processing_time
        )
    
    def crosstab(
        self,
        df: pd.DataFrame,
        row_col: str,
        col_col: str,
        normalize: bool = False
    ) -> pd.DataFrame:
        """Create cross-tabulation."""
        result = pd.crosstab(
            df[row_col],
            df[col_col],
            normalize='all' if normalize else False,
            margins=True
        )
        return result
    
    def _detect_index(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect index columns."""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return cat_cols[:1] if cat_cols else [df.columns[0]]
    
    def _detect_values(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect value columns."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return num_cols[:1] if num_cols else [df.columns[-1]]


# ============================================================================
# Factory Functions
# ============================================================================

def get_pivot_engine(config: PivotConfig = None) -> PivotAnalysisEngine:
    """Get pivot analysis engine."""
    return PivotAnalysisEngine(config=config)


def quick_pivot(
    df: pd.DataFrame,
    index: str,
    values: str,
    columns: str = None,
    aggfunc: str = "sum"
) -> Dict[str, Any]:
    """Quick pivot table."""
    engine = PivotAnalysisEngine(verbose=False)
    result = engine.create_pivot(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
    return result.to_dict()
