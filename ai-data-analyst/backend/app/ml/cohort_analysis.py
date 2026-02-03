# AI Enterprise Data Analyst - Cohort Analysis Engine
# Production-grade cohort analysis for retention and behavior tracking
# Handles: any transaction/event data, flexible cohort definitions

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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

class CohortType(str, Enum):
    """Types of cohort analysis."""
    RETENTION = "retention"  # Customer retention over time
    REVENUE = "revenue"  # Revenue per cohort
    BEHAVIOR = "behavior"  # Behavioral metrics
    SIZE = "size"  # Cohort size over time


class CohortPeriod(str, Enum):
    """Cohort period granularity."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CohortConfig:
    """Configuration for cohort analysis."""
    cohort_type: CohortType = CohortType.RETENTION
    period: CohortPeriod = CohortPeriod.MONTHLY
    
    # Column mappings
    customer_id_col: Optional[str] = None
    date_col: Optional[str] = None
    value_col: Optional[str] = None  # For revenue cohorts
    
    # Analysis parameters
    max_periods: int = 12  # Maximum periods to analyze


@dataclass
class CohortResult:
    """Complete cohort analysis result."""
    cohort_type: CohortType
    period: CohortPeriod
    n_cohorts: int
    n_customers: int
    
    # Cohort matrix
    cohort_matrix: pd.DataFrame = None  # Main cohort table
    cohort_sizes: Dict[str, int] = field(default_factory=dict)
    
    # Summary statistics
    avg_retention: Dict[int, float] = field(default_factory=dict)
    cohort_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "cohort_type": self.cohort_type.value,
                "period": self.period.value,
                "n_cohorts": self.n_cohorts,
                "n_customers": self.n_customers
            },
            "cohort_matrix": self.cohort_matrix.to_dict() if self.cohort_matrix is not None else {},
            "cohort_sizes": self.cohort_sizes,
            "avg_retention": {str(k): round(v, 2) for k, v in self.avg_retention.items()},
            "cohort_performance": self.cohort_performance
        }


# ============================================================================
# Cohort Analysis Engine
# ============================================================================

class CohortAnalysisEngine:
    """
    Complete Cohort Analysis engine.
    
    Features:
    - Retention cohorts
    - Revenue cohorts
    - Flexible period granularity
    - Automatic cohort creation
    """
    
    def __init__(self, config: CohortConfig = None, verbose: bool = True):
        self.config = config or CohortConfig()
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        customer_id_col: str = None,
        date_col: str = None,
        value_col: str = None
    ) -> CohortResult:
        """Perform cohort analysis on transaction/event data."""
        start_time = datetime.now()
        
        # Auto-detect columns
        customer_id_col = customer_id_col or self._detect_customer_col(df)
        date_col = date_col or self._detect_date_col(df)
        
        if self.config.cohort_type == CohortType.REVENUE:
            value_col = value_col or self._detect_value_col(df)
        
        if self.verbose:
            logger.info(f"Cohort analysis: customer={customer_id_col}, date={date_col}")
        
        # Prepare data
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[customer_id_col, date_col])
        
        # Create cohort column (first activity period)
        df['CohortPeriod'] = df.groupby(customer_id_col)[date_col].transform('min').dt.to_period(self.config.period.value)
        df['ActivityPeriod'] = df[date_col].dt.to_period(self.config.period.value)
        
        # Calculate period index
        df['PeriodIndex'] = (df['ActivityPeriod'] - df['CohortPeriod']).apply(lambda x: x.n if hasattr(x, 'n') else 0)
        
        # Filter to max periods
        df = df[df['PeriodIndex'] <= self.config.max_periods]
        
        # Calculate cohort matrix based on type
        if self.config.cohort_type == CohortType.RETENTION:
            cohort_matrix = self._calculate_retention(df, customer_id_col)
        elif self.config.cohort_type == CohortType.REVENUE:
            cohort_matrix = self._calculate_revenue(df, customer_id_col, value_col)
        elif self.config.cohort_type == CohortType.SIZE:
            cohort_matrix = self._calculate_size(df, customer_id_col)
        else:
            cohort_matrix = self._calculate_retention(df, customer_id_col)
        
        # Get cohort sizes
        cohort_sizes = df.groupby('CohortPeriod')[customer_id_col].nunique().to_dict()
        cohort_sizes = {str(k): int(v) for k, v in cohort_sizes.items()}
        
        # Average retention by period
        avg_retention = {}
        for col in cohort_matrix.columns:
            avg_retention[col] = cohort_matrix[col].mean()
        
        # Cohort performance
        cohort_performance = {}
        for cohort in cohort_matrix.index:
            cohort_performance[str(cohort)] = {
                'initial_size': cohort_sizes.get(str(cohort), 0),
                'period_0': float(cohort_matrix.loc[cohort, 0]) if 0 in cohort_matrix.columns else 0,
                'period_1': float(cohort_matrix.loc[cohort, 1]) if 1 in cohort_matrix.columns else 0,
                'avg_retention': float(cohort_matrix.loc[cohort].mean())
            }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CohortResult(
            cohort_type=self.config.cohort_type,
            period=self.config.period,
            n_cohorts=len(cohort_matrix),
            n_customers=df[customer_id_col].nunique(),
            cohort_matrix=cohort_matrix,
            cohort_sizes=cohort_sizes,
            avg_retention=avg_retention,
            cohort_performance=cohort_performance,
            processing_time_sec=processing_time
        )
    
    def _calculate_retention(
        self,
        df: pd.DataFrame,
        customer_col: str
    ) -> pd.DataFrame:
        """Calculate retention cohort matrix."""
        # Count unique customers per cohort and period
        cohort_data = df.groupby(['CohortPeriod', 'PeriodIndex'])[customer_col].nunique().reset_index()
        cohort_data.columns = ['CohortPeriod', 'PeriodIndex', 'Customers']
        
        # Pivot to matrix
        cohort_matrix = cohort_data.pivot(index='CohortPeriod', columns='PeriodIndex', values='Customers')
        
        # Convert to retention percentage
        cohort_sizes = cohort_matrix[0]
        retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
        
        return retention_matrix.round(2)
    
    def _calculate_revenue(
        self,
        df: pd.DataFrame,
        customer_col: str,
        value_col: str
    ) -> pd.DataFrame:
        """Calculate revenue cohort matrix."""
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
        
        # Sum revenue per cohort and period
        cohort_data = df.groupby(['CohortPeriod', 'PeriodIndex'])[value_col].sum().reset_index()
        cohort_data.columns = ['CohortPeriod', 'PeriodIndex', 'Revenue']
        
        # Pivot to matrix
        cohort_matrix = cohort_data.pivot(index='CohortPeriod', columns='PeriodIndex', values='Revenue')
        
        return cohort_matrix.round(2)
    
    def _calculate_size(
        self,
        df: pd.DataFrame,
        customer_col: str
    ) -> pd.DataFrame:
        """Calculate cohort size matrix."""
        cohort_data = df.groupby(['CohortPeriod', 'PeriodIndex'])[customer_col].nunique().reset_index()
        cohort_data.columns = ['CohortPeriod', 'PeriodIndex', 'Customers']
        
        cohort_matrix = cohort_data.pivot(index='CohortPeriod', columns='PeriodIndex', values='Customers')
        
        return cohort_matrix
    
    def _detect_customer_col(self, df: pd.DataFrame) -> str:
        """Auto-detect customer column."""
        patterns = ['customer', 'user', 'client', 'member', 'account', 'id']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[0]
    
    def _detect_date_col(self, df: pd.DataFrame) -> str:
        """Auto-detect date column."""
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        patterns = ['date', 'time', 'created', 'timestamp']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    def _detect_value_col(self, df: pd.DataFrame) -> str:
        """Auto-detect value column."""
        patterns = ['amount', 'revenue', 'total', 'value', 'price', 'sales']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(p in col.lower() for p in patterns):
                return col
        return numeric_cols[0] if len(numeric_cols) > 0 else None


# ============================================================================
# Factory Functions
# ============================================================================

def get_cohort_engine(config: CohortConfig = None) -> CohortAnalysisEngine:
    """Get cohort analysis engine."""
    return CohortAnalysisEngine(config=config)


def quick_cohort(
    df: pd.DataFrame,
    cohort_type: str = "retention",
    period: str = "M"
) -> Dict[str, Any]:
    """Quick cohort analysis."""
    config = CohortConfig(
        cohort_type=CohortType(cohort_type),
        period=CohortPeriod(period)
    )
    engine = CohortAnalysisEngine(config=config, verbose=False)
    result = engine.analyze(df)
    return result.to_dict()
