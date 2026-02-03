# AI Enterprise Data Analyst - Cohort Retention Engine
# Production-grade cohort retention analysis
# Handles: any user-level transaction data

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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
# Data Classes
# ============================================================================

@dataclass
class CohortSummary:
    """Summary for a single cohort."""
    cohort_id: str
    cohort_size: int
    retention_rates: List[float]  # Retention rate for each period
    avg_retention: float
    churn_rate: float
    lifetime_periods: float  # Average customer lifetime in periods


@dataclass
class RetentionMatrix:
    """Retention matrix data."""
    cohorts: List[str]
    periods: List[int]
    matrix: pd.DataFrame  # Retention rates
    counts_matrix: pd.DataFrame  # Absolute counts


@dataclass
class CohortRetentionResult:
    """Complete cohort retention result."""
    n_cohorts: int = 0
    n_users: int = 0
    cohort_period: str = "monthly"
    
    # Matrices
    retention_matrix: RetentionMatrix = None
    
    # Summaries
    cohorts: List[CohortSummary] = field(default_factory=list)
    
    # Aggregated metrics
    avg_retention_by_period: Dict[int, float] = field(default_factory=dict)
    overall_avg_retention: float = 0.0
    
    # Trends
    retention_trend: str = ""  # improving, declining, stable
    best_cohort: str = ""
    worst_cohort: str = ""
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_cohorts": self.n_cohorts,
                "n_users": self.n_users,
                "overall_avg_retention": round(self.overall_avg_retention, 2),
                "retention_trend": self.retention_trend,
                "best_cohort": self.best_cohort,
                "worst_cohort": self.worst_cohort
            },
            "retention_by_period": {
                f"period_{k}": round(v, 2) 
                for k, v in self.avg_retention_by_period.items()
            },
            "cohorts": [
                {
                    "cohort": c.cohort_id,
                    "size": c.cohort_size,
                    "avg_retention": round(c.avg_retention, 2),
                    "churn_rate": round(c.churn_rate, 2)
                }
                for c in self.cohorts[:20]
            ],
            "matrix_preview": self.retention_matrix.matrix.head(10).to_dict() if self.retention_matrix else {}
        }


# ============================================================================
# Cohort Retention Engine
# ============================================================================

class CohortRetentionEngine:
    """
    Production-grade Cohort Retention engine.
    
    Features:
    - Flexible cohort definition
    - Retention matrix generation
    - Trend analysis
    - Cohort comparison
    - Churn calculation
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        user_col: str,
        date_col: str,
        cohort_period: str = "M",  # M=monthly, W=weekly, Q=quarterly
        observation_period: str = None
    ) -> CohortRetentionResult:
        """Perform cohort retention analysis."""
        start_time = datetime.now()
        
        if observation_period is None:
            observation_period = cohort_period
        
        if self.verbose:
            logger.info(f"Cohort retention: {df[user_col].nunique()} users")
        
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_work = df_work.dropna(subset=[date_col])
        
        # Define cohort as first transaction period
        first_activity = df_work.groupby(user_col)[date_col].min().reset_index()
        first_activity.columns = [user_col, 'first_activity']
        df_work = df_work.merge(first_activity, on=user_col)
        
        # Create cohort label
        df_work['cohort'] = df_work['first_activity'].dt.to_period(cohort_period).astype(str)
        df_work['activity_period'] = df_work[date_col].dt.to_period(observation_period)
        df_work['first_period'] = df_work['first_activity'].dt.to_period(observation_period)
        
        # Calculate period number (0 = first period)
        df_work['period_number'] = (
            df_work['activity_period'].astype(int) - 
            df_work['first_period'].astype(int)
        )
        
        # Build retention matrix
        cohort_data = df_work.groupby(['cohort', 'period_number'])[user_col].nunique().reset_index()
        cohort_data.columns = ['cohort', 'period_number', 'users']
        
        # Pivot to matrix
        retention_pivot = cohort_data.pivot(
            index='cohort', columns='period_number', values='users'
        ).fillna(0)
        
        # Convert to retention rates
        cohort_sizes = retention_pivot[0]
        retention_rates = retention_pivot.div(cohort_sizes, axis=0) * 100
        
        # Create matrix objects
        retention_matrix = RetentionMatrix(
            cohorts=retention_rates.index.tolist(),
            periods=retention_rates.columns.tolist(),
            matrix=retention_rates,
            counts_matrix=retention_pivot
        )
        
        # Generate cohort summaries
        cohorts = []
        for cohort_id in retention_rates.index:
            rates = retention_rates.loc[cohort_id].dropna().values
            size = int(cohort_sizes.get(cohort_id, 0))
            
            avg_retention = float(np.mean(rates[1:])) if len(rates) > 1 else 0
            churn_rate = 100 - avg_retention
            
            # Lifetime estimate (simplified)
            lifetime = sum(r / 100 for r in rates) if len(rates) > 0 else 0
            
            cohorts.append(CohortSummary(
                cohort_id=str(cohort_id),
                cohort_size=size,
                retention_rates=rates.tolist(),
                avg_retention=avg_retention,
                churn_rate=churn_rate,
                lifetime_periods=lifetime
            ))
        
        # Average retention by period
        avg_by_period = {}
        for period in retention_rates.columns:
            period_rates = retention_rates[period].dropna()
            if len(period_rates) > 0:
                avg_by_period[int(period)] = float(period_rates.mean())
        
        # Overall average (excluding period 0)
        period_1_plus = [v for k, v in avg_by_period.items() if k > 0]
        overall_avg = float(np.mean(period_1_plus)) if period_1_plus else 0
        
        # Trend analysis
        retention_trend = self._detect_trend(cohorts)
        
        # Best/worst
        if cohorts:
            sorted_cohorts = sorted(cohorts, key=lambda x: -x.avg_retention)
            best_cohort = sorted_cohorts[0].cohort_id
            worst_cohort = sorted_cohorts[-1].cohort_id
        else:
            best_cohort = worst_cohort = ""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CohortRetentionResult(
            n_cohorts=len(cohorts),
            n_users=df_work[user_col].nunique(),
            cohort_period=cohort_period,
            retention_matrix=retention_matrix,
            cohorts=cohorts,
            avg_retention_by_period=avg_by_period,
            overall_avg_retention=overall_avg,
            retention_trend=retention_trend,
            best_cohort=best_cohort,
            worst_cohort=worst_cohort,
            processing_time_sec=processing_time
        )
    
    def _detect_trend(self, cohorts: List[CohortSummary]) -> str:
        """Detect retention trend across cohorts."""
        if len(cohorts) < 3:
            return "insufficient_data"
        
        # Look at 3 most recent cohorts
        recent = cohorts[-3:]
        recent_rates = [c.avg_retention for c in recent]
        
        if all(recent_rates[i] <= recent_rates[i+1] for i in range(len(recent_rates)-1)):
            return "improving"
        elif all(recent_rates[i] >= recent_rates[i+1] for i in range(len(recent_rates)-1)):
            return "declining"
        return "stable"
    
    def get_retention_curve(
        self,
        result: CohortRetentionResult,
        cohort_id: str = None
    ) -> Dict[int, float]:
        """Get retention curve for specific cohort or average."""
        if cohort_id:
            for c in result.cohorts:
                if c.cohort_id == cohort_id:
                    return {i: r for i, r in enumerate(c.retention_rates)}
        return result.avg_retention_by_period


# ============================================================================
# Factory Functions
# ============================================================================

def get_retention_engine() -> CohortRetentionEngine:
    """Get cohort retention engine."""
    return CohortRetentionEngine()


def quick_retention(
    df: pd.DataFrame,
    user_col: str,
    date_col: str
) -> Dict[str, Any]:
    """Quick cohort retention analysis."""
    engine = CohortRetentionEngine(verbose=False)
    result = engine.analyze(df, user_col, date_col)
    return result.to_dict()


def get_retention_matrix(
    df: pd.DataFrame,
    user_col: str,
    date_col: str
) -> pd.DataFrame:
    """Get retention matrix as DataFrame."""
    engine = CohortRetentionEngine(verbose=False)
    result = engine.analyze(df, user_col, date_col)
    return result.retention_matrix.matrix if result.retention_matrix else pd.DataFrame()
