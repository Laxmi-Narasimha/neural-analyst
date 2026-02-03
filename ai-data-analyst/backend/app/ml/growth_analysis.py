# AI Enterprise Data Analyst - Growth Analysis Engine
# Production-grade growth rate and trend analysis
# Handles: any time series data with growth calculations

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
# Enums
# ============================================================================

class GrowthType(str, Enum):
    """Types of growth calculations."""
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    CAGR = "cagr"
    MOM = "month_over_month"
    YOY = "year_over_year"
    QOQ = "quarter_over_quarter"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PeriodGrowth:
    """Growth for a single period."""
    period: str
    value: float
    absolute_change: float
    pct_change: float
    cumulative_growth: float


@dataclass
class GrowthMetrics:
    """Summary growth metrics."""
    total_growth_pct: float
    avg_growth_pct: float
    cagr: float
    highest_growth: float
    lowest_growth: float
    volatility: float  # std of growth rates
    positive_periods: int
    negative_periods: int


@dataclass
class GrowthResult:
    """Complete growth analysis result."""
    n_periods: int = 0
    start_value: float = 0.0
    end_value: float = 0.0
    
    # Metrics
    metrics: GrowthMetrics = None
    
    # Period-by-period
    periods: List[PeriodGrowth] = field(default_factory=list)
    
    # Trend
    trend: str = ""  # accelerating, decelerating, stable, volatile
    
    # Projections
    projected_next: Optional[float] = None
    projected_growth_rate: Optional[float] = None
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_periods": self.n_periods,
                "start_value": round(self.start_value, 2),
                "end_value": round(self.end_value, 2),
                "total_growth_pct": round(self.metrics.total_growth_pct, 2) if self.metrics else 0,
                "avg_growth_pct": round(self.metrics.avg_growth_pct, 2) if self.metrics else 0,
                "cagr": round(self.metrics.cagr, 2) if self.metrics else 0
            },
            "trend": self.trend,
            "periods": [
                {
                    "period": p.period,
                    "value": round(p.value, 2),
                    "pct_change": round(p.pct_change, 2)
                }
                for p in self.periods[-12:]  # Last 12 periods
            ],
            "projection": {
                "next_value": round(self.projected_next, 2) if self.projected_next else None,
                "growth_rate": round(self.projected_growth_rate, 2) if self.projected_growth_rate else None
            }
        }


# ============================================================================
# Growth Analysis Engine
# ============================================================================

class GrowthAnalysisEngine:
    """
    Production-grade Growth Analysis engine.
    
    Features:
    - Multiple growth calculations
    - CAGR calculation
    - Trend detection
    - Growth projections
    - Volatility analysis
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame = None,
        values: List[float] = None,
        value_col: str = None,
        date_col: str = None,
        periods: List[str] = None
    ) -> GrowthResult:
        """Analyze growth."""
        start_time = datetime.now()
        
        # Extract values
        if df is not None:
            if value_col is None:
                value_col = df.select_dtypes(include=[np.number]).columns[0]
            
            if date_col and date_col in df.columns:
                df_sorted = df.sort_values(date_col)
                periods = df_sorted[date_col].astype(str).tolist()
            else:
                periods = [f"Period_{i}" for i in range(len(df))]
            
            values = df_sorted[value_col].tolist() if date_col else df[value_col].tolist()
        
        if values is None or len(values) < 2:
            return GrowthResult(n_periods=len(values) if values else 0)
        
        if periods is None:
            periods = [f"Period_{i}" for i in range(len(values))]
        
        if self.verbose:
            logger.info(f"Growth analysis: {len(values)} periods")
        
        # Calculate period-by-period growth
        period_data = []
        cumulative = 0
        
        for i, (val, period) in enumerate(zip(values, periods)):
            if i == 0:
                abs_change = 0
                pct_change = 0
            else:
                prev_val = values[i - 1]
                abs_change = val - prev_val
                pct_change = ((val - prev_val) / abs(prev_val) * 100) if prev_val != 0 else 0
            
            cumulative = ((val - values[0]) / abs(values[0]) * 100) if values[0] != 0 else 0
            
            period_data.append(PeriodGrowth(
                period=str(period),
                value=float(val),
                absolute_change=float(abs_change),
                pct_change=float(pct_change),
                cumulative_growth=float(cumulative)
            ))
        
        # Calculate metrics
        growth_rates = [p.pct_change for p in period_data[1:]]  # Exclude first period
        
        if len(growth_rates) == 0:
            return GrowthResult(n_periods=len(values))
        
        n_years = len(values) / 12 if len(values) >= 12 else 1
        
        # CAGR
        if values[0] > 0 and values[-1] > 0:
            cagr = ((values[-1] / values[0]) ** (1 / n_years) - 1) * 100
        else:
            cagr = 0
        
        total_growth = ((values[-1] - values[0]) / abs(values[0]) * 100) if values[0] != 0 else 0
        
        metrics = GrowthMetrics(
            total_growth_pct=total_growth,
            avg_growth_pct=float(np.mean(growth_rates)),
            cagr=cagr,
            highest_growth=float(max(growth_rates)),
            lowest_growth=float(min(growth_rates)),
            volatility=float(np.std(growth_rates)),
            positive_periods=sum(1 for r in growth_rates if r > 0),
            negative_periods=sum(1 for r in growth_rates if r < 0)
        )
        
        # Detect trend
        trend = self._detect_trend(growth_rates)
        
        # Simple projection
        projected_next = None
        projected_growth = None
        if len(growth_rates) >= 3:
            recent_avg = np.mean(growth_rates[-3:])
            projected_growth = recent_avg
            projected_next = values[-1] * (1 + recent_avg / 100)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GrowthResult(
            n_periods=len(values),
            start_value=values[0],
            end_value=values[-1],
            metrics=metrics,
            periods=period_data,
            trend=trend,
            projected_next=projected_next,
            projected_growth_rate=projected_growth,
            processing_time_sec=processing_time
        )
    
    def _detect_trend(self, growth_rates: List[float]) -> str:
        """Detect growth trend."""
        if len(growth_rates) < 3:
            return "insufficient_data"
        
        # Check last 3 periods
        recent = growth_rates[-3:]
        
        # Calculate acceleration
        if len(growth_rates) >= 6:
            first_half = np.mean(growth_rates[:len(growth_rates)//2])
            second_half = np.mean(growth_rates[len(growth_rates)//2:])
            
            if second_half > first_half + 2:
                return "accelerating"
            elif second_half < first_half - 2:
                return "decelerating"
        
        # Check volatility
        volatility = np.std(growth_rates)
        if volatility > abs(np.mean(growth_rates)) * 2:
            return "volatile"
        
        return "stable"
    
    def calculate_cagr(
        self,
        start_value: float,
        end_value: float,
        n_years: float
    ) -> float:
        """Calculate Compound Annual Growth Rate."""
        if start_value <= 0 or end_value <= 0 or n_years <= 0:
            return 0
        return ((end_value / start_value) ** (1 / n_years) - 1) * 100
    
    def calculate_mom_growth(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: str
    ) -> pd.DataFrame:
        """Calculate month-over-month growth."""
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_sorted = df_work.sort_values(date_col)
        
        df_sorted['mom_growth'] = df_sorted[value_col].pct_change() * 100
        
        return df_sorted
    
    def calculate_yoy_growth(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: str
    ) -> pd.DataFrame:
        """Calculate year-over-year growth."""
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_sorted = df_work.sort_values(date_col)
        
        df_sorted['yoy_growth'] = df_sorted[value_col].pct_change(periods=12) * 100
        
        return df_sorted


# ============================================================================
# Factory Functions
# ============================================================================

def get_growth_engine() -> GrowthAnalysisEngine:
    """Get growth analysis engine."""
    return GrowthAnalysisEngine()


def quick_growth(
    values: List[float],
    periods: List[str] = None
) -> Dict[str, Any]:
    """Quick growth analysis."""
    engine = GrowthAnalysisEngine(verbose=False)
    result = engine.analyze(values=values, periods=periods)
    return result.to_dict()


def calculate_cagr(
    start_value: float,
    end_value: float,
    n_years: float
) -> float:
    """Calculate CAGR."""
    engine = GrowthAnalysisEngine(verbose=False)
    return engine.calculate_cagr(start_value, end_value, n_years)
