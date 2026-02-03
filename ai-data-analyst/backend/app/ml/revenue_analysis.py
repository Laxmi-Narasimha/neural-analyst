# AI Enterprise Data Analyst - Revenue Analysis Engine
# Production-grade revenue metrics and analysis
# Handles: any revenue data, period comparisons, growth metrics

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
# Data Classes
# ============================================================================

@dataclass
class RevenuePeriod:
    """Revenue for a specific period."""
    period: str
    revenue: float
    growth_pct: Optional[float]
    yoy_growth: Optional[float]


@dataclass
class RevenueMetrics:
    """Revenue analysis metrics."""
    total_revenue: float
    average_revenue: float
    revenue_variance: float
    
    # Growth
    period_over_period_growth: float
    yoy_growth: Optional[float]
    
    # Breakdown
    by_segment: Dict[str, float] = field(default_factory=dict)
    by_product: Dict[str, float] = field(default_factory=dict)
    by_region: Dict[str, float] = field(default_factory=dict)


@dataclass
class RevenueResult:
    """Complete revenue analysis result."""
    n_periods: int = 0
    date_range: str = ""
    
    # Metrics
    metrics: RevenueMetrics = None
    
    # Time series
    periods: List[RevenuePeriod] = field(default_factory=list)
    
    # Top contributors
    top_segments: List[tuple] = field(default_factory=list)
    top_products: List[tuple] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_periods": self.n_periods,
                "date_range": self.date_range,
                "total_revenue": round(self.metrics.total_revenue, 2),
                "average_revenue": round(self.metrics.average_revenue, 2)
            },
            "growth": {
                "period_over_period": f"{self.metrics.period_over_period_growth:.1f}%",
                "yoy": f"{self.metrics.yoy_growth:.1f}%" if self.metrics.yoy_growth else None
            },
            "breakdown": {
                "by_segment": {k: round(v, 2) for k, v in list(self.metrics.by_segment.items())[:10]},
                "by_product": {k: round(v, 2) for k, v in list(self.metrics.by_product.items())[:10]}
            },
            "periods": [
                {
                    "period": p.period,
                    "revenue": round(p.revenue, 2),
                    "growth_pct": round(p.growth_pct, 1) if p.growth_pct else None
                }
                for p in self.periods[:24]
            ]
        }


# ============================================================================
# Revenue Analysis Engine
# ============================================================================

class RevenueAnalysisEngine:
    """
    Revenue Analysis engine.
    
    Features:
    - Period-over-period analysis
    - Year-over-year comparison
    - Revenue breakdown
    - Growth metrics
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        revenue_col: str = None,
        date_col: str = None,
        segment_col: str = None,
        product_col: str = None
    ) -> RevenueResult:
        """Analyze revenue data."""
        start_time = datetime.now()
        
        # Auto-detect columns
        if revenue_col is None:
            revenue_col = self._detect_revenue_col(df)
        if date_col is None:
            date_col = self._detect_date_col(df)
        
        if self.verbose:
            logger.info(f"Revenue analysis: {revenue_col} by {date_col}")
        
        # Handle date
        df = df.copy()
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df['period'] = df[date_col].dt.to_period('M').astype(str)
        else:
            df['period'] = 'All'
        
        # Period analysis
        period_data = df.groupby('period')[revenue_col].sum().reset_index()
        period_data = period_data.sort_values('period')
        
        periods = []
        prev_revenue = None
        
        for _, row in period_data.iterrows():
            revenue = row[revenue_col]
            growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue and prev_revenue > 0 else None
            
            periods.append(RevenuePeriod(
                period=row['period'],
                revenue=float(revenue),
                growth_pct=growth,
                yoy_growth=None
            ))
            
            prev_revenue = revenue
        
        # Calculate YoY for periods with 12-month history
        for i, period in enumerate(periods):
            if i >= 12:
                yoy = (period.revenue - periods[i-12].revenue) / periods[i-12].revenue * 100
                period.yoy_growth = yoy
        
        # Metrics
        total = df[revenue_col].sum()
        avg = df.groupby('period')[revenue_col].sum().mean()
        var = df.groupby('period')[revenue_col].sum().var()
        
        pop_growth = periods[-1].growth_pct if periods and periods[-1].growth_pct else 0
        yoy_growth = periods[-1].yoy_growth if periods and len(periods) >= 12 else None
        
        # Breakdowns
        by_segment = df.groupby(segment_col)[revenue_col].sum().to_dict() if segment_col and segment_col in df.columns else {}
        by_product = df.groupby(product_col)[revenue_col].sum().to_dict() if product_col and product_col in df.columns else {}
        
        metrics = RevenueMetrics(
            total_revenue=float(total),
            average_revenue=float(avg),
            revenue_variance=float(var) if not np.isnan(var) else 0,
            period_over_period_growth=pop_growth,
            yoy_growth=yoy_growth,
            by_segment=by_segment,
            by_product=by_product
        )
        
        # Top contributors
        top_segments = sorted(by_segment.items(), key=lambda x: -x[1])[:5]
        top_products = sorted(by_product.items(), key=lambda x: -x[1])[:5]
        
        # Date range
        if date_col in df.columns:
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            date_range = "All periods"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RevenueResult(
            n_periods=len(periods),
            date_range=date_range,
            metrics=metrics,
            periods=periods,
            top_segments=top_segments,
            top_products=top_products,
            processing_time_sec=processing_time
        )
    
    def _detect_revenue_col(self, df: pd.DataFrame) -> str:
        patterns = ['revenue', 'sales', 'amount', 'total', 'value']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                if df[col].dtype in [np.float64, np.int64]:
                    return col
        return df.select_dtypes(include=[np.number]).columns[0]
    
    def _detect_date_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        patterns = ['date', 'time', 'period']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def get_revenue_engine() -> RevenueAnalysisEngine:
    """Get revenue analysis engine."""
    return RevenueAnalysisEngine()


def quick_revenue(
    df: pd.DataFrame,
    revenue_col: str = None
) -> Dict[str, Any]:
    """Quick revenue analysis."""
    engine = RevenueAnalysisEngine(verbose=False)
    result = engine.analyze(df, revenue_col)
    return result.to_dict()
