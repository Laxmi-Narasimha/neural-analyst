# AI Enterprise Data Analyst - BI Metrics Engine
# Business intelligence metrics, KPIs, and dashboards

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# BI Metric Types
# ============================================================================

class MetricType(str, Enum):
    """Types of business metrics."""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    RATIO = "ratio"
    GROWTH = "growth"
    PERCENTAGE = "percentage"
    CUMULATIVE = "cumulative"


class TrendDirection(str, Enum):
    """Trend direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class Metric:
    """Single metric definition and value."""
    
    name: str
    value: float
    metric_type: MetricType
    
    # Comparison
    previous_value: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    trend: TrendDirection = TrendDirection.STABLE
    
    # Context
    unit: str = ""
    target: Optional[float] = None
    is_on_target: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 2),
            "unit": self.unit,
            "metric_type": self.metric_type.value,
            "previous_value": round(self.previous_value, 2) if self.previous_value else None,
            "change": round(self.change, 2) if self.change else None,
            "change_pct": round(self.change_pct, 2) if self.change_pct else None,
            "trend": self.trend.value,
            "target": self.target,
            "is_on_target": self.is_on_target
        }


@dataclass
class KPI:
    """Key Performance Indicator."""
    
    name: str
    current_value: float
    target_value: float
    
    period_start: datetime
    period_end: datetime
    
    achievement_pct: float = 0.0
    status: str = "on_track"
    historical: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current_value": round(self.current_value, 2),
            "target_value": round(self.target_value, 2),
            "achievement_pct": round(self.achievement_pct, 2),
            "status": self.status,
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            }
        }


# ============================================================================
# Common Business Metrics
# ============================================================================

class BusinessMetrics:
    """Common business metric calculations."""
    
    @staticmethod
    def revenue(df: pd.DataFrame, amount_col: str) -> float:
        """Total revenue."""
        return float(df[amount_col].sum())
    
    @staticmethod
    def aov(df: pd.DataFrame, amount_col: str, order_col: str = None) -> float:
        """Average Order Value."""
        if order_col:
            return float(df.groupby(order_col)[amount_col].sum().mean())
        return float(df[amount_col].mean())
    
    @staticmethod
    def conversion_rate(conversions: int, total: int) -> float:
        """Conversion rate."""
        return conversions / total * 100 if total > 0 else 0
    
    @staticmethod
    def growth_rate(current: float, previous: float) -> float:
        """Growth rate percentage."""
        return ((current - previous) / previous * 100) if previous != 0 else 0
    
    @staticmethod
    def retention_rate(retained: int, total: int) -> float:
        """Retention rate."""
        return retained / total * 100 if total > 0 else 0
    
    @staticmethod
    def churn_rate(churned: int, total: int) -> float:
        """Churn rate."""
        return churned / total * 100 if total > 0 else 0
    
    @staticmethod
    def cac(marketing_cost: float, new_customers: int) -> float:
        """Customer Acquisition Cost."""
        return marketing_cost / new_customers if new_customers > 0 else 0
    
    @staticmethod
    def ltv(avg_revenue: float, lifespan_months: float, margin: float = 1.0) -> float:
        """Customer Lifetime Value."""
        return avg_revenue * lifespan_months * margin
    
    @staticmethod
    def arpu(revenue: float, users: int) -> float:
        """Average Revenue Per User."""
        return revenue / users if users > 0 else 0
    
    @staticmethod
    def dau_mau(daily_active: int, monthly_active: int) -> float:
        """DAU/MAU Stickiness ratio."""
        return daily_active / monthly_active * 100 if monthly_active > 0 else 0


# ============================================================================
# Time-Based Metrics
# ============================================================================

class TimeSeriesMetrics:
    """Time-based metric calculations."""
    
    @staticmethod
    def period_comparison(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        period: str = "M"  # M=month, W=week, D=day
    ) -> pd.DataFrame:
        """Compare metrics across periods."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['period'] = df[date_col].dt.to_period(period)
        
        grouped = df.groupby('period')[value_col].agg(['sum', 'mean', 'count'])
        grouped['prev_sum'] = grouped['sum'].shift(1)
        grouped['growth_pct'] = (grouped['sum'] - grouped['prev_sum']) / grouped['prev_sum'] * 100
        
        return grouped.reset_index()
    
    @staticmethod
    def yoy_growth(
        df: pd.DataFrame,
        date_col: str,
        value_col: str
    ) -> pd.DataFrame:
        """Year-over-Year growth."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        
        pivoted = df.pivot_table(
            index='month',
            columns='year',
            values=value_col,
            aggfunc='sum'
        )
        
        # Calculate YoY growth for most recent year
        if len(pivoted.columns) >= 2:
            current_year = pivoted.columns[-1]
            prev_year = pivoted.columns[-2]
            pivoted['yoy_growth'] = (pivoted[current_year] - pivoted[prev_year]) / pivoted[prev_year] * 100
        
        return pivoted
    
    @staticmethod
    def rolling_metrics(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        window: int = 7
    ) -> pd.DataFrame:
        """Calculate rolling metrics."""
        df = df.copy()
        df = df.sort_values(date_col)
        
        df[f'rolling_avg_{window}d'] = df[value_col].rolling(window).mean()
        df[f'rolling_sum_{window}d'] = df[value_col].rolling(window).sum()
        df[f'rolling_std_{window}d'] = df[value_col].rolling(window).std()
        
        return df


# ============================================================================
# Dashboard Builder
# ============================================================================

class DashboardBuilder:
    """Build dashboard with multiple metrics."""
    
    def __init__(self):
        self._metrics: list[Metric] = []
        self._kpis: list[KPI] = []
    
    def add_metric(
        self,
        name: str,
        value: float,
        previous_value: float = None,
        target: float = None,
        unit: str = "",
        metric_type: MetricType = MetricType.COUNT
    ) -> "DashboardBuilder":
        """Add a metric to the dashboard."""
        change = value - previous_value if previous_value is not None else None
        change_pct = (value - previous_value) / previous_value * 100 if previous_value else None
        
        trend = TrendDirection.STABLE
        if change is not None:
            trend = TrendDirection.UP if change > 0 else TrendDirection.DOWN if change < 0 else TrendDirection.STABLE
        
        is_on_target = value >= target if target else True
        
        self._metrics.append(Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            previous_value=previous_value,
            change=change,
            change_pct=change_pct,
            trend=trend,
            unit=unit,
            target=target,
            is_on_target=is_on_target
        ))
        
        return self
    
    def add_kpi(
        self,
        name: str,
        current: float,
        target: float,
        period_start: datetime = None,
        period_end: datetime = None
    ) -> "DashboardBuilder":
        """Add a KPI to the dashboard."""
        achievement = (current / target * 100) if target > 0 else 0
        
        status = "on_track"
        if achievement >= 100:
            status = "achieved"
        elif achievement >= 80:
            status = "on_track"
        elif achievement >= 50:
            status = "at_risk"
        else:
            status = "off_track"
        
        self._kpis.append(KPI(
            name=name,
            current_value=current,
            target_value=target,
            period_start=period_start or datetime.utcnow(),
            period_end=period_end or datetime.utcnow(),
            achievement_pct=achievement,
            status=status
        ))
        
        return self
    
    def build(self) -> dict[str, Any]:
        """Build the dashboard."""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": [m.to_dict() for m in self._metrics],
            "kpis": [k.to_dict() for k in self._kpis],
            "summary": {
                "total_metrics": len(self._metrics),
                "total_kpis": len(self._kpis),
                "kpis_on_track": sum(1 for k in self._kpis if k.status in ["achieved", "on_track"]),
                "metrics_trending_up": sum(1 for m in self._metrics if m.trend == TrendDirection.UP)
            }
        }


# ============================================================================
# BI Metrics Engine
# ============================================================================

class BIMetricsEngine:
    """
    Business Intelligence metrics engine.
    
    Features:
    - Common business metrics
    - Time-based analysis
    - KPI tracking
    - Dashboard building
    - Trend detection
    """
    
    def __init__(self):
        self.business = BusinessMetrics()
        self.time_series = TimeSeriesMetrics()
    
    def create_dashboard(self) -> DashboardBuilder:
        """Create a new dashboard builder."""
        return DashboardBuilder()
    
    def calculate_core_metrics(
        self,
        df: pd.DataFrame,
        revenue_col: str = None,
        user_col: str = None,
        date_col: str = None
    ) -> dict[str, Metric]:
        """Calculate core business metrics."""
        metrics = {}
        
        if revenue_col and revenue_col in df.columns:
            metrics["total_revenue"] = Metric(
                name="Total Revenue",
                value=self.business.revenue(df, revenue_col),
                metric_type=MetricType.SUM,
                unit="$"
            )
            
            metrics["aov"] = Metric(
                name="Average Order Value",
                value=self.business.aov(df, revenue_col),
                metric_type=MetricType.AVERAGE,
                unit="$"
            )
        
        if user_col and user_col in df.columns:
            metrics["unique_users"] = Metric(
                name="Unique Users",
                value=float(df[user_col].nunique()),
                metric_type=MetricType.COUNT
            )
            
            if revenue_col:
                metrics["arpu"] = Metric(
                    name="ARPU",
                    value=self.business.arpu(
                        df[revenue_col].sum(),
                        df[user_col].nunique()
                    ),
                    metric_type=MetricType.AVERAGE,
                    unit="$"
                )
        
        return metrics
    
    def analyze_trends(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        period: str = "M"
    ) -> dict[str, Any]:
        """Analyze trends over time."""
        comparison = self.time_series.period_comparison(df, date_col, value_col, period)
        
        # Calculate trend
        if len(comparison) >= 2:
            recent = comparison['sum'].iloc[-1]
            previous = comparison['sum'].iloc[-2]
            growth = (recent - previous) / previous * 100 if previous > 0 else 0
            
            trend = "up" if growth > 5 else "down" if growth < -5 else "stable"
        else:
            growth = 0
            trend = "stable"
        
        return {
            "period_data": comparison.to_dict(orient="records"),
            "latest_growth_pct": round(growth, 2),
            "trend": trend,
            "periods_analyzed": len(comparison)
        }
    
    def compare_cohorts(
        self,
        df: pd.DataFrame,
        cohort_col: str,
        value_col: str
    ) -> dict[str, Any]:
        """Compare metrics across cohorts."""
        cohort_stats = df.groupby(cohort_col)[value_col].agg([
            'count', 'sum', 'mean', 'median', 'std'
        ]).reset_index()
        
        cohort_stats.columns = [cohort_col, 'count', 'total', 'average', 'median', 'std']
        
        return {
            "cohorts": cohort_stats.to_dict(orient="records"),
            "best_performing": cohort_stats.loc[cohort_stats['total'].idxmax(), cohort_col],
            "worst_performing": cohort_stats.loc[cohort_stats['total'].idxmin(), cohort_col]
        }


# Factory function
def get_bi_metrics_engine() -> BIMetricsEngine:
    """Get BI metrics engine instance."""
    return BIMetricsEngine()
