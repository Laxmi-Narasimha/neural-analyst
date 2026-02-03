# AI Enterprise Data Analyst - Metric Calculator Engine
# Production-grade business and technical metric calculations
# Handles: generic metrics with flexible formulas

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

class MetricType(str, Enum):
    """Types of metrics."""
    RATIO = "ratio"
    RATE = "rate"
    PERCENTAGE = "percentage"
    AVERAGE = "average"
    SUM = "sum"
    COUNT = "count"
    GROWTH = "growth"
    CUSTOM = "custom"


class AggregationLevel(str, Enum):
    """Aggregation levels."""
    TOTAL = "total"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    metric_type: MetricType
    formula: str  # Description of how it's calculated
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    target_col: Optional[str] = None
    higher_is_better: bool = True
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


@dataclass
class MetricValue:
    """Calculated metric value."""
    metric_name: str
    value: float
    formatted_value: str
    period: Optional[str] = None
    
    # Comparison
    vs_target: Optional[float] = None
    vs_previous: Optional[float] = None
    
    # Status
    status: str = "normal"  # normal, warning, critical, excellent


@dataclass
class MetricResult:
    """Complete metric calculation result."""
    n_metrics: int = 0
    
    # Current values
    metrics: Dict[str, MetricValue] = field(default_factory=dict)
    
    # Trend over time
    trends: Dict[str, List[MetricValue]] = field(default_factory=dict)
    
    # Summary
    excellent_metrics: List[str] = field(default_factory=list)
    warning_metrics: List[str] = field(default_factory=list)
    critical_metrics: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_metrics": self.n_metrics,
            "metrics": {
                name: {
                    "value": mv.value,
                    "formatted": mv.formatted_value,
                    "status": mv.status,
                    "vs_target": round(mv.vs_target, 2) if mv.vs_target else None,
                    "vs_previous": round(mv.vs_previous, 2) if mv.vs_previous else None
                }
                for name, mv in self.metrics.items()
            },
            "summary": {
                "excellent": self.excellent_metrics,
                "warning": self.warning_metrics,
                "critical": self.critical_metrics
            }
        }


# ============================================================================
# Common Business Metrics Library
# ============================================================================

class MetricLibrary:
    """Library of common business metrics."""
    
    @staticmethod
    def get_common_metrics() -> Dict[str, MetricDefinition]:
        """Get common business metrics."""
        return {
            "conversion_rate": MetricDefinition(
                name="Conversion Rate",
                metric_type=MetricType.RATIO,
                formula="Conversions / Total Visits * 100",
                higher_is_better=True
            ),
            "bounce_rate": MetricDefinition(
                name="Bounce Rate",
                metric_type=MetricType.RATIO,
                formula="Single Page Sessions / Total Sessions * 100",
                higher_is_better=False
            ),
            "average_order_value": MetricDefinition(
                name="Average Order Value",
                metric_type=MetricType.AVERAGE,
                formula="Total Revenue / Number of Orders",
                higher_is_better=True
            ),
            "customer_acquisition_cost": MetricDefinition(
                name="Customer Acquisition Cost",
                metric_type=MetricType.RATIO,
                formula="Total Marketing Spend / New Customers",
                higher_is_better=False
            ),
            "retention_rate": MetricDefinition(
                name="Retention Rate",
                metric_type=MetricType.RATIO,
                formula="(End Customers - New Customers) / Start Customers * 100",
                higher_is_better=True
            ),
            "churn_rate": MetricDefinition(
                name="Churn Rate",
                metric_type=MetricType.RATIO,
                formula="Lost Customers / Total Customers * 100",
                higher_is_better=False
            ),
            "gross_margin": MetricDefinition(
                name="Gross Margin",
                metric_type=MetricType.RATIO,
                formula="(Revenue - COGS) / Revenue * 100",
                higher_is_better=True
            ),
            "operating_margin": MetricDefinition(
                name="Operating Margin",
                metric_type=MetricType.RATIO,
                formula="Operating Income / Revenue * 100",
                higher_is_better=True
            ),
            "roi": MetricDefinition(
                name="Return on Investment",
                metric_type=MetricType.RATIO,
                formula="(Gain - Cost) / Cost * 100",
                higher_is_better=True
            ),
            "roas": MetricDefinition(
                name="Return on Ad Spend",
                metric_type=MetricType.RATIO,
                formula="Revenue from Ads / Ad Spend",
                higher_is_better=True
            )
        }


# ============================================================================
# Metric Calculator Engine
# ============================================================================

class MetricCalculatorEngine:
    """
    Production-grade Metric Calculator engine.
    
    Features:
    - Flexible metric definitions
    - Target comparison
    - Period-over-period change
    - Status assessment
    - Trend analysis
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.definitions: Dict[str, MetricDefinition] = {}
    
    def define_metric(
        self,
        name: str,
        metric_type: MetricType,
        formula: str = "",
        higher_is_better: bool = True,
        target: float = None,
        warning_threshold: float = None,
        critical_threshold: float = None
    ):
        """Define a custom metric."""
        self.definitions[name] = MetricDefinition(
            name=name,
            metric_type=metric_type,
            formula=formula,
            higher_is_better=higher_is_better,
            target_value=target,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold
        )
        return self
    
    def calculate_ratio(
        self,
        df: pd.DataFrame,
        metric_name: str,
        numerator_col: str,
        denominator_col: str,
        multiply_by: float = 100,
        definition: MetricDefinition = None
    ) -> MetricValue:
        """Calculate ratio metric."""
        numerator = df[numerator_col].sum()
        denominator = df[denominator_col].sum()
        
        value = (numerator / denominator * multiply_by) if denominator > 0 else 0
        
        return self._create_metric_value(
            metric_name, value, definition, is_percentage=(multiply_by == 100)
        )
    
    def calculate_average(
        self,
        df: pd.DataFrame,
        metric_name: str,
        value_col: str,
        weight_col: str = None,
        definition: MetricDefinition = None
    ) -> MetricValue:
        """Calculate average metric."""
        if weight_col:
            weights = df[weight_col]
            value = np.average(df[value_col], weights=weights)
        else:
            value = df[value_col].mean()
        
        return self._create_metric_value(metric_name, float(value), definition)
    
    def calculate_sum(
        self,
        df: pd.DataFrame,
        metric_name: str,
        value_col: str,
        definition: MetricDefinition = None
    ) -> MetricValue:
        """Calculate sum metric."""
        value = df[value_col].sum()
        return self._create_metric_value(metric_name, float(value), definition)
    
    def calculate_count(
        self,
        df: pd.DataFrame,
        metric_name: str,
        condition_col: str = None,
        condition_value: Any = None,
        definition: MetricDefinition = None
    ) -> MetricValue:
        """Calculate count metric."""
        if condition_col and condition_value is not None:
            value = (df[condition_col] == condition_value).sum()
        else:
            value = len(df)
        
        return self._create_metric_value(metric_name, float(value), definition)
    
    def calculate_growth(
        self,
        current: float,
        previous: float,
        metric_name: str = "Growth",
        definition: MetricDefinition = None
    ) -> MetricValue:
        """Calculate growth rate."""
        if previous == 0:
            value = 0 if current == 0 else 100
        else:
            value = (current - previous) / abs(previous) * 100
        
        mv = self._create_metric_value(metric_name, value, definition, is_percentage=True)
        mv.vs_previous = value
        return mv
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        metric_configs: List[Dict[str, Any]],
        date_col: str = None,
        compare_to_previous: bool = True
    ) -> MetricResult:
        """Calculate multiple metrics."""
        start_time = datetime.now()
        
        metrics = {}
        trends = {}
        
        for config in metric_configs:
            name = config['name']
            calc_type = config.get('type', 'sum')
            
            try:
                if calc_type == 'ratio':
                    mv = self.calculate_ratio(
                        df, name,
                        config['numerator'], config['denominator'],
                        config.get('multiply_by', 100)
                    )
                elif calc_type == 'average':
                    mv = self.calculate_average(
                        df, name, config['column'],
                        config.get('weight_column')
                    )
                elif calc_type == 'count':
                    mv = self.calculate_count(
                        df, name, config.get('condition_column'),
                        config.get('condition_value')
                    )
                else:  # sum
                    mv = self.calculate_sum(df, name, config['column'])
                
                metrics[name] = mv
                
                # Trend analysis if date column provided
                if date_col and date_col in df.columns:
                    trends[name] = self._calculate_trend(
                        df, date_col, config, calc_type
                    )
                
            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")
        
        # Categorize by status
        excellent = [n for n, m in metrics.items() if m.status == 'excellent']
        warning = [n for n, m in metrics.items() if m.status == 'warning']
        critical = [n for n, m in metrics.items() if m.status == 'critical']
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return MetricResult(
            n_metrics=len(metrics),
            metrics=metrics,
            trends=trends,
            excellent_metrics=excellent,
            warning_metrics=warning,
            critical_metrics=critical,
            processing_time_sec=processing_time
        )
    
    def _create_metric_value(
        self,
        name: str,
        value: float,
        definition: MetricDefinition = None,
        is_percentage: bool = False
    ) -> MetricValue:
        """Create metric value with formatting and status."""
        # Format
        if np.isnan(value) or np.isinf(value):
            value = 0
        
        if is_percentage:
            formatted = f"{value:.2f}%"
        elif value >= 1000000:
            formatted = f"{value / 1000000:.2f}M"
        elif value >= 1000:
            formatted = f"{value / 1000:.2f}K"
        else:
            formatted = f"{value:.2f}"
        
        # Status
        status = "normal"
        vs_target = None
        
        if definition and definition.target_value:
            vs_target = ((value - definition.target_value) / definition.target_value * 100)
            
            if definition.higher_is_better:
                if value >= definition.target_value:
                    status = "excellent"
                elif definition.warning_threshold and value < definition.warning_threshold:
                    status = "warning"
                elif definition.critical_threshold and value < definition.critical_threshold:
                    status = "critical"
            else:
                if value <= definition.target_value:
                    status = "excellent"
                elif definition.warning_threshold and value > definition.warning_threshold:
                    status = "warning"
                elif definition.critical_threshold and value > definition.critical_threshold:
                    status = "critical"
        
        return MetricValue(
            metric_name=name,
            value=value,
            formatted_value=formatted,
            vs_target=vs_target,
            status=status
        )
    
    def _calculate_trend(
        self,
        df: pd.DataFrame,
        date_col: str,
        config: Dict[str, Any],
        calc_type: str
    ) -> List[MetricValue]:
        """Calculate metric trend over time."""
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_work['period'] = df_work[date_col].dt.to_period('M').astype(str)
        
        trend = []
        for period in df_work['period'].unique():
            period_df = df_work[df_work['period'] == period]
            
            try:
                if calc_type == 'ratio':
                    mv = self.calculate_ratio(
                        period_df, config['name'],
                        config['numerator'], config['denominator']
                    )
                elif calc_type == 'average':
                    mv = self.calculate_average(
                        period_df, config['name'], config['column']
                    )
                else:
                    mv = self.calculate_sum(
                        period_df, config['name'], config['column']
                    )
                
                mv.period = period
                trend.append(mv)
            except:
                pass
        
        # Sort by period
        trend.sort(key=lambda x: x.period or '')
        return trend


# ============================================================================
# Factory Functions
# ============================================================================

def get_metric_engine() -> MetricCalculatorEngine:
    """Get metric calculator engine."""
    return MetricCalculatorEngine()


def quick_metrics(
    df: pd.DataFrame,
    configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Quick metric calculation."""
    engine = MetricCalculatorEngine(verbose=False)
    result = engine.calculate_all(df, configs)
    return result.to_dict()


def get_metric_library() -> Dict[str, MetricDefinition]:
    """Get common metrics library."""
    return MetricLibrary.get_common_metrics()
