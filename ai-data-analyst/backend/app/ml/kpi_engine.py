# AI Enterprise Data Analyst - KPI Engine
# Production-grade KPI calculations and tracking
# Handles: custom KPI definitions, targets, thresholds

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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

class KPIStatus(str, Enum):
    """KPI status relative to target."""
    EXCEEDING = "exceeding"
    ON_TARGET = "on_target"
    BELOW_TARGET = "below_target"
    CRITICAL = "critical"


class KPITrend(str, Enum):
    """KPI trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class KPIDefinition:
    """Definition of a KPI."""
    name: str
    formula: str  # Column name or calculation
    target: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    higher_is_better: bool = True
    unit: str = ""
    description: str = ""


@dataclass
class KPIValue:
    """Calculated KPI value."""
    name: str
    value: float
    target: Optional[float]
    variance: float
    variance_pct: float
    status: KPIStatus
    trend: Optional[KPITrend]
    previous_value: Optional[float]
    unit: str


@dataclass
class KPIResult:
    """Complete KPI analysis result."""
    kpis: List[KPIValue] = field(default_factory=list)
    
    # Summary
    total_kpis: int = 0
    exceeding_count: int = 0
    on_target_count: int = 0
    below_target_count: int = 0
    critical_count: int = 0
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_kpis": self.total_kpis,
                "exceeding": self.exceeding_count,
                "on_target": self.on_target_count,
                "below_target": self.below_target_count,
                "critical": self.critical_count
            },
            "kpis": [
                {
                    "name": k.name,
                    "value": round(k.value, 2),
                    "target": round(k.target, 2) if k.target else None,
                    "variance": round(k.variance, 2),
                    "variance_pct": round(k.variance_pct, 1),
                    "status": k.status.value,
                    "trend": k.trend.value if k.trend else None,
                    "unit": k.unit
                }
                for k in self.kpis
            ]
        }


# ============================================================================
# Built-in KPI Calculations
# ============================================================================

class BuiltInKPIs:
    """Common business KPI calculations."""
    
    @staticmethod
    def revenue_growth(current: float, previous: float) -> float:
        if previous == 0:
            return 0
        return (current - previous) / previous * 100
    
    @staticmethod
    def conversion_rate(conversions: int, visitors: int) -> float:
        if visitors == 0:
            return 0
        return conversions / visitors * 100
    
    @staticmethod
    def average_order_value(revenue: float, orders: int) -> float:
        if orders == 0:
            return 0
        return revenue / orders
    
    @staticmethod
    def customer_acquisition_cost(marketing_cost: float, new_customers: int) -> float:
        if new_customers == 0:
            return 0
        return marketing_cost / new_customers
    
    @staticmethod
    def gross_margin(revenue: float, cogs: float) -> float:
        if revenue == 0:
            return 0
        return (revenue - cogs) / revenue * 100
    
    @staticmethod
    def churn_rate(lost_customers: int, start_customers: int) -> float:
        if start_customers == 0:
            return 0
        return lost_customers / start_customers * 100


# ============================================================================
# KPI Engine
# ============================================================================

class KPIEngine:
    """
    KPI Calculation engine.
    
    Features:
    - Custom KPI definitions
    - Target tracking
    - Status and trend analysis
    - Threshold alerts
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.definitions: Dict[str, KPIDefinition] = {}
        self.builtin = BuiltInKPIs()
    
    def register_kpi(self, definition: KPIDefinition):
        """Register a KPI definition."""
        self.definitions[definition.name] = definition
    
    def calculate(
        self,
        df: pd.DataFrame,
        previous_df: pd.DataFrame = None
    ) -> KPIResult:
        """Calculate all registered KPIs."""
        start_time = datetime.now()
        
        kpis = []
        
        for name, defn in self.definitions.items():
            kpi_value = self._calculate_single(df, defn, previous_df)
            kpis.append(kpi_value)
        
        # Summary
        exceeding = sum(1 for k in kpis if k.status == KPIStatus.EXCEEDING)
        on_target = sum(1 for k in kpis if k.status == KPIStatus.ON_TARGET)
        below = sum(1 for k in kpis if k.status == KPIStatus.BELOW_TARGET)
        critical = sum(1 for k in kpis if k.status == KPIStatus.CRITICAL)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return KPIResult(
            kpis=kpis,
            total_kpis=len(kpis),
            exceeding_count=exceeding,
            on_target_count=on_target,
            below_target_count=below,
            critical_count=critical,
            processing_time_sec=processing_time
        )
    
    def calculate_from_dict(
        self,
        values: Dict[str, float],
        targets: Dict[str, float] = None,
        previous_values: Dict[str, float] = None
    ) -> KPIResult:
        """Calculate KPIs from dictionaries."""
        start_time = datetime.now()
        
        targets = targets or {}
        previous_values = previous_values or {}
        
        kpis = []
        
        for name, value in values.items():
            target = targets.get(name)
            prev = previous_values.get(name)
            
            variance = value - target if target else 0
            variance_pct = (variance / abs(target) * 100) if target and target != 0 else 0
            
            status = self._determine_status(value, target, True)
            trend = self._determine_trend(value, prev, True) if prev else None
            
            kpis.append(KPIValue(
                name=name,
                value=value,
                target=target,
                variance=variance,
                variance_pct=variance_pct,
                status=status,
                trend=trend,
                previous_value=prev,
                unit=""
            ))
        
        exceeding = sum(1 for k in kpis if k.status == KPIStatus.EXCEEDING)
        on_target = sum(1 for k in kpis if k.status == KPIStatus.ON_TARGET)
        below = sum(1 for k in kpis if k.status == KPIStatus.BELOW_TARGET)
        critical = sum(1 for k in kpis if k.status == KPIStatus.CRITICAL)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return KPIResult(
            kpis=kpis,
            total_kpis=len(kpis),
            exceeding_count=exceeding,
            on_target_count=on_target,
            below_target_count=below,
            critical_count=critical,
            processing_time_sec=processing_time
        )
    
    def _calculate_single(
        self,
        df: pd.DataFrame,
        defn: KPIDefinition,
        previous_df: pd.DataFrame = None
    ) -> KPIValue:
        """Calculate a single KPI."""
        # Get current value
        if defn.formula in df.columns:
            value = float(df[defn.formula].sum())
        else:
            # Try to evaluate as expression
            try:
                value = float(eval(defn.formula, {"df": df, "np": np, "pd": pd}))
            except:
                value = 0.0
        
        # Get previous value
        prev_value = None
        if previous_df is not None:
            if defn.formula in previous_df.columns:
                prev_value = float(previous_df[defn.formula].sum())
        
        # Calculate variance
        variance = value - defn.target if defn.target else 0
        variance_pct = (variance / abs(defn.target) * 100) if defn.target and defn.target != 0 else 0
        
        # Determine status
        status = self._determine_status(value, defn.target, defn.higher_is_better, 
                                         defn.warning_threshold, defn.critical_threshold)
        
        # Determine trend
        trend = self._determine_trend(value, prev_value, defn.higher_is_better) if prev_value else None
        
        return KPIValue(
            name=defn.name,
            value=value,
            target=defn.target,
            variance=variance,
            variance_pct=variance_pct,
            status=status,
            trend=trend,
            previous_value=prev_value,
            unit=defn.unit
        )
    
    def _determine_status(
        self,
        value: float,
        target: Optional[float],
        higher_is_better: bool,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None
    ) -> KPIStatus:
        """Determine KPI status."""
        if target is None:
            return KPIStatus.ON_TARGET
        
        if higher_is_better:
            if value >= target:
                return KPIStatus.EXCEEDING
            elif critical_threshold and value < critical_threshold:
                return KPIStatus.CRITICAL
            elif warning_threshold and value < warning_threshold:
                return KPIStatus.BELOW_TARGET
            elif value >= target * 0.95:
                return KPIStatus.ON_TARGET
            else:
                return KPIStatus.BELOW_TARGET
        else:
            if value <= target:
                return KPIStatus.EXCEEDING
            elif critical_threshold and value > critical_threshold:
                return KPIStatus.CRITICAL
            elif warning_threshold and value > warning_threshold:
                return KPIStatus.BELOW_TARGET
            elif value <= target * 1.05:
                return KPIStatus.ON_TARGET
            else:
                return KPIStatus.BELOW_TARGET
    
    def _determine_trend(
        self,
        current: float,
        previous: float,
        higher_is_better: bool
    ) -> KPITrend:
        """Determine KPI trend."""
        if previous == 0:
            return KPITrend.STABLE
        
        change_pct = (current - previous) / abs(previous) * 100
        
        if abs(change_pct) < 2:
            return KPITrend.STABLE
        
        if higher_is_better:
            return KPITrend.IMPROVING if change_pct > 0 else KPITrend.DECLINING
        else:
            return KPITrend.IMPROVING if change_pct < 0 else KPITrend.DECLINING


# ============================================================================
# Factory Functions
# ============================================================================

def get_kpi_engine() -> KPIEngine:
    """Get KPI engine."""
    return KPIEngine()


def quick_kpis(
    values: Dict[str, float],
    targets: Dict[str, float] = None
) -> Dict[str, Any]:
    """Quick KPI calculation."""
    engine = KPIEngine(verbose=False)
    result = engine.calculate_from_dict(values, targets)
    return result.to_dict()
