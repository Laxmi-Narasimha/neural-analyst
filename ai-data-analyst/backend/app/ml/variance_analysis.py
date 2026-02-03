# AI Enterprise Data Analyst - Variance Analysis Engine
# Production-grade budget vs actual analysis
# Handles: any financial data, variance calculations

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

class VarianceType(str, Enum):
    """Type of variance."""
    FAVORABLE = "favorable"  # Better than budget
    UNFAVORABLE = "unfavorable"  # Worse than budget
    NEUTRAL = "neutral"


class VarianceCategory(str, Enum):
    """Variance category."""
    VOLUME = "volume"  # Due to quantity
    PRICE = "price"  # Due to price
    MIX = "mix"  # Due to product mix
    EFFICIENCY = "efficiency"  # Due to usage
    TOTAL = "total"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VarianceItem:
    """Single variance item."""
    category: str
    budget: float
    actual: float
    variance_amount: float
    variance_pct: float
    variance_type: VarianceType


@dataclass
class VarianceResult:
    """Complete variance analysis result."""
    n_items: int = 0
    total_budget: float = 0.0
    total_actual: float = 0.0
    total_variance: float = 0.0
    total_variance_pct: float = 0.0
    
    # Items
    items: List[VarianceItem] = field(default_factory=list)
    
    # Summary
    favorable_count: int = 0
    unfavorable_count: int = 0
    largest_favorable: Optional[VarianceItem] = None
    largest_unfavorable: Optional[VarianceItem] = None
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_budget": round(self.total_budget, 2),
                "total_actual": round(self.total_actual, 2),
                "total_variance": round(self.total_variance, 2),
                "total_variance_pct": round(self.total_variance_pct, 2),
                "favorable_count": self.favorable_count,
                "unfavorable_count": self.unfavorable_count
            },
            "items": [
                {
                    "category": v.category,
                    "budget": round(v.budget, 2),
                    "actual": round(v.actual, 2),
                    "variance": round(v.variance_amount, 2),
                    "variance_pct": round(v.variance_pct, 2),
                    "type": v.variance_type.value
                }
                for v in sorted(self.items, key=lambda x: abs(x.variance_amount), reverse=True)[:20]
            ]
        }


# ============================================================================
# Variance Analysis Engine
# ============================================================================

class VarianceAnalysisEngine:
    """
    Variance Analysis engine.
    
    Features:
    - Budget vs actual comparison
    - Favorable/unfavorable classification
    - Percentage and absolute variance
    - Multi-level analysis
    """
    
    def __init__(self, is_revenue: bool = True, verbose: bool = True):
        self.is_revenue = is_revenue  # True if higher actual is favorable
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame = None,
        category_col: str = None,
        budget_col: str = None,
        actual_col: str = None,
        budget_values: Dict[str, float] = None,
        actual_values: Dict[str, float] = None
    ) -> VarianceResult:
        """Perform variance analysis."""
        start_time = datetime.now()
        
        # Get data
        if budget_values and actual_values:
            categories = set(budget_values.keys()) | set(actual_values.keys())
            data = []
            for cat in categories:
                data.append({
                    'category': cat,
                    'budget': budget_values.get(cat, 0),
                    'actual': actual_values.get(cat, 0)
                })
            df = pd.DataFrame(data)
            category_col, budget_col, actual_col = 'category', 'budget', 'actual'
        elif df is not None:
            if category_col is None or budget_col is None or actual_col is None:
                category_col, budget_col, actual_col = self._detect_columns(df)
        else:
            raise ValueError("Provide DataFrame or budget/actual values")
        
        if self.verbose:
            logger.info(f"Variance analysis for {len(df)} items")
        
        items = []
        
        for _, row in df.iterrows():
            budget = float(row[budget_col])
            actual = float(row[actual_col])
            variance = actual - budget
            
            if budget != 0:
                variance_pct = variance / abs(budget) * 100
            else:
                variance_pct = 100 if actual > 0 else 0
            
            # Determine if favorable
            if self.is_revenue:
                variance_type = VarianceType.FAVORABLE if variance >= 0 else VarianceType.UNFAVORABLE
            else:  # For expenses, negative variance is favorable
                variance_type = VarianceType.FAVORABLE if variance <= 0 else VarianceType.UNFAVORABLE
            
            items.append(VarianceItem(
                category=str(row[category_col]),
                budget=budget,
                actual=actual,
                variance_amount=variance,
                variance_pct=variance_pct,
                variance_type=variance_type
            ))
        
        # Totals
        total_budget = sum(i.budget for i in items)
        total_actual = sum(i.actual for i in items)
        total_variance = total_actual - total_budget
        total_variance_pct = total_variance / abs(total_budget) * 100 if total_budget != 0 else 0
        
        favorable = [i for i in items if i.variance_type == VarianceType.FAVORABLE]
        unfavorable = [i for i in items if i.variance_type == VarianceType.UNFAVORABLE]
        
        largest_fav = max(favorable, key=lambda x: abs(x.variance_amount)) if favorable else None
        largest_unfav = max(unfavorable, key=lambda x: abs(x.variance_amount)) if unfavorable else None
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return VarianceResult(
            n_items=len(items),
            total_budget=total_budget,
            total_actual=total_actual,
            total_variance=total_variance,
            total_variance_pct=total_variance_pct,
            items=items,
            favorable_count=len(favorable),
            unfavorable_count=len(unfavorable),
            largest_favorable=largest_fav,
            largest_unfavorable=largest_unfav,
            processing_time_sec=processing_time
        )
    
    def _detect_columns(self, df: pd.DataFrame):
        """Auto-detect columns."""
        cols = df.columns.tolist()
        
        cat_col = cols[0]
        for c in cols:
            if df[c].dtype == 'object':
                cat_col = c
                break
        
        budget_col = None
        actual_col = None
        
        for c in cols:
            cl = c.lower()
            if 'budget' in cl or 'plan' in cl or 'target' in cl:
                budget_col = c
            elif 'actual' in cl or 'real' in cl:
                actual_col = c
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if budget_col is None and len(num_cols) >= 1:
            budget_col = num_cols[0]
        if actual_col is None and len(num_cols) >= 2:
            actual_col = num_cols[1]
        
        return cat_col, budget_col or num_cols[0], actual_col or num_cols[-1]


# ============================================================================
# Factory Functions
# ============================================================================

def get_variance_engine(is_revenue: bool = True) -> VarianceAnalysisEngine:
    """Get variance analysis engine."""
    return VarianceAnalysisEngine(is_revenue=is_revenue)


def quick_variance(
    budget: Dict[str, float],
    actual: Dict[str, float],
    is_revenue: bool = True
) -> Dict[str, Any]:
    """Quick variance analysis."""
    engine = VarianceAnalysisEngine(is_revenue=is_revenue, verbose=False)
    result = engine.analyze(budget_values=budget, actual_values=actual)
    return result.to_dict()
