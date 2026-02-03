# AI Enterprise Data Analyst - Pareto Analysis Engine
# Production-grade Pareto (80/20) analysis
# Handles: any categorical/numeric data, ABC classification

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

class ABCClass(str, Enum):
    """ABC classification."""
    A = "A"  # Top contributors (typically 80% of value)
    B = "B"  # Medium contributors
    C = "C"  # Low contributors


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ParetoItem:
    """Single item in Pareto analysis."""
    category: str
    value: float
    percentage: float
    cumulative_percentage: float
    abc_class: ABCClass
    rank: int


@dataclass
class ParetoResult:
    """Complete Pareto analysis result."""
    n_categories: int = 0
    total_value: float = 0.0
    
    # Pareto items
    items: List[ParetoItem] = field(default_factory=list)
    
    # 80/20 analysis
    categories_for_80pct: int = 0
    pct_categories_for_80pct: float = 0.0
    
    # ABC classification
    class_a_count: int = 0
    class_b_count: int = 0
    class_c_count: int = 0
    class_a_value_pct: float = 0.0
    class_b_value_pct: float = 0.0
    class_c_value_pct: float = 0.0
    
    # Pareto efficiency
    pareto_ratio: float = 0.0  # 80/20 ratio
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_categories": self.n_categories,
                "total_value": round(self.total_value, 2),
                "categories_for_80pct": self.categories_for_80pct,
                "pct_categories_for_80pct": round(self.pct_categories_for_80pct, 1),
                "pareto_ratio": round(self.pareto_ratio, 2)
            },
            "abc_classification": {
                "class_a": {"count": self.class_a_count, "value_pct": round(self.class_a_value_pct, 1)},
                "class_b": {"count": self.class_b_count, "value_pct": round(self.class_b_value_pct, 1)},
                "class_c": {"count": self.class_c_count, "value_pct": round(self.class_c_value_pct, 1)}
            },
            "top_items": [
                {
                    "category": item.category,
                    "value": round(item.value, 2),
                    "percentage": round(item.percentage, 2),
                    "cumulative_pct": round(item.cumulative_percentage, 2),
                    "class": item.abc_class.value
                }
                for item in self.items[:20]
            ]
        }


# ============================================================================
# Pareto Analysis Engine
# ============================================================================

class ParetoAnalysisEngine:
    """
    Pareto Analysis engine.
    
    Features:
    - 80/20 rule analysis
    - ABC classification
    - Cumulative contribution
    - Pareto efficiency ratio
    """
    
    def __init__(
        self,
        class_a_threshold: float = 0.80,
        class_b_threshold: float = 0.95,
        verbose: bool = True
    ):
        self.class_a_threshold = class_a_threshold
        self.class_b_threshold = class_b_threshold
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame = None,
        category_col: str = None,
        value_col: str = None,
        values: Dict[str, float] = None
    ) -> ParetoResult:
        """Perform Pareto analysis."""
        start_time = datetime.now()
        
        # Get data
        if values is not None:
            data = pd.Series(values)
        elif df is not None:
            if category_col is None or value_col is None:
                category_col, value_col = self._detect_columns(df)
            data = df.groupby(category_col)[value_col].sum()
        else:
            raise ValueError("Provide either DataFrame or values dict")
        
        if self.verbose:
            logger.info(f"Pareto analysis for {len(data)} categories")
        
        # Sort descending
        data = data.sort_values(ascending=False)
        total = data.sum()
        
        # Calculate items
        items = []
        cumulative = 0
        
        for rank, (category, value) in enumerate(data.items(), 1):
            pct = value / total * 100
            cumulative += pct
            
            # ABC classification
            if cumulative <= self.class_a_threshold * 100:
                abc_class = ABCClass.A
            elif cumulative <= self.class_b_threshold * 100:
                abc_class = ABCClass.B
            else:
                abc_class = ABCClass.C
            
            items.append(ParetoItem(
                category=str(category),
                value=float(value),
                percentage=pct,
                cumulative_percentage=cumulative,
                abc_class=abc_class,
                rank=rank
            ))
        
        # Calculate metrics
        categories_for_80 = sum(1 for item in items if item.cumulative_percentage <= 80)
        if categories_for_80 == 0:
            categories_for_80 = 1
        
        class_a_items = [i for i in items if i.abc_class == ABCClass.A]
        class_b_items = [i for i in items if i.abc_class == ABCClass.B]
        class_c_items = [i for i in items if i.abc_class == ABCClass.C]
        
        class_a_value = sum(i.value for i in class_a_items)
        class_b_value = sum(i.value for i in class_b_items)
        class_c_value = sum(i.value for i in class_c_items)
        
        # Pareto ratio: value % / category %
        pareto_ratio = 80 / (categories_for_80 / len(items) * 100) if items else 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ParetoResult(
            n_categories=len(items),
            total_value=total,
            items=items,
            categories_for_80pct=categories_for_80,
            pct_categories_for_80pct=categories_for_80 / len(items) * 100 if items else 0,
            class_a_count=len(class_a_items),
            class_b_count=len(class_b_items),
            class_c_count=len(class_c_items),
            class_a_value_pct=class_a_value / total * 100 if total > 0 else 0,
            class_b_value_pct=class_b_value / total * 100 if total > 0 else 0,
            class_c_value_pct=class_c_value / total * 100 if total > 0 else 0,
            pareto_ratio=pareto_ratio,
            processing_time_sec=processing_time
        )
    
    def _detect_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Auto-detect category and value columns."""
        # Category: first object/string column
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        category_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]
        
        # Value: first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        value_col = num_cols[0] if len(num_cols) > 0 else df.columns[1]
        
        return category_col, value_col


# ============================================================================
# Factory Functions
# ============================================================================

def get_pareto_engine(
    class_a_threshold: float = 0.80
) -> ParetoAnalysisEngine:
    """Get Pareto analysis engine."""
    return ParetoAnalysisEngine(class_a_threshold=class_a_threshold)


def quick_pareto(
    df: pd.DataFrame = None,
    values: Dict[str, float] = None
) -> Dict[str, Any]:
    """Quick Pareto analysis."""
    engine = ParetoAnalysisEngine(verbose=False)
    result = engine.analyze(df=df, values=values)
    return result.to_dict()
