# AI Enterprise Data Analyst - Automated Insights Engine
# Production-grade automatic insight generation
# Handles: any data, generates actionable insights

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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

class InsightType(str, Enum):
    """Types of insights."""
    TREND = "trend"
    OUTLIER = "outlier"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    SUMMARY = "summary"


class InsightPriority(str, Enum):
    """Insight priority."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Insight:
    """Single insight."""
    insight_type: InsightType
    priority: InsightPriority
    title: str
    description: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    recommendation: Optional[str] = None
    confidence: float = 0.0


@dataclass
class InsightsResult:
    """Complete insights result."""
    n_insights: int = 0
    insights: List[Insight] = field(default_factory=list)
    
    # Categorized
    high_priority: List[Insight] = field(default_factory=list)
    by_type: Dict[str, List[Insight]] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_insights": self.n_insights,
            "high_priority": [
                {
                    "type": i.insight_type.value,
                    "title": i.title,
                    "description": i.description,
                    "recommendation": i.recommendation
                }
                for i in self.high_priority[:5]
            ],
            "all_insights": [
                {
                    "type": i.insight_type.value,
                    "priority": i.priority.value,
                    "title": i.title,
                    "description": i.description
                }
                for i in self.insights[:20]
            ]
        }


# ============================================================================
# Automated Insights Engine
# ============================================================================

class AutomatedInsightsEngine:
    """
    Automated Insights engine.
    
    Features:
    - Trend detection
    - Outlier identification
    - Correlation insights
    - Distribution analysis
    - Pattern recognition
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def generate(self, df: pd.DataFrame) -> InsightsResult:
        """Generate insights from data."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Generating insights for {len(df)} rows, {len(df.columns)} columns")
        
        insights = []
        
        # Summary insights
        insights.extend(self._summary_insights(df))
        
        # Numeric column insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            insights.extend(self._numeric_insights(df, col))
        
        # Categorical insights
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            insights.extend(self._categorical_insights(df, col))
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            insights.extend(self._correlation_insights(df, numeric_cols))
        
        # Sort by priority
        priority_order = {InsightPriority.HIGH: 0, InsightPriority.MEDIUM: 1, InsightPriority.LOW: 2}
        insights.sort(key=lambda x: (priority_order[x.priority], -x.confidence))
        
        high_priority = [i for i in insights if i.priority == InsightPriority.HIGH]
        
        by_type = {}
        for i in insights:
            t = i.insight_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(i)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return InsightsResult(
            n_insights=len(insights),
            insights=insights,
            high_priority=high_priority,
            by_type=by_type,
            processing_time_sec=processing_time
        )
    
    def _summary_insights(self, df: pd.DataFrame) -> List[Insight]:
        """Generate summary insights."""
        insights = []
        
        # Missing data
        missing_pct = df.isna().sum().sum() / df.size * 100
        if missing_pct > 10:
            insights.append(Insight(
                insight_type=InsightType.SUMMARY,
                priority=InsightPriority.HIGH,
                title="High Missing Data Rate",
                description=f"{missing_pct:.1f}% of data is missing",
                metric_value=missing_pct,
                recommendation="Consider data imputation or investigate data collection issues",
                confidence=0.9
            ))
        
        # Dataset size
        if len(df) < 100:
            insights.append(Insight(
                insight_type=InsightType.SUMMARY,
                priority=InsightPriority.MEDIUM,
                title="Small Dataset",
                description=f"Only {len(df)} rows - statistical analyses may be less reliable",
                metric_value=len(df),
                recommendation="Collect more data for robust analysis",
                confidence=0.8
            ))
        
        return insights
    
    def _numeric_insights(self, df: pd.DataFrame, col: str) -> List[Insight]:
        """Generate insights for numeric column."""
        insights = []
        data = df[col].dropna()
        
        if len(data) < 5:
            return insights
        
        # Outliers
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_low = data < (q1 - 1.5 * iqr)
        outlier_high = data > (q3 + 1.5 * iqr)
        outlier_pct = (outlier_low | outlier_high).sum() / len(data) * 100
        
        if outlier_pct > 5:
            insights.append(Insight(
                insight_type=InsightType.OUTLIER,
                priority=InsightPriority.MEDIUM,
                title=f"Outliers in {col}",
                description=f"{outlier_pct:.1f}% of values are outliers",
                metric_name=col,
                metric_value=outlier_pct,
                recommendation="Investigate outlier causes or consider robust methods",
                confidence=0.85
            ))
        
        # Skewness
        skewness = scipy_stats.skew(data)
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insights.append(Insight(
                insight_type=InsightType.DISTRIBUTION,
                priority=InsightPriority.LOW,
                title=f"Skewed Distribution in {col}",
                description=f"Data is skewed to the {direction} (skewness: {skewness:.2f})",
                metric_name=col,
                metric_value=skewness,
                recommendation="Consider log transformation for analysis",
                confidence=0.8
            ))
        
        # Zeros
        zero_pct = (data == 0).sum() / len(data) * 100
        if zero_pct > 20:
            insights.append(Insight(
                insight_type=InsightType.PATTERN,
                priority=InsightPriority.MEDIUM,
                title=f"High Zero Rate in {col}",
                description=f"{zero_pct:.1f}% of values are zero",
                metric_name=col,
                metric_value=zero_pct,
                recommendation="Investigate if zeros are valid or missing values",
                confidence=0.75
            ))
        
        return insights
    
    def _categorical_insights(self, df: pd.DataFrame, col: str) -> List[Insight]:
        """Generate insights for categorical column."""
        insights = []
        data = df[col].dropna()
        
        if len(data) == 0:
            return insights
        
        value_counts = data.value_counts()
        unique_ratio = len(value_counts) / len(data)
        
        # High cardinality
        if unique_ratio > 0.5 and len(value_counts) > 50:
            insights.append(Insight(
                insight_type=InsightType.PATTERN,
                priority=InsightPriority.LOW,
                title=f"High Cardinality in {col}",
                description=f"{len(value_counts)} unique values - may need grouping",
                metric_name=col,
                metric_value=len(value_counts),
                recommendation="Consider grouping rare categories",
                confidence=0.7
            ))
        
        # Dominant value
        top_pct = value_counts.iloc[0] / len(data) * 100
        if top_pct > 80:
            insights.append(Insight(
                insight_type=InsightType.PATTERN,
                priority=InsightPriority.MEDIUM,
                title=f"Dominant Value in {col}",
                description=f"'{value_counts.index[0]}' represents {top_pct:.1f}% of data",
                metric_name=col,
                metric_value=top_pct,
                recommendation="Limited predictive value for this feature",
                confidence=0.85
            ))
        
        return insights
    
    def _correlation_insights(
        self,
        df: pd.DataFrame,
        numeric_cols: pd.Index
    ) -> List[Insight]:
        """Generate correlation insights."""
        insights = []
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                
                if abs(corr) > 0.7:
                    direction = "positive" if corr > 0 else "negative"
                    insights.append(Insight(
                        insight_type=InsightType.CORRELATION,
                        priority=InsightPriority.HIGH if abs(corr) > 0.85 else InsightPriority.MEDIUM,
                        title=f"Strong {direction.title()} Correlation",
                        description=f"{col1} and {col2} are strongly {direction}ly correlated (r={corr:.2f})",
                        metric_value=corr,
                        recommendation="May indicate multicollinearity - consider removing one feature",
                        confidence=abs(corr)
                    ))
        
        return insights


# ============================================================================
# Factory Functions
# ============================================================================

def get_insights_engine() -> AutomatedInsightsEngine:
    """Get automated insights engine."""
    return AutomatedInsightsEngine()


def quick_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick automated insights."""
    engine = AutomatedInsightsEngine(verbose=False)
    result = engine.generate(df)
    return result.to_dict()
