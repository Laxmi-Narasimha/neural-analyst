# AI Enterprise Data Analyst - Automated Insights Engine
# Generate actionable insights from data automatically

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Insight Types
# ============================================================================

class InsightType(str, Enum):
    """Types of automated insights."""
    ANOMALY = "anomaly"
    TREND = "trend"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    OUTLIER = "outlier"
    SEASONALITY = "seasonality"
    COMPARISON = "comparison"
    CHANGE_POINT = "change_point"
    SEGMENT = "segment"
    PREDICTION = "prediction"


class InsightPriority(str, Enum):
    """Insight priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """Single data insight."""
    
    insight_type: InsightType
    priority: InsightPriority
    
    title: str
    description: str
    
    # Evidence
    metric: str = ""
    value: Any = None
    expected_value: Any = None
    deviation: float = 0.0
    
    # Affected data
    columns: list[str] = field(default_factory=list)
    affected_rows: int = 0
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.insight_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "metric": self.metric,
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation": round(self.deviation, 2) if self.deviation else None,
            "columns": self.columns,
            "affected_rows": self.affected_rows,
            "recommendations": self.recommendations,
            "confidence": round(self.confidence, 2)
        }


@dataclass
class InsightReport:
    """Collection of insights."""
    
    insights: list[Insight]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.insights if i.priority == InsightPriority.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.insights if i.priority == InsightPriority.HIGH)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_insights": len(self.insights),
            "critical": self.critical_count,
            "high": self.high_count,
            "generated_at": self.generated_at.isoformat(),
            "insights": [i.to_dict() for i in sorted(
                self.insights,
                key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.priority.value]
            )]
        }


# ============================================================================
# Insight Detectors
# ============================================================================

class AnomalyInsightDetector:
    """Detect anomaly-based insights."""
    
    def detect(self, df: pd.DataFrame) -> list[Insight]:
        """Detect anomalies in numeric columns."""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # Z-score based anomalies
            mean = series.mean()
            std = series.std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((series - mean) / std)
            anomalies = z_scores > 3
            n_anomalies = anomalies.sum()
            
            if n_anomalies > 0 and n_anomalies < len(series) * 0.1:
                max_anomaly = series[anomalies].max()
                
                insights.append(Insight(
                    insight_type=InsightType.ANOMALY,
                    priority=InsightPriority.HIGH if n_anomalies > 5 else InsightPriority.MEDIUM,
                    title=f"Anomalies detected in {col}",
                    description=f"Found {n_anomalies} unusual values in column '{col}' that deviate significantly from the mean ({mean:.2f}).",
                    metric=col,
                    value=float(max_anomaly),
                    expected_value=float(mean),
                    deviation=float((max_anomaly - mean) / std),
                    columns=[col],
                    affected_rows=int(n_anomalies),
                    recommendations=[
                        f"Investigate the {n_anomalies} anomalous records",
                        "Consider whether these are data errors or genuine outliers",
                        "Apply outlier treatment if needed for analysis"
                    ],
                    confidence=0.85
                ))
        
        return insights


class TrendInsightDetector:
    """Detect trend-based insights."""
    
    def detect(
        self,
        df: pd.DataFrame,
        date_col: str = None,
        value_col: str = None
    ) -> list[Insight]:
        """Detect trends in time series data."""
        insights = []
        
        # Try to find date column
        if date_col is None:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
            else:
                return insights
        
        if date_col not in df.columns:
            return insights
        
        # Analyze each numeric column
        numeric_cols = [value_col] if value_col else df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            try:
                sorted_df = df.sort_values(date_col)
                series = sorted_df[col].dropna()
                
                if len(series) < 10:
                    continue
                
                # Calculate trend
                x = np.arange(len(series))
                slope, intercept = np.polyfit(x, series.values, 1)
                
                # Determine trend direction and significance
                pct_change = slope * len(series) / series.mean() * 100 if series.mean() != 0 else 0
                
                if abs(pct_change) > 10:
                    direction = "increasing" if slope > 0 else "decreasing"
                    priority = InsightPriority.HIGH if abs(pct_change) > 25 else InsightPriority.MEDIUM
                    
                    insights.append(Insight(
                        insight_type=InsightType.TREND,
                        priority=priority,
                        title=f"{col} is {direction}",
                        description=f"The '{col}' metric shows a {direction} trend, changing by approximately {abs(pct_change):.1f}% over the observed period.",
                        metric=col,
                        value=float(series.iloc[-1]),
                        expected_value=float(series.iloc[0]),
                        deviation=pct_change,
                        columns=[col, date_col],
                        recommendations=[
                            f"Monitor {col} closely",
                            f"Investigate factors driving the {direction} trend",
                            "Consider setting alerts for threshold breaches"
                        ],
                        confidence=0.8
                    ))
            
            except Exception as e:
                logger.warning(f"Trend detection error for {col}: {e}")
        
        return insights


class CorrelationInsightDetector:
    """Detect correlation-based insights."""
    
    def detect(self, df: pd.DataFrame, threshold: float = 0.7) -> list[Insight]:
        """Detect strong correlations between columns."""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return insights
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr = corr_matrix.loc[col1, col2]
                
                if abs(corr) >= threshold:
                    direction = "positive" if corr > 0 else "negative"
                    
                    insights.append(Insight(
                        insight_type=InsightType.CORRELATION,
                        priority=InsightPriority.MEDIUM,
                        title=f"Strong {direction} correlation: {col1} ↔ {col2}",
                        description=f"'{col1}' and '{col2}' have a strong {direction} correlation ({corr:.2f}). Changes in one tend to be associated with {'similar' if corr > 0 else 'opposite'} changes in the other.",
                        metric=f"{col1}_vs_{col2}",
                        value=float(corr),
                        columns=[col1, col2],
                        recommendations=[
                            "Consider if this relationship is causal or coincidental",
                            "One variable may be useful for predicting the other",
                            "Check for multicollinearity if using both in models"
                        ],
                        confidence=0.9
                    ))
        
        return insights


class DistributionInsightDetector:
    """Detect distribution-based insights."""
    
    def detect(self, df: pd.DataFrame) -> list[Insight]:
        """Detect distribution anomalies."""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 20:
                continue
            
            # Check for skewness
            try:
                from scipy.stats import skew, kurtosis
                
                sk = skew(series)
                kurt = kurtosis(series)
                
                if abs(sk) > 2:
                    direction = "right (positive)" if sk > 0 else "left (negative)"
                    
                    insights.append(Insight(
                        insight_type=InsightType.DISTRIBUTION,
                        priority=InsightPriority.LOW,
                        title=f"{col} is highly skewed",
                        description=f"The distribution of '{col}' is highly skewed to the {direction} (skewness: {sk:.2f}). This may affect statistical analyses.",
                        metric=col,
                        value=float(sk),
                        columns=[col],
                        recommendations=[
                            "Consider log transformation for normalization",
                            "Use median instead of mean for central tendency",
                            "Non-parametric tests may be more appropriate"
                        ],
                        confidence=0.8
                    ))
                
                if kurt > 3:
                    insights.append(Insight(
                        insight_type=InsightType.DISTRIBUTION,
                        priority=InsightPriority.LOW,
                        title=f"{col} has heavy tails",
                        description=f"'{col}' has a leptokurtic distribution (kurtosis: {kurt:.2f}) with heavier tails than normal, indicating more extreme values.",
                        metric=col,
                        value=float(kurt),
                        columns=[col],
                        recommendations=[
                            "Investigate extreme values in the tails",
                            "Consider robust statistics",
                            "Standard deviation may overstate variability"
                        ],
                        confidence=0.75
                    ))
            
            except Exception:
                pass
        
        return insights


class MissingDataInsightDetector:
    """Detect missing data patterns."""
    
    def detect(self, df: pd.DataFrame) -> list[Insight]:
        """Detect missing data issues."""
        insights = []
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
        
        # High missing data
        high_missing = missing_pct[missing_pct > 20]
        
        for col, pct in high_missing.items():
            priority = InsightPriority.CRITICAL if pct > 50 else InsightPriority.HIGH
            
            insights.append(Insight(
                insight_type=InsightType.ANOMALY,
                priority=priority,
                title=f"High missing data in {col}",
                description=f"Column '{col}' has {pct:.1f}% missing values ({int(missing[col])} out of {len(df)} rows).",
                metric=col,
                value=float(pct),
                columns=[col],
                affected_rows=int(missing[col]),
                recommendations=[
                    "Investigate why data is missing",
                    "Consider imputation strategies",
                    f"May need to drop column if {pct:.0f}% > 50%"
                ],
                confidence=1.0
            ))
        
        return insights


# ============================================================================
# Insight Engine
# ============================================================================

class InsightEngine:
    """
    Automated insight generation engine.
    
    Features:
    - Anomaly detection
    - Trend analysis
    - Correlation discovery
    - Distribution analysis
    - Missing data patterns
    - Prioritized recommendations
    """
    
    def __init__(self):
        self.detectors = {
            "anomaly": AnomalyInsightDetector(),
            "trend": TrendInsightDetector(),
            "correlation": CorrelationInsightDetector(),
            "distribution": DistributionInsightDetector(),
            "missing": MissingDataInsightDetector()
        }
    
    def analyze(
        self,
        df: pd.DataFrame,
        date_col: str = None
    ) -> InsightReport:
        """Generate comprehensive insights from data."""
        all_insights = []
        
        # Run all detectors
        all_insights.extend(self.detectors["anomaly"].detect(df))
        all_insights.extend(self.detectors["trend"].detect(df, date_col))
        all_insights.extend(self.detectors["correlation"].detect(df))
        all_insights.extend(self.detectors["distribution"].detect(df))
        all_insights.extend(self.detectors["missing"].detect(df))
        
        return InsightReport(insights=all_insights)
    
    def get_top_insights(
        self,
        df: pd.DataFrame,
        n: int = 5
    ) -> list[Insight]:
        """Get top N priority insights."""
        report = self.analyze(df)
        return report.insights[:n]
    
    def summarize(self, report: InsightReport) -> str:
        """Generate executive summary of insights."""
        critical = report.critical_count
        high = report.high_count
        total = len(report.insights)
        
        summary = f"Analysis found {total} insights"
        
        if critical > 0:
            summary += f", including {critical} CRITICAL issues requiring immediate attention"
        
        if high > 0:
            summary += f" and {high} HIGH priority items"
        
        summary += "."
        
        # Top insight
        if report.insights:
            top = report.insights[0]
            summary += f"\n\nTop Insight: {top.title}\n{top.description}"
        
        return summary


# Factory function
def get_insight_engine() -> InsightEngine:
    """Get insight engine instance."""
    return InsightEngine()
