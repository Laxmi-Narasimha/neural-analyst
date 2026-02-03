# AI Enterprise Data Analyst - Trend Analysis Engine
# Production-grade trend detection and analysis
# Handles: any time series data, multiple trend detection methods

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums and Types
# ============================================================================

class TrendDirection(str, Enum):
    """Trend direction."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class TrendStrength(str, Enum):
    """Trend strength."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrendConfig:
    """Configuration for trend analysis."""
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    
    # Smoothing
    smoothing_window: int = 7
    
    # Change point detection
    min_segment_length: int = 10


@dataclass
class TrendInfo:
    """Trend information."""
    direction: TrendDirection
    strength: TrendStrength
    slope: float
    r_squared: float
    p_value: float
    percentage_change: float
    start_value: float
    end_value: float


@dataclass
class ChangePoint:
    """Change point in time series."""
    index: int
    date: Optional[datetime]
    value: float
    direction_before: TrendDirection
    direction_after: TrendDirection
    magnitude: float


@dataclass
class TrendResult:
    """Complete trend analysis result."""
    n_observations: int = 0
    date_range: Tuple[datetime, datetime] = (None, None)
    
    # Overall trend
    overall_trend: TrendInfo = None
    
    # Segments
    segments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Change points
    change_points: List[ChangePoint] = field(default_factory=list)
    
    # Supporting data
    smoothed_values: List[float] = field(default_factory=list)
    trend_line: List[float] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_observations": self.n_observations,
                "date_range": [d.isoformat() if d else None for d in self.date_range]
            },
            "overall_trend": {
                "direction": self.overall_trend.direction.value,
                "strength": self.overall_trend.strength.value,
                "slope": round(self.overall_trend.slope, 4),
                "r_squared": round(self.overall_trend.r_squared, 4),
                "percentage_change": round(self.overall_trend.percentage_change, 2)
            } if self.overall_trend else {},
            "change_points": [
                {
                    "index": cp.index,
                    "date": cp.date.isoformat() if cp.date else None,
                    "direction_change": f"{cp.direction_before.value} → {cp.direction_after.value}",
                    "magnitude": round(cp.magnitude, 2)
                }
                for cp in self.change_points[:10]
            ],
            "segments": self.segments[:10]
        }


# ============================================================================
# Trend Analysis Engine
# ============================================================================

class TrendAnalysisEngine:
    """
    Complete Trend Analysis engine.
    
    Features:
    - Linear trend detection
    - Non-parametric trend testing
    - Change point detection
    - Trend segmentation
    """
    
    def __init__(self, config: TrendConfig = None, verbose: bool = True):
        self.config = config or TrendConfig()
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        date_col: str = None,
        value_col: str = None
    ) -> TrendResult:
        """Perform trend analysis."""
        start_time = datetime.now()
        
        # Auto-detect columns
        date_col = date_col or self.config.date_col or self._detect_date_col(df)
        value_col = value_col or self.config.value_col or self._detect_value_col(df)
        
        if self.verbose:
            logger.info(f"Trend analysis: date={date_col}, value={value_col}")
        
        # Prepare data
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
        
        dates = df[date_col].values
        values = df[value_col].values.astype(float)
        
        if len(values) < 3:
            raise DataProcessingException("Need at least 3 data points for trend analysis")
        
        # Smooth values
        smoothed = self._smooth_values(values)
        
        # Calculate overall trend
        overall_trend = self._calculate_trend(values)
        
        # Detect change points
        change_points = self._detect_change_points(values, dates)
        
        # Calculate trend line
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        trend_line = slope * x + intercept
        
        # Segments
        segments = self._calculate_segments(values, dates, change_points)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TrendResult(
            n_observations=len(values),
            date_range=(pd.Timestamp(dates[0]).to_pydatetime(), pd.Timestamp(dates[-1]).to_pydatetime()),
            overall_trend=overall_trend,
            segments=segments,
            change_points=change_points,
            smoothed_values=smoothed.tolist(),
            trend_line=trend_line.tolist(),
            processing_time_sec=processing_time
        )
    
    def _calculate_trend(self, values: np.ndarray) -> TrendInfo:
        """Calculate trend from values."""
        n = len(values)
        x = np.arange(n)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, values)
        r_squared = r_value ** 2
        
        # Mann-Kendall test for trend significance
        try:
            tau, mk_p = scipy_stats.kendalltau(x, values)
        except:
            tau, mk_p = 0, 1.0
        
        # Determine direction
        if mk_p > 0.05:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Calculate percentage change
        start_val = values[0]
        end_val = values[-1]
        pct_change = (end_val - start_val) / abs(start_val) * 100 if start_val != 0 else 0
        
        # Determine strength
        strength = self._get_trend_strength(r_squared, abs(tau))
        
        return TrendInfo(
            direction=direction,
            strength=strength,
            slope=slope,
            r_squared=r_squared,
            p_value=mk_p,
            percentage_change=pct_change,
            start_value=start_val,
            end_value=end_val
        )
    
    def _get_trend_strength(self, r_squared: float, tau: float) -> TrendStrength:
        """Get trend strength from R² and Kendall tau."""
        combined = (r_squared + abs(tau)) / 2
        
        if combined >= 0.8:
            return TrendStrength.VERY_STRONG
        elif combined >= 0.6:
            return TrendStrength.STRONG
        elif combined >= 0.4:
            return TrendStrength.MODERATE
        elif combined >= 0.2:
            return TrendStrength.WEAK
        return TrendStrength.NONE
    
    def _smooth_values(self, values: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing."""
        window = min(self.config.smoothing_window, len(values) // 3)
        if window < 2:
            return values
        
        smoothed = pd.Series(values).rolling(window=window, center=True).mean()
        return smoothed.fillna(method='bfill').fillna(method='ffill').values
    
    def _detect_change_points(
        self,
        values: np.ndarray,
        dates: np.ndarray
    ) -> List[ChangePoint]:
        """Detect change points using PELT-like approach."""
        change_points = []
        n = len(values)
        min_seg = self.config.min_segment_length
        
        if n < 2 * min_seg:
            return change_points
        
        # Simple change point detection using moving window
        window = max(min_seg, n // 10)
        
        for i in range(window, n - window):
            left = values[i - window:i]
            right = values[i:i + window]
            
            # T-test for difference
            try:
                _, p = scipy_stats.ttest_ind(left, right)
                if p < 0.01:
                    # Get trends before and after
                    left_slope = np.polyfit(np.arange(len(left)), left, 1)[0]
                    right_slope = np.polyfit(np.arange(len(right)), right, 1)[0]
                    
                    dir_before = TrendDirection.INCREASING if left_slope > 0 else TrendDirection.DECREASING
                    dir_after = TrendDirection.INCREASING if right_slope > 0 else TrendDirection.DECREASING
                    
                    # Avoid duplicate nearby change points
                    if not change_points or i - change_points[-1].index > window:
                        change_points.append(ChangePoint(
                            index=i,
                            date=pd.Timestamp(dates[i]).to_pydatetime() if dates is not None else None,
                            value=values[i],
                            direction_before=dir_before,
                            direction_after=dir_after,
                            magnitude=abs(np.mean(right) - np.mean(left))
                        ))
            except:
                continue
        
        return change_points
    
    def _calculate_segments(
        self,
        values: np.ndarray,
        dates: np.ndarray,
        change_points: List[ChangePoint]
    ) -> List[Dict[str, Any]]:
        """Calculate trend segments."""
        segments = []
        
        # Add boundaries
        indices = [0] + [cp.index for cp in change_points] + [len(values)]
        
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            
            if end_idx - start_idx < 3:
                continue
            
            segment_values = values[start_idx:end_idx]
            segment_trend = self._calculate_trend(segment_values)
            
            segments.append({
                "start_index": start_idx,
                "end_index": end_idx,
                "start_date": pd.Timestamp(dates[start_idx]).isoformat() if dates is not None else None,
                "end_date": pd.Timestamp(dates[end_idx - 1]).isoformat() if dates is not None else None,
                "direction": segment_trend.direction.value,
                "slope": round(segment_trend.slope, 4),
                "r_squared": round(segment_trend.r_squared, 4)
            })
        
        return segments
    
    def _detect_date_col(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        patterns = ['date', 'time', 'timestamp', 'dt']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[0]
    
    def _detect_value_col(self, df: pd.DataFrame) -> str:
        patterns = ['value', 'amount', 'count', 'total', 'metric']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(p in col.lower() for p in patterns):
                return col
        return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]


# ============================================================================
# Factory Functions
# ============================================================================

def get_trend_engine(config: TrendConfig = None) -> TrendAnalysisEngine:
    """Get trend analysis engine."""
    return TrendAnalysisEngine(config=config)


def quick_trend(
    df: pd.DataFrame,
    date_col: str = None,
    value_col: str = None
) -> Dict[str, Any]:
    """Quick trend analysis."""
    engine = TrendAnalysisEngine(verbose=False)
    result = engine.analyze(df, date_col, value_col)
    return result.to_dict()
