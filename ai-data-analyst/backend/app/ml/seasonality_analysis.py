# AI Enterprise Data Analyst - Seasonality Analysis Engine
# Production-grade seasonal pattern detection
# Handles: any time series, multiple seasonality types

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

class SeasonalityType(str, Enum):
    """Types of seasonality."""
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class SeasonalStrength(str, Enum):
    """Seasonality strength."""
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SeasonalPattern:
    """Detected seasonal pattern."""
    period: int
    seasonality_type: SeasonalityType
    strength: SeasonalStrength
    strength_value: float
    seasonal_indices: Dict[int, float] = field(default_factory=dict)


@dataclass
class SeasonalityResult:
    """Complete seasonality analysis result."""
    n_observations: int = 0
    data_frequency: str = ""
    
    # Detected patterns
    patterns: List[SeasonalPattern] = field(default_factory=list)
    
    # Primary seasonality
    primary_pattern: Optional[SeasonalPattern] = None
    
    # Peak/trough analysis
    peak_periods: List[int] = field(default_factory=list)
    trough_periods: List[int] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "data_frequency": self.data_frequency,
            "primary_pattern": {
                "type": self.primary_pattern.seasonality_type.value,
                "period": self.primary_pattern.period,
                "strength": self.primary_pattern.strength.value,
                "strength_value": round(self.primary_pattern.strength_value, 4)
            } if self.primary_pattern else None,
            "seasonal_indices": self.primary_pattern.seasonal_indices if self.primary_pattern else {},
            "peak_periods": self.peak_periods[:5],
            "trough_periods": self.trough_periods[:5],
            "all_patterns": [
                {
                    "type": p.seasonality_type.value,
                    "period": p.period,
                    "strength": p.strength.value
                }
                for p in self.patterns
            ]
        }


# ============================================================================
# Seasonality Analysis Engine
# ============================================================================

class SeasonalityAnalysisEngine:
    """
    Seasonality Analysis engine.
    
    Features:
    - Automatic frequency detection
    - Multiple seasonality detection
    - Seasonal strength measurement
    - Seasonal index calculation
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame = None,
        date_col: str = None,
        value_col: str = None,
        values: np.ndarray = None,
        frequency: str = None
    ) -> SeasonalityResult:
        """Analyze seasonality in time series."""
        start_time = datetime.now()
        
        # Get values
        if values is None and df is not None:
            if date_col is None:
                date_col = self._detect_date_col(df)
            if value_col is None:
                value_col = self._detect_value_col(df)
            
            df = df.sort_values(date_col)
            values = df[value_col].values
        
        values = np.array(values).astype(float)
        values = values[~np.isnan(values)]
        
        if self.verbose:
            logger.info(f"Analyzing seasonality for {len(values)} observations")
        
        # Detect frequency
        if frequency is None:
            frequency = self._detect_frequency(len(values))
        
        # Test different periods
        periods_to_test = self._get_periods_to_test(frequency, len(values))
        
        patterns = []
        for period, stype in periods_to_test:
            strength = self._calculate_seasonal_strength(values, period)
            
            if strength > 0.1:
                indices = self._calculate_seasonal_indices(values, period)
                patterns.append(SeasonalPattern(
                    period=period,
                    seasonality_type=stype,
                    strength=self._classify_strength(strength),
                    strength_value=strength,
                    seasonal_indices=indices
                ))
        
        # Sort by strength
        patterns.sort(key=lambda x: -x.strength_value)
        
        primary = patterns[0] if patterns else None
        
        # Peak/trough analysis
        peak_periods = []
        trough_periods = []
        
        if primary:
            indices = primary.seasonal_indices
            sorted_idx = sorted(indices.items(), key=lambda x: -x[1])
            peak_periods = [p for p, _ in sorted_idx[:3]]
            trough_periods = [p for p, _ in sorted_idx[-3:]]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SeasonalityResult(
            n_observations=len(values),
            data_frequency=frequency,
            patterns=patterns,
            primary_pattern=primary,
            peak_periods=peak_periods,
            trough_periods=trough_periods,
            processing_time_sec=processing_time
        )
    
    def _calculate_seasonal_strength(self, values: np.ndarray, period: int) -> float:
        """Calculate strength of seasonality for given period."""
        n = len(values)
        if n < 2 * period:
            return 0.0
        
        # Autocorrelation at lag = period
        mean = np.mean(values)
        var = np.var(values)
        
        if var == 0:
            return 0.0
        
        autocorr = np.correlate(values - mean, values - mean, mode='full')
        autocorr = autocorr[n - 1:] / (var * n)
        
        if len(autocorr) > period:
            return float(abs(autocorr[period]))
        
        return 0.0
    
    def _calculate_seasonal_indices(
        self,
        values: np.ndarray,
        period: int
    ) -> Dict[int, float]:
        """Calculate seasonal indices for each period position."""
        n = len(values)
        indices = {}
        
        for p in range(period):
            period_values = values[p::period]
            if len(period_values) > 0:
                indices[p + 1] = float(np.mean(period_values))
        
        # Normalize to mean = 100
        total_mean = np.mean(list(indices.values()))
        if total_mean > 0:
            indices = {k: v / total_mean * 100 for k, v in indices.items()}
        
        return indices
    
    def _classify_strength(self, strength: float) -> SeasonalStrength:
        """Classify seasonal strength."""
        if strength >= 0.6:
            return SeasonalStrength.VERY_STRONG
        elif strength >= 0.4:
            return SeasonalStrength.STRONG
        elif strength >= 0.2:
            return SeasonalStrength.MODERATE
        elif strength >= 0.1:
            return SeasonalStrength.WEAK
        return SeasonalStrength.NONE
    
    def _detect_frequency(self, n: int) -> str:
        """Detect likely data frequency."""
        if n > 365 * 2:
            return "daily"
        elif n > 52 * 2:
            return "weekly"
        elif n > 12 * 2:
            return "monthly"
        elif n > 4 * 2:
            return "quarterly"
        return "yearly"
    
    def _get_periods_to_test(
        self,
        frequency: str,
        n: int
    ) -> List[Tuple[int, SeasonalityType]]:
        """Get periods to test based on frequency."""
        periods = []
        
        if frequency == "daily":
            if n >= 14:
                periods.append((7, SeasonalityType.WEEKLY))
            if n >= 60:
                periods.append((30, SeasonalityType.MONTHLY))
            if n >= 730:
                periods.append((365, SeasonalityType.YEARLY))
        elif frequency == "weekly":
            if n >= 8:
                periods.append((4, SeasonalityType.MONTHLY))
            if n >= 104:
                periods.append((52, SeasonalityType.YEARLY))
        elif frequency == "monthly":
            if n >= 6:
                periods.append((3, SeasonalityType.QUARTERLY))
            if n >= 24:
                periods.append((12, SeasonalityType.YEARLY))
        elif frequency == "quarterly":
            if n >= 8:
                periods.append((4, SeasonalityType.YEARLY))
        
        return periods
    
    def _detect_date_col(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return df.columns[0]
    
    def _detect_value_col(self, df: pd.DataFrame) -> str:
        num_cols = df.select_dtypes(include=[np.number]).columns
        return num_cols[0] if len(num_cols) > 0 else df.columns[1]


# ============================================================================
# Factory Functions
# ============================================================================

def get_seasonality_engine() -> SeasonalityAnalysisEngine:
    """Get seasonality analysis engine."""
    return SeasonalityAnalysisEngine()


def quick_seasonality(
    df: pd.DataFrame,
    date_col: str = None,
    value_col: str = None
) -> Dict[str, Any]:
    """Quick seasonality analysis."""
    engine = SeasonalityAnalysisEngine(verbose=False)
    result = engine.analyze(df, date_col, value_col)
    return result.to_dict()
