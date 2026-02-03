# AI Enterprise Data Analyst - Time Series Decomposition Engine
# Production-grade time series component extraction
# Handles: trend, seasonality, residuals with multiple methods

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class DecompositionModel(str, Enum):
    """Decomposition model type."""
    ADDITIVE = "additive"  # Y = T + S + R
    MULTIPLICATIVE = "multiplicative"  # Y = T * S * R


class DecompositionMethod(str, Enum):
    """Decomposition method."""
    CLASSICAL = "classical"  # Moving average based
    STL = "stl"  # Seasonal and Trend decomposition using Loess
    X11 = "x11"  # X11 method


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DecompositionComponents:
    """Time series components."""
    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    
    # Seasonal indices
    seasonal_indices: Dict[int, float] = field(default_factory=dict)
    
    # Deseasonalized series
    deseasonalized: np.ndarray = None


@dataclass
class DecompositionResult:
    """Complete decomposition result."""
    n_observations: int = 0
    period: int = 1
    model: DecompositionModel = DecompositionModel.ADDITIVE
    method: DecompositionMethod = DecompositionMethod.CLASSICAL
    
    # Components
    components: DecompositionComponents = None
    
    # Quality metrics
    trend_strength: float = 0.0  # 0-1
    seasonal_strength: float = 0.0  # 0-1
    residual_variance: float = 0.0
    
    # Dates if available
    dates: List[datetime] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "period": self.period,
            "model": self.model.value,
            "method": self.method.value,
            "quality": {
                "trend_strength": round(self.trend_strength, 4),
                "seasonal_strength": round(self.seasonal_strength, 4),
                "residual_variance": round(self.residual_variance, 4)
            },
            "seasonal_indices": {
                str(k): round(v, 4) for k, v in self.components.seasonal_indices.items()
            } if self.components else {},
            "trend_preview": self.components.trend[:10].tolist() if self.components else [],
            "seasonal_preview": self.components.seasonal[:10].tolist() if self.components else []
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get components as DataFrame."""
        if self.components is None:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'observed': self.components.observed,
            'trend': self.components.trend,
            'seasonal': self.components.seasonal,
            'residual': self.components.residual
        })
        
        if self.components.deseasonalized is not None:
            df['deseasonalized'] = self.components.deseasonalized
        
        if self.dates:
            df.index = self.dates
        
        return df


# ============================================================================
# Time Series Decomposition Engine
# ============================================================================

class TimeSeriesDecompositionEngine:
    """
    Production-grade Time Series Decomposition engine.
    
    Features:
    - Classical decomposition
    - STL decomposition
    - Auto period detection
    - Trend and seasonal strength
    - Deseasonalization
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def decompose(
        self,
        data: pd.Series = None,
        df: pd.DataFrame = None,
        value_col: str = None,
        date_col: str = None,
        period: int = None,
        model: DecompositionModel = DecompositionModel.ADDITIVE,
        method: DecompositionMethod = DecompositionMethod.CLASSICAL
    ) -> DecompositionResult:
        """Decompose time series."""
        start_time = datetime.now()
        
        # Get values
        if data is None and df is not None:
            if value_col is None:
                value_col = df.select_dtypes(include=[np.number]).columns[0]
            data = df[value_col]
            
            if date_col:
                dates = pd.to_datetime(df[date_col]).tolist()
            else:
                dates = []
        else:
            dates = data.index.tolist() if hasattr(data.index, 'tolist') else []
        
        values = np.array(data.dropna().values, dtype=float)
        n = len(values)
        
        if n < 4:
            return self._create_insufficient_result(n, start_time)
        
        # Auto-detect period
        if period is None:
            period = self._detect_period(values)
        
        if self.verbose:
            logger.info(f"Decomposing {n} observations with period={period}, model={model.value}")
        
        # Handle multiplicative with zeros
        if model == DecompositionModel.MULTIPLICATIVE:
            if (values <= 0).any():
                model = DecompositionModel.ADDITIVE
                logger.warning("Switched to additive model due to non-positive values")
        
        # Perform decomposition
        if method == DecompositionMethod.STL:
            components = self._stl_decomposition(values, period, model)
        else:
            components = self._classical_decomposition(values, period, model)
        
        # Calculate seasonal indices
        seasonal_indices = self._calculate_seasonal_indices(
            components.seasonal, period
        )
        components.seasonal_indices = seasonal_indices
        
        # Deseasonalize
        if model == DecompositionModel.ADDITIVE:
            components.deseasonalized = values - components.seasonal
        else:
            components.deseasonalized = values / components.seasonal
        
        # Quality metrics
        trend_strength = self._calculate_trend_strength(
            components.trend, components.residual
        )
        seasonal_strength = self._calculate_seasonal_strength(
            components.seasonal, components.residual
        )
        residual_variance = float(np.nanvar(components.residual))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DecompositionResult(
            n_observations=n,
            period=period,
            model=model,
            method=method,
            components=components,
            trend_strength=trend_strength,
            seasonal_strength=seasonal_strength,
            residual_variance=residual_variance,
            dates=dates[:n],
            processing_time_sec=processing_time
        )
    
    def _classical_decomposition(
        self,
        values: np.ndarray,
        period: int,
        model: DecompositionModel
    ) -> DecompositionComponents:
        """Classical moving average decomposition."""
        n = len(values)
        
        # Trend: centered moving average
        if period % 2 == 0:
            # Even period: 2x moving average
            trend = self._moving_average(values, period)
            trend = self._moving_average(trend, 2)
        else:
            trend = self._moving_average(values, period)
        
        # Detrended series
        if model == DecompositionModel.MULTIPLICATIVE:
            detrended = values / trend
            detrended[~np.isfinite(detrended)] = 1.0
        else:
            detrended = values - trend
            detrended[np.isnan(detrended)] = 0.0
        
        # Seasonal: average of detrended by position
        seasonal = np.zeros(n)
        for i in range(period):
            indices = list(range(i, n, period))
            avg = np.nanmean(detrended[indices])
            for idx in indices:
                seasonal[idx] = avg
        
        # Normalize seasonal
        if model == DecompositionModel.MULTIPLICATIVE:
            seasonal_mean = np.mean(seasonal[:period])
            if seasonal_mean != 0:
                seasonal = seasonal / seasonal_mean
        else:
            seasonal = seasonal - np.mean(seasonal[:period])
        
        # Residual
        if model == DecompositionModel.MULTIPLICATIVE:
            with np.errstate(divide='ignore', invalid='ignore'):
                residual = values / (trend * seasonal)
                residual[~np.isfinite(residual)] = 1.0
        else:
            residual = values - trend - seasonal
        
        return DecompositionComponents(
            observed=values,
            trend=trend,
            seasonal=seasonal,
            residual=residual
        )
    
    def _stl_decomposition(
        self,
        values: np.ndarray,
        period: int,
        model: DecompositionModel
    ) -> DecompositionComponents:
        """STL decomposition using statsmodels if available."""
        try:
            from statsmodels.tsa.seasonal import STL
            
            # For multiplicative, take log
            if model == DecompositionModel.MULTIPLICATIVE:
                work_values = np.log(values + 1e-10)
            else:
                work_values = values
            
            stl = STL(work_values, period=period, robust=True)
            result = stl.fit()
            
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Transform back for multiplicative
            if model == DecompositionModel.MULTIPLICATIVE:
                trend = np.exp(trend)
                seasonal = np.exp(seasonal)
                residual = np.exp(residual)
            
            return DecompositionComponents(
                observed=values,
                trend=trend,
                seasonal=seasonal,
                residual=residual
            )
        
        except ImportError:
            return self._classical_decomposition(values, period, model)
    
    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Calculate centered moving average."""
        n = len(values)
        result = np.full(n, np.nan)
        
        half = window // 2
        for i in range(half, n - half):
            result[i] = np.mean(values[i - half:i + half + 1])
        
        return result
    
    def _detect_period(self, values: np.ndarray) -> int:
        """Auto-detect seasonal period using autocorrelation."""
        n = len(values)
        
        if n < 8:
            return 1
        
        # Try common periods
        candidate_periods = [7, 12, 4, 52, 24, 30, 365]
        
        # Calculate autocorrelation
        mean = np.mean(values)
        var = np.var(values)
        
        if var == 0:
            return 1
        
        best_period = 1
        best_acf = 0
        
        for period in candidate_periods:
            if period >= n // 2:
                continue
            
            acf = np.correlate(values - mean, values - mean, mode='full')
            acf = acf[n - 1:] / (var * n)
            
            if len(acf) > period:
                if abs(acf[period]) > best_acf:
                    best_acf = abs(acf[period])
                    best_period = period
        
        return best_period if best_acf > 0.3 else 1
    
    def _calculate_seasonal_indices(
        self,
        seasonal: np.ndarray,
        period: int
    ) -> Dict[int, float]:
        """Calculate seasonal indices."""
        indices = {}
        
        for i in range(period):
            positions = list(range(i, len(seasonal), period))
            if positions:
                indices[i + 1] = float(np.mean(seasonal[positions]))
        
        return indices
    
    def _calculate_trend_strength(
        self,
        trend: np.ndarray,
        residual: np.ndarray
    ) -> float:
        """Calculate trend strength (0-1)."""
        valid_trend = trend[~np.isnan(trend)]
        valid_residual = residual[~np.isnan(residual)]
        
        if len(valid_trend) == 0 or len(valid_residual) == 0:
            return 0.0
        
        var_residual = np.var(valid_residual)
        var_trend_residual = np.var(valid_trend + valid_residual[:len(valid_trend)])
        
        if var_trend_residual == 0:
            return 0.0
        
        return max(0, 1 - var_residual / var_trend_residual)
    
    def _calculate_seasonal_strength(
        self,
        seasonal: np.ndarray,
        residual: np.ndarray
    ) -> float:
        """Calculate seasonal strength (0-1)."""
        valid_seasonal = seasonal[~np.isnan(seasonal)]
        valid_residual = residual[~np.isnan(residual)]
        
        if len(valid_seasonal) == 0 or len(valid_residual) == 0:
            return 0.0
        
        var_residual = np.var(valid_residual)
        var_seasonal_residual = np.var(valid_seasonal + valid_residual[:len(valid_seasonal)])
        
        if var_seasonal_residual == 0:
            return 0.0
        
        return max(0, 1 - var_residual / var_seasonal_residual)
    
    def _create_insufficient_result(
        self,
        n: int,
        start_time: datetime
    ) -> DecompositionResult:
        """Create result for insufficient data."""
        return DecompositionResult(
            n_observations=n,
            period=1,
            processing_time_sec=(datetime.now() - start_time).total_seconds()
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_decomposition_engine() -> TimeSeriesDecompositionEngine:
    """Get time series decomposition engine."""
    return TimeSeriesDecompositionEngine()


def quick_decompose(
    data: pd.Series,
    period: int = None
) -> Dict[str, Any]:
    """Quick time series decomposition."""
    engine = TimeSeriesDecompositionEngine(verbose=False)
    result = engine.decompose(data=data, period=period)
    return result.to_dict()


def deseasonalize(
    data: pd.Series,
    period: int = None
) -> pd.Series:
    """Quick deseasonalization."""
    engine = TimeSeriesDecompositionEngine(verbose=False)
    result = engine.decompose(data=data, period=period)
    if result.components:
        return pd.Series(result.components.deseasonalized, index=data.index)
    return data
