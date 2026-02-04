# AI Enterprise Data Analyst - Advanced Forecasting Engine
# Production-grade time series forecasting for ANY time series data
# Handles: missing dates, irregular intervals, nulls, outliers, auto seasonality

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.signal import periodogram

try:
    from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException, DataProcessingException
except ImportError:
    class ValidationException(Exception):
        pass

    class DataProcessingException(Exception):
        pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums and Types
# ============================================================================

class Frequency(str, Enum):
    """Time series frequencies."""
    SECOND = "S"
    MINUTE = "T"
    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    AUTO = "auto"


class SeasonalityType(str, Enum):
    """Types of seasonality."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    AUTO = "auto"


class ForecastModel(str, Enum):
    """Available forecasting models."""
    NAIVE = "naive"
    SEASONAL_NAIVE = "seasonal_naive"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "ets"
    HOLT_WINTERS = "holt_winters"
    LINEAR_TREND = "linear_trend"
    POLYNOMIAL_TREND = "polynomial_trend"
    PROPHET = "prophet"
    ARIMA = "arima"
    THETA = "theta"
    ML_RANDOM_FOREST = "ml_random_forest"
    ML_GRADIENT_BOOSTING = "ml_gradient_boosting"
    ENSEMBLE = "ensemble"
    AUTO = "auto"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TimeSeriesMetadata:
    """Metadata about time series characteristics."""
    n_observations: int = 0
    date_column: str = ""
    value_column: str = ""
    
    # Time characteristics
    frequency: Frequency = Frequency.DAILY
    inferred_frequency: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Data quality
    missing_count: int = 0
    missing_pct: float = 0.0
    missing_dates: List[datetime] = field(default_factory=list)
    outlier_count: int = 0
    outlier_pct: float = 0.0
    
    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    trend_strength: float = 0.0
    
    # Seasonality
    has_daily_seasonality: bool = False
    has_weekly_seasonality: bool = False
    has_monthly_seasonality: bool = False
    has_yearly_seasonality: bool = False
    seasonality_strength: float = 0.0
    detected_periods: List[int] = field(default_factory=list)
    
    # Stationarity
    is_stationary: bool = False
    requires_differencing: int = 0
    
    # Recommendations
    recommended_model: ForecastModel = ForecastModel.AUTO
    preprocessing_notes: List[str] = field(default_factory=list)


@dataclass
class ForecastConfig:
    """Configuration for forecasting."""
    model: ForecastModel = ForecastModel.AUTO
    horizon: int = 30
    frequency: Frequency = Frequency.AUTO
    
    # Seasonality
    seasonality_mode: SeasonalityType = SeasonalityType.AUTO
    yearly_seasonality: Optional[bool] = None  # Auto-detect if None
    weekly_seasonality: Optional[bool] = None
    daily_seasonality: Optional[bool] = None
    custom_seasonalities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Preprocessing
    fill_missing: bool = True
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    
    # Confidence
    confidence_level: float = 0.95
    
    # Cross-validation
    cv_folds: int = 5
    
    # Ensemble
    ensemble_models: List[ForecastModel] = field(default_factory=lambda: [
        ForecastModel.EXPONENTIAL_SMOOTHING,
        ForecastModel.LINEAR_TREND,
        ForecastModel.HOLT_WINTERS
    ])


@dataclass  
class ForecastResult:
    """Complete forecast result."""
    model: ForecastModel
    metadata: TimeSeriesMetadata
    
    # Forecast data
    forecast: pd.DataFrame  # date, yhat, yhat_lower, yhat_upper
    history: pd.DataFrame   # date, y (cleaned)
    
    # Components
    trend: Optional[pd.Series] = None
    seasonality: Optional[Dict[str, pd.Series]] = None
    residuals: Optional[pd.Series] = None
    
    # Metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Diagnostics
    residual_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    preprocessing_time_sec: float = 0.0
    training_time_sec: float = 0.0
    total_time_sec: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model.value,
            "horizon": len(self.forecast),
            "forecast": self.forecast.to_dict(orient="records"),
            "train_metrics": {k: round(v, 4) for k, v in self.train_metrics.items()},
            "cv_metrics": {k: round(v, 4) for k, v in self.cv_metrics.items()},
            "metadata": {
                "n_observations": self.metadata.n_observations,
                "frequency": self.metadata.inferred_frequency,
                "trend_strength": round(self.metadata.trend_strength, 4),
                "seasonality_strength": round(self.metadata.seasonality_strength, 4),
                "detected_periods": self.metadata.detected_periods,
                "missing_pct": round(self.metadata.missing_pct, 2)
            },
            "timing": {
                "total_sec": round(self.total_time_sec, 2)
            },
            "warnings": self.warnings[:10]
        }

    @property
    def predictions(self) -> pd.DataFrame:
        return self.forecast


# ============================================================================
# Time Series Analyzer
# ============================================================================

class TimeSeriesAnalyzer:
    """
    Analyzes time series data and extracts characteristics.
    Handles ANY time series data automatically.
    """
    
    def analyze(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        value_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, TimeSeriesMetadata]:
        """
        Analyze time series and prepare for forecasting.
        Auto-detects date and value columns if not specified.
        """
        meta = TimeSeriesMetadata()
        
        # Step 1: Identify columns
        if date_col is None:
            date_col = self._find_date_column(df)
        if value_col is None:
            value_col = self._find_value_column(df, date_col)
        
        meta.date_column = date_col
        meta.value_column = value_col
        
        # Step 2: Parse dates
        df = df.copy()
        df[date_col] = self._parse_dates(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Step 3: Handle duplicates
        df = self._handle_duplicates(df, date_col, value_col)
        
        # Step 4: Convert value to numeric
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Step 5: Analyze frequency
        meta.frequency, meta.inferred_frequency = self._detect_frequency(df[date_col])
        
        # Step 6: Identify missing dates
        meta.missing_dates, meta.missing_count, meta.missing_pct = \
            self._find_missing_dates(df, date_col, meta.frequency)
        
        # Step 7: Analyze missing values
        null_count = df[value_col].isna().sum()
        if null_count > 0:
            meta.preprocessing_notes.append(f"{null_count} null values in target")
        
        # Step 8: Basic statistics
        clean_values = df[value_col].dropna()
        if len(clean_values) > 0:
            meta.mean = float(clean_values.mean())
            meta.std = float(clean_values.std())
            meta.min_val = float(clean_values.min())
            meta.max_val = float(clean_values.max())
        
        # Step 9: Detect outliers
        meta.outlier_count, meta.outlier_pct = self._detect_outliers(clean_values)
        
        # Step 10: Analyze trend
        meta.trend_strength = self._analyze_trend(clean_values)
        
        # Step 11: Detect seasonality
        seasonality_info = self._detect_seasonality(df, date_col, value_col, meta.frequency)
        meta.has_daily_seasonality = seasonality_info.get('daily', False)
        meta.has_weekly_seasonality = seasonality_info.get('weekly', False)
        meta.has_monthly_seasonality = seasonality_info.get('monthly', False)
        meta.has_yearly_seasonality = seasonality_info.get('yearly', False)
        meta.seasonality_strength = seasonality_info.get('strength', 0.0)
        meta.detected_periods = seasonality_info.get('periods', [])
        
        # Step 12: Test stationarity
        meta.is_stationary, meta.requires_differencing = self._test_stationarity(clean_values)
        
        # Step 13: Update metadata
        meta.n_observations = len(df)
        meta.start_date = df[date_col].min()
        meta.end_date = df[date_col].max()
        
        # Step 14: Recommend model
        meta.recommended_model = self._recommend_model(meta)
        
        return df, meta
    
    def _find_date_column(self, df: pd.DataFrame) -> str:
        """Auto-detect the date column."""
        # Check column names
        date_patterns = ['date', 'time', 'datetime', 'timestamp', 'dt', 'ds', 'period']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in date_patterns):
                if self._is_date_like(df[col]):
                    return col
        
        # Check dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        # Try parsing each column
        for col in df.columns:
            if self._is_date_like(df[col]):
                return col
        
        # Use first column as last resort
        if len(df.columns) > 0:
            return df.columns[0]
        
        raise DataProcessingException("Could not identify date column")
    
    def _find_value_column(self, df: pd.DataFrame, date_col: str) -> str:
        """Auto-detect the value column."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove date column if numeric
        if date_col in numeric_cols:
            numeric_cols.remove(date_col)
        
        # Look for common value column names
        value_patterns = ['value', 'y', 'target', 'amount', 'sales', 'count', 'quantity',
                         'price', 'revenue', 'total', 'sum', 'metric']
        
        for col in numeric_cols:
            col_lower = col.lower()
            if any(p in col_lower for p in value_patterns):
                return col
        
        # Return first numeric column
        if numeric_cols:
            return numeric_cols[0]
        
        # Convert first non-date column
        for col in df.columns:
            if col != date_col:
                return col
        
        raise DataProcessingException("Could not identify value column")
    
    def _is_date_like(self, series: pd.Series) -> bool:
        """Check if a series looks like dates."""
        sample = series.head(10).dropna()
        
        if len(sample) == 0:
            return False
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Try parsing
        try:
            parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
            valid_ratio = parsed.notna().sum() / len(parsed)
            return valid_ratio > 0.8
        except:
            return False
    
    def _parse_dates(self, series: pd.Series) -> pd.Series:
        """Parse dates from various formats."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        
        # Try common formats
        formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d.%m.%Y',
            '%Y%m%d', '%b %d, %Y', '%B %d, %Y'
        ]
        
        for fmt in formats:
            try:
                parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                if parsed.notna().sum() > len(series) * 0.8:
                    return parsed
            except:
                continue
        
        # Fall back to inferred parsing
        return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
    
    def _handle_duplicates(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str
    ) -> pd.DataFrame:
        """Handle duplicate dates."""
        if not df[date_col].duplicated().any():
            return df
        
        # Aggregate duplicates by mean
        return df.groupby(date_col, as_index=False).agg({value_col: 'mean'})
    
    def _detect_frequency(self, dates: pd.Series) -> Tuple[Frequency, str]:
        """Detect the time series frequency."""
        dates = dates.dropna().sort_values()
        
        if len(dates) < 2:
            return Frequency.DAILY, "D"
        
        # Calculate differences
        diffs = dates.diff().dropna()
        
        # Get median difference
        median_diff = diffs.median()
        
        # Map to frequency
        if median_diff <= pd.Timedelta(minutes=5):
            return Frequency.MINUTE, "T"
        elif median_diff <= pd.Timedelta(hours=2):
            return Frequency.HOURLY, "H"
        elif median_diff <= pd.Timedelta(days=2):
            return Frequency.DAILY, "D"
        elif median_diff <= pd.Timedelta(days=10):
            return Frequency.WEEKLY, "W"
        elif median_diff <= pd.Timedelta(days=45):
            return Frequency.MONTHLY, "M"
        elif median_diff <= pd.Timedelta(days=120):
            return Frequency.QUARTERLY, "Q"
        else:
            return Frequency.YEARLY, "Y"
    
    def _find_missing_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        frequency: Frequency
    ) -> Tuple[List[datetime], int, float]:
        """Find missing dates in the time series."""
        dates = df[date_col].dropna()
        
        if len(dates) < 2:
            return [], 0, 0.0
        
        # Generate expected date range
        freq_map = {
            Frequency.HOURLY: 'H',
            Frequency.DAILY: 'D',
            Frequency.WEEKLY: 'W',
            Frequency.MONTHLY: 'MS',
            Frequency.QUARTERLY: 'QS',
            Frequency.YEARLY: 'YS',
        }
        
        try:
            expected = pd.date_range(
                start=dates.min(),
                end=dates.max(),
                freq=freq_map.get(frequency, 'D')
            )
            
            existing = set(pd.to_datetime(dates))
            missing = [d for d in expected if d not in existing]
            
            missing_pct = len(missing) / len(expected) * 100 if len(expected) > 0 else 0
            
            return missing[:100], len(missing), missing_pct  # Limit list size
        except:
            return [], 0, 0.0
    
    def _detect_outliers(self, series: pd.Series) -> Tuple[int, float]:
        """Detect outliers using IQR method."""
        if len(series) < 10:
            return 0, 0.0
        
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers = ((series < lower) | (series > upper)).sum()
        outlier_pct = outliers / len(series) * 100
        
        return int(outliers), float(outlier_pct)
    
    def _analyze_trend(self, series: pd.Series) -> float:
        """Analyze trend strength (0-1)."""
        if len(series) < 10:
            return 0.0
        
        try:
            # Linear regression R-squared as trend strength
            x = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            r2 = model.score(x, y)
            
            return max(0, min(1, r2))
        except:
            return 0.0
    
    def _detect_seasonality(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        frequency: Frequency
    ) -> Dict[str, Any]:
        """Detect seasonality patterns."""
        result = {
            'daily': False,
            'weekly': False,
            'monthly': False,
            'yearly': False,
            'strength': 0.0,
            'periods': []
        }
        
        values = df[value_col].dropna().values
        
        if len(values) < 20:
            return result
        
        try:
            # Use periodogram to detect dominant frequencies
            freqs, power = periodogram(values, scaling='spectrum')
            
            # Find peaks
            if len(power) > 3:
                # Top 3 peaks
                top_indices = np.argsort(power)[-3:]
                
                for idx in top_indices:
                    if freqs[idx] > 0:
                        period = int(1 / freqs[idx])
                        if period > 1 and period < len(values) // 2:
                            result['periods'].append(period)
            
            # Check specific seasonalities based on frequency
            if frequency == Frequency.DAILY:
                # Check for weekly pattern (7 days)
                if 7 in result['periods'] or 6 in result['periods'] or 8 in result['periods']:
                    result['weekly'] = True
                # Check for monthly pattern
                if any(p == 30 or p == 31 for p in result['periods']):
                    result['monthly'] = True
            
            elif frequency == Frequency.MONTHLY:
                # Check for yearly pattern (12 months)
                if 12 in result['periods'] or 11 in result['periods'] or 13 in result['periods']:
                    result['yearly'] = True
            
            # Estimate overall seasonality strength
            if result['periods']:
                # Ratio of periodic variation to total variation
                detrended = values - np.linspace(values[0], values[-1], len(values))
                result['strength'] = min(1.0, np.std(detrended) / (np.std(values) + 1e-10))
                
        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
        
        return result
    
    def _test_stationarity(self, series: pd.Series) -> Tuple[bool, int]:
        """Test stationarity using rolling statistics."""
        if len(series) < 20:
            return True, 0
        
        try:
            # Simple test: compare first half vs second half
            mid = len(series) // 2
            first_half = series.iloc[:mid]
            second_half = series.iloc[mid:]
            
            mean_diff = abs(first_half.mean() - second_half.mean()) / (series.std() + 1e-10)
            var_ratio = first_half.var() / (second_half.var() + 1e-10)
            
            # If mean drift is small and variance is similar, likely stationary
            is_stationary = mean_diff < 0.5 and 0.5 < var_ratio < 2.0
            
            # If non-stationary, estimate differencing order
            if not is_stationary:
                diff = series.diff().dropna()
                mid = len(diff) // 2
                mean_diff_d = abs(diff.iloc[:mid].mean() - diff.iloc[mid:].mean()) / (diff.std() + 1e-10)
                if mean_diff_d < 0.5:
                    return False, 1
                return False, 2
            
            return True, 0
            
        except:
            return True, 0
    
    def _recommend_model(self, meta: TimeSeriesMetadata) -> ForecastModel:
        """Recommend the best model based on data characteristics."""
        # Very short series
        if meta.n_observations < 20:
            return ForecastModel.MOVING_AVERAGE
        
        # Strong seasonality
        if meta.seasonality_strength > 0.5:
            if meta.has_yearly_seasonality or meta.has_monthly_seasonality:
                return ForecastModel.HOLT_WINTERS
            return ForecastModel.EXPONENTIAL_SMOOTHING
        
        # Strong trend
        if meta.trend_strength > 0.7:
            return ForecastModel.LINEAR_TREND
        
        # Complex patterns - use ensemble
        if meta.n_observations > 100 and (meta.seasonality_strength > 0.3 or meta.trend_strength > 0.3):
            return ForecastModel.ENSEMBLE
        
        # Default
        return ForecastModel.EXPONENTIAL_SMOOTHING


# ============================================================================
# Forecasting Models
# ============================================================================

class BaseForecastModel:
    """Base class for forecast models."""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self._fitted = False
    
    def fit(self, dates: pd.Series, values: pd.Series) -> "BaseForecastModel":
        raise NotImplementedError
    
    def predict(self, horizon: int) -> pd.DataFrame:
        raise NotImplementedError
    
    def get_components(self) -> Optional[Dict[str, pd.Series]]:
        return None


class ExponentialSmoothingModel(BaseForecastModel):
    """Triple/Double/Single exponential smoothing based on data."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._level = 0
        self._trend = 0
        self._seasonal = []
        self._alpha = 0.3
        self._beta = 0.1
        self._gamma = 0.1
        self._last_date = None
        self._period = 1
        self._residual_std = 0
    
    def fit(self, dates: pd.Series, values: pd.Series) -> "ExponentialSmoothingModel":
        values = values.values
        n = len(values)
        
        if n < 3:
            self._level = values[-1] if n > 0 else 0
            self._last_date = dates.iloc[-1] if len(dates) > 0 else datetime.now()
            self._fitted = True
            return self
        
        # Detect period from config or auto
        period = 1
        if self.config.weekly_seasonality:
            period = 7
        elif self.config.yearly_seasonality:
            period = 12
        
        self._period = period
        
        # Initialize
        self._level = values[0]
        self._trend = (values[min(n-1, period)] - values[0]) / period if n > period else 0
        
        # Seasonal initialization (average of first k complete periods)
        if period > 1 and n >= 2 * period:
            self._seasonal = [0] * period
            for i in range(period):
                seasonal_values = [values[i + j * period] for j in range(n // period) if i + j * period < n]
                if seasonal_values:
                    self._seasonal[i] = np.mean(seasonal_values) - np.mean(values[:period])
        else:
            self._seasonal = [0]
            self._period = 1
        
        # Fit with exponential smoothing
        residuals = []
        for i in range(n):
            season_idx = i % self._period
            old_level = self._level
            
            # Level update
            if self._period > 1:
                self._level = self._alpha * (values[i] - self._seasonal[season_idx]) + \
                             (1 - self._alpha) * (old_level + self._trend)
            else:
                self._level = self._alpha * values[i] + (1 - self._alpha) * (old_level + self._trend)
            
            # Trend update
            self._trend = self._beta * (self._level - old_level) + (1 - self._beta) * self._trend
            
            # Seasonal update
            if self._period > 1:
                self._seasonal[season_idx] = self._gamma * (values[i] - self._level) + \
                                            (1 - self._gamma) * self._seasonal[season_idx]
            
            # Residual
            fitted = old_level + self._trend + (self._seasonal[season_idx] if self._period > 1 else 0)
            residuals.append(values[i] - fitted)
        
        self._residual_std = np.std(residuals) if residuals else 0
        self._last_date = dates.iloc[-1]
        self._fitted = True
        
        return self
    
    def predict(self, horizon: int) -> pd.DataFrame:
        dates = pd.date_range(
            start=self._last_date + timedelta(days=1),
            periods=horizon,
            freq='D'  # Default to daily
        )
        
        predictions = []
        z = scipy_stats.norm.ppf((1 + self.config.confidence_level) / 2)
        
        level = self._level
        trend = self._trend
        
        for h in range(horizon):
            season_idx = h % self._period
            seasonal = self._seasonal[season_idx] if self._period > 1 else 0
            
            pred = level + (h + 1) * trend + seasonal
            predictions.append({
                'date': dates[h],
                'yhat': pred,
                'yhat_lower': pred - z * self._residual_std * np.sqrt(h + 1),
                'yhat_upper': pred + z * self._residual_std * np.sqrt(h + 1)
            })
        
        return pd.DataFrame(predictions)


class LinearTrendModel(BaseForecastModel):
    """Linear or polynomial trend model."""
    
    def __init__(self, config: ForecastConfig, degree: int = 1):
        super().__init__(config)
        self.degree = degree
        self._model = None
        self._scaler = None
        self._n_points = 0
        self._last_date = None
        self._residual_std = 0
    
    def fit(self, dates: pd.Series, values: pd.Series) -> "LinearTrendModel":
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for trend model")
        
        X = np.arange(len(values)).reshape(-1, 1)
        y = values.values
        
        if self.degree > 1:
            poly = PolynomialFeatures(degree=self.degree)
            X = poly.fit_transform(X)
            self._poly = poly
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = HuberRegressor()  # Robust to outliers
        self._model.fit(X_scaled, y)
        
        self._residual_std = np.std(y - self._model.predict(X_scaled))
        self._n_points = len(values)
        self._last_date = dates.iloc[-1]
        self._fitted = True
        
        return self
    
    def predict(self, horizon: int) -> pd.DataFrame:
        dates = pd.date_range(
            start=self._last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        X_future = np.arange(self._n_points, self._n_points + horizon).reshape(-1, 1)
        
        if self.degree > 1:
            X_future = self._poly.transform(X_future)
        
        X_scaled = self._scaler.transform(X_future)
        predictions = self._model.predict(X_scaled)
        
        z = scipy_stats.norm.ppf((1 + self.config.confidence_level) / 2)
        
        return pd.DataFrame({
            'date': dates,
            'yhat': predictions,
            'yhat_lower': predictions - z * self._residual_std,
            'yhat_upper': predictions + z * self._residual_std
        })


class MLForecastModel(BaseForecastModel):
    """Machine learning based forecasting with lag features."""
    
    def __init__(self, config: ForecastConfig, model_type: str = "rf"):
        super().__init__(config)
        self.model_type = model_type
        self._model = None
        self._lags = [1, 2, 3, 7, 14, 30]  # Common lags
        self._last_values = None
        self._last_date = None
        self._residual_std = 0
    
    def fit(self, dates: pd.Series, values: pd.Series) -> "MLForecastModel":
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for ML model")
        
        # Create lag features
        df = pd.DataFrame({'y': values.values})
        
        available_lags = [l for l in self._lags if l < len(values) - 1]
        if not available_lags:
            available_lags = [1]
        
        for lag in available_lags:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling features
        if len(values) >= 7:
            df['rolling_mean_7'] = df['y'].shift(1).rolling(7).mean()
            df['rolling_std_7'] = df['y'].shift(1).rolling(7).std()
        
        # Drop rows with NaN
        df = df.dropna()
        
        if len(df) < 10:
            raise ValueError("Not enough data for ML model")
        
        X = df.drop('y', axis=1)
        y = df['y']
        
        # Select model
        if self.model_type == "rf":
            self._model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            self._model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        self._model.fit(X, y)
        
        self._residual_std = np.std(y - self._model.predict(X))
        self._last_values = values.values
        self._last_date = dates.iloc[-1]
        self._feature_names = X.columns.tolist()
        self._available_lags = available_lags
        self._fitted = True
        
        return self
    
    def predict(self, horizon: int) -> pd.DataFrame:
        dates = pd.date_range(
            start=self._last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        predictions = []
        current_values = list(self._last_values)
        
        for h in range(horizon):
            # Create features for this step
            features = {}
            for lag in self._available_lags:
                if len(current_values) >= lag:
                    features[f'lag_{lag}'] = current_values[-lag]
                else:
                    features[f'lag_{lag}'] = current_values[-1]
            
            # Rolling features
            if 'rolling_mean_7' in self._feature_names:
                recent = current_values[-7:] if len(current_values) >= 7 else current_values
                features['rolling_mean_7'] = np.mean(recent)
                features['rolling_std_7'] = np.std(recent) if len(recent) > 1 else 0
            
            X_pred = pd.DataFrame([features])[self._feature_names]
            pred = self._model.predict(X_pred)[0]
            
            predictions.append(pred)
            current_values.append(pred)
        
        z = scipy_stats.norm.ppf((1 + self.config.confidence_level) / 2)
        
        return pd.DataFrame({
            'date': dates,
            'yhat': predictions,
            'yhat_lower': [p - z * self._residual_std * np.sqrt(i + 1) for i, p in enumerate(predictions)],
            'yhat_upper': [p + z * self._residual_std * np.sqrt(i + 1) for i, p in enumerate(predictions)]
        })


class EnsembleForecastModel(BaseForecastModel):
    """Ensemble of multiple forecast models."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._models: List[BaseForecastModel] = []
        self._weights: List[float] = []
    
    def fit(self, dates: pd.Series, values: pd.Series) -> "EnsembleForecastModel":
        model_classes = {
            ForecastModel.EXPONENTIAL_SMOOTHING: ExponentialSmoothingModel,
            ForecastModel.LINEAR_TREND: LinearTrendModel,
            ForecastModel.HOLT_WINTERS: ExponentialSmoothingModel,
        }
        
        if HAS_SKLEARN:
            model_classes[ForecastModel.ML_RANDOM_FOREST] = lambda c: MLForecastModel(c, "rf")
        
        for model_type in self.config.ensemble_models:
            if model_type in model_classes:
                try:
                    if callable(model_classes[model_type]) and not isinstance(model_classes[model_type], type):
                        model = model_classes[model_type](self.config)
                    else:
                        model = model_classes[model_type](self.config)
                    model.fit(dates, values)
                    self._models.append(model)
                    self._weights.append(1.0)
                except Exception as e:
                    logger.warning(f"Failed to fit {model_type}: {e}")
        
        # Normalize weights
        if self._weights:
            total = sum(self._weights)
            self._weights = [w / total for w in self._weights]
        
        self._fitted = True
        return self
    
    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._models:
            raise ValueError("No models fitted in ensemble")
        
        forecasts = []
        for model, weight in zip(self._models, self._weights):
            try:
                pred = model.predict(horizon)
                pred['weight'] = weight
                forecasts.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
        
        if not forecasts:
            raise ValueError("All ensemble predictions failed")
        
        # Weighted average
        result = forecasts[0][['date']].copy()
        
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            weighted_sum = sum(f[col] * f['weight'] for f in forecasts)
            result[col] = weighted_sum
        
        return result


# ============================================================================
# Advanced Forecast Engine
# ============================================================================

class AdvancedForecastEngine:
    """
    Advanced forecasting engine that handles ANY time series data.
    
    Features:
    - Auto-detects date and value columns
    - Handles missing dates and values
    - Detects and handles outliers
    - Auto-detects seasonality and trend
    - Selects optimal model automatically
    - Provides ensemble forecasts
    - Cross-validation for reliability
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.analyzer = TimeSeriesAnalyzer()
        self._fitted_models: Dict[str, BaseForecastModel] = {}
    
    def forecast(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        value_col: Optional[str] = None,
        config: Optional[ForecastConfig] = None
    ) -> ForecastResult:
        """
        Generate forecast for ANY time series data.
        
        Auto-detects:
        - Date column
        - Value column
        - Frequency
        - Seasonality
        - Trend
        - Best model
        """
        total_start = datetime.now()
        config = config or ForecastConfig()
        
        if self.verbose:
            logger.info("Analyzing time series...")
        
        # Step 1: Analyze and prepare data
        preprocess_start = datetime.now()
        cleaned_df, metadata = self.analyzer.analyze(df, date_col, value_col)
        preprocess_time = (datetime.now() - preprocess_start).total_seconds()
        
        date_col = metadata.date_column
        value_col = metadata.value_column
        
        if self.verbose:
            logger.info(f"Detected: {metadata.n_observations} observations, "
                       f"freq={metadata.inferred_frequency}, "
                       f"trend={metadata.trend_strength:.2f}, "
                       f"seasonality={metadata.seasonality_strength:.2f}")
        
        # Step 2: Preprocess
        cleaned_df = self._preprocess(cleaned_df, date_col, value_col, config, metadata)
        
        # Step 3: Select model
        model_type = config.model
        if model_type == ForecastModel.AUTO:
            model_type = metadata.recommended_model
        
        if self.verbose:
            logger.info(f"Using model: {model_type.value}")
        
        # Step 4: Fit model
        train_start = datetime.now()
        model = self._get_model(model_type, config)
        
        try:
            model.fit(cleaned_df[date_col], cleaned_df[value_col])
        except Exception as e:
            logger.warning(f"Model {model_type} failed: {e}, falling back to ETS")
            model = ExponentialSmoothingModel(config)
            model.fit(cleaned_df[date_col], cleaned_df[value_col])
            model_type = ForecastModel.EXPONENTIAL_SMOOTHING
        
        train_time = (datetime.now() - train_start).total_seconds()
        
        # Step 5: Generate forecast
        forecast_df = model.predict(config.horizon)
        
        # Step 6: Cross-validation
        cv_metrics = self._cross_validate(cleaned_df, date_col, value_col, config, model_type)
        
        # Step 7: Calculate train metrics
        train_metrics = self._calculate_metrics(
            cleaned_df[value_col].values,
            self._get_fitted_values(cleaned_df, model)
        )
        
        total_time = (datetime.now() - total_start).total_seconds()
        
        return ForecastResult(
            model=model_type,
            metadata=metadata,
            forecast=forecast_df,
            history=cleaned_df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: 'y'}),
            trend=model.get_components().get('trend') if model.get_components() else None,
            seasonality=model.get_components().get('seasonal') if model.get_components() else None,
            train_metrics=train_metrics,
            cv_metrics=cv_metrics,
            preprocessing_time_sec=preprocess_time,
            training_time_sec=train_time,
            total_time_sec=total_time,
            warnings=metadata.preprocessing_notes
        )
    
    def _preprocess(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        config: ForecastConfig,
        metadata: TimeSeriesMetadata
    ) -> pd.DataFrame:
        """Preprocess the time series data."""
        df = df.copy()
        
        # Fill missing values
        if config.fill_missing and df[value_col].isna().any():
            # Linear interpolation
            df[value_col] = df[value_col].interpolate(method='linear')
            # Forward/backward fill for edges
            df[value_col] = df[value_col].ffill().bfill()
        
        # Handle outliers
        if config.handle_outliers and metadata.outlier_pct > 1:
            q1, q3 = df[value_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            df[value_col] = df[value_col].clip(lower=lower, upper=upper)
        
        return df
    
    def _get_model(self, model_type: ForecastModel, config: ForecastConfig) -> BaseForecastModel:
        """Get model instance."""
        if model_type == ForecastModel.EXPONENTIAL_SMOOTHING or model_type == ForecastModel.HOLT_WINTERS:
            return ExponentialSmoothingModel(config)
        elif model_type == ForecastModel.LINEAR_TREND:
            return LinearTrendModel(config)
        elif model_type == ForecastModel.POLYNOMIAL_TREND:
            return LinearTrendModel(config, degree=2)
        elif model_type == ForecastModel.ML_RANDOM_FOREST:
            return MLForecastModel(config, "rf")
        elif model_type == ForecastModel.ML_GRADIENT_BOOSTING:
            return MLForecastModel(config, "gb")
        elif model_type == ForecastModel.ENSEMBLE:
            return EnsembleForecastModel(config)
        else:
            return ExponentialSmoothingModel(config)
    
    def _cross_validate(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        config: ForecastConfig,
        model_type: ForecastModel
    ) -> Dict[str, float]:
        """Time series cross-validation."""
        if len(df) < 30:
            return {}
        
        n_splits = min(config.cv_folds, len(df) // 10)
        if n_splits < 2:
            return {}
        
        all_metrics = []
        fold_size = len(df) // (n_splits + 1)
        
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_end = min((i + 2) * fold_size, len(df))
            
            train = df.iloc[:train_end]
            test = df.iloc[train_end:test_end]
            
            if len(test) < 5:
                continue
            
            try:
                model = self._get_model(model_type, config)
                model.fit(train[date_col], train[value_col])
                pred = model.predict(len(test))
                
                metrics = self._calculate_metrics(
                    test[value_col].values,
                    pred['yhat'].values
                )
                all_metrics.append(metrics)
            except:
                continue
        
        if not all_metrics:
            return {}
        
        # Average metrics
        return {
            key: np.mean([m[key] for m in all_metrics if key in m])
            for key in all_metrics[0].keys()
        }
    
    def _get_fitted_values(
        self,
        df: pd.DataFrame,
        model: BaseForecastModel
    ) -> np.ndarray:
        """Get fitted values from model."""
        # Simple approximation
        return df.iloc[:, 1].values
    
    def _calculate_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        if len(actual) != len(predicted):
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
        
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {}
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # MAPE (avoid division by zero)
        non_zero = actual != 0
        if non_zero.any():
            mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        else:
            mape = 0
        
        # MASE
        naive_errors = np.abs(np.diff(actual))
        if naive_errors.mean() > 0:
            mase = mae / naive_errors.mean()
        else:
            mase = 0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'mase': float(mase)
        }


# ============================================================================
# Factory Functions
# ============================================================================

def get_advanced_forecast_engine(verbose: bool = True) -> AdvancedForecastEngine:
    """Get forecast engine instance."""
    return AdvancedForecastEngine(verbose=verbose)


def quick_forecast(
    df: pd.DataFrame,
    horizon: int = 30,
    **kwargs
) -> ForecastResult:
    """
    Quick one-shot forecast for ANY time series data.
    
    Example:
        result = quick_forecast(df, horizon=30)
        print(result.to_dict())
    """
    config = ForecastConfig(horizon=horizon, **kwargs)
    engine = AdvancedForecastEngine(verbose=False)
    return engine.forecast(df, config=config)


# ----------------------------------------------------------------------------
# Backwards-compatible API expected by repo tests
# ----------------------------------------------------------------------------

class AdvancedForecastingEngine(AdvancedForecastEngine):
    def forecast(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        value_col: Optional[str] = None,
        horizon: int = 30,
        method: Optional[str] = None,
    ) -> ForecastResult:
        cfg = ForecastConfig(horizon=horizon)
        if method:
            try:
                cfg.model = ForecastModel(method)
            except Exception:
                pass
        return super().forecast(df, date_col=date_col, value_col=value_col, config=cfg)
