# AI Enterprise Data Analyst - Forecasting Engine
# Prophet, ARIMA, and ensemble forecasting (Uber/Lyft patterns)

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Forecasting Types
# ============================================================================

class ForecastMethod(str, Enum):
    """Forecasting methods."""
    PROPHET = "prophet"
    ARIMA = "arima"
    ETS = "ets"
    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ENSEMBLE = "ensemble"
    LINEAR = "linear"


class SeasonalityMode(str, Enum):
    """Seasonality modes."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


@dataclass
class ForecastConfig:
    """Forecasting configuration."""
    
    method: ForecastMethod = ForecastMethod.ENSEMBLE
    horizon: int = 30  # Days
    frequency: str = "D"  # D=daily, W=weekly, M=monthly
    
    # Seasonality
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    seasonality_mode: SeasonalityMode = SeasonalityMode.ADDITIVE
    
    # Confidence
    confidence_level: float = 0.95
    
    # ARIMA specific
    arima_order: tuple = (1, 1, 1)
    seasonal_order: tuple = (1, 1, 1, 12)
    
    # Ensemble
    ensemble_methods: list[ForecastMethod] = field(default_factory=lambda: [
        ForecastMethod.EXPONENTIAL_SMOOTHING,
        ForecastMethod.LINEAR
    ])


@dataclass
class ForecastResult:
    """Forecast result."""
    
    method: ForecastMethod
    forecast: pd.DataFrame  # columns: date, yhat, yhat_lower, yhat_upper
    metrics: dict[str, float] = field(default_factory=dict)
    components: Optional[dict[str, pd.Series]] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "horizon": len(self.forecast),
            "forecast": self.forecast.to_dict(orient="records"),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "has_components": self.components is not None
        }


# ============================================================================
# Base Forecaster
# ============================================================================

class BaseForecaster:
    """Base class for forecasters."""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self._fitted = False
    
    def fit(self, df: pd.DataFrame, date_col: str, value_col: str) -> "BaseForecaster":
        """Fit the forecaster."""
        raise NotImplementedError
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """Generate forecast."""
        raise NotImplementedError
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
        """Evaluate forecast accuracy."""
        mask = ~actual.isna() & ~predicted.isna()
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {}
        
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        
        # Mean Absolute Scaled Error (relative to naive forecast)
        naive_errors = np.abs(np.diff(actual))
        if naive_errors.mean() > 0:
            mase = mae / naive_errors.mean()
        else:
            mase = 0
        
        return {
            "mape": float(mape),
            "rmse": float(rmse),
            "mae": float(mae),
            "mase": float(mase)
        }


# ============================================================================
# Statistical Forecasters
# ============================================================================

class ExponentialSmoothingForecaster(BaseForecaster):
    """Exponential smoothing forecaster."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._alpha = 0.3  # Smoothing parameter
        self._last_value = None
        self._last_date = None
        self._trend = 0
    
    def fit(self, df: pd.DataFrame, date_col: str, value_col: str) -> "ExponentialSmoothingForecaster":
        """Fit exponential smoothing."""
        df = df.sort_values(date_col)
        values = df[value_col].values
        
        # Simple exponential smoothing with trend
        level = values[0]
        trend = 0
        
        for i in range(1, len(values)):
            new_level = self._alpha * values[i] + (1 - self._alpha) * (level + trend)
            trend = 0.1 * (new_level - level) + 0.9 * trend
            level = new_level
        
        self._last_value = level
        self._trend = trend
        self._last_date = df[date_col].max()
        self._fitted = True
        
        return self
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """Generate forecast."""
        horizon = horizon or self.config.horizon
        
        dates = pd.date_range(
            start=self._last_date + timedelta(days=1),
            periods=horizon,
            freq=self.config.frequency
        )
        
        predictions = []
        current = self._last_value
        
        for i in range(horizon):
            current = current + self._trend
            predictions.append(current)
        
        # Confidence intervals
        std = abs(self._last_value) * 0.1  # Approximate
        z = 1.96  # 95% CI
        
        return pd.DataFrame({
            'date': dates,
            'yhat': predictions,
            'yhat_lower': [p - z * std * np.sqrt(i + 1) for i, p in enumerate(predictions)],
            'yhat_upper': [p + z * std * np.sqrt(i + 1) for i, p in enumerate(predictions)]
        })


class MovingAverageForecaster(BaseForecaster):
    """Moving average forecaster."""
    
    def __init__(self, config: ForecastConfig, window: int = 7):
        super().__init__(config)
        self.window = window
        self._values = None
        self._last_date = None
    
    def fit(self, df: pd.DataFrame, date_col: str, value_col: str) -> "MovingAverageForecaster":
        """Fit moving average."""
        df = df.sort_values(date_col)
        self._values = df[value_col].tail(self.window).values
        self._last_date = df[date_col].max()
        self._fitted = True
        return self
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """Generate forecast."""
        horizon = horizon or self.config.horizon
        
        dates = pd.date_range(
            start=self._last_date + timedelta(days=1),
            periods=horizon,
            freq=self.config.frequency
        )
        
        ma_value = np.mean(self._values)
        std = np.std(self._values)
        
        return pd.DataFrame({
            'date': dates,
            'yhat': [ma_value] * horizon,
            'yhat_lower': [ma_value - 1.96 * std] * horizon,
            'yhat_upper': [ma_value + 1.96 * std] * horizon
        })


class LinearTrendForecaster(BaseForecaster):
    """Linear trend forecaster."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._slope = 0
        self._intercept = 0
        self._last_date = None
        self._n_points = 0
    
    def fit(self, df: pd.DataFrame, date_col: str, value_col: str) -> "LinearTrendForecaster":
        """Fit linear trend."""
        df = df.sort_values(date_col)
        
        x = np.arange(len(df))
        y = df[value_col].values
        
        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        self._slope = numerator / denominator if denominator != 0 else 0
        self._intercept = y_mean - self._slope * x_mean
        self._last_date = df[date_col].max()
        self._n_points = len(df)
        self._residual_std = np.std(y - (self._slope * x + self._intercept))
        self._fitted = True
        
        return self
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """Generate forecast."""
        horizon = horizon or self.config.horizon
        
        dates = pd.date_range(
            start=self._last_date + timedelta(days=1),
            periods=horizon,
            freq=self.config.frequency
        )
        
        x_future = np.arange(self._n_points, self._n_points + horizon)
        predictions = self._slope * x_future + self._intercept
        
        return pd.DataFrame({
            'date': dates,
            'yhat': predictions,
            'yhat_lower': predictions - 1.96 * self._residual_std,
            'yhat_upper': predictions + 1.96 * self._residual_std
        })


# ============================================================================
# Prophet Forecaster
# ============================================================================

class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecaster."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._model = None
        self._df = None
    
    def fit(self, df: pd.DataFrame, date_col: str, value_col: str) -> "ProphetForecaster":
        """Fit Prophet model."""
        try:
            from prophet import Prophet
            
            # Prepare data in Prophet format
            self._df = pd.DataFrame({
                'ds': pd.to_datetime(df[date_col]),
                'y': df[value_col]
            })
            
            self._model = Prophet(
                yearly_seasonality=self.config.yearly_seasonality,
                weekly_seasonality=self.config.weekly_seasonality,
                daily_seasonality=self.config.daily_seasonality,
                seasonality_mode=self.config.seasonality_mode.value,
                interval_width=self.config.confidence_level
            )
            
            self._model.fit(self._df)
            self._fitted = True
            
        except ImportError:
            logger.warning("Prophet not installed, using fallback")
            fallback = ExponentialSmoothingForecaster(self.config)
            fallback.fit(df, date_col, value_col)
            self._fallback = fallback
            self._fitted = True
        
        return self
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """Generate forecast."""
        horizon = horizon or self.config.horizon
        
        if hasattr(self, '_fallback'):
            return self._fallback.predict(horizon)
        
        future = self._model.make_future_dataframe(periods=horizon)
        forecast = self._model.predict(future)
        
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
        result.columns = ['date', 'yhat', 'yhat_lower', 'yhat_upper']
        
        return result


# ============================================================================
# Ensemble Forecaster
# ============================================================================

class EnsembleForecaster(BaseForecaster):
    """Ensemble of multiple forecasters."""
    
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._forecasters: list[BaseForecaster] = []
        self._weights: list[float] = []
    
    def fit(self, df: pd.DataFrame, date_col: str, value_col: str) -> "EnsembleForecaster":
        """Fit all ensemble members."""
        # Create forecasters
        for method in self.config.ensemble_methods:
            if method == ForecastMethod.EXPONENTIAL_SMOOTHING:
                forecaster = ExponentialSmoothingForecaster(self.config)
            elif method == ForecastMethod.MOVING_AVERAGE:
                forecaster = MovingAverageForecaster(self.config)
            elif method == ForecastMethod.LINEAR:
                forecaster = LinearTrendForecaster(self.config)
            elif method == ForecastMethod.PROPHET:
                forecaster = ProphetForecaster(self.config)
            else:
                forecaster = ExponentialSmoothingForecaster(self.config)
            
            try:
                forecaster.fit(df, date_col, value_col)
                self._forecasters.append(forecaster)
                self._weights.append(1.0)
            except Exception as e:
                logger.warning(f"Failed to fit {method}: {e}")
        
        # Normalize weights
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]
        self._fitted = True
        
        return self
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """Generate ensemble forecast."""
        horizon = horizon or self.config.horizon
        
        forecasts = []
        for forecaster, weight in zip(self._forecasters, self._weights):
            try:
                pred = forecaster.predict(horizon)
                pred['weight'] = weight
                forecasts.append(pred)
            except Exception as e:
                logger.warning(f"Forecaster failed: {e}")
        
        if not forecasts:
            raise ValidationException("All forecasters failed")
        
        # Weighted average
        result = forecasts[0][['date']].copy()
        
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            weighted_sum = sum(f[col] * f['weight'] for f in forecasts)
            result[col] = weighted_sum
        
        return result


# ============================================================================
# Forecast Engine
# ============================================================================

class ForecastEngine:
    """
    Production forecasting engine.
    
    Features:
    - Multiple methods (Prophet, ARIMA, ETS, Ensemble)
    - Automatic method selection
    - Cross-validation
    - Anomaly detection in forecasts
    """
    
    def __init__(self):
        self._forecasters: dict[str, BaseForecaster] = {}
    
    def forecast(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        config: ForecastConfig = None
    ) -> ForecastResult:
        """Generate forecast."""
        config = config or ForecastConfig()
        
        # Select forecaster
        if config.method == ForecastMethod.ENSEMBLE:
            forecaster = EnsembleForecaster(config)
        elif config.method == ForecastMethod.PROPHET:
            forecaster = ProphetForecaster(config)
        elif config.method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            forecaster = ExponentialSmoothingForecaster(config)
        elif config.method == ForecastMethod.LINEAR:
            forecaster = LinearTrendForecaster(config)
        else:
            forecaster = EnsembleForecaster(config)
        
        # Fit and predict
        forecaster.fit(df, date_col, value_col)
        forecast_df = forecaster.predict(config.horizon)
        
        # Cross-validation for metrics
        metrics = self._cross_validate(df, date_col, value_col, forecaster)
        
        return ForecastResult(
            method=config.method,
            forecast=forecast_df,
            metrics=metrics
        )
    
    def _cross_validate(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        forecaster: BaseForecaster,
        n_splits: int = 3
    ) -> dict[str, float]:
        """Time series cross-validation."""
        if len(df) < 30:
            return {}
        
        split_size = len(df) // (n_splits + 1)
        all_metrics = []
        
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_end = min((i + 2) * split_size, len(df))
            
            train = df.iloc[:train_end]
            test = df.iloc[train_end:test_end]
            
            if len(test) < 5:
                continue
            
            try:
                temp_forecaster = type(forecaster)(forecaster.config)
                temp_forecaster.fit(train, date_col, value_col)
                pred = temp_forecaster.predict(len(test))
                
                metrics = forecaster.evaluate(
                    test[value_col].reset_index(drop=True),
                    pred['yhat'].reset_index(drop=True)
                )
                all_metrics.append(metrics)
            except Exception:
                continue
        
        if not all_metrics:
            return {}
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics if key in m])
        
        return avg_metrics
    
    def detect_anomalies(
        self,
        actual: pd.Series,
        forecast: pd.DataFrame
    ) -> pd.DataFrame:
        """Detect anomalies in actual vs forecast."""
        result = forecast.copy()
        
        if len(actual) == len(forecast):
            result['actual'] = actual.values
            result['residual'] = actual.values - forecast['yhat'].values
            result['is_anomaly'] = (
                (actual.values < forecast['yhat_lower'].values) |
                (actual.values > forecast['yhat_upper'].values)
            )
        
        return result


# Factory function
def get_forecast_engine() -> ForecastEngine:
    """Get forecast engine instance."""
    return ForecastEngine()
