# AI Enterprise Data Analyst - Time Series Agent
# Agent for time series analysis and forecasting

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
)
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


class TrendType(str, Enum):
    """Types of trends in time series."""
    UPWARD = "upward"
    DOWNWARD = "downward"
    STATIONARY = "stationary"
    VOLATILE = "volatile"


class SeasonalityType(str, Enum):
    """Types of seasonality."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    NONE = "none"


@dataclass
class TimeSeriesAnalysis:
    """Result of time series analysis."""
    
    # Basic info
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    n_observations: int = 0
    frequency: Optional[str] = None
    
    # Trend analysis
    trend: TrendType = TrendType.STATIONARY
    trend_slope: float = 0.0
    trend_strength: float = 0.0
    
    # Stationarity
    is_stationary: bool = False
    adf_statistic: Optional[float] = None
    adf_pvalue: Optional[float] = None
    
    # Seasonality
    has_seasonality: bool = False
    seasonality_type: SeasonalityType = SeasonalityType.NONE
    seasonality_strength: float = 0.0
    
    # Decomposition
    decomposition: dict[str, list[float]] = field(default_factory=dict)
    
    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    autocorrelation: list[float] = field(default_factory=list)
    
    # Anomalies
    anomaly_indices: list[int] = field(default_factory=list)
    anomaly_percentage: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "period": {
                "start": self.start_date.isoformat() if self.start_date else None,
                "end": self.end_date.isoformat() if self.end_date else None,
                "observations": self.n_observations,
                "frequency": self.frequency
            },
            "trend": {
                "type": self.trend.value,
                "slope": round(self.trend_slope, 6),
                "strength": round(self.trend_strength, 4)
            },
            "stationarity": {
                "is_stationary": self.is_stationary,
                "adf_statistic": round(self.adf_statistic, 4) if self.adf_statistic else None,
                "adf_pvalue": round(self.adf_pvalue, 6) if self.adf_pvalue else None
            },
            "seasonality": {
                "has_seasonality": self.has_seasonality,
                "type": self.seasonality_type.value,
                "strength": round(self.seasonality_strength, 4)
            },
            "statistics": {
                "mean": round(self.mean, 4),
                "std": round(self.std, 4),
                "min": round(self.min_value, 4),
                "max": round(self.max_value, 4)
            },
            "anomalies": {
                "count": len(self.anomaly_indices),
                "percentage": round(self.anomaly_percentage, 2)
            }
        }


@dataclass
class ForecastResult:
    """Result of time series forecasting."""
    
    model_name: str = ""
    training_samples: int = 0
    forecast_periods: int = 0
    
    # Predictions
    predictions: list[float] = field(default_factory=list)
    lower_bound: list[float] = field(default_factory=list)
    upper_bound: list[float] = field(default_factory=list)
    confidence_level: float = 0.95
    
    # Metrics
    train_metrics: dict[str, float] = field(default_factory=dict)
    
    # Model info
    model_params: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "training_samples": self.training_samples,
            "forecast_periods": self.forecast_periods,
            "predictions": [round(p, 4) for p in self.predictions[:10]],
            "confidence_level": self.confidence_level,
            "metrics": {k: round(v, 4) for k, v in self.train_metrics.items()},
            "model_params": self.model_params
        }


class TimeSeriesEngine:
    """
    Time series analysis and forecasting engine.
    
    Features:
    - Trend and seasonality detection
    - Stationarity tests (ADF, KPSS)
    - Decomposition
    - Anomaly detection
    - Forecasting (ARIMA, Holt-Winters, Prophet)
    """
    
    def analyze(
        self,
        series: pd.Series,
        datetime_index: Optional[pd.DatetimeIndex] = None
    ) -> TimeSeriesAnalysis:
        """Perform comprehensive time series analysis."""
        analysis = TimeSeriesAnalysis()
        
        # Convert to numeric and handle missing
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(series) < 10:
            return analysis
        
        # Basic info
        analysis.n_observations = len(series)
        analysis.mean = float(series.mean())
        analysis.std = float(series.std())
        analysis.min_value = float(series.min())
        analysis.max_value = float(series.max())
        
        if datetime_index is not None and len(datetime_index) > 0:
            analysis.start_date = datetime_index[0].to_pydatetime() if hasattr(datetime_index[0], 'to_pydatetime') else datetime_index[0]
            analysis.end_date = datetime_index[-1].to_pydatetime() if hasattr(datetime_index[-1], 'to_pydatetime') else datetime_index[-1]
            analysis.frequency = self._detect_frequency(datetime_index)
        
        # Trend analysis
        self._analyze_trend(series, analysis)
        
        # Stationarity tests
        self._test_stationarity(series, analysis)
        
        # Seasonality detection
        self._detect_seasonality(series, analysis)
        
        # Anomaly detection
        self._detect_anomalies(series, analysis)
        
        # Autocorrelation
        if HAS_STATSMODELS:
            try:
                acf_values = acf(series, nlags=min(20, len(series) // 2), fft=True)
                analysis.autocorrelation = acf_values.tolist()
            except:
                pass
        
        return analysis
    
    def _detect_frequency(self, datetime_index: pd.DatetimeIndex) -> str:
        """Detect time series frequency."""
        if len(datetime_index) < 2:
            return "unknown"
        
        # Calculate median difference
        diffs = pd.Series(datetime_index).diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            return "hourly"
        elif median_diff <= pd.Timedelta(days=1):
            return "daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "weekly"
        elif median_diff <= pd.Timedelta(days=31):
            return "monthly"
        else:
            return "yearly"
    
    def _analyze_trend(self, series: pd.Series, analysis: TimeSeriesAnalysis) -> None:
        """Analyze trend in the series."""
        n = len(series)
        x = np.arange(n)
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        analysis.trend_slope = slope
        analysis.trend_strength = abs(r_value)
        
        # Classify trend
        if abs(slope) < 0.001 * series.std():
            analysis.trend = TrendType.STATIONARY
        elif slope > 0:
            analysis.trend = TrendType.UPWARD
        else:
            analysis.trend = TrendType.DOWNWARD
        
        # Check volatility
        rolling_std = series.rolling(window=min(10, n // 5)).std()
        if rolling_std.std() > series.std() * 0.5:
            analysis.trend = TrendType.VOLATILE
    
    def _test_stationarity(self, series: pd.Series, analysis: TimeSeriesAnalysis) -> None:
        """Test for stationarity using ADF test."""
        if not HAS_STATSMODELS or len(series) < 20:
            return
        
        try:
            adf_result = adfuller(series.values, autolag='AIC')
            analysis.adf_statistic = adf_result[0]
            analysis.adf_pvalue = adf_result[1]
            analysis.is_stationary = adf_result[1] < 0.05
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
    
    def _detect_seasonality(self, series: pd.Series, analysis: TimeSeriesAnalysis) -> None:
        """Detect seasonality in the series."""
        if not HAS_STATSMODELS or len(series) < 24:
            return
        
        # Try different seasonal periods
        periods_to_try = [7, 12, 24, 52, 365]
        
        for period in periods_to_try:
            if len(series) < 2 * period:
                continue
            
            try:
                decomposition = seasonal_decompose(
                    series.values,
                    period=period,
                    extrapolate_trend='freq'
                )
                
                # Calculate seasonality strength
                seasonal_var = np.var(decomposition.seasonal)
                residual_var = np.var(decomposition.resid[~np.isnan(decomposition.resid)])
                
                if seasonal_var + residual_var > 0:
                    strength = 1 - (residual_var / (seasonal_var + residual_var))
                    
                    if strength > 0.5 and strength > analysis.seasonality_strength:
                        analysis.has_seasonality = True
                        analysis.seasonality_strength = strength
                        
                        # Map period to type
                        if period == 7:
                            analysis.seasonality_type = SeasonalityType.WEEKLY
                        elif period == 12:
                            analysis.seasonality_type = SeasonalityType.MONTHLY
                        elif period == 52 or period == 365:
                            analysis.seasonality_type = SeasonalityType.YEARLY
                        
                        # Store decomposition
                        analysis.decomposition = {
                            "trend": decomposition.trend[~np.isnan(decomposition.trend)].tolist()[:100],
                            "seasonal": decomposition.seasonal[:100].tolist(),
                            "residual": decomposition.resid[~np.isnan(decomposition.resid)].tolist()[:100]
                        }
            except:
                continue
    
    def _detect_anomalies(self, series: pd.Series, analysis: TimeSeriesAnalysis) -> None:
        """Detect anomalies using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        anomaly_mask = (series < lower) | (series > upper)
        analysis.anomaly_indices = series[anomaly_mask].index.tolist()
        analysis.anomaly_percentage = anomaly_mask.sum() / len(series) * 100
    
    def forecast(
        self,
        series: pd.Series,
        periods: int = 30,
        model: str = "auto"
    ) -> ForecastResult:
        """Generate forecast for the time series."""
        result = ForecastResult(forecast_periods=periods, training_samples=len(series))
        
        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(series) < 10:
            return result
        
        # Choose model
        if model == "auto":
            model = self._select_best_model(series)
        
        # Fit and forecast
        if model == "prophet" and HAS_PROPHET:
            return self._forecast_prophet(series, periods)
        elif model == "arima" and HAS_STATSMODELS:
            return self._forecast_arima(series, periods)
        elif model == "holtwinters" and HAS_STATSMODELS:
            return self._forecast_holtwinters(series, periods)
        else:
            # Simple exponential smoothing fallback
            return self._forecast_simple(series, periods)
    
    def _select_best_model(self, series: pd.Series) -> str:
        """Select best forecasting model based on data characteristics."""
        if len(series) > 100 and HAS_PROPHET:
            return "prophet"
        elif HAS_STATSMODELS:
            return "holtwinters"
        else:
            return "simple"
    
    def _forecast_prophet(self, series: pd.Series, periods: int) -> ForecastResult:
        """Forecast using Prophet."""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
            'y': series.values
        })
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return ForecastResult(
            model_name="Prophet",
            training_samples=len(series),
            forecast_periods=periods,
            predictions=forecast['yhat'].tail(periods).tolist(),
            lower_bound=forecast['yhat_lower'].tail(periods).tolist(),
            upper_bound=forecast['yhat_upper'].tail(periods).tolist(),
            train_metrics={"mape": self._calculate_mape(series.values, forecast['yhat'].head(len(series)).values)}
        )
    
    def _forecast_arima(self, series: pd.Series, periods: int) -> ForecastResult:
        """Forecast using ARIMA."""
        try:
            model = ARIMA(series.values, order=(1, 1, 1))
            fitted = model.fit()
            
            forecast = fitted.get_forecast(steps=periods)
            predictions = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            return ForecastResult(
                model_name="ARIMA(1,1,1)",
                training_samples=len(series),
                forecast_periods=periods,
                predictions=predictions.tolist(),
                lower_bound=conf_int[:, 0].tolist(),
                upper_bound=conf_int[:, 1].tolist(),
                train_metrics={"aic": fitted.aic, "bic": fitted.bic}
            )
        except:
            return self._forecast_simple(series, periods)
    
    def _forecast_holtwinters(self, series: pd.Series, periods: int) -> ForecastResult:
        """Forecast using Holt-Winters Exponential Smoothing."""
        try:
            model = ExponentialSmoothing(
                series.values,
                trend='add',
                seasonal=None,
                damped_trend=True
            )
            fitted = model.fit()
            
            forecast = fitted.forecast(periods)
            
            return ForecastResult(
                model_name="Holt-Winters",
                training_samples=len(series),
                forecast_periods=periods,
                predictions=forecast.tolist(),
                train_metrics={"sse": fitted.sse}
            )
        except:
            return self._forecast_simple(series, periods)
    
    def _forecast_simple(self, series: pd.Series, periods: int) -> ForecastResult:
        """Simple exponential smoothing forecast."""
        alpha = 0.3
        level = series.iloc[0]
        
        # Fit
        for val in series:
            level = alpha * val + (1 - alpha) * level
        
        # Forecast
        predictions = [level] * periods
        std = series.std()
        
        return ForecastResult(
            model_name="Simple Exponential Smoothing",
            training_samples=len(series),
            forecast_periods=periods,
            predictions=predictions,
            lower_bound=[p - 1.96 * std for p in predictions],
            upper_bound=[p + 1.96 * std for p in predictions]
        )
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


class TimeSeriesAgent(BaseAgent[dict[str, Any]]):
    """
    Time Series Agent for temporal data analysis.
    
    Capabilities:
    - Trend and seasonality detection
    - Stationarity testing
    - Time series decomposition
    - Anomaly detection
    - Forecasting with multiple models
    """
    
    name: str = "TimeSeriesAgent"
    description: str = "Time series analysis and forecasting"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.engine = TimeSeriesEngine()
    
    def _register_tools(self) -> None:
        """Register time series tools."""
        
        self.register_tool(AgentTool(
            name="analyze_timeseries",
            description="Analyze a time series for trends, seasonality, and stationarity",
            function=self._analyze_timeseries,
            parameters={
                "data": {"type": "array", "items": {"type": "number"}},
                "datetime_index": {"type": "array", "items": {"type": "string"}}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="forecast",
            description="Generate forecasts for a time series",
            function=self._forecast,
            parameters={
                "data": {"type": "array", "items": {"type": "number"}},
                "periods": {"type": "integer", "default": 30},
                "model": {"type": "string", "enum": ["auto", "prophet", "arima", "holtwinters"]}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="detect_anomalies",
            description="Detect anomalies in time series data",
            function=self._detect_anomalies,
            parameters={
                "data": {"type": "array", "items": {"type": "number"}}
            },
            required_params=["data"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute time series analysis."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a time series expert. Provide analysis and recommendations."
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "models_available": ["Prophet", "ARIMA", "Holt-Winters"]
        }
    
    async def _analyze_timeseries(
        self,
        data: list[float],
        datetime_index: list[str] = None
    ) -> dict[str, Any]:
        """Analyze time series."""
        series = pd.Series(data)
        
        dt_index = None
        if datetime_index:
            try:
                dt_index = pd.to_datetime(datetime_index)
            except:
                pass
        
        analysis = self.engine.analyze(series, dt_index)
        
        return {
            "status": "success",
            "analysis": analysis.to_dict()
        }
    
    async def _forecast(
        self,
        data: list[float],
        periods: int = 30,
        model: str = "auto"
    ) -> dict[str, Any]:
        """Generate forecast."""
        series = pd.Series(data)
        result = self.engine.forecast(series, periods, model)
        
        return {
            "status": "success",
            "forecast": result.to_dict()
        }
    
    async def _detect_anomalies(
        self,
        data: list[float]
    ) -> dict[str, Any]:
        """Detect anomalies in time series."""
        series = pd.Series(data)
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        anomalies = series[(series < lower) | (series > upper)]
        
        return {
            "status": "success",
            "anomaly_count": len(anomalies),
            "anomaly_indices": anomalies.index.tolist(),
            "anomaly_values": anomalies.tolist(),
            "bounds": {"lower": lower, "upper": upper}
        }


# Factory function
def get_timeseries_agent() -> TimeSeriesAgent:
    """Get time series agent instance."""
    return TimeSeriesAgent()
