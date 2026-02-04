# AI Enterprise Data Analyst - Univariate Analysis Engine
# Production-grade single variable analysis with complete edge case handling
# Handles: numeric, categorical, datetime, text - robust to all data issues

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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
# Enums
# ============================================================================

class VariableType(str, Enum):
    """Variable types."""
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    TEXT = "text"
    BINARY = "binary"
    CONSTANT = "constant"
    IDENTIFIER = "identifier"


class DistributionShape(str, Enum):
    """Distribution shape."""
    NORMAL = "normal"
    SKEWED_RIGHT = "skewed_right"
    SKEWED_LEFT = "skewed_left"
    BIMODAL = "bimodal"
    UNIFORM = "uniform"
    HEAVY_TAILED = "heavy_tailed"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NumericAnalysis:
    """Complete numeric variable analysis."""
    # Central tendency
    mean: float
    median: float
    mode: Optional[float]
    trimmed_mean: float  # 5% trimmed
    
    # Dispersion
    std: float
    variance: float
    mad: float  # Median Absolute Deviation
    iqr: float
    range_val: float
    cv: float  # Coefficient of variation
    
    # Position
    min_val: float
    max_val: float
    q1: float
    q3: float
    percentiles: Dict[int, float]
    
    # Shape
    skewness: float
    kurtosis: float
    distribution_shape: DistributionShape
    is_normal: bool
    normality_pvalue: float
    
    # Outliers
    outlier_count: int
    outlier_pct: float
    outlier_indices: List[int]
    
    # Missing
    missing_count: int
    missing_pct: float
    
    # Special values
    zero_count: int
    negative_count: int
    infinite_count: int


@dataclass
class CategoricalAnalysis:
    """Complete categorical variable analysis."""
    # Cardinality
    unique_count: int
    unique_ratio: float
    
    # Frequency
    value_counts: Dict[str, int]
    value_percentages: Dict[str, float]
    mode: str
    mode_frequency: int
    mode_percentage: float
    
    # Diversity
    entropy: float
    gini_impurity: float
    
    # Missing
    missing_count: int
    missing_pct: float
    
    # Rare values
    rare_values: List[str]  # Less than 1%
    rare_count: int


@dataclass
class DatetimeAnalysis:
    """Complete datetime variable analysis."""
    min_date: datetime
    max_date: datetime
    range_days: int
    
    # Distribution
    weekday_distribution: Dict[str, int]
    month_distribution: Dict[int, int]
    year_distribution: Dict[int, int]
    hour_distribution: Optional[Dict[int, int]]
    
    # Patterns
    most_common_weekday: str
    most_common_month: int
    most_common_hour: Optional[int]
    
    # Gaps
    has_gaps: bool
    max_gap_days: int
    
    # Missing
    missing_count: int
    missing_pct: float


@dataclass
class UnivariateResult:
    """Complete univariate analysis result."""
    column_name: str
    variable_type: VariableType
    n_total: int
    n_valid: int
    n_missing: int
    missing_pct: float
    
    # Type-specific analysis
    numeric_analysis: Optional[NumericAnalysis] = None
    categorical_analysis: Optional[CategoricalAnalysis] = None
    datetime_analysis: Optional[DatetimeAnalysis] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "column": self.column_name,
            "type": self.variable_type.value,
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "missing_pct": round(self.missing_pct, 2),
            "recommendations": self.recommendations,
            "warnings": self.warnings
        }
        
        if self.numeric_analysis:
            na = self.numeric_analysis
            result["numeric"] = {
                "mean": self._safe_round(na.mean),
                "median": self._safe_round(na.median),
                "std": self._safe_round(na.std),
                "min": self._safe_round(na.min_val),
                "max": self._safe_round(na.max_val),
                "skewness": self._safe_round(na.skewness),
                "kurtosis": self._safe_round(na.kurtosis),
                "distribution_shape": na.distribution_shape.value,
                "is_normal": na.is_normal,
                "outlier_pct": round(na.outlier_pct, 2),
                "zero_count": na.zero_count
            }
        
        if self.categorical_analysis:
            ca = self.categorical_analysis
            result["categorical"] = {
                "unique_count": ca.unique_count,
                "unique_ratio": round(ca.unique_ratio, 4),
                "mode": ca.mode,
                "mode_pct": round(ca.mode_percentage, 2),
                "entropy": round(ca.entropy, 4),
                "rare_count": ca.rare_count,
                "top_values": dict(list(ca.value_counts.items())[:10])
            }
        
        if self.datetime_analysis:
            da = self.datetime_analysis
            result["datetime"] = {
                "min_date": da.min_date.isoformat() if da.min_date else None,
                "max_date": da.max_date.isoformat() if da.max_date else None,
                "range_days": da.range_days,
                "most_common_weekday": da.most_common_weekday,
                "has_gaps": da.has_gaps
            }
        
        return result

    @property
    def std(self) -> float:
        na = self.numeric_analysis
        if na is None or na.std is None:
            return 0.0
        try:
            return float(na.std)
        except Exception:
            return 0.0
    
    def _safe_round(self, val: float, decimals: int = 4) -> Optional[float]:
        if val is None or np.isnan(val) or np.isinf(val):
            return None
        return round(val, decimals)


# ============================================================================
# Univariate Analysis Engine
# ============================================================================

class UnivariateAnalysisEngine:
    """
    Production-grade Univariate Analysis engine.
    
    Features:
    - Automatic type detection
    - Complete edge case handling
    - Robust statistics for non-normal data
    - Comprehensive outlier detection
    - Missing value analysis
    - Actionable recommendations
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        data: Union[pd.Series, pd.DataFrame],
        column: str = None
    ) -> UnivariateResult:
        """Analyze a single variable with complete edge case handling."""
        start_time = datetime.now()
        
        # Extract series
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.columns[0]
            series = data[column].copy()
        else:
            series = data.copy()
            column = series.name or "variable"
        
        if self.verbose:
            logger.info(f"Univariate analysis: {column}, n={len(series)}")
        
        # Basic counts
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_valid = n_total - n_missing
        missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0
        
        # Edge case: all missing
        if n_valid == 0:
            return self._create_empty_result(column, n_total, missing_pct, start_time)
        
        # Detect variable type
        var_type = self._detect_type(series)
        
        # Type-specific analysis
        numeric_analysis = None
        categorical_analysis = None
        datetime_analysis = None
        recommendations = []
        warnings_list = []
        
        if var_type in [VariableType.NUMERIC_CONTINUOUS, VariableType.NUMERIC_DISCRETE]:
            numeric_analysis = self._analyze_numeric(series)
            recommendations, warnings_list = self._numeric_recommendations(numeric_analysis, series)
        
        elif var_type in [VariableType.CATEGORICAL_NOMINAL, VariableType.CATEGORICAL_ORDINAL, 
                          VariableType.BINARY]:
            categorical_analysis = self._analyze_categorical(series)
            recommendations, warnings_list = self._categorical_recommendations(categorical_analysis)
        
        elif var_type == VariableType.DATETIME:
            datetime_analysis = self._analyze_datetime(series)
            recommendations, warnings_list = self._datetime_recommendations(datetime_analysis)
        
        elif var_type == VariableType.CONSTANT:
            warnings_list.append("Column has only one unique value - no variance")
            recommendations.append("Consider removing this constant column")
        
        elif var_type == VariableType.IDENTIFIER:
            warnings_list.append("Column appears to be an identifier (unique for each row)")
            recommendations.append("Exclude from analysis unless needed as join key")
        
        # Missing data recommendations
        if missing_pct > 50:
            warnings_list.append(f"High missing rate: {missing_pct:.1f}%")
            recommendations.append("Consider dropping column or investigating data source")
        elif missing_pct > 10:
            recommendations.append("Consider imputation strategy for missing values")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return UnivariateResult(
            column_name=column,
            variable_type=var_type,
            n_total=n_total,
            n_valid=n_valid,
            n_missing=n_missing,
            missing_pct=missing_pct,
            numeric_analysis=numeric_analysis,
            categorical_analysis=categorical_analysis,
            datetime_analysis=datetime_analysis,
            recommendations=recommendations,
            warnings=warnings_list,
            processing_time_sec=processing_time
        )
    
    def analyze_all(self, df: pd.DataFrame) -> List[UnivariateResult]:
        """Analyze all columns in DataFrame."""
        results = []
        for col in df.columns:
            try:
                result = self.analyze(df, col)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing {col}: {e}")
        return results
    
    def _detect_type(self, series: pd.Series) -> VariableType:
        """Detect variable type with edge case handling."""
        clean = series.dropna()
        
        if len(clean) == 0:
            return VariableType.CONSTANT
        
        n_unique = clean.nunique()
        n_total = len(clean)
        
        # Constant check
        if n_unique == 1:
            return VariableType.CONSTANT
        
        # Binary check
        if n_unique == 2:
            return VariableType.BINARY
        
        # Identifier check (near-unique)
        if n_unique / n_total > 0.95 and n_total > 20:
            return VariableType.IDENTIFIER
        
        # Datetime check
        if pd.api.types.is_datetime64_any_dtype(series):
            return VariableType.DATETIME
        
        # Numeric check
        if pd.api.types.is_numeric_dtype(series):
            # Discrete vs continuous
            if pd.api.types.is_integer_dtype(series) and n_unique < 20:
                return VariableType.NUMERIC_DISCRETE
            return VariableType.NUMERIC_CONTINUOUS
        
        # Try to convert to numeric
        numeric_converted = pd.to_numeric(clean, errors='coerce')
        if numeric_converted.notna().sum() / len(clean) > 0.8:
            return VariableType.NUMERIC_CONTINUOUS
        
        # Try to convert to datetime
        try:
            datetime_converted = pd.to_datetime(clean, errors='coerce', infer_datetime_format=True)
            if datetime_converted.notna().sum() / len(clean) > 0.8:
                return VariableType.DATETIME
        except:
            pass
        
        # Categorical
        if n_unique / n_total < 0.5:
            return VariableType.CATEGORICAL_NOMINAL
        
        return VariableType.TEXT
    
    def _analyze_numeric(self, series: pd.Series) -> NumericAnalysis:
        """Complete numeric analysis with robust statistics."""
        clean = series.dropna()
        values = clean.values.astype(float)
        
        # Handle edge case: all same value
        if len(np.unique(values)) == 1:
            val = values[0]
            return NumericAnalysis(
                mean=val, median=val, mode=val, trimmed_mean=val,
                std=0, variance=0, mad=0, iqr=0, range_val=0, cv=0,
                min_val=val, max_val=val, q1=val, q3=val,
                percentiles={p: val for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
                skewness=0, kurtosis=0, distribution_shape=DistributionShape.UNKNOWN,
                is_normal=False, normality_pvalue=0,
                outlier_count=0, outlier_pct=0, outlier_indices=[],
                missing_count=int(series.isna().sum()),
                missing_pct=series.isna().sum() / len(series) * 100,
                zero_count=int((values == 0).sum()),
                negative_count=int((values < 0).sum()),
                infinite_count=0
            )
        
        # Handle infinities
        infinite_mask = np.isinf(values)
        infinite_count = int(infinite_mask.sum())
        values_finite = values[~infinite_mask]
        
        if len(values_finite) < 2:
            # Edge case: too few finite values
            return self._create_minimal_numeric(series, infinite_count)
        
        # Central tendency
        mean = float(np.mean(values_finite))
        median = float(np.median(values_finite))
        
        # Mode - handle multimodal
        try:
            mode_result = scipy_stats.mode(values_finite, keepdims=True)
            mode = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else median
        except:
            mode = median
        
        # Trimmed mean (robust to outliers)
        trimmed_mean = float(scipy_stats.trim_mean(values_finite, 0.05))
        
        # Dispersion
        std = float(np.std(values_finite, ddof=1))
        variance = float(np.var(values_finite, ddof=1))
        mad = float(np.median(np.abs(values_finite - median)))
        
        q1 = float(np.percentile(values_finite, 25))
        q3 = float(np.percentile(values_finite, 75))
        iqr = q3 - q1
        
        min_val = float(np.min(values_finite))
        max_val = float(np.max(values_finite))
        range_val = max_val - min_val
        
        cv = (std / abs(mean)) if mean != 0 else 0
        
        # Percentiles
        percentiles = {
            p: float(np.percentile(values_finite, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        
        # Shape
        skewness = float(scipy_stats.skew(values_finite))
        kurtosis = float(scipy_stats.kurtosis(values_finite))
        
        # Normality test
        if len(values_finite) >= 8:
            try:
                if len(values_finite) < 5000:
                    _, normality_p = scipy_stats.shapiro(values_finite[:5000])
                else:
                    _, normality_p = scipy_stats.normaltest(values_finite)
            except:
                normality_p = 0
        else:
            normality_p = 0
        
        is_normal = normality_p > 0.05
        
        # Distribution shape
        distribution_shape = self._determine_shape(skewness, kurtosis, is_normal)
        
        # Outliers (IQR method)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (values_finite < lower_bound) | (values_finite > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = outlier_count / len(values_finite) * 100
        
        # Get outlier indices in original series
        clean_indices = clean.index.tolist()
        outlier_indices = [clean_indices[i] for i, is_outlier in enumerate(outlier_mask) if is_outlier]
        
        # Special values
        zero_count = int((values_finite == 0).sum())
        negative_count = int((values_finite < 0).sum())
        
        return NumericAnalysis(
            mean=mean, median=median, mode=mode, trimmed_mean=trimmed_mean,
            std=std, variance=variance, mad=mad, iqr=iqr, range_val=range_val, cv=cv,
            min_val=min_val, max_val=max_val, q1=q1, q3=q3, percentiles=percentiles,
            skewness=skewness, kurtosis=kurtosis, distribution_shape=distribution_shape,
            is_normal=is_normal, normality_pvalue=normality_p,
            outlier_count=outlier_count, outlier_pct=outlier_pct, outlier_indices=outlier_indices[:100],
            missing_count=int(series.isna().sum()),
            missing_pct=series.isna().sum() / len(series) * 100,
            zero_count=zero_count, negative_count=negative_count, infinite_count=infinite_count
        )
    
    def _determine_shape(self, skewness: float, kurtosis: float, is_normal: bool) -> DistributionShape:
        """Determine distribution shape from statistics."""
        if is_normal and abs(skewness) < 0.5:
            return DistributionShape.NORMAL
        
        if skewness > 1:
            return DistributionShape.SKEWED_RIGHT
        elif skewness < -1:
            return DistributionShape.SKEWED_LEFT
        
        if kurtosis > 3:
            return DistributionShape.HEAVY_TAILED
        elif kurtosis < -1:
            return DistributionShape.UNIFORM
        
        return DistributionShape.UNKNOWN
    
    def _analyze_categorical(self, series: pd.Series) -> CategoricalAnalysis:
        """Complete categorical analysis."""
        clean = series.dropna().astype(str)
        n_total = len(clean)
        
        if n_total == 0:
            return self._create_empty_categorical(series)
        
        # Value counts
        vc = clean.value_counts()
        value_counts = vc.to_dict()
        value_percentages = (vc / n_total * 100).to_dict()
        
        # Cardinality
        unique_count = len(vc)
        unique_ratio = unique_count / n_total
        
        # Mode
        mode = str(vc.index[0])
        mode_frequency = int(vc.iloc[0])
        mode_percentage = mode_frequency / n_total * 100
        
        # Entropy (Shannon)
        probs = vc.values / n_total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        
        # Gini impurity
        gini = float(1 - np.sum(probs ** 2))
        
        # Rare values (less than 1%)
        rare_mask = vc / n_total < 0.01
        rare_values = vc[rare_mask].index.tolist()
        rare_count = len(rare_values)
        
        return CategoricalAnalysis(
            unique_count=unique_count, unique_ratio=unique_ratio,
            value_counts=value_counts, value_percentages=value_percentages,
            mode=mode, mode_frequency=mode_frequency, mode_percentage=mode_percentage,
            entropy=entropy, gini_impurity=gini,
            missing_count=int(series.isna().sum()),
            missing_pct=series.isna().sum() / len(series) * 100,
            rare_values=rare_values[:20], rare_count=rare_count
        )
    
    def _analyze_datetime(self, series: pd.Series) -> DatetimeAnalysis:
        """Complete datetime analysis."""
        clean = pd.to_datetime(series, errors='coerce').dropna()
        
        if len(clean) == 0:
            return self._create_empty_datetime(series)
        
        min_date = clean.min().to_pydatetime()
        max_date = clean.max().to_pydatetime()
        range_days = (max_date - min_date).days
        
        # Distributions
        weekday_dist = clean.dt.day_name().value_counts().to_dict()
        month_dist = clean.dt.month.value_counts().to_dict()
        year_dist = clean.dt.year.value_counts().to_dict()
        
        # Hour distribution (if time component exists)
        has_time = (clean.dt.hour.nunique() > 1) or (clean.dt.minute.nunique() > 1)
        hour_dist = clean.dt.hour.value_counts().to_dict() if has_time else None
        
        # Most common
        most_common_weekday = clean.dt.day_name().mode().iloc[0] if len(clean) > 0 else ""
        most_common_month = int(clean.dt.month.mode().iloc[0]) if len(clean) > 0 else 0
        most_common_hour = int(clean.dt.hour.mode().iloc[0]) if has_time and len(clean) > 0 else None
        
        # Gap analysis
        sorted_dates = clean.sort_values()
        diffs = sorted_dates.diff().dropna()
        max_gap = int(diffs.max().days) if len(diffs) > 0 else 0
        has_gaps = max_gap > diffs.median().days * 3 if len(diffs) > 0 else False
        
        return DatetimeAnalysis(
            min_date=min_date, max_date=max_date, range_days=range_days,
            weekday_distribution=weekday_dist, month_distribution=month_dist,
            year_distribution=year_dist, hour_distribution=hour_dist,
            most_common_weekday=most_common_weekday, most_common_month=most_common_month,
            most_common_hour=most_common_hour,
            has_gaps=has_gaps, max_gap_days=max_gap,
            missing_count=int(series.isna().sum()),
            missing_pct=series.isna().sum() / len(series) * 100
        )
    
    def _numeric_recommendations(
        self,
        analysis: NumericAnalysis,
        series: pd.Series
    ) -> Tuple[List[str], List[str]]:
        """Generate recommendations for numeric variable."""
        recs = []
        warns = []
        
        if analysis.outlier_pct > 10:
            warns.append(f"High outlier rate: {analysis.outlier_pct:.1f}%")
            recs.append("Consider outlier treatment (winsorization, capping, or removal)")
        
        if abs(analysis.skewness) > 2:
            warns.append(f"Highly skewed distribution (skewness: {analysis.skewness:.2f})")
            recs.append("Consider log or Box-Cox transformation")
        
        if analysis.zero_count / len(series.dropna()) > 0.5:
            warns.append(f"High zero rate: {analysis.zero_count / len(series.dropna()) * 100:.1f}%")
            recs.append("Consider zero-inflated models if this is target variable")
        
        if analysis.cv > 1:
            recs.append("High coefficient of variation - consider standardization")
        
        if not analysis.is_normal and series.dropna().nunique() > 10:
            recs.append("Non-normal distribution - use non-parametric tests")
        
        return recs, warns
    
    def _categorical_recommendations(
        self,
        analysis: CategoricalAnalysis
    ) -> Tuple[List[str], List[str]]:
        """Generate recommendations for categorical variable."""
        recs = []
        warns = []
        
        if analysis.unique_ratio > 0.5:
            warns.append("High cardinality categorical variable")
            recs.append("Consider grouping rare categories or feature hashing")
        
        if analysis.mode_percentage > 90:
            warns.append(f"Dominant category: {analysis.mode} ({analysis.mode_percentage:.1f}%)")
            recs.append("Limited predictive value - consider binary encoding")
        
        if analysis.rare_count > 10:
            recs.append("Consider grouping rare categories into 'Other'")
        
        if analysis.entropy < 0.5:
            recs.append("Low entropy - limited information content")
        
        return recs, warns
    
    def _datetime_recommendations(
        self,
        analysis: DatetimeAnalysis
    ) -> Tuple[List[str], List[str]]:
        """Generate recommendations for datetime variable."""
        recs = []
        warns = []
        
        if analysis.has_gaps:
            warns.append(f"Large gaps detected (max: {analysis.max_gap_days} days)")
            recs.append("Investigate data collection gaps")
        
        if analysis.range_days < 30:
            warns.append("Short date range - limited temporal patterns")
        
        recs.append("Consider extracting: day of week, month, quarter, year features")
        
        if analysis.hour_distribution:
            recs.append("Time component available - consider hour-of-day features")
        
        return recs, warns
    
    def _create_empty_result(
        self,
        column: str,
        n_total: int,
        missing_pct: float,
        start_time: datetime
    ) -> UnivariateResult:
        """Create result for all-missing column."""
        return UnivariateResult(
            column_name=column,
            variable_type=VariableType.CONSTANT,
            n_total=n_total,
            n_valid=0,
            n_missing=n_total,
            missing_pct=missing_pct,
            recommendations=["Column is entirely missing - consider removal"],
            warnings=["All values are missing"],
            processing_time_sec=(datetime.now() - start_time).total_seconds()
        )
    
    def _create_minimal_numeric(self, series: pd.Series, infinite_count: int) -> NumericAnalysis:
        """Create minimal numeric analysis for edge cases."""
        return NumericAnalysis(
            mean=0, median=0, mode=None, trimmed_mean=0,
            std=0, variance=0, mad=0, iqr=0, range_val=0, cv=0,
            min_val=0, max_val=0, q1=0, q3=0,
            percentiles={}, skewness=0, kurtosis=0,
            distribution_shape=DistributionShape.UNKNOWN,
            is_normal=False, normality_pvalue=0,
            outlier_count=0, outlier_pct=0, outlier_indices=[],
            missing_count=int(series.isna().sum()),
            missing_pct=series.isna().sum() / len(series) * 100,
            zero_count=0, negative_count=0, infinite_count=infinite_count
        )
    
    def _create_empty_categorical(self, series: pd.Series) -> CategoricalAnalysis:
        """Create empty categorical analysis."""
        return CategoricalAnalysis(
            unique_count=0, unique_ratio=0,
            value_counts={}, value_percentages={},
            mode="", mode_frequency=0, mode_percentage=0,
            entropy=0, gini_impurity=0,
            missing_count=int(series.isna().sum()),
            missing_pct=100,
            rare_values=[], rare_count=0
        )
    
    def _create_empty_datetime(self, series: pd.Series) -> DatetimeAnalysis:
        """Create empty datetime analysis."""
        return DatetimeAnalysis(
            min_date=None, max_date=None, range_days=0,
            weekday_distribution={}, month_distribution={},
            year_distribution={}, hour_distribution=None,
            most_common_weekday="", most_common_month=0, most_common_hour=None,
            has_gaps=False, max_gap_days=0,
            missing_count=int(series.isna().sum()),
            missing_pct=100
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_univariate_engine() -> UnivariateAnalysisEngine:
    """Get univariate analysis engine."""
    return UnivariateAnalysisEngine()


def quick_univariate(
    data: Union[pd.Series, pd.DataFrame],
    column: str = None
) -> Dict[str, Any]:
    """Quick univariate analysis."""
    engine = UnivariateAnalysisEngine(verbose=False)
    result = engine.analyze(data, column)
    return result.to_dict()


def analyze_all_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze all columns in DataFrame."""
    engine = UnivariateAnalysisEngine(verbose=False)
    results = engine.analyze_all(df)
    return [r.to_dict() for r in results]
