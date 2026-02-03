# AI Enterprise Data Analyst - Distribution Analysis Engine
# Production-grade distribution fitting and analysis
# Handles: any numeric data, multiple distribution tests

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import (
    norm, lognorm, expon, gamma, beta, uniform, weibull_min,
    poisson, nbinom, binom
)

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

class DistributionType(str, Enum):
    """Distribution types."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    UNIFORM = "uniform"
    WEIBULL = "weibull"
    POISSON = "poisson"
    BIMODAL = "bimodal"
    SKEWED = "skewed"
    HEAVY_TAILED = "heavy_tailed"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FittedDistribution:
    """Result of distribution fitting."""
    distribution: DistributionType
    params: Dict[str, float]
    ks_statistic: float
    ks_p_value: float
    aic: float
    is_good_fit: bool


@dataclass
class DistributionStats:
    """Distribution statistics."""
    mean: float
    median: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    min_value: float
    max_value: float
    range_value: float
    iqr: float
    q1: float
    q3: float


@dataclass
class DistributionResult:
    """Complete distribution analysis result."""
    n_observations: int = 0
    
    # Basic stats
    stats: DistributionStats = None
    
    # Best fit distribution
    best_fit: FittedDistribution = None
    
    # All fitted distributions
    fitted_distributions: List[FittedDistribution] = field(default_factory=list)
    
    # Normality tests
    is_normal: bool = False
    normality_test_p_value: float = 0.0
    
    # Distribution characteristics
    distribution_type: DistributionType = DistributionType.UNKNOWN
    
    # Percentiles
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "stats": {
                "mean": round(self.stats.mean, 4),
                "median": round(self.stats.median, 4),
                "std": round(self.stats.std, 4),
                "skewness": round(self.stats.skewness, 4),
                "kurtosis": round(self.stats.kurtosis, 4),
                "iqr": round(self.stats.iqr, 4)
            } if self.stats else {},
            "is_normal": self.is_normal,
            "normality_p_value": round(self.normality_test_p_value, 4),
            "distribution_type": self.distribution_type.value,
            "best_fit": {
                "distribution": self.best_fit.distribution.value,
                "ks_p_value": round(self.best_fit.ks_p_value, 4),
                "is_good_fit": self.best_fit.is_good_fit
            } if self.best_fit else None,
            "percentiles": {str(k): round(v, 4) for k, v in self.percentiles.items()}
        }


# ============================================================================
# Distribution Analysis Engine
# ============================================================================

class DistributionAnalysisEngine:
    """
    Complete Distribution Analysis engine.
    
    Features:
    - Distribution fitting
    - Normality testing
    - Skewness and kurtosis analysis
    - Percentile calculation
    """
    
    DISTRIBUTIONS = [
        ('normal', norm),
        ('lognormal', lognorm),
        ('exponential', expon),
        ('gamma', gamma),
        ('uniform', uniform),
        ('weibull', weibull_min)
    ]
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        data: pd.Series | np.ndarray | pd.DataFrame,
        column: str = None
    ) -> DistributionResult:
        """Perform distribution analysis."""
        start_time = datetime.now()
        
        # Extract data
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.select_dtypes(include=[np.number]).columns[0]
            values = data[column].dropna().values
        elif isinstance(data, pd.Series):
            values = data.dropna().values
        else:
            values = np.array(data)
            values = values[~np.isnan(values)]
        
        if len(values) < 10:
            raise DataProcessingException("Need at least 10 data points for distribution analysis")
        
        if self.verbose:
            logger.info(f"Analyzing distribution of {len(values)} values")
        
        # Calculate stats
        stats = self._calculate_stats(values)
        
        # Normality test
        is_normal, norm_p = self._test_normality(values)
        
        # Determine distribution type
        dist_type = self._determine_distribution_type(stats, is_normal)
        
        # Fit distributions
        fitted = self._fit_distributions(values)
        fitted.sort(key=lambda x: -x.ks_p_value)
        
        best_fit = fitted[0] if fitted else None
        
        # Calculate percentiles
        percentiles = {
            1: float(np.percentile(values, 1)),
            5: float(np.percentile(values, 5)),
            10: float(np.percentile(values, 10)),
            25: float(np.percentile(values, 25)),
            50: float(np.percentile(values, 50)),
            75: float(np.percentile(values, 75)),
            90: float(np.percentile(values, 90)),
            95: float(np.percentile(values, 95)),
            99: float(np.percentile(values, 99))
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DistributionResult(
            n_observations=len(values),
            stats=stats,
            best_fit=best_fit,
            fitted_distributions=fitted,
            is_normal=is_normal,
            normality_test_p_value=norm_p,
            distribution_type=dist_type,
            percentiles=percentiles,
            processing_time_sec=processing_time
        )
    
    def _calculate_stats(self, values: np.ndarray) -> DistributionStats:
        """Calculate distribution statistics."""
        return DistributionStats(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values)),
            variance=float(np.var(values)),
            skewness=float(scipy_stats.skew(values)),
            kurtosis=float(scipy_stats.kurtosis(values)),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            range_value=float(np.max(values) - np.min(values)),
            iqr=float(np.percentile(values, 75) - np.percentile(values, 25)),
            q1=float(np.percentile(values, 25)),
            q3=float(np.percentile(values, 75))
        )
    
    def _test_normality(self, values: np.ndarray) -> Tuple[bool, float]:
        """Test for normality."""
        if len(values) < 20:
            _, p = scipy_stats.shapiro(values)
        elif len(values) < 5000:
            _, p = scipy_stats.normaltest(values)
        else:
            # Use subsample for large datasets
            sample = np.random.choice(values, size=5000, replace=False)
            _, p = scipy_stats.normaltest(sample)
        
        return p > 0.05, p
    
    def _determine_distribution_type(
        self,
        stats: DistributionStats,
        is_normal: bool
    ) -> DistributionType:
        """Determine distribution type from statistics."""
        if is_normal:
            return DistributionType.NORMAL
        
        skewness = stats.skewness
        kurtosis = stats.kurtosis
        
        if abs(skewness) > 2:
            if skewness > 0:
                return DistributionType.LOGNORMAL
            return DistributionType.SKEWED
        
        if kurtosis > 3:
            return DistributionType.HEAVY_TAILED
        
        if stats.min_value >= 0 and skewness > 0.5:
            return DistributionType.EXPONENTIAL
        
        return DistributionType.UNKNOWN
    
    def _fit_distributions(self, values: np.ndarray) -> List[FittedDistribution]:
        """Fit various distributions."""
        fitted = []
        
        for name, dist in self.DISTRIBUTIONS:
            try:
                # Fit distribution
                if name == 'lognormal' and np.min(values) <= 0:
                    continue
                
                params = dist.fit(values)
                
                # KS test
                ks_stat, ks_p = scipy_stats.kstest(values, name, args=params)
                
                # AIC
                log_likelihood = np.sum(dist.logpdf(values, *params))
                n_params = len(params)
                aic = 2 * n_params - 2 * log_likelihood
                
                fitted.append(FittedDistribution(
                    distribution=DistributionType(name) if name in [d.value for d in DistributionType] else DistributionType.UNKNOWN,
                    params={f"param_{i}": p for i, p in enumerate(params)},
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    aic=aic,
                    is_good_fit=ks_p > 0.05
                ))
            except Exception:
                continue
        
        return fitted


# ============================================================================
# Factory Functions
# ============================================================================

def get_distribution_engine() -> DistributionAnalysisEngine:
    """Get distribution analysis engine."""
    return DistributionAnalysisEngine()


def quick_distribution(
    data: pd.Series | np.ndarray
) -> Dict[str, Any]:
    """Quick distribution analysis."""
    engine = DistributionAnalysisEngine(verbose=False)
    result = engine.analyze(data)
    return result.to_dict()
