# AI Enterprise Data Analyst - Correlation Analysis Engine
# Production-grade correlation analysis for any data
# Handles: multiple correlation methods, significance testing

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

class CorrelationMethod(str, Enum):
    """Correlation methods."""
    PEARSON = "pearson"  # Linear relationships
    SPEARMAN = "spearman"  # Monotonic relationships
    KENDALL = "kendall"  # Rank correlation
    PHIK = "phik"  # Categorical correlation
    AUTO = "auto"  # Auto-select


class CorrelationStrength(str, Enum):
    """Correlation strength classification."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEGLIGIBLE = "negligible"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CorrelationPair:
    """Single correlation pair."""
    var1: str
    var2: str
    correlation: float
    p_value: float
    method: CorrelationMethod
    strength: CorrelationStrength
    is_significant: bool


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""
    method: CorrelationMethod = CorrelationMethod.AUTO
    significance_level: float = 0.05
    min_correlation: float = 0.1  # Minimum to report


@dataclass
class CorrelationResult:
    """Complete correlation analysis result."""
    n_variables: int = 0
    n_pairs: int = 0
    
    # Correlation matrix
    correlation_matrix: pd.DataFrame = None
    p_value_matrix: pd.DataFrame = None
    
    # Top correlations
    top_positive: List[CorrelationPair] = field(default_factory=list)
    top_negative: List[CorrelationPair] = field(default_factory=list)
    significant_pairs: List[CorrelationPair] = field(default_factory=list)
    
    # Multi-collinearity
    high_collinearity_pairs: List[CorrelationPair] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_variables": self.n_variables,
                "n_pairs": self.n_pairs,
                "n_significant": len(self.significant_pairs)
            },
            "correlation_matrix": self.correlation_matrix.to_dict() if self.correlation_matrix is not None else {},
            "top_positive": [
                {"var1": p.var1, "var2": p.var2, "correlation": round(p.correlation, 4), "strength": p.strength.value}
                for p in self.top_positive[:10]
            ],
            "top_negative": [
                {"var1": p.var1, "var2": p.var2, "correlation": round(p.correlation, 4), "strength": p.strength.value}
                for p in self.top_negative[:10]
            ],
            "high_collinearity": [
                {"var1": p.var1, "var2": p.var2, "correlation": round(p.correlation, 4)}
                for p in self.high_collinearity_pairs[:10]
            ]
        }


# ============================================================================
# Correlation Analysis Engine
# ============================================================================

class CorrelationAnalysisEngine:
    """
    Complete Correlation Analysis engine.
    
    Features:
    - Multiple correlation methods
    - Significance testing
    - Auto-detection of method
    - Collinearity detection
    """
    
    def __init__(self, config: CorrelationConfig = None, verbose: bool = True):
        self.config = config or CorrelationConfig()
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: CorrelationMethod = None
    ) -> CorrelationResult:
        """Perform correlation analysis."""
        start_time = datetime.now()
        
        method = method or self.config.method
        
        # Select numeric columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            columns = [c for c in columns if c in df.select_dtypes(include=[np.number]).columns]
        
        if len(columns) < 2:
            raise DataProcessingException("Need at least 2 numeric columns for correlation")
        
        if self.verbose:
            logger.info(f"Analyzing correlations for {len(columns)} variables")
        
        data = df[columns].dropna()
        
        # Auto-select method
        if method == CorrelationMethod.AUTO:
            method = self._auto_select_method(data)
            if self.verbose:
                logger.info(f"Auto-selected method: {method.value}")
        
        # Calculate correlation matrix
        if method == CorrelationMethod.PEARSON:
            corr_matrix = data.corr(method='pearson')
        elif method == CorrelationMethod.SPEARMAN:
            corr_matrix = data.corr(method='spearman')
        elif method == CorrelationMethod.KENDALL:
            corr_matrix = data.corr(method='kendall')
        else:
            corr_matrix = data.corr(method='pearson')
        
        # Calculate p-values
        p_matrix = self._calculate_p_values(data, method)
        
        # Extract pairs
        all_pairs = self._extract_pairs(corr_matrix, p_matrix, method)
        
        # Filter and sort
        significant = [p for p in all_pairs if p.is_significant]
        top_positive = sorted([p for p in all_pairs if p.correlation > 0], key=lambda x: -x.correlation)
        top_negative = sorted([p for p in all_pairs if p.correlation < 0], key=lambda x: x.correlation)
        high_collinearity = [p for p in all_pairs if abs(p.correlation) > 0.8]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CorrelationResult(
            n_variables=len(columns),
            n_pairs=len(all_pairs),
            correlation_matrix=corr_matrix,
            p_value_matrix=p_matrix,
            top_positive=top_positive,
            top_negative=top_negative,
            significant_pairs=significant,
            high_collinearity_pairs=high_collinearity,
            processing_time_sec=processing_time
        )
    
    def _auto_select_method(self, data: pd.DataFrame) -> CorrelationMethod:
        """Auto-select correlation method."""
        # Check for normality
        normal_count = 0
        for col in data.columns[:5]:  # Sample first 5 columns
            _, p = scipy_stats.normaltest(data[col].dropna())
            if p > 0.05:
                normal_count += 1
        
        # If mostly normal, use Pearson
        if normal_count >= len(data.columns[:5]) / 2:
            return CorrelationMethod.PEARSON
        else:
            return CorrelationMethod.SPEARMAN
    
    def _calculate_p_values(
        self,
        data: pd.DataFrame,
        method: CorrelationMethod
    ) -> pd.DataFrame:
        """Calculate p-value matrix."""
        columns = data.columns
        n = len(columns)
        p_matrix = pd.DataFrame(np.ones((n, n)), index=columns, columns=columns)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    x = data[col1].values
                    y = data[col2].values
                    
                    # Remove NaN pairs
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[mask], y[mask]
                    
                    if len(x) < 3:
                        continue
                    
                    if method == CorrelationMethod.PEARSON:
                        _, p = scipy_stats.pearsonr(x, y)
                    elif method == CorrelationMethod.SPEARMAN:
                        _, p = scipy_stats.spearmanr(x, y)
                    elif method == CorrelationMethod.KENDALL:
                        _, p = scipy_stats.kendalltau(x, y)
                    else:
                        _, p = scipy_stats.pearsonr(x, y)
                    
                    p_matrix.loc[col1, col2] = p
                    p_matrix.loc[col2, col1] = p
        
        return p_matrix
    
    def _extract_pairs(
        self,
        corr_matrix: pd.DataFrame,
        p_matrix: pd.DataFrame,
        method: CorrelationMethod
    ) -> List[CorrelationPair]:
        """Extract correlation pairs."""
        pairs = []
        columns = corr_matrix.columns
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    p_val = p_matrix.loc[col1, col2]
                    
                    if abs(corr) < self.config.min_correlation:
                        continue
                    
                    pairs.append(CorrelationPair(
                        var1=col1,
                        var2=col2,
                        correlation=corr,
                        p_value=p_val,
                        method=method,
                        strength=self._get_strength(corr),
                        is_significant=p_val < self.config.significance_level
                    ))
        
        return pairs
    
    def _get_strength(self, corr: float) -> CorrelationStrength:
        """Get correlation strength."""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif abs_corr >= 0.6:
            return CorrelationStrength.STRONG
        elif abs_corr >= 0.4:
            return CorrelationStrength.MODERATE
        elif abs_corr >= 0.2:
            return CorrelationStrength.WEAK
        return CorrelationStrength.NEGLIGIBLE


# ============================================================================
# Factory Functions
# ============================================================================

def get_correlation_engine(config: CorrelationConfig = None) -> CorrelationAnalysisEngine:
    """Get correlation analysis engine."""
    return CorrelationAnalysisEngine(config=config)


def quick_correlation(
    df: pd.DataFrame,
    method: str = "auto"
) -> Dict[str, Any]:
    """Quick correlation analysis."""
    config = CorrelationConfig(method=CorrelationMethod(method))
    engine = CorrelationAnalysisEngine(config=config, verbose=False)
    result = engine.analyze(df)
    return result.to_dict()
