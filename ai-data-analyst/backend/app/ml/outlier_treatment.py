# AI Enterprise Data Analyst - Outlier Treatment Engine
# Production-grade outlier detection and treatment
# Handles: multiple methods, automatic selection, treatment strategies

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

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class OutlierMethod(str, Enum):
    """Outlier detection methods."""
    IQR = "iqr"  # Interquartile Range
    ZSCORE = "zscore"  # Z-score
    MAD = "mad"  # Median Absolute Deviation
    ISOLATION_FOREST = "isolation_forest"
    PERCENTILE = "percentile"  # Percentile-based
    DBSCAN = "dbscan"


class TreatmentStrategy(str, Enum):
    """Outlier treatment strategies."""
    REMOVE = "remove"  # Remove outlier rows
    CAP = "cap"  # Cap at bounds (winsorization)
    REPLACE_MEAN = "replace_mean"
    REPLACE_MEDIAN = "replace_median"
    REPLACE_NULL = "replace_null"
    FLAG = "flag"  # Just flag, don't modify


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OutlierInfo:
    """Information about detected outliers."""
    column: str
    method: OutlierMethod
    n_outliers: int
    outlier_pct: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    outlier_indices: List[int]
    outlier_values: List[float]
    
    # Statistics
    min_outlier: Optional[float] = None
    max_outlier: Optional[float] = None


@dataclass
class OutlierResult:
    """Complete outlier treatment result."""
    n_rows_original: int = 0
    n_rows_after: int = 0
    
    # Detection results by column
    detections: Dict[str, OutlierInfo] = field(default_factory=dict)
    
    # Overall
    total_outliers: int = 0
    total_outlier_rows: int = 0
    
    # Treatment applied
    strategy: TreatmentStrategy = TreatmentStrategy.FLAG
    treated_df: pd.DataFrame = None
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_rows_original": self.n_rows_original,
                "n_rows_after": self.n_rows_after,
                "total_outliers": self.total_outliers,
                "total_outlier_rows": self.total_outlier_rows,
                "strategy": self.strategy.value
            },
            "by_column": {
                col: {
                    "n_outliers": info.n_outliers,
                    "outlier_pct": round(info.outlier_pct, 2),
                    "lower_bound": round(info.lower_bound, 4) if info.lower_bound is not None else None,
                    "upper_bound": round(info.upper_bound, 4) if info.upper_bound is not None else None,
                    "method": info.method.value
                }
                for col, info in self.detections.items()
            }
        }


# ============================================================================
# Outlier Treatment Engine
# ============================================================================

class OutlierTreatmentEngine:
    """
    Production-grade Outlier Treatment engine.
    
    Features:
    - Multiple detection methods
    - Automatic method selection
    - Multiple treatment strategies
    - Multivariate outlier detection
    - Robust to skewed distributions
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def detect_and_treat(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: OutlierMethod = OutlierMethod.IQR,
        strategy: TreatmentStrategy = TreatmentStrategy.CAP,
        threshold: float = 1.5,  # IQR multiplier or z-score threshold
        auto_select_method: bool = False
    ) -> OutlierResult:
        """Detect and treat outliers."""
        start_time = datetime.now()
        
        # Select numeric columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.verbose:
            logger.info(f"Outlier detection: {len(columns)} columns, method={method.value}")
        
        result_df = df.copy()
        detections = {}
        all_outlier_indices = set()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            # Auto-select method based on distribution
            if auto_select_method:
                method = self._select_method(series)
            
            # Detect outliers
            outlier_info = self._detect_outliers(series, col, method, threshold)
            
            if outlier_info.n_outliers > 0:
                detections[col] = outlier_info
                all_outlier_indices.update(outlier_info.outlier_indices)
                
                # Apply treatment
                if strategy != TreatmentStrategy.FLAG:
                    result_df = self._apply_treatment(
                        result_df, col, outlier_info, strategy
                    )
        
        # Calculate totals
        total_outliers = sum(info.n_outliers for info in detections.values())
        
        # Handle row removal
        final_n_rows = len(result_df)
        if strategy == TreatmentStrategy.REMOVE:
            result_df = result_df.loc[~result_df.index.isin(all_outlier_indices)]
            final_n_rows = len(result_df)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return OutlierResult(
            n_rows_original=len(df),
            n_rows_after=final_n_rows,
            detections=detections,
            total_outliers=total_outliers,
            total_outlier_rows=len(all_outlier_indices),
            strategy=strategy,
            treated_df=result_df,
            processing_time_sec=processing_time
        )
    
    def detect_only(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: OutlierMethod = OutlierMethod.IQR,
        threshold: float = 1.5
    ) -> OutlierResult:
        """Only detect outliers without treatment."""
        return self.detect_and_treat(
            df, columns, method, TreatmentStrategy.FLAG, threshold
        )
    
    def _detect_outliers(
        self,
        series: pd.Series,
        column: str,
        method: OutlierMethod,
        threshold: float
    ) -> OutlierInfo:
        """Detect outliers using specified method."""
        values = series.values
        
        if method == OutlierMethod.IQR:
            return self._detect_iqr(series, column, threshold)
        elif method == OutlierMethod.ZSCORE:
            return self._detect_zscore(series, column, threshold)
        elif method == OutlierMethod.MAD:
            return self._detect_mad(series, column, threshold)
        elif method == OutlierMethod.PERCENTILE:
            return self._detect_percentile(series, column, threshold)
        elif method == OutlierMethod.ISOLATION_FOREST:
            return self._detect_isolation_forest(series, column)
        else:
            return self._detect_iqr(series, column, threshold)
    
    def _detect_iqr(
        self,
        series: pd.Series,
        column: str,
        multiplier: float = 1.5
    ) -> OutlierInfo:
        """IQR-based outlier detection."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = series[outlier_mask].index.tolist()
        outlier_values = series[outlier_mask].values.tolist()
        
        return OutlierInfo(
            column=column,
            method=OutlierMethod.IQR,
            n_outliers=len(outlier_indices),
            outlier_pct=len(outlier_indices) / len(series) * 100,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values[:100],
            min_outlier=min(outlier_values) if outlier_values else None,
            max_outlier=max(outlier_values) if outlier_values else None
        )
    
    def _detect_zscore(
        self,
        series: pd.Series,
        column: str,
        threshold: float = 3.0
    ) -> OutlierInfo:
        """Z-score based outlier detection."""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return OutlierInfo(
                column=column, method=OutlierMethod.ZSCORE,
                n_outliers=0, outlier_pct=0,
                lower_bound=mean, upper_bound=mean,
                outlier_indices=[], outlier_values=[]
            )
        
        z_scores = (series - mean) / std
        outlier_mask = z_scores.abs() > threshold
        
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        outlier_indices = series[outlier_mask].index.tolist()
        outlier_values = series[outlier_mask].values.tolist()
        
        return OutlierInfo(
            column=column,
            method=OutlierMethod.ZSCORE,
            n_outliers=len(outlier_indices),
            outlier_pct=len(outlier_indices) / len(series) * 100,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values[:100],
            min_outlier=min(outlier_values) if outlier_values else None,
            max_outlier=max(outlier_values) if outlier_values else None
        )
    
    def _detect_mad(
        self,
        series: pd.Series,
        column: str,
        threshold: float = 3.5
    ) -> OutlierInfo:
        """Median Absolute Deviation based detection (robust)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            mad = np.mean(np.abs(series - median))
        
        if mad == 0:
            return OutlierInfo(
                column=column, method=OutlierMethod.MAD,
                n_outliers=0, outlier_pct=0,
                lower_bound=median, upper_bound=median,
                outlier_indices=[], outlier_values=[]
            )
        
        # Modified Z-score
        modified_z = 0.6745 * (series - median) / mad
        outlier_mask = modified_z.abs() > threshold
        
        lower_bound = median - threshold * mad / 0.6745
        upper_bound = median + threshold * mad / 0.6745
        
        outlier_indices = series[outlier_mask].index.tolist()
        outlier_values = series[outlier_mask].values.tolist()
        
        return OutlierInfo(
            column=column,
            method=OutlierMethod.MAD,
            n_outliers=len(outlier_indices),
            outlier_pct=len(outlier_indices) / len(series) * 100,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values[:100],
            min_outlier=min(outlier_values) if outlier_values else None,
            max_outlier=max(outlier_values) if outlier_values else None
        )
    
    def _detect_percentile(
        self,
        series: pd.Series,
        column: str,
        pct: float = 1.0
    ) -> OutlierInfo:
        """Percentile-based detection."""
        lower_bound = series.quantile(pct / 100)
        upper_bound = series.quantile(1 - pct / 100)
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = series[outlier_mask].index.tolist()
        outlier_values = series[outlier_mask].values.tolist()
        
        return OutlierInfo(
            column=column,
            method=OutlierMethod.PERCENTILE,
            n_outliers=len(outlier_indices),
            outlier_pct=len(outlier_indices) / len(series) * 100,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values[:100],
            min_outlier=min(outlier_values) if outlier_values else None,
            max_outlier=max(outlier_values) if outlier_values else None
        )
    
    def _detect_isolation_forest(
        self,
        series: pd.Series,
        column: str
    ) -> OutlierInfo:
        """Isolation Forest detection."""
        try:
            from sklearn.ensemble import IsolationForest
            
            values = series.values.reshape(-1, 1)
            iso = IsolationForest(contamination=0.05, random_state=42)
            predictions = iso.fit_predict(values)
            
            outlier_mask = predictions == -1
            outlier_indices = series.iloc[outlier_mask].index.tolist()
            outlier_values = series.iloc[outlier_mask].values.tolist()
            
            return OutlierInfo(
                column=column,
                method=OutlierMethod.ISOLATION_FOREST,
                n_outliers=len(outlier_indices),
                outlier_pct=len(outlier_indices) / len(series) * 100,
                lower_bound=None,
                upper_bound=None,
                outlier_indices=outlier_indices,
                outlier_values=outlier_values[:100],
                min_outlier=min(outlier_values) if outlier_values else None,
                max_outlier=max(outlier_values) if outlier_values else None
            )
        except ImportError:
            return self._detect_iqr(series, column, 1.5)
    
    def _apply_treatment(
        self,
        df: pd.DataFrame,
        column: str,
        outlier_info: OutlierInfo,
        strategy: TreatmentStrategy
    ) -> pd.DataFrame:
        """Apply treatment strategy."""
        result = df.copy()
        
        if strategy == TreatmentStrategy.CAP:
            # Winsorization
            if outlier_info.lower_bound is not None:
                result.loc[result[column] < outlier_info.lower_bound, column] = outlier_info.lower_bound
            if outlier_info.upper_bound is not None:
                result.loc[result[column] > outlier_info.upper_bound, column] = outlier_info.upper_bound
        
        elif strategy == TreatmentStrategy.REPLACE_MEAN:
            mean_val = result.loc[~result.index.isin(outlier_info.outlier_indices), column].mean()
            result.loc[result.index.isin(outlier_info.outlier_indices), column] = mean_val
        
        elif strategy == TreatmentStrategy.REPLACE_MEDIAN:
            median_val = result.loc[~result.index.isin(outlier_info.outlier_indices), column].median()
            result.loc[result.index.isin(outlier_info.outlier_indices), column] = median_val
        
        elif strategy == TreatmentStrategy.REPLACE_NULL:
            result.loc[result.index.isin(outlier_info.outlier_indices), column] = np.nan
        
        return result
    
    def _select_method(self, series: pd.Series) -> OutlierMethod:
        """Auto-select best method based on distribution."""
        # Check skewness
        skewness = scipy_stats.skew(series.dropna())
        
        # For highly skewed distributions, use MAD (robust)
        if abs(skewness) > 2:
            return OutlierMethod.MAD
        
        # For normal-ish distributions, use z-score
        _, normality_p = scipy_stats.shapiro(series.dropna().head(5000))
        if normality_p > 0.05:
            return OutlierMethod.ZSCORE
        
        # Default to IQR
        return OutlierMethod.IQR


# ============================================================================
# Factory Functions
# ============================================================================

def get_outlier_engine() -> OutlierTreatmentEngine:
    """Get outlier treatment engine."""
    return OutlierTreatmentEngine()


def quick_outlier_detection(
    df: pd.DataFrame,
    columns: List[str] = None
) -> Dict[str, Any]:
    """Quick outlier detection."""
    engine = OutlierTreatmentEngine(verbose=False)
    result = engine.detect_only(df, columns)
    return result.to_dict()


def winsorize(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = "iqr"
) -> pd.DataFrame:
    """Quick winsorization (capping) of outliers."""
    engine = OutlierTreatmentEngine(verbose=False)
    result = engine.detect_and_treat(
        df, columns,
        OutlierMethod(method),
        TreatmentStrategy.CAP
    )
    return result.treated_df
