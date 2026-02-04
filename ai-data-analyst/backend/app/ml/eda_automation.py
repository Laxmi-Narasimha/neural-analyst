# AI Enterprise Data Analyst - EDA Automation Engine
# Production-grade automated exploratory data analysis
# Handles: any DataFrame with comprehensive profiling

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str
    dtype: str
    n_total: int
    n_missing: int
    missing_pct: float
    n_unique: int
    unique_pct: float
    
    # Type-specific
    is_numeric: bool = False
    is_categorical: bool = False
    is_datetime: bool = False
    is_text: bool = False
    is_binary: bool = False
    is_constant: bool = False
    is_id: bool = False
    
    # Numeric stats
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    
    # Categorical stats
    mode: Optional[str] = None
    mode_freq: Optional[int] = None
    top_values: Dict[str, int] = field(default_factory=dict)
    
    # Quality issues
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CorrelationFinding:
    """Significant correlation finding."""
    var1: str
    var2: str
    correlation: float
    strength: str
    direction: str


@dataclass
class EDAResult:
    """Complete EDA result."""
    n_rows: int = 0
    n_columns: int = 0
    
    # Memory
    memory_mb: float = 0.0
    
    # Column profiles
    columns: List[ColumnProfile] = field(default_factory=list)
    
    # Column type counts
    n_numeric: int = 0
    n_categorical: int = 0
    n_datetime: int = 0
    n_text: int = 0
    
    # Data quality
    total_missing: int = 0
    total_missing_pct: float = 0.0
    duplicate_rows: int = 0
    duplicate_pct: float = 0.0
    
    # Correlations
    high_correlations: List[CorrelationFinding] = field(default_factory=list)
    
    # Overall
    quality_score: float = 100.0
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overview": {
                "rows": self.n_rows,
                "columns": self.n_columns,
                "memory_mb": round(self.memory_mb, 2),
                "quality_score": round(self.quality_score, 1)
            },
            "column_types": {
                "numeric": self.n_numeric,
                "categorical": self.n_categorical,
                "datetime": self.n_datetime,
                "text": self.n_text
            },
            "data_quality": {
                "total_missing_pct": round(self.total_missing_pct, 2),
                "duplicate_rows": self.duplicate_rows,
                "duplicate_pct": round(self.duplicate_pct, 2)
            },
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "missing_pct": round(c.missing_pct, 2),
                    "unique_pct": round(c.unique_pct, 2),
                    "issues": c.issues[:3]
                }
                for c in self.columns
            ],
            "high_correlations": [
                {
                    "vars": f"{c.var1} ~ {c.var2}",
                    "correlation": round(c.correlation, 3),
                    "strength": c.strength
                }
                for c in self.high_correlations[:10]
            ],
            "key_insights": self.key_insights[:10],
            "recommendations": self.recommendations[:10]
        }

    @property
    def insights(self) -> List[str]:
        return self.key_insights


# ============================================================================
# EDA Automation Engine
# ============================================================================

class EDAAutomationEngine:
    """
    Production-grade EDA Automation engine.
    
    Features:
    - Automatic column type detection
    - Complete profiling for each column
    - Correlation analysis
    - Data quality assessment
    - Issue detection
    - Actionable recommendations
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        include_correlations: bool = True,
        sample_size: int = None
    ) -> EDAResult:
        """Perform complete EDA."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"EDA: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Sample if too large
        if sample_size and len(df) > sample_size:
            df_work = df.sample(n=sample_size, random_state=42)
        else:
            df_work = df
        
        # Memory
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Profile columns
        columns = []
        n_numeric = n_categorical = n_datetime = n_text = 0
        
        for col in df_work.columns:
            profile = self._profile_column(df_work[col], col)
            columns.append(profile)
            
            if profile.is_numeric:
                n_numeric += 1
            if profile.is_categorical:
                n_categorical += 1
            if profile.is_datetime:
                n_datetime += 1
            if profile.is_text:
                n_text += 1
        
        # Missing values
        total_missing = df_work.isna().sum().sum()
        total_missing_pct = total_missing / df_work.size * 100
        
        # Duplicates
        duplicate_rows = int(df_work.duplicated().sum())
        duplicate_pct = duplicate_rows / len(df_work) * 100
        
        # Correlations
        high_correlations = []
        if include_correlations and n_numeric > 1:
            high_correlations = self._find_correlations(df_work)
        
        # Quality score
        quality_score = self._calculate_quality_score(
            total_missing_pct, duplicate_pct, columns
        )
        
        # Insights
        key_insights = self._generate_insights(df_work, columns, high_correlations)
        
        # Recommendations
        recommendations = self._generate_recommendations(columns, total_missing_pct, duplicate_pct)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return EDAResult(
            n_rows=len(df),
            n_columns=len(df.columns),
            memory_mb=memory_mb,
            columns=columns,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_datetime=n_datetime,
            n_text=n_text,
            total_missing=total_missing,
            total_missing_pct=total_missing_pct,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            high_correlations=high_correlations,
            quality_score=quality_score,
            key_insights=key_insights,
            recommendations=recommendations,
            processing_time_sec=processing_time
        )
    
    def _profile_column(self, series: pd.Series, name: str) -> ColumnProfile:
        """Profile a single column."""
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_unique = series.nunique()
        
        profile = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            n_total=n_total,
            n_missing=n_missing,
            missing_pct=n_missing / n_total * 100 if n_total > 0 else 0,
            n_unique=n_unique,
            unique_pct=n_unique / n_total * 100 if n_total > 0 else 0
        )
        
        clean = series.dropna()
        
        # Detect type
        if n_unique == 1:
            profile.is_constant = True
            profile.issues.append("Constant value - no variance")
        elif n_unique == 2:
            profile.is_binary = True
        elif n_unique / n_total > 0.95 and n_total > 20:
            profile.is_id = True
        
        if pd.api.types.is_datetime64_any_dtype(series):
            profile.is_datetime = True
            if len(clean) > 0:
                profile.min_val = clean.min()
                profile.max_val = clean.max()
        
        elif pd.api.types.is_numeric_dtype(series):
            profile.is_numeric = True
            if len(clean) > 0:
                profile.mean = float(clean.mean())
                profile.median = float(clean.median())
                profile.std = float(clean.std())
                profile.min_val = float(clean.min())
                profile.max_val = float(clean.max())
                
                try:
                    profile.skewness = float(scipy_stats.skew(clean.dropna()))
                except:
                    profile.skewness = 0
                
                # Detection issues
                if profile.std == 0:
                    profile.issues.append("Zero variance")
                
                if abs(profile.skewness or 0) > 2:
                    profile.issues.append(f"Highly skewed ({profile.skewness:.2f})")
                    profile.recommendations.append("Consider log transformation")
                
                # Check for outliers
                q1, q3 = clean.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                if outliers / len(clean) > 0.05:
                    profile.issues.append(f"Outliers: {outliers / len(clean) * 100:.1f}%")
        
        else:  # Categorical or text
            if n_unique > 50 and n_unique / n_total < 0.5:
                profile.is_text = True
            else:
                profile.is_categorical = True
            
            if len(clean) > 0:
                vc = clean.value_counts()
                profile.mode = str(vc.index[0])
                profile.mode_freq = int(vc.iloc[0])
                profile.top_values = vc.head(5).to_dict()
                
                # High cardinality
                if n_unique > 100:
                    profile.issues.append("High cardinality")
                    profile.recommendations.append("Consider grouping or encoding")
        
        # Missing value issues
        if profile.missing_pct > 50:
            profile.issues.append(f"High missing: {profile.missing_pct:.1f}%")
            profile.recommendations.append("Consider removing or imputing")
        elif profile.missing_pct > 10:
            profile.recommendations.append("Consider imputation strategy")
        
        return profile
    
    def _find_correlations(self, df: pd.DataFrame) -> List[CorrelationFinding]:
        """Find high correlations."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return []
        
        corr_matrix = numeric_df.corr()
        findings = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                
                if np.isnan(corr):
                    continue
                
                abs_corr = abs(corr)
                
                if abs_corr > 0.5:
                    if abs_corr > 0.8:
                        strength = "very_strong"
                    elif abs_corr > 0.6:
                        strength = "strong"
                    else:
                        strength = "moderate"
                    
                    direction = "positive" if corr > 0 else "negative"
                    
                    findings.append(CorrelationFinding(
                        var1=col1,
                        var2=col2,
                        correlation=corr,
                        strength=strength,
                        direction=direction
                    ))
        
        findings.sort(key=lambda x: -abs(x.correlation))
        return findings
    
    def _calculate_quality_score(
        self,
        missing_pct: float,
        duplicate_pct: float,
        columns: List[ColumnProfile]
    ) -> float:
        """Calculate overall data quality score."""
        score = 100.0
        
        # Penalize for missing
        score -= min(30, missing_pct)
        
        # Penalize for duplicates
        score -= min(20, duplicate_pct * 2)
        
        # Penalize for problematic columns
        for col in columns:
            if col.is_constant:
                score -= 2
            if len(col.issues) > 2:
                score -= 3
        
        return max(0, min(100, score))
    
    def _generate_insights(
        self,
        df: pd.DataFrame,
        columns: List[ColumnProfile],
        correlations: List[CorrelationFinding]
    ) -> List[str]:
        """Generate key insights."""
        insights = []
        
        insights.append(f"Dataset has {len(df):,} rows and {len(df.columns)} columns")
        
        # Missing
        missing_cols = [c for c in columns if c.missing_pct > 10]
        if missing_cols:
            insights.append(f"{len(missing_cols)} columns have >10% missing values")
        
        # Numeric summary
        numeric_cols = [c for c in columns if c.is_numeric]
        if numeric_cols:
            skewed = [c for c in numeric_cols if c.skewness and abs(c.skewness) > 2]
            if skewed:
                insights.append(f"{len(skewed)} numeric columns are highly skewed")
        
        # Correlations
        very_strong = [c for c in correlations if c.strength == "very_strong"]
        if very_strong:
            insights.append(f"{len(very_strong)} variable pairs have very strong correlation (>0.8)")
        
        # High cardinality
        high_card = [c for c in columns if c.is_categorical and c.n_unique > 50]
        if high_card:
            insights.append(f"{len(high_card)} categorical columns have high cardinality (>50)")
        
        return insights
    
    def _generate_recommendations(
        self,
        columns: List[ColumnProfile],
        missing_pct: float,
        duplicate_pct: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        if duplicate_pct > 1:
            recs.append(f"Remove {duplicate_pct:.1f}% duplicate rows")
        
        high_missing = [c.name for c in columns if c.missing_pct > 50]
        if high_missing:
            recs.append(f"Consider dropping columns with >50% missing: {high_missing[:3]}")
        
        constant_cols = [c.name for c in columns if c.is_constant]
        if constant_cols:
            recs.append(f"Remove constant columns: {constant_cols[:3]}")
        
        id_cols = [c.name for c in columns if c.is_id]
        if id_cols:
            recs.append(f"Exclude ID columns from analysis: {id_cols[:3]}")
        
        skewed_cols = [c.name for c in columns if c.is_numeric and c.skewness and abs(c.skewness) > 2]
        if skewed_cols:
            recs.append(f"Apply transformation to skewed columns: {skewed_cols[:3]}")
        
        return recs


# ============================================================================
# Factory Functions
# ============================================================================

def get_eda_engine() -> EDAAutomationEngine:
    """Get EDA automation engine."""
    return EDAAutomationEngine()


def quick_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick EDA analysis."""
    engine = EDAAutomationEngine(verbose=False)
    result = engine.analyze(df)
    return result.to_dict()


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Get profiling summary as DataFrame."""
    engine = EDAAutomationEngine(verbose=False)
    result = engine.analyze(df, include_correlations=False)
    
    data = []
    for col in result.columns:
        data.append({
            'column': col.name,
            'dtype': col.dtype,
            'missing_pct': round(col.missing_pct, 2),
            'unique_pct': round(col.unique_pct, 2),
            'is_numeric': col.is_numeric,
            'is_categorical': col.is_categorical,
            'issues': '; '.join(col.issues[:2])
        })
    
    return pd.DataFrame(data)
