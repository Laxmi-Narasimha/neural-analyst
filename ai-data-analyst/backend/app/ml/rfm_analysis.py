# AI Enterprise Data Analyst - RFM Analysis Engine
# Production-grade RFM (Recency, Frequency, Monetary) segmentation
# Handles: any transaction data, flexible date/amount columns, auto scoring

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
# Enums and Types
# ============================================================================

class RFMScoringMethod(str, Enum):
    """RFM scoring methods."""
    QUANTILE = "quantile"  # Score by quantiles (default)
    CUSTOM = "custom"  # Custom breakpoints
    KMEANS = "kmeans"  # K-means clustering
    PERCENTILE = "percentile"  # Percentile-based


class CustomerSegment(str, Enum):
    """Standard customer segments based on RFM."""
    CHAMPIONS = "Champions"
    LOYAL_CUSTOMERS = "Loyal Customers"
    POTENTIAL_LOYALISTS = "Potential Loyalists"
    NEW_CUSTOMERS = "New Customers"
    PROMISING = "Promising"
    NEED_ATTENTION = "Need Attention"
    ABOUT_TO_SLEEP = "About to Sleep"
    AT_RISK = "At Risk"
    CANT_LOSE = "Can't Lose Them"
    HIBERNATING = "Hibernating"
    LOST = "Lost"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RFMScore:
    """Individual customer RFM scores."""
    customer_id: Any
    recency_value: float  # Days since last purchase
    frequency_value: int  # Number of purchases
    monetary_value: float  # Total spend
    
    recency_score: int = 0  # 1-5 score
    frequency_score: int = 0
    monetary_score: int = 0
    
    rfm_score: str = ""  # Combined e.g., "555"
    rfm_total: int = 0  # Sum of scores
    
    segment: str = ""
    percentile_rank: float = 0.0


@dataclass
class RFMConfig:
    """Configuration for RFM analysis."""
    # Column mappings
    customer_id_col: Optional[str] = None
    date_col: Optional[str] = None
    amount_col: Optional[str] = None
    
    # Analysis date (defaults to max date in data)
    analysis_date: Optional[datetime] = None
    
    # Scoring
    scoring_method: RFMScoringMethod = RFMScoringMethod.QUANTILE
    n_bins: int = 5  # Number of score levels
    
    # Custom breakpoints (if using CUSTOM method)
    recency_breakpoints: List[float] = field(default_factory=list)
    frequency_breakpoints: List[float] = field(default_factory=list)
    monetary_breakpoints: List[float] = field(default_factory=list)
    
    # Segment definitions (rfm_score pattern -> segment name)
    custom_segments: Optional[Dict[str, str]] = None


@dataclass
class RFMResult:
    """Complete RFM analysis result."""
    # Summary
    n_customers: int = 0
    analysis_date: Optional[datetime] = None
    date_range: Tuple[datetime, datetime] = (None, None)
    
    # Customer scores
    customer_rfm: pd.DataFrame = None  # Full RFM data
    
    # Segment summary
    segment_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Score distributions
    score_distributions: Dict[str, Dict[int, int]] = field(default_factory=dict)
    
    # Statistics
    rfm_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Correlation between RFM components
    correlation_matrix: Optional[pd.DataFrame] = None
    
    # Timing
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_customers": self.n_customers,
                "analysis_date": self.analysis_date.isoformat() if self.analysis_date else None,
                "date_range": [d.isoformat() if d else None for d in self.date_range]
            },
            "segment_summary": self.segment_summary,
            "score_distributions": self.score_distributions,
            "rfm_stats": {
                k: {sk: round(sv, 2) for sk, sv in v.items()}
                for k, v in self.rfm_stats.items()
            },
            "top_customers": self.customer_rfm.head(20).to_dict(orient="records") if self.customer_rfm is not None else [],
            "processing_time_sec": round(self.processing_time_sec, 2)
        }


# ============================================================================
# Column Detector
# ============================================================================

class RFMColumnDetector:
    """Auto-detect RFM-relevant columns."""
    
    def detect_columns(
        self,
        df: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Auto-detect customer_id, date, and amount columns.
        Returns: (customer_id_col, date_col, amount_col)
        """
        customer_col = self._find_customer_col(df)
        date_col = self._find_date_col(df)
        amount_col = self._find_amount_col(df)
        
        return customer_col, date_col, amount_col
    
    def _find_customer_col(self, df: pd.DataFrame) -> Optional[str]:
        """Find customer ID column."""
        patterns = ['customer', 'client', 'user', 'member', 'account', 'buyer', 
                   'cust_id', 'customer_id', 'user_id', 'client_id']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns):
                return col
        
        # Look for high cardinality column (likely ID)
        for col in df.columns:
            if df[col].nunique() / len(df) > 0.1:
                if not pd.api.types.is_float_dtype(df[col]):
                    return col
        
        return df.columns[0] if len(df.columns) > 0 else None
    
    def _find_date_col(self, df: pd.DataFrame) -> Optional[str]:
        """Find date column."""
        patterns = ['date', 'time', 'order_date', 'purchase_date', 'transaction_date',
                   'created', 'timestamp', 'dt']
        
        # First check dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        # Then check names
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns):
                # Try to parse as date
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    return col
                except:
                    continue
        
        return None
    
    def _find_amount_col(self, df: pd.DataFrame) -> Optional[str]:
        """Find monetary amount column."""
        patterns = ['amount', 'total', 'revenue', 'sales', 'price', 'value', 
                   'spend', 'purchase', 'order_value', 'transaction_amount']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check names first
        for col in numeric_cols:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns):
                return col
        
        # Return first numeric column that looks like money
        for col in numeric_cols:
            if df[col].mean() > 0 and df[col].std() > 0:
                return col
        
        return numeric_cols[0] if numeric_cols else None


# ============================================================================
# RFM Scorer
# ============================================================================

class RFMScorer:
    """Score customers based on RFM values."""
    
    def __init__(self, config: RFMConfig):
        self.config = config
        self._r_bins = None
        self._f_bins = None
        self._m_bins = None
    
    def score(
        self,
        rfm_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Score RFM values."""
        df = rfm_df.copy()
        n_bins = self.config.n_bins
        
        if self.config.scoring_method == RFMScoringMethod.QUANTILE:
            # Recency: lower is better, so reverse scoring
            df['R_Score'] = pd.qcut(
                df['Recency'].rank(method='first'), 
                q=n_bins, 
                labels=range(n_bins, 0, -1),
                duplicates='drop'
            ).astype(int)
            
            # Frequency: higher is better
            df['F_Score'] = pd.qcut(
                df['Frequency'].rank(method='first'),
                q=n_bins,
                labels=range(1, n_bins + 1),
                duplicates='drop'
            ).astype(int)
            
            # Monetary: higher is better
            df['M_Score'] = pd.qcut(
                df['Monetary'].rank(method='first'),
                q=n_bins,
                labels=range(1, n_bins + 1),
                duplicates='drop'
            ).astype(int)
            
        elif self.config.scoring_method == RFMScoringMethod.PERCENTILE:
            # Percentile-based scoring
            df['R_Score'] = n_bins - (df['Recency'].rank(pct=True) * n_bins).astype(int)
            df['R_Score'] = df['R_Score'].clip(1, n_bins)
            
            df['F_Score'] = (df['Frequency'].rank(pct=True) * n_bins).astype(int).clip(1, n_bins)
            df['M_Score'] = (df['Monetary'].rank(pct=True) * n_bins).astype(int).clip(1, n_bins)
        
        elif self.config.scoring_method == RFMScoringMethod.CUSTOM:
            df['R_Score'] = self._custom_score(
                df['Recency'], 
                self.config.recency_breakpoints,
                reverse=True
            )
            df['F_Score'] = self._custom_score(
                df['Frequency'],
                self.config.frequency_breakpoints
            )
            df['M_Score'] = self._custom_score(
                df['Monetary'],
                self.config.monetary_breakpoints
            )
        
        # Combined scores
        df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
        df['RFM_Total'] = df['R_Score'] + df['F_Score'] + df['M_Score']
        
        # Percentile rank
        df['Percentile_Rank'] = df['RFM_Total'].rank(pct=True) * 100
        
        return df
    
    def _custom_score(
        self,
        series: pd.Series,
        breakpoints: List[float],
        reverse: bool = False
    ) -> pd.Series:
        """Score based on custom breakpoints."""
        if not breakpoints:
            return pd.qcut(series.rank(method='first'), q=self.config.n_bins, 
                          labels=range(1, self.config.n_bins + 1)).astype(int)
        
        bins = [-np.inf] + sorted(breakpoints) + [np.inf]
        labels = range(1, len(bins))
        if reverse:
            labels = range(len(bins) - 1, 0, -1)
        
        return pd.cut(series, bins=bins, labels=list(labels)).astype(int)


# ============================================================================
# Segment Mapper
# ============================================================================

class SegmentMapper:
    """Map RFM scores to customer segments."""
    
    # Default segment definitions (R, F score patterns)
    DEFAULT_SEGMENTS = {
        # Champions: High R, High F
        (5, 5): CustomerSegment.CHAMPIONS,
        (5, 4): CustomerSegment.CHAMPIONS,
        (4, 5): CustomerSegment.CHAMPIONS,
        
        # Loyal Customers: High F
        (4, 4): CustomerSegment.LOYAL_CUSTOMERS,
        (3, 5): CustomerSegment.LOYAL_CUSTOMERS,
        (3, 4): CustomerSegment.LOYAL_CUSTOMERS,
        
        # Potential Loyalists: High R, Medium F
        (5, 3): CustomerSegment.POTENTIAL_LOYALISTS,
        (4, 3): CustomerSegment.POTENTIAL_LOYALISTS,
        (5, 2): CustomerSegment.POTENTIAL_LOYALISTS,
        
        # New Customers: Very High R, Low F
        (5, 1): CustomerSegment.NEW_CUSTOMERS,
        (4, 1): CustomerSegment.NEW_CUSTOMERS,
        
        # Promising: Medium R, Low F
        (4, 2): CustomerSegment.PROMISING,
        (3, 1): CustomerSegment.PROMISING,
        
        # Need Attention: Medium R, Medium F
        (3, 3): CustomerSegment.NEED_ATTENTION,
        (3, 2): CustomerSegment.NEED_ATTENTION,
        (2, 3): CustomerSegment.NEED_ATTENTION,
        
        # About to Sleep: Low-Medium R, Low F
        (2, 2): CustomerSegment.ABOUT_TO_SLEEP,
        (2, 1): CustomerSegment.ABOUT_TO_SLEEP,
        
        # At Risk: Low R, High F (used to be good customers)
        (2, 5): CustomerSegment.AT_RISK,
        (2, 4): CustomerSegment.AT_RISK,
        (1, 5): CustomerSegment.CANT_LOSE,
        (1, 4): CustomerSegment.CANT_LOSE,
        
        # Hibernating: Low R, Low-Medium F
        (1, 3): CustomerSegment.HIBERNATING,
        (1, 2): CustomerSegment.HIBERNATING,
        
        # Lost: Lowest R, Lowest F
        (1, 1): CustomerSegment.LOST,
    }
    
    def __init__(self, custom_segments: Dict[str, str] = None):
        self.custom_segments = custom_segments
    
    def map_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map RFM scores to segments."""
        df = df.copy()
        
        def get_segment(row):
            r, f = int(row['R_Score']), int(row['F_Score'])
            
            # Check custom segments first
            if self.custom_segments:
                rfm = row['RFM_Score']
                if rfm in self.custom_segments:
                    return self.custom_segments[rfm]
            
            # Use default mapping
            key = (r, f)
            if key in self.DEFAULT_SEGMENTS:
                return self.DEFAULT_SEGMENTS[key].value
            
            # Fallback based on total score
            total = row['RFM_Total']
            if total >= 12:
                return CustomerSegment.CHAMPIONS.value
            elif total >= 9:
                return CustomerSegment.LOYAL_CUSTOMERS.value
            elif total >= 6:
                return CustomerSegment.NEED_ATTENTION.value
            else:
                return CustomerSegment.HIBERNATING.value
        
        df['Segment'] = df.apply(get_segment, axis=1)
        
        return df


# ============================================================================
# RFM Analysis Engine
# ============================================================================

class RFMAnalysisEngine:
    """
    Complete RFM analysis engine.
    
    Features:
    - Auto-detects customer, date, amount columns
    - Flexible scoring methods
    - Automatic customer segmentation
    - Comprehensive statistics and insights
    """
    
    def __init__(self, config: RFMConfig = None, verbose: bool = True):
        self.config = config or RFMConfig()
        self.verbose = verbose
        self.column_detector = RFMColumnDetector()
    
    def analyze(
        self,
        df: pd.DataFrame,
        customer_id_col: str = None,
        date_col: str = None,
        amount_col: str = None,
        analysis_date: datetime = None
    ) -> RFMResult:
        """
        Perform complete RFM analysis.
        
        Handles ANY transaction data automatically.
        """
        start_time = datetime.now()
        
        # Auto-detect columns if not provided
        if customer_id_col is None or date_col is None or amount_col is None:
            detected = self.column_detector.detect_columns(df)
            customer_id_col = customer_id_col or detected[0]
            date_col = date_col or detected[1]
            amount_col = amount_col or detected[2]
        
        if self.verbose:
            logger.info(f"Using columns: customer={customer_id_col}, date={date_col}, amount={amount_col}")
        
        # Validate columns
        for col in [customer_id_col, date_col, amount_col]:
            if col not in df.columns:
                raise DataProcessingException(f"Column '{col}' not found in data")
        
        # Prepare data
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        
        # Remove nulls
        df = df.dropna(subset=[customer_id_col, date_col, amount_col])
        
        if len(df) == 0:
            raise DataProcessingException("No valid data after cleaning")
        
        # Set analysis date
        if analysis_date is None:
            analysis_date = df[date_col].max() + timedelta(days=1)
        
        date_range = (df[date_col].min(), df[date_col].max())
        
        if self.verbose:
            logger.info(f"Analyzing {len(df)} transactions, {df[customer_id_col].nunique()} customers")
        
        # Calculate RFM values
        rfm_df = self._calculate_rfm(df, customer_id_col, date_col, amount_col, analysis_date)
        
        # Score RFM
        scorer = RFMScorer(self.config)
        rfm_df = scorer.score(rfm_df)
        
        # Map segments
        mapper = SegmentMapper(self.config.custom_segments)
        rfm_df = mapper.map_segments(rfm_df)
        
        # Calculate statistics
        rfm_stats = self._calculate_stats(rfm_df)
        score_distributions = self._get_score_distributions(rfm_df)
        segment_summary = self._get_segment_summary(rfm_df)
        
        # Correlation
        correlation = rfm_df[['Recency', 'Frequency', 'Monetary']].corr()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RFMResult(
            n_customers=len(rfm_df),
            analysis_date=analysis_date,
            date_range=date_range,
            customer_rfm=rfm_df,
            segment_summary=segment_summary,
            score_distributions=score_distributions,
            rfm_stats=rfm_stats,
            correlation_matrix=correlation,
            processing_time_sec=processing_time
        )
    
    def _calculate_rfm(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str,
        analysis_date: datetime
    ) -> pd.DataFrame:
        """Calculate RFM values for each customer."""
        rfm = df.groupby(customer_col).agg({
            date_col: 'max',  # Most recent purchase
            customer_col: 'count',  # Frequency
            amount_col: 'sum'  # Total monetary value
        }).rename(columns={
            date_col: 'LastPurchase',
            customer_col: 'Frequency',
            amount_col: 'Monetary'
        })
        
        # Calculate recency
        rfm['Recency'] = (analysis_date - rfm['LastPurchase']).dt.days
        
        # Reset index
        rfm = rfm.reset_index()
        rfm = rfm.rename(columns={customer_col: 'CustomerID'})
        
        return rfm
    
    def _calculate_stats(self, rfm_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate RFM statistics."""
        stats = {}
        
        for metric in ['Recency', 'Frequency', 'Monetary']:
            stats[metric] = {
                'mean': float(rfm_df[metric].mean()),
                'median': float(rfm_df[metric].median()),
                'std': float(rfm_df[metric].std()),
                'min': float(rfm_df[metric].min()),
                'max': float(rfm_df[metric].max()),
                'q25': float(rfm_df[metric].quantile(0.25)),
                'q75': float(rfm_df[metric].quantile(0.75))
            }
        
        return stats
    
    def _get_score_distributions(
        self,
        rfm_df: pd.DataFrame
    ) -> Dict[str, Dict[int, int]]:
        """Get distribution of scores."""
        return {
            'R_Score': rfm_df['R_Score'].value_counts().sort_index().to_dict(),
            'F_Score': rfm_df['F_Score'].value_counts().sort_index().to_dict(),
            'M_Score': rfm_df['M_Score'].value_counts().sort_index().to_dict()
        }
    
    def _get_segment_summary(
        self,
        rfm_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by segment."""
        summary = {}
        
        for segment in rfm_df['Segment'].unique():
            segment_df = rfm_df[rfm_df['Segment'] == segment]
            
            summary[segment] = {
                'count': int(len(segment_df)),
                'percentage': round(len(segment_df) / len(rfm_df) * 100, 1),
                'avg_recency': round(segment_df['Recency'].mean(), 1),
                'avg_frequency': round(segment_df['Frequency'].mean(), 1),
                'avg_monetary': round(segment_df['Monetary'].mean(), 2),
                'total_revenue': round(segment_df['Monetary'].sum(), 2)
            }
        
        return summary


# ============================================================================
# Factory Functions
# ============================================================================

def get_rfm_engine(config: RFMConfig = None) -> RFMAnalysisEngine:
    """Get RFM analysis engine."""
    return RFMAnalysisEngine(config=config)


def quick_rfm(
    df: pd.DataFrame,
    customer_col: str = None,
    date_col: str = None,
    amount_col: str = None
) -> Dict[str, Any]:
    """
    Quick RFM analysis on transaction data.
    
    Example:
        result = quick_rfm(transactions_df)
        print(result['segment_summary'])
    """
    engine = RFMAnalysisEngine(verbose=False)
    result = engine.analyze(df, customer_col, date_col, amount_col)
    return result.to_dict()
