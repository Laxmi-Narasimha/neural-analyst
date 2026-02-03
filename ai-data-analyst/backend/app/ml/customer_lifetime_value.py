# AI Enterprise Data Analyst - Customer Lifetime Value Engine
# Production-grade CLV calculation with multiple models
# Handles: any transaction data, probabilistic & simple models

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import minimize

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

class CLVModel(str, Enum):
    """CLV calculation models."""
    SIMPLE = "simple"  # Average * Time
    HISTORICAL = "historical"  # Sum of past transactions
    COHORT = "cohort"  # Cohort-based average
    PROBABILISTIC = "probabilistic"  # BG/NBD + Gamma-Gamma
    ML_BASED = "ml_based"  # ML prediction


class TransactionType(str, Enum):
    """Transaction type for modeling."""
    CONTRACTUAL = "contractual"  # Subscription-based
    NON_CONTRACTUAL = "non_contractual"  # Discrete purchases


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CustomerCLV:
    """Individual customer CLV data."""
    customer_id: Any
    historical_value: float
    predicted_clv: float
    expected_purchases: float
    probability_alive: float
    recency: int
    frequency: int
    monetary_value: float
    tenure: int
    segment: str = ""


@dataclass
class CLVConfig:
    """Configuration for CLV analysis."""
    model: CLVModel = CLVModel.PROBABILISTIC
    
    # Time horizon for prediction (months)
    prediction_horizon: int = 12
    
    # Discount rate (annual)
    discount_rate: float = 0.10
    
    # Column mappings
    customer_id_col: Optional[str] = None
    date_col: Optional[str] = None
    amount_col: Optional[str] = None
    
    # Model parameters
    penalizer_coef: float = 0.0  # L2 regularization


@dataclass
class CLVResult:
    """Complete CLV analysis result."""
    n_customers: int = 0
    total_historical_value: float = 0.0
    total_predicted_clv: float = 0.0
    
    # Customer-level data
    customer_clv: pd.DataFrame = None
    
    # Summary statistics
    clv_stats: Dict[str, float] = field(default_factory=dict)
    
    # Segment breakdown
    segment_clv: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Model parameters (for probabilistic)
    model_params: Dict[str, float] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_customers": self.n_customers,
                "total_historical_value": round(self.total_historical_value, 2),
                "total_predicted_clv": round(self.total_predicted_clv, 2),
                "avg_clv": round(self.total_predicted_clv / max(1, self.n_customers), 2)
            },
            "clv_stats": {k: round(v, 2) for k, v in self.clv_stats.items()},
            "segment_clv": self.segment_clv,
            "top_customers": self.customer_clv.nlargest(20, 'predicted_clv').to_dict(orient='records') if self.customer_clv is not None else [],
            "model_params": self.model_params
        }


# ============================================================================
# CLV Models
# ============================================================================

class SimpleCLVModel:
    """Simple CLV: Average Order Value * Purchase Frequency * Customer Lifespan."""
    
    def calculate(
        self,
        rfm_df: pd.DataFrame,
        prediction_months: int = 12
    ) -> pd.DataFrame:
        """Calculate simple CLV."""
        df = rfm_df.copy()
        
        # Average order value
        df['AOV'] = df['Monetary'] / df['Frequency'].clip(lower=1)
        
        # Purchase frequency (per month)
        df['Purchase_Rate'] = df['Frequency'] / (df['Tenure'] / 30).clip(lower=1)
        
        # Simple CLV
        df['predicted_clv'] = df['AOV'] * df['Purchase_Rate'] * prediction_months
        
        return df


class HistoricalCLVModel:
    """Historical CLV: Sum of all past transactions."""
    
    def calculate(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical CLV."""
        df = rfm_df.copy()
        df['predicted_clv'] = df['Monetary']
        return df


class CohortCLVModel:
    """Cohort-based CLV calculation."""
    
    def calculate(
        self,
        transactions_df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str
    ) -> pd.DataFrame:
        """Calculate CLV based on cohort averages."""
        df = transactions_df.copy()
        
        # Determine cohort (first purchase month)
        first_purchase = df.groupby(customer_col)[date_col].min().reset_index()
        first_purchase['Cohort'] = first_purchase[date_col].dt.to_period('M')
        
        # Calculate cohort average CLV
        customer_totals = df.groupby(customer_col)[amount_col].sum().reset_index()
        customer_totals.columns = [customer_col, 'Monetary']
        
        # Merge cohort info
        customer_totals = customer_totals.merge(first_purchase[[customer_col, 'Cohort']])
        
        # Cohort average
        cohort_avg = customer_totals.groupby('Cohort')['Monetary'].mean().to_dict()
        
        customer_totals['cohort_avg_clv'] = customer_totals['Cohort'].map(cohort_avg)
        customer_totals['predicted_clv'] = customer_totals['Monetary']
        
        return customer_totals


class ProbabilisticCLVModel:
    """
    Probabilistic CLV using simplified BG/NBD and Gamma-Gamma models.
    
    BG/NBD: Predicts expected number of future transactions
    Gamma-Gamma: Predicts expected monetary value per transaction
    """
    
    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params = {}
    
    def calculate(
        self,
        rfm_df: pd.DataFrame,
        prediction_months: int = 12,
        discount_rate: float = 0.10
    ) -> pd.DataFrame:
        """Calculate probabilistic CLV."""
        df = rfm_df.copy()
        
        # Ensure required columns
        required = ['Recency', 'Frequency', 'Monetary', 'Tenure']
        for col in required:
            if col not in df.columns:
                raise DataProcessingException(f"Missing required column: {col}")
        
        # Filter for repeat customers (frequency > 0)
        df_repeat = df[df['Frequency'] > 1].copy()
        
        if len(df_repeat) < 10:
            # Not enough data for probabilistic model, fall back to simple
            logger.warning("Not enough repeat customers, using simple CLV")
            simple_model = SimpleCLVModel()
            return simple_model.calculate(df, prediction_months)
        
        # Calculate probability alive (simplified)
        df['p_alive'] = self._probability_alive(df)
        
        # Expected future transactions (simplified)
        df['expected_purchases'] = self._expected_purchases(df, prediction_months)
        
        # Expected monetary value
        avg_monetary = df['Monetary'].mean() / df['Frequency'].mean()
        df['expected_avg_value'] = df['Monetary'] / df['Frequency'].clip(lower=1)
        
        # CLV with discount
        monthly_discount = (1 + discount_rate) ** (1/12) - 1
        discount_factor = sum(1 / (1 + monthly_discount) ** i for i in range(1, prediction_months + 1))
        
        df['predicted_clv'] = df['p_alive'] * df['expected_purchases'] * df['expected_avg_value'] * discount_factor
        
        return df
    
    def _probability_alive(self, df: pd.DataFrame) -> pd.Series:
        """
        Simplified probability that customer is still active.
        Based on recency relative to tenure.
        """
        # If recency is high relative to historical inter-purchase time, probability is lower
        avg_gap = df['Tenure'] / df['Frequency'].clip(lower=1)
        recency_ratio = df['Recency'] / avg_gap.clip(lower=1)
        
        # Sigmoid-like transformation
        p_alive = 1 / (1 + np.exp((recency_ratio - 2) * 0.5))
        
        return p_alive.clip(0, 1)
    
    def _expected_purchases(
        self,
        df: pd.DataFrame,
        months: int
    ) -> pd.Series:
        """Estimate expected purchases in future period."""
        # Purchase rate
        purchase_rate = df['Frequency'] / (df['Tenure'] / 30).clip(lower=1)
        
        # Adjust by probability alive
        expected = purchase_rate * months * df.get('p_alive', 1)
        
        return expected.clip(lower=0)


# ============================================================================
# CLV Analysis Engine
# ============================================================================

class CLVAnalysisEngine:
    """
    Complete Customer Lifetime Value analysis engine.
    
    Features:
    - Multiple CLV models (simple, historical, probabilistic)
    - Auto-detects transaction columns
    - Segment-based analysis
    - Discount rate adjustment
    """
    
    def __init__(self, config: CLVConfig = None, verbose: bool = True):
        self.config = config or CLVConfig()
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        customer_id_col: str = None,
        date_col: str = None,
        amount_col: str = None
    ) -> CLVResult:
        """
        Perform CLV analysis on transaction data.
        """
        start_time = datetime.now()
        
        # Auto-detect columns
        from app.ml.rfm_analysis import RFMColumnDetector
        detector = RFMColumnDetector()
        detected = detector.detect_columns(df)
        
        customer_id_col = customer_id_col or self.config.customer_id_col or detected[0]
        date_col = date_col or self.config.date_col or detected[1]
        amount_col = amount_col or self.config.amount_col or detected[2]
        
        if self.verbose:
            logger.info(f"CLV Analysis: customer={customer_id_col}, date={date_col}, amount={amount_col}")
        
        # Prepare data
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        df = df.dropna(subset=[customer_id_col, date_col, amount_col])
        
        # Calculate RFM metrics
        rfm_df = self._calculate_rfm_metrics(df, customer_id_col, date_col, amount_col)
        
        # Apply CLV model
        if self.config.model == CLVModel.SIMPLE:
            model = SimpleCLVModel()
            clv_df = model.calculate(rfm_df, self.config.prediction_horizon)
        elif self.config.model == CLVModel.HISTORICAL:
            model = HistoricalCLVModel()
            clv_df = model.calculate(rfm_df)
        elif self.config.model == CLVModel.PROBABILISTIC:
            model = ProbabilisticCLVModel(self.config.penalizer_coef)
            clv_df = model.calculate(
                rfm_df, 
                self.config.prediction_horizon,
                self.config.discount_rate
            )
        else:
            model = SimpleCLVModel()
            clv_df = model.calculate(rfm_df, self.config.prediction_horizon)
        
        # Calculate segments based on CLV
        clv_df = self._segment_by_clv(clv_df)
        
        # Statistics
        clv_stats = {
            'mean_clv': float(clv_df['predicted_clv'].mean()),
            'median_clv': float(clv_df['predicted_clv'].median()),
            'std_clv': float(clv_df['predicted_clv'].std()),
            'min_clv': float(clv_df['predicted_clv'].min()),
            'max_clv': float(clv_df['predicted_clv'].max()),
            'total_clv': float(clv_df['predicted_clv'].sum())
        }
        
        # Segment breakdown
        segment_clv = self._get_segment_breakdown(clv_df)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CLVResult(
            n_customers=len(clv_df),
            total_historical_value=float(clv_df['Monetary'].sum()),
            total_predicted_clv=float(clv_df['predicted_clv'].sum()),
            customer_clv=clv_df,
            clv_stats=clv_stats,
            segment_clv=segment_clv,
            processing_time_sec=processing_time
        )
    
    def _calculate_rfm_metrics(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str
    ) -> pd.DataFrame:
        """Calculate RFM metrics for CLV."""
        analysis_date = df[date_col].max() + timedelta(days=1)
        first_date = df[date_col].min()
        
        rfm = df.groupby(customer_col).agg({
            date_col: ['min', 'max', 'count'],
            amount_col: 'sum'
        })
        
        rfm.columns = ['FirstPurchase', 'LastPurchase', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()
        rfm.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase', 'Frequency', 'Monetary']
        
        rfm['Recency'] = (analysis_date - rfm['LastPurchase']).dt.days
        rfm['Tenure'] = (rfm['LastPurchase'] - rfm['FirstPurchase']).dt.days + 1
        
        return rfm
    
    def _segment_by_clv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Segment customers by CLV value."""
        df = df.copy()
        
        # Quartile-based segmentation
        try:
            df['CLV_Segment'] = pd.qcut(
                df['predicted_clv'],
                q=4,
                labels=['Low Value', 'Medium Value', 'High Value', 'Premium'],
                duplicates='drop'
            )
        except:
            df['CLV_Segment'] = 'Medium Value'
        
        return df
    
    def _get_segment_breakdown(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get CLV breakdown by segment."""
        if 'CLV_Segment' not in df.columns:
            return {}
        
        summary = {}
        for segment in df['CLV_Segment'].unique():
            seg_df = df[df['CLV_Segment'] == segment]
            summary[str(segment)] = {
                'count': int(len(seg_df)),
                'percentage': round(len(seg_df) / len(df) * 100, 1),
                'avg_clv': round(seg_df['predicted_clv'].mean(), 2),
                'total_clv': round(seg_df['predicted_clv'].sum(), 2)
            }
        
        return summary


# ============================================================================
# Factory Functions
# ============================================================================

def get_clv_engine(config: CLVConfig = None) -> CLVAnalysisEngine:
    """Get CLV analysis engine."""
    return CLVAnalysisEngine(config=config)


def quick_clv(
    df: pd.DataFrame,
    prediction_months: int = 12
) -> Dict[str, Any]:
    """
    Quick CLV analysis on transaction data.
    
    Example:
        result = quick_clv(transactions_df, prediction_months=12)
        print(result['summary'])
    """
    config = CLVConfig(prediction_horizon=prediction_months)
    engine = CLVAnalysisEngine(config=config, verbose=False)
    result = engine.analyze(df)
    return result.to_dict()
