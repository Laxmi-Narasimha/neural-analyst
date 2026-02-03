# AI Enterprise Data Analyst - Customer Analytics Engine
# Cohort analysis, RFM, CLV, churn prediction (Shopify/Stripe patterns)

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
# Customer Analytics Types
# ============================================================================

class RFMSegment(str, Enum):
    """RFM customer segments."""
    CHAMPIONS = "champions"
    LOYAL_CUSTOMERS = "loyal_customers"
    POTENTIAL_LOYALISTS = "potential_loyalists"
    NEW_CUSTOMERS = "new_customers"
    PROMISING = "promising"
    NEED_ATTENTION = "need_attention"
    ABOUT_TO_SLEEP = "about_to_sleep"
    AT_RISK = "at_risk"
    CANT_LOSE = "cant_lose"
    HIBERNATING = "hibernating"
    LOST = "lost"


@dataclass
class RFMScore:
    """RFM analysis result for a customer."""
    
    customer_id: Any
    recency: int  # Days since last purchase
    frequency: int  # Number of purchases
    monetary: float  # Total spend
    
    r_score: int = 0  # 1-5
    f_score: int = 0
    m_score: int = 0
    rfm_score: str = ""
    segment: Optional[RFMSegment] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "recency_days": self.recency,
            "frequency": self.frequency,
            "monetary": round(self.monetary, 2),
            "r_score": self.r_score,
            "f_score": self.f_score,
            "m_score": self.m_score,
            "rfm_score": self.rfm_score,
            "segment": self.segment.value if self.segment else None
        }


@dataclass
class CohortMetrics:
    """Cohort analysis metrics."""
    
    cohort: str  # e.g., "2024-01"
    period: int  # Periods since acquisition
    users: int
    retention_rate: float
    revenue: float = 0.0
    avg_order_value: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "period": self.period,
            "users": self.users,
            "retention_rate": round(self.retention_rate, 4),
            "revenue": round(self.revenue, 2),
            "avg_order_value": round(self.avg_order_value, 2)
        }


@dataclass
class CLVResult:
    """Customer Lifetime Value result."""
    
    customer_id: Any
    historical_clv: float
    predicted_clv: float
    probability_alive: float
    expected_purchases: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "historical_clv": round(self.historical_clv, 2),
            "predicted_clv": round(self.predicted_clv, 2),
            "probability_alive": round(self.probability_alive, 4),
            "expected_purchases": round(self.expected_purchases, 2)
        }


# ============================================================================
# RFM Analyzer
# ============================================================================

class RFMAnalyzer:
    """
    RFM (Recency, Frequency, Monetary) analysis.
    
    Standard e-commerce customer segmentation technique.
    """
    
    def __init__(self, n_segments: int = 5):
        self.n_segments = n_segments
    
    def analyze(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str,
        reference_date: datetime = None
    ) -> list[RFMScore]:
        """Perform RFM analysis."""
        reference_date = reference_date or datetime.now()
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,
            customer_col: 'count',
            amount_col: 'sum'
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm = rfm.reset_index()
        
        # Score each dimension
        rfm['r_score'] = pd.qcut(
            rfm['recency'].rank(method='first'),
            q=self.n_segments,
            labels=range(self.n_segments, 0, -1)
        ).astype(int)
        
        rfm['f_score'] = pd.qcut(
            rfm['frequency'].rank(method='first'),
            q=self.n_segments,
            labels=range(1, self.n_segments + 1)
        ).astype(int)
        
        rfm['m_score'] = pd.qcut(
            rfm['monetary'].rank(method='first'),
            q=self.n_segments,
            labels=range(1, self.n_segments + 1)
        ).astype(int)
        
        # Create RFM string
        rfm['rfm_score'] = (
            rfm['r_score'].astype(str) +
            rfm['f_score'].astype(str) +
            rfm['m_score'].astype(str)
        )
        
        # Segment customers
        results = []
        for _, row in rfm.iterrows():
            segment = self._get_segment(row['r_score'], row['f_score'], row['m_score'])
            
            results.append(RFMScore(
                customer_id=row[customer_col],
                recency=int(row['recency']),
                frequency=int(row['frequency']),
                monetary=float(row['monetary']),
                r_score=int(row['r_score']),
                f_score=int(row['f_score']),
                m_score=int(row['m_score']),
                rfm_score=row['rfm_score'],
                segment=segment
            ))
        
        return results
    
    def _get_segment(self, r: int, f: int, m: int) -> RFMSegment:
        """Determine customer segment from RFM scores."""
        rfm_sum = r + f + m
        
        if r >= 4 and f >= 4:
            return RFMSegment.CHAMPIONS
        elif r >= 3 and f >= 4:
            return RFMSegment.LOYAL_CUSTOMERS
        elif r >= 4 and f >= 2:
            return RFMSegment.POTENTIAL_LOYALISTS
        elif r >= 4 and f == 1:
            return RFMSegment.NEW_CUSTOMERS
        elif r >= 3 and f >= 2:
            return RFMSegment.PROMISING
        elif r == 3 and f <= 2:
            return RFMSegment.NEED_ATTENTION
        elif r == 2 and f >= 2:
            return RFMSegment.ABOUT_TO_SLEEP
        elif r == 2 and f >= 4:
            return RFMSegment.AT_RISK
        elif r == 1 and f >= 4:
            return RFMSegment.CANT_LOSE
        elif r == 2 and f <= 2:
            return RFMSegment.HIBERNATING
        else:
            return RFMSegment.LOST
    
    def get_segment_summary(self, rfm_scores: list[RFMScore]) -> dict[str, dict]:
        """Get summary by segment."""
        segments = {}
        
        for score in rfm_scores:
            if score.segment:
                seg = score.segment.value
                if seg not in segments:
                    segments[seg] = {
                        "count": 0,
                        "total_revenue": 0,
                        "avg_frequency": 0,
                        "avg_recency": 0
                    }
                segments[seg]["count"] += 1
                segments[seg]["total_revenue"] += score.monetary
                segments[seg]["avg_frequency"] += score.frequency
                segments[seg]["avg_recency"] += score.recency
        
        # Calculate averages
        for seg in segments:
            count = segments[seg]["count"]
            if count > 0:
                segments[seg]["avg_frequency"] /= count
                segments[seg]["avg_recency"] /= count
        
        return segments


# ============================================================================
# Cohort Analyzer
# ============================================================================

class CohortAnalyzer:
    """
    Cohort analysis for retention and revenue tracking.
    """
    
    def analyze(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        cohort_period: str = "M"  # M=monthly, W=weekly
    ) -> pd.DataFrame:
        """Perform cohort analysis."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Determine cohort (first transaction date)
        cohorts = df.groupby(customer_col)[date_col].min().reset_index()
        cohorts.columns = [customer_col, 'cohort_date']
        
        df = df.merge(cohorts, on=customer_col)
        
        # Cohort period
        df['cohort'] = df['cohort_date'].dt.to_period(cohort_period)
        df['transaction_period'] = df[date_col].dt.to_period(cohort_period)
        
        # Period number (0, 1, 2, ...)
        df['period'] = (df['transaction_period'] - df['cohort']).apply(lambda x: x.n)
        
        # Cohort size (users who started in each cohort)
        cohort_sizes = df.groupby('cohort')[customer_col].nunique()
        
        # Active users in each period
        cohort_pivot = df.groupby(['cohort', 'period'])[customer_col].nunique().unstack(fill_value=0)
        
        # Retention rates
        retention = cohort_pivot.div(cohort_sizes, axis=0)
        
        return retention
    
    def get_cohort_metrics(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str = None,
        cohort_period: str = "M"
    ) -> list[CohortMetrics]:
        """Get detailed cohort metrics."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Determine cohorts
        cohorts = df.groupby(customer_col)[date_col].min().reset_index()
        cohorts.columns = [customer_col, 'cohort_date']
        df = df.merge(cohorts, on=customer_col)
        
        df['cohort'] = df['cohort_date'].dt.to_period(cohort_period).astype(str)
        df['transaction_period'] = df[date_col].dt.to_period(cohort_period)
        df['period'] = ((df[date_col].dt.to_period(cohort_period) - 
                         df['cohort_date'].dt.to_period(cohort_period)).apply(lambda x: x.n))
        
        # Cohort sizes
        cohort_sizes = df.groupby('cohort')[customer_col].nunique()
        
        results = []
        for cohort in df['cohort'].unique():
            cohort_data = df[df['cohort'] == cohort]
            cohort_size = cohort_sizes[cohort]
            
            for period in sorted(cohort_data['period'].unique()):
                period_data = cohort_data[cohort_data['period'] == period]
                users = period_data[customer_col].nunique()
                
                metrics = CohortMetrics(
                    cohort=cohort,
                    period=int(period),
                    users=users,
                    retention_rate=users / cohort_size if cohort_size > 0 else 0
                )
                
                if amount_col and amount_col in period_data.columns:
                    metrics.revenue = float(period_data[amount_col].sum())
                    metrics.avg_order_value = float(period_data[amount_col].mean())
                
                results.append(metrics)
        
        return results


# ============================================================================
# CLV Calculator
# ============================================================================

class CLVCalculator:
    """
    Customer Lifetime Value calculation.
    
    Methods:
    - Historical (actual spend)
    - Predictive (BG/NBD model approximation)
    """
    
    def __init__(self, discount_rate: float = 0.1):
        self.discount_rate = discount_rate
    
    def calculate_historical(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str
    ) -> dict[Any, float]:
        """Calculate historical CLV."""
        clv = df.groupby(customer_col)[amount_col].sum()
        return clv.to_dict()
    
    def calculate_predictive(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str,
        prediction_period: int = 12  # months
    ) -> list[CLVResult]:
        """
        Calculate predictive CLV using simplified BG/NBD approach.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        now = df[date_col].max()
        
        # Calculate customer metrics
        customer_stats = df.groupby(customer_col).agg({
            date_col: ['min', 'max', 'count'],
            amount_col: ['sum', 'mean']
        })
        
        customer_stats.columns = ['first_purchase', 'last_purchase', 'frequency', 'total_spend', 'avg_order']
        customer_stats = customer_stats.reset_index()
        
        results = []
        for _, row in customer_stats.iterrows():
            customer_id = row[customer_col]
            
            # Recency and tenure
            recency = (now - row['last_purchase']).days
            tenure = (row['last_purchase'] - row['first_purchase']).days
            
            # Historical CLV
            historical = float(row['total_spend'])
            
            # Probability of being alive (simplified)
            if tenure > 0:
                purchase_rate = row['frequency'] / (tenure / 30)  # Per month
                prob_alive = max(0, 1 - recency / (recency + 30 / purchase_rate)) if purchase_rate > 0 else 0.5
            else:
                purchase_rate = 0.5
                prob_alive = 0.8 if recency < 90 else 0.3
            
            # Expected purchases in prediction period
            expected_purchases = prob_alive * purchase_rate * prediction_period
            
            # Predicted CLV
            predicted = expected_purchases * row['avg_order']
            
            # Discount future value
            monthly_rate = self.discount_rate / 12
            if monthly_rate > 0:
                discount_factor = (1 - (1 + monthly_rate) ** (-prediction_period)) / monthly_rate
                predicted *= discount_factor / prediction_period
            
            results.append(CLVResult(
                customer_id=customer_id,
                historical_clv=historical,
                predicted_clv=float(predicted),
                probability_alive=float(prob_alive),
                expected_purchases=float(expected_purchases)
            ))
        
        return results


# ============================================================================
# Churn Predictor
# ============================================================================

class ChurnPredictor:
    """
    Customer churn prediction.
    """
    
    def __init__(self):
        self._model = None
        self._features = []
        self._fitted = False
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str = None,
        reference_date: datetime = None
    ) -> pd.DataFrame:
        """Prepare churn prediction features."""
        reference_date = reference_date or datetime.now()
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        features = df.groupby(customer_col).agg({
            date_col: ['min', 'max', 'count'],
        })
        
        features.columns = ['first_date', 'last_date', 'transaction_count']
        features = features.reset_index()
        
        # Recency
        features['recency_days'] = (reference_date - features['last_date']).dt.days
        
        # Tenure
        features['tenure_days'] = (features['last_date'] - features['first_date']).dt.days
        
        # Frequency
        features['avg_days_between'] = features['tenure_days'] / (features['transaction_count'] + 1)
        
        if amount_col and amount_col in df.columns:
            amount_stats = df.groupby(customer_col)[amount_col].agg(['sum', 'mean', 'std'])
            amount_stats.columns = ['total_spend', 'avg_spend', 'spend_std']
            features = features.merge(amount_stats.reset_index(), on=customer_col)
            features['spend_std'] = features['spend_std'].fillna(0)
        
        return features
    
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_cols: list[str] = None
    ) -> dict[str, float]:
        """Train churn prediction model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            self._features = feature_cols or [
                c for c in features.columns 
                if c not in ['customer_id', 'first_date', 'last_date']
            ]
            
            X = features[self._features].fillna(0)
            
            self._model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            scores = cross_val_score(self._model, X, target, cv=5, scoring='roc_auc')
            
            self._model.fit(X, target)
            self._fitted = True
            
            return {
                "auc_mean": float(scores.mean()),
                "auc_std": float(scores.std()),
                "feature_importance": dict(zip(self._features, self._model.feature_importances_.tolist()))
            }
            
        except ImportError:
            return {"error": "sklearn not installed"}
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict churn probability."""
        if not self._fitted:
            raise ValidationException("Model not trained")
        
        X = features[self._features].fillna(0)
        probas = self._model.predict_proba(X)[:, 1]
        
        result = features.copy()
        result['churn_probability'] = probas
        result['churn_risk'] = pd.cut(
            probas,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        return result


# ============================================================================
# Customer Analytics Engine
# ============================================================================

class CustomerAnalyticsEngine:
    """
    Complete customer analytics suite.
    
    Features:
    - RFM segmentation
    - Cohort analysis
    - Customer Lifetime Value
    - Churn prediction
    """
    
    def __init__(self):
        self.rfm = RFMAnalyzer()
        self.cohort = CohortAnalyzer()
        self.clv = CLVCalculator()
        self.churn = ChurnPredictor()
    
    def full_analysis(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        amount_col: str
    ) -> dict[str, Any]:
        """Run full customer analytics."""
        results = {}
        
        # RFM
        rfm_scores = self.rfm.analyze(df, customer_col, date_col, amount_col)
        results['rfm'] = {
            "scores": [s.to_dict() for s in rfm_scores[:100]],
            "segments": self.rfm.get_segment_summary(rfm_scores)
        }
        
        # Cohort
        cohort_metrics = self.cohort.get_cohort_metrics(
            df, customer_col, date_col, amount_col
        )
        results['cohort'] = [m.to_dict() for m in cohort_metrics]
        
        # CLV
        clv_results = self.clv.calculate_predictive(
            df, customer_col, date_col, amount_col
        )
        results['clv'] = {
            "results": [c.to_dict() for c in clv_results[:100]],
            "total_predicted": sum(c.predicted_clv for c in clv_results),
            "avg_clv": np.mean([c.predicted_clv for c in clv_results])
        }
        
        return results


# Factory function
def get_customer_analytics_engine() -> CustomerAnalyticsEngine:
    """Get customer analytics engine instance."""
    return CustomerAnalyticsEngine()
