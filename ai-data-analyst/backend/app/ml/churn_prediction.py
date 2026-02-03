# AI Enterprise Data Analyst - Churn Prediction Engine
# Production-grade churn prediction and analysis
# Handles: any customer data, multiple churn definitions

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        classification_report, roc_auc_score, precision_recall_curve,
        confusion_matrix, f1_score, accuracy_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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

class ChurnDefinition(str, Enum):
    """How to define churn."""
    INACTIVITY = "inactivity"  # No activity for N days
    EXPLICIT = "explicit"  # Explicit churn flag in data
    CONTRACTUAL = "contractual"  # Subscription cancellation


class ChurnModel(str, Enum):
    """Churn prediction models."""
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ChurnConfig:
    """Configuration for churn analysis."""
    # Churn definition
    churn_definition: ChurnDefinition = ChurnDefinition.INACTIVITY
    inactivity_days: int = 90  # Days without activity = churned
    
    # Model settings
    model_type: ChurnModel = ChurnModel.RANDOM_FOREST
    
    # Column mappings
    customer_id_col: Optional[str] = None
    date_col: Optional[str] = None
    churn_col: Optional[str] = None  # For explicit churn
    
    # Features to use (if None, auto-detect)
    feature_columns: Optional[List[str]] = None


@dataclass
class ChurnResult:
    """Complete churn analysis result."""
    n_customers: int = 0
    churn_rate: float = 0.0
    
    # Predictions
    customer_predictions: pd.DataFrame = None
    
    # Model performance
    model_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Risk segments
    risk_segments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Churn drivers
    churn_drivers: List[Dict[str, Any]] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_customers": self.n_customers,
                "churn_rate": round(self.churn_rate * 100, 2)
            },
            "model_metrics": {k: round(v, 4) for k, v in self.model_metrics.items()},
            "feature_importance": {k: round(v, 4) for k, v in 
                                  sorted(self.feature_importance.items(), key=lambda x: -x[1])[:15]},
            "risk_segments": self.risk_segments,
            "churn_drivers": self.churn_drivers[:10],
            "high_risk_customers": self.customer_predictions.nlargest(20, 'churn_probability').to_dict(orient='records') if self.customer_predictions is not None else []
        }


# ============================================================================
# Feature Engineering for Churn
# ============================================================================

class ChurnFeatureEngineer:
    """Create features for churn prediction."""
    
    def create_features(
        self,
        df: pd.DataFrame,
        customer_col: str,
        date_col: str,
        value_col: str = None,
        analysis_date: datetime = None
    ) -> pd.DataFrame:
        """Create behavioral features for each customer."""
        if analysis_date is None:
            analysis_date = pd.to_datetime(df[date_col]).max()
        
        features = df.groupby(customer_col).agg({
            date_col: ['min', 'max', 'count']
        })
        features.columns = ['first_activity', 'last_activity', 'activity_count']
        features = features.reset_index()
        
        # Recency
        features['recency_days'] = (analysis_date - features['last_activity']).dt.days
        
        # Tenure
        features['tenure_days'] = (features['last_activity'] - features['first_activity']).dt.days + 1
        
        # Frequency (activities per month)
        features['frequency_monthly'] = features['activity_count'] / (features['tenure_days'] / 30).clip(lower=1)
        
        # If value column exists
        if value_col and value_col in df.columns:
            value_agg = df.groupby(customer_col)[value_col].agg(['sum', 'mean', 'std'])
            value_agg.columns = ['total_value', 'avg_value', 'std_value']
            value_agg = value_agg.reset_index()
            features = features.merge(value_agg, on=customer_col, how='left')
            features['std_value'] = features['std_value'].fillna(0)
        
        # Recent activity trend
        recent_30 = df[pd.to_datetime(df[date_col]) >= (analysis_date - timedelta(days=30))]
        recent_90 = df[pd.to_datetime(df[date_col]) >= (analysis_date - timedelta(days=90))]
        
        recent_30_count = recent_30.groupby(customer_col).size().reset_index(name='activity_last_30')
        recent_90_count = recent_90.groupby(customer_col).size().reset_index(name='activity_last_90')
        
        features = features.merge(recent_30_count, on=customer_col, how='left')
        features = features.merge(recent_90_count, on=customer_col, how='left')
        features['activity_last_30'] = features['activity_last_30'].fillna(0)
        features['activity_last_90'] = features['activity_last_90'].fillna(0)
        
        # Trend indicator
        features['activity_trend'] = features['activity_last_30'] / (features['activity_last_90'] / 3).clip(lower=0.1)
        
        return features


# ============================================================================
# Churn Prediction Engine
# ============================================================================

class ChurnPredictionEngine:
    """
    Complete Churn Prediction engine.
    
    Features:
    - Multiple churn definitions
    - Automatic feature engineering
    - Multiple model options
    - Risk segmentation
    """
    
    def __init__(self, config: ChurnConfig = None, verbose: bool = True):
        self.config = config or ChurnConfig()
        self.verbose = verbose
        self.feature_engineer = ChurnFeatureEngineer()
        self._model = None
        self._scaler = None
        self._feature_names = []
    
    def analyze(
        self,
        df: pd.DataFrame,
        customer_id_col: str = None,
        date_col: str = None,
        churn_col: str = None
    ) -> ChurnResult:
        """Perform churn analysis and prediction."""
        start_time = datetime.now()
        
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for churn prediction")
        
        # Auto-detect columns
        customer_id_col = customer_id_col or self._detect_customer_col(df)
        date_col = date_col or self._detect_date_col(df)
        
        if self.verbose:
            logger.info(f"Churn analysis: customer={customer_id_col}, date={date_col}")
        
        # Prepare data
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[customer_id_col, date_col])
        
        analysis_date = df[date_col].max()
        
        # Create features
        features_df = self.feature_engineer.create_features(
            df, customer_id_col, date_col, 
            self._detect_value_col(df),
            analysis_date
        )
        
        # Define churn
        if self.config.churn_definition == ChurnDefinition.INACTIVITY:
            features_df['churned'] = (features_df['recency_days'] > self.config.inactivity_days).astype(int)
        elif self.config.churn_definition == ChurnDefinition.EXPLICIT and churn_col:
            churn_mapping = df.groupby(customer_id_col)[churn_col].max().reset_index()
            features_df = features_df.merge(churn_mapping, on=customer_id_col, how='left')
            features_df['churned'] = features_df[churn_col].fillna(0).astype(int)
        else:
            features_df['churned'] = (features_df['recency_days'] > self.config.inactivity_days).astype(int)
        
        churn_rate = features_df['churned'].mean()
        
        if self.verbose:
            logger.info(f"Churn rate: {churn_rate:.2%}")
        
        # Prepare features for modeling
        feature_cols = ['recency_days', 'tenure_days', 'frequency_monthly', 'activity_count',
                       'activity_last_30', 'activity_last_90', 'activity_trend']
        
        if 'total_value' in features_df.columns:
            feature_cols.extend(['total_value', 'avg_value', 'std_value'])
        
        available_features = [c for c in feature_cols if c in features_df.columns]
        self._feature_names = available_features
        
        X = features_df[available_features].fillna(0)
        y = features_df['churned']
        
        # Train model
        model_metrics, feature_importance = self._train_model(X, y)
        
        # Predict probabilities
        X_scaled = self._scaler.transform(X)
        features_df['churn_probability'] = self._model.predict_proba(X_scaled)[:, 1]
        features_df['churn_prediction'] = (features_df['churn_probability'] > 0.5).astype(int)
        
        # Risk segmentation
        features_df['risk_segment'] = pd.cut(
            features_df['churn_probability'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical']
        )
        
        risk_segments = self._get_risk_segments(features_df)
        
        # Churn drivers
        churn_drivers = self._analyze_churn_drivers(features_df, available_features)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChurnResult(
            n_customers=len(features_df),
            churn_rate=churn_rate,
            customer_predictions=features_df,
            model_metrics=model_metrics,
            feature_importance=feature_importance,
            risk_segments=risk_segments,
            churn_drivers=churn_drivers,
            processing_time_sec=processing_time
        )
    
    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train churn prediction model."""
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 10 else None
        )
        
        # Select and train model
        if self.config.model_type == ChurnModel.LOGISTIC:
            self._model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.config.model_type == ChurnModel.RANDOM_FOREST:
            self._model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.config.model_type == ChurnModel.GRADIENT_BOOSTING:
            self._model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            self._model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        self._model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Feature importance
        if hasattr(self._model, 'feature_importances_'):
            importance = dict(zip(self._feature_names, self._model.feature_importances_))
        elif hasattr(self._model, 'coef_'):
            importance = dict(zip(self._feature_names, np.abs(self._model.coef_[0])))
        else:
            importance = {f: 1.0 / len(self._feature_names) for f in self._feature_names}
        
        return metrics, importance
    
    def _get_risk_segments(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get summary by risk segment."""
        segments = {}
        for segment in df['risk_segment'].unique():
            seg_df = df[df['risk_segment'] == segment]
            segments[str(segment)] = {
                'count': int(len(seg_df)),
                'percentage': round(len(seg_df) / len(df) * 100, 1),
                'avg_churn_prob': round(seg_df['churn_probability'].mean(), 3),
                'avg_recency': round(seg_df['recency_days'].mean(), 1)
            }
        return segments
    
    def _analyze_churn_drivers(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze what drives churn."""
        drivers = []
        churned = df[df['churned'] == 1]
        active = df[df['churned'] == 0]
        
        for col in feature_cols:
            churned_mean = churned[col].mean()
            active_mean = active[col].mean()
            
            if active_mean != 0:
                diff_pct = (churned_mean - active_mean) / abs(active_mean) * 100
            else:
                diff_pct = 0
            
            drivers.append({
                'feature': col,
                'churned_avg': round(churned_mean, 2),
                'active_avg': round(active_mean, 2),
                'difference_pct': round(diff_pct, 1),
                'impact': 'Higher' if diff_pct > 0 else 'Lower'
            })
        
        return sorted(drivers, key=lambda x: abs(x['difference_pct']), reverse=True)
    
    def _detect_customer_col(self, df: pd.DataFrame) -> str:
        patterns = ['customer', 'user', 'client', 'member', 'id']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[0]
    
    def _detect_date_col(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        patterns = ['date', 'time', 'created']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[1]
    
    def _detect_value_col(self, df: pd.DataFrame) -> Optional[str]:
        patterns = ['amount', 'revenue', 'total', 'value', 'price']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(p in col.lower() for p in patterns):
                return col
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def get_churn_engine(config: ChurnConfig = None) -> ChurnPredictionEngine:
    """Get churn prediction engine."""
    return ChurnPredictionEngine(config=config)


def quick_churn(
    df: pd.DataFrame,
    inactivity_days: int = 90
) -> Dict[str, Any]:
    """Quick churn analysis."""
    config = ChurnConfig(inactivity_days=inactivity_days)
    engine = ChurnPredictionEngine(config=config, verbose=False)
    result = engine.analyze(df)
    return result.to_dict()
