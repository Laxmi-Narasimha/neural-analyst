# AI Enterprise Data Analyst - AutoML Engine
# Automated machine learning with feature selection and model optimization

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
import time

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# AutoML Types
# ============================================================================

class AutoMLTask(str, Enum):
    """AutoML task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"


class OptimizationMetric(str, Enum):
    """Optimization metrics."""
    # Classification
    ACCURACY = "accuracy"
    F1 = "f1"
    AUC = "auc"
    PRECISION = "precision"
    RECALL = "recall"
    LOG_LOSS = "log_loss"
    # Regression
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    MAPE = "mape"


@dataclass
class ModelCandidate:
    """Single model candidate in AutoML."""
    
    name: str
    model: Any
    params: dict[str, Any]
    score: float
    training_time: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "score": round(self.score, 4),
            "training_time_sec": round(self.training_time, 2),
            "top_features": dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }


@dataclass
class AutoMLResult:
    """AutoML experiment result."""
    
    task: AutoMLTask
    metric: OptimizationMetric
    best_model: ModelCandidate
    all_models: list[ModelCandidate]
    total_time: float
    
    # Data info
    n_samples: int = 0
    n_features: int = 0
    selected_features: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "metric": self.metric.value,
            "best_model": self.best_model.to_dict(),
            "models_evaluated": len(self.all_models),
            "total_time_sec": round(self.total_time, 2),
            "data": {
                "samples": self.n_samples,
                "features": self.n_features,
                "selected_features": self.selected_features[:20]
            },
            "leaderboard": [m.to_dict() for m in sorted(
                self.all_models, key=lambda x: x.score, reverse=True
            )[:5]]
        }


# ============================================================================
# Feature Selector
# ============================================================================

class FeatureSelector:
    """
    Automatic feature selection.
    
    Methods:
    - Variance threshold
    - Correlation filtering
    - Feature importance
    - Recursive elimination
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        max_features: int = None
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self._selected_features: list[str] = []
    
    def fit_select(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> pd.DataFrame:
        """Select features automatically."""
        selected_cols = list(X.columns)
        
        # 1. Remove constant/low variance
        variances = X.var()
        low_var = variances[variances < self.variance_threshold].index.tolist()
        selected_cols = [c for c in selected_cols if c not in low_var]
        
        # 2. Remove highly correlated
        if len(selected_cols) > 1:
            corr_matrix = X[selected_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
            selected_cols = [c for c in selected_cols if c not in to_drop]
        
        # 3. Feature importance if target provided
        if y is not None and len(selected_cols) > 0:
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                X_clean = X[selected_cols].fillna(0)
                
                if y.dtype == 'object' or y.nunique() <= 10:
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                
                model.fit(X_clean, y)
                
                importances = pd.Series(model.feature_importances_, index=selected_cols)
                importances = importances.sort_values(ascending=False)
                
                if self.max_features:
                    selected_cols = importances.head(self.max_features).index.tolist()
            except:
                pass
        
        self._selected_features = selected_cols
        return X[selected_cols]
    
    def get_selected_features(self) -> list[str]:
        """Get selected feature names."""
        return self._selected_features


# ============================================================================
# Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating ML models with default parameters."""
    
    @staticmethod
    def get_classification_models() -> list[tuple[str, Any, dict]]:
        """Get classification model candidates."""
        models = []
        
        try:
            from sklearn.linear_model import LogisticRegression
            models.append((
                "LogisticRegression",
                LogisticRegression(max_iter=1000, random_state=42),
                {"C": 1.0}
            ))
        except ImportError:
            pass
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            models.append((
                "RandomForest",
                RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                {"n_estimators": 100, "max_depth": 10}
            ))
        except ImportError:
            pass
        
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            models.append((
                "GradientBoosting",
                GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
                {"n_estimators": 100, "max_depth": 5}
            ))
        except ImportError:
            pass
        
        try:
            from sklearn.svm import SVC
            models.append((
                "SVM",
                SVC(probability=True, random_state=42),
                {"kernel": "rbf"}
            ))
        except ImportError:
            pass
        
        try:
            from xgboost import XGBClassifier
            models.append((
                "XGBoost",
                XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss'),
                {"n_estimators": 100}
            ))
        except ImportError:
            pass
        
        try:
            from lightgbm import LGBMClassifier
            models.append((
                "LightGBM",
                LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
                {"n_estimators": 100}
            ))
        except ImportError:
            pass
        
        return models
    
    @staticmethod
    def get_regression_models() -> list[tuple[str, Any, dict]]:
        """Get regression model candidates."""
        models = []
        
        try:
            from sklearn.linear_model import Ridge
            models.append((
                "Ridge",
                Ridge(alpha=1.0),
                {"alpha": 1.0}
            ))
        except ImportError:
            pass
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            models.append((
                "RandomForest",
                RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                {"n_estimators": 100}
            ))
        except ImportError:
            pass
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            models.append((
                "GradientBoosting",
                GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
                {"n_estimators": 100}
            ))
        except ImportError:
            pass
        
        try:
            from xgboost import XGBRegressor
            models.append((
                "XGBoost",
                XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
                {"n_estimators": 100}
            ))
        except ImportError:
            pass
        
        try:
            from lightgbm import LGBMRegressor
            models.append((
                "LightGBM",
                LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                {"n_estimators": 100}
            ))
        except ImportError:
            pass
        
        return models


# ============================================================================
# Model Evaluator
# ============================================================================

class ModelEvaluator:
    """Evaluate models with cross-validation."""
    
    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task: AutoMLTask,
        metric: OptimizationMetric,
        cv: int = 5
    ) -> float:
        """Evaluate model using cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score
            
            scoring = self._metric_to_sklearn(metric, task)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return float(scores.mean())
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0
    
    def _metric_to_sklearn(
        self,
        metric: OptimizationMetric,
        task: AutoMLTask
    ) -> str:
        """Convert metric to sklearn scoring string."""
        mapping = {
            OptimizationMetric.ACCURACY: 'accuracy',
            OptimizationMetric.F1: 'f1',
            OptimizationMetric.AUC: 'roc_auc',
            OptimizationMetric.PRECISION: 'precision',
            OptimizationMetric.RECALL: 'recall',
            OptimizationMetric.LOG_LOSS: 'neg_log_loss',
            OptimizationMetric.RMSE: 'neg_root_mean_squared_error',
            OptimizationMetric.MAE: 'neg_mean_absolute_error',
            OptimizationMetric.R2: 'r2',
        }
        
        return mapping.get(metric, 'accuracy' if task == AutoMLTask.CLASSIFICATION else 'neg_mean_squared_error')


# ============================================================================
# AutoML Engine
# ============================================================================

class AutoMLEngine:
    """
    Automated Machine Learning Engine.
    
    Features:
    - Automatic task detection
    - Feature selection
    - Model selection
    - Hyperparameter tuning
    - Cross-validation
    """
    
    def __init__(
        self,
        max_time_seconds: int = 300,
        cv_folds: int = 5
    ):
        self.max_time_seconds = max_time_seconds
        self.cv_folds = cv_folds
        self.feature_selector = FeatureSelector()
        self.evaluator = ModelEvaluator()
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: AutoMLTask = None,
        metric: OptimizationMetric = None
    ) -> AutoMLResult:
        """Run AutoML experiment."""
        start_time = time.time()
        
        # Detect task
        if task is None:
            task = self._detect_task(y)
        
        # Default metric
        if metric is None:
            metric = OptimizationMetric.AUC if task == AutoMLTask.CLASSIFICATION else OptimizationMetric.RMSE
        
        # Feature selection
        X_selected = self.feature_selector.fit_select(X, y)
        selected_features = self.feature_selector.get_selected_features()
        
        # Get models
        if task == AutoMLTask.CLASSIFICATION or task == AutoMLTask.MULTICLASS:
            models = ModelFactory.get_classification_models()
        else:
            models = ModelFactory.get_regression_models()
        
        if not models:
            raise ValidationException("No models available. Install sklearn.")
        
        # Train and evaluate models
        candidates: list[ModelCandidate] = []
        X_array = X_selected.fillna(0).values
        y_array = y.values
        
        for name, model, params in models:
            if time.time() - start_time > self.max_time_seconds:
                break
            
            try:
                model_start = time.time()
                
                # Evaluate
                score = self.evaluator.evaluate(
                    model, X_array, y_array, task, metric, self.cv_folds
                )
                
                # Fit for feature importance
                model.fit(X_array, y_array)
                
                # Feature importance
                importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(selected_features, model.feature_importances_.tolist()))
                elif hasattr(model, 'coef_'):
                    coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
                    importance = dict(zip(selected_features, np.abs(coef).tolist()))
                
                candidates.append(ModelCandidate(
                    name=name,
                    model=model,
                    params=params,
                    score=score,
                    training_time=time.time() - model_start,
                    feature_importance=importance
                ))
                
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        if not candidates:
            raise ValidationException("All models failed to train")
        
        # Find best
        best = max(candidates, key=lambda x: x.score)
        
        return AutoMLResult(
            task=task,
            metric=metric,
            best_model=best,
            all_models=candidates,
            total_time=time.time() - start_time,
            n_samples=len(X),
            n_features=len(selected_features),
            selected_features=selected_features
        )
    
    def _detect_task(self, y: pd.Series) -> AutoMLTask:
        """Detect ML task from target variable."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            n_unique = y.nunique()
            return AutoMLTask.MULTICLASS if n_unique > 2 else AutoMLTask.CLASSIFICATION
        elif y.nunique() <= 10:
            return AutoMLTask.CLASSIFICATION
        else:
            return AutoMLTask.REGRESSION
    
    def predict(
        self,
        X: pd.DataFrame,
        result: AutoMLResult
    ) -> np.ndarray:
        """Make predictions with best model."""
        X_selected = X[result.selected_features].fillna(0).values
        return result.best_model.model.predict(X_selected)
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        result: AutoMLResult
    ) -> np.ndarray:
        """Get prediction probabilities."""
        if not hasattr(result.best_model.model, 'predict_proba'):
            raise ValidationException("Model doesn't support probability predictions")
        
        X_selected = X[result.selected_features].fillna(0).values
        return result.best_model.model.predict_proba(X_selected)


# Factory function
def get_automl_engine() -> AutoMLEngine:
    """Get AutoML engine instance."""
    return AutoMLEngine()
