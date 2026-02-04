# AI Enterprise Data Analyst - Model Comparison Engine
# Production-grade model comparison and selection
# Handles: multiple models, cross-validation, statistical comparison

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class ModelType(str, Enum):
    """Types of models."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


class ComparisonCriterion(str, Enum):
    """Criteria for model comparison."""
    ACCURACY = "accuracy"
    F1 = "f1"
    AUC = "auc"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    name: str
    metrics: Dict[str, float]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    train_time_sec: float
    
    # Ranking
    rank: int = 0
    is_best: bool = False


@dataclass
class StatisticalComparison:
    """Statistical comparison between two models."""
    model1: str
    model2: str
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    winner: Optional[str]


@dataclass
class ModelComparisonResult:
    """Complete model comparison result."""
    n_models: int = 0
    primary_metric: str = ""
    task_type: ModelType = ModelType.CLASSIFIER
    
    # Model results
    models: List[ModelMetrics] = field(default_factory=list)
    
    # Best model
    best_model: str = ""
    best_score: float = 0.0
    
    # Statistical comparisons
    comparisons: List[StatisticalComparison] = field(default_factory=list)
    
    # Rankings
    rankings: Dict[str, int] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_models": self.n_models,
                "primary_metric": self.primary_metric,
                "task_type": self.task_type.value,
                "best_model": self.best_model,
                "best_score": round(self.best_score, 4)
            },
            "models": [
                {
                    "name": m.name,
                    "rank": m.rank,
                    "cv_mean": round(m.cv_mean, 4),
                    "cv_std": round(m.cv_std, 4),
                    "metrics": {k: round(v, 4) for k, v in m.metrics.items()},
                    "is_best": m.is_best
                }
                for m in self.models
            ],
            "rankings": self.rankings,
            "statistical_tests": [
                {
                    "comparison": f"{c.model1} vs {c.model2}",
                    "p_value": round(c.p_value, 4),
                    "significant": c.is_significant,
                    "winner": c.winner
                }
                for c in self.comparisons[:10]
            ]
        }


# ============================================================================
# Model Comparison Engine
# ============================================================================

class ModelComparisonEngine:
    """
    Production-grade Model Comparison engine.
    
    Features:
    - Cross-validation comparison
    - Multiple metrics
    - Statistical significance testing
    - Automatic best model selection
    - Ranking by multiple criteria
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        alpha: float = 0.05,
        verbose: bool = True
    ):
        self.n_folds = n_folds
        self.alpha = alpha
        self.verbose = verbose
    
    def compare(
        self,
        *args: Any,
        task_type: ModelType = None,
        primary_metric: ComparisonCriterion = None,
    ) -> ModelComparisonResult:
        """
        Compare multiple models.

        Supports both call styles:
        - compare(models, X, y)  (used by repo tests)
        - compare(X, y, models)  (legacy/internal)
        """
        start_time = datetime.now()

        if len(args) != 3:
            raise ValueError("compare expects 3 positional arguments: (models, X, y) or (X, y, models)")

        a0, a1, a2 = args
        if isinstance(a0, dict):
            models: Dict[str, Any] = a0
            X: pd.DataFrame = a1
            y: pd.Series = a2
        else:
            X = a0
            y = a1
            models = a2
        
        # Auto-detect task type
        if task_type is None:
            task_type = self._detect_task_type(y)
        
        # Set default metric
        if primary_metric is None:
            primary_metric = (ComparisonCriterion.ACCURACY if task_type == ModelType.CLASSIFIER 
                             else ComparisonCriterion.RMSE)
        
        if self.verbose:
            logger.info(f"Comparing {len(models)} models using {primary_metric.value}")
        
        # Evaluate each model
        model_results = []
        
        for name, model in models.items():
            try:
                metrics, cv_scores, train_time = self._evaluate_model(
                    model, X, y, task_type
                )
                
                model_results.append(ModelMetrics(
                    name=name,
                    metrics=metrics,
                    cv_scores=cv_scores,
                    cv_mean=float(np.mean(cv_scores)),
                    cv_std=float(np.std(cv_scores)),
                    train_time_sec=train_time
                ))
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")
        
        if not model_results:
            return ModelComparisonResult(n_models=0)
        
        # Rank models
        higher_is_better = primary_metric not in [
            ComparisonCriterion.RMSE, ComparisonCriterion.MAE
        ]
        
        model_results.sort(
            key=lambda x: x.cv_mean,
            reverse=higher_is_better
        )
        
        for i, m in enumerate(model_results):
            m.rank = i + 1
            m.is_best = i == 0
        
        best_model = model_results[0]
        
        # Statistical comparisons
        comparisons = []
        for i, m1 in enumerate(model_results):
            for m2 in model_results[i+1:]:
                comp = self._statistical_test(m1, m2, higher_is_better)
                comparisons.append(comp)
        
        # Rankings dict
        rankings = {m.name: m.rank for m in model_results}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ModelComparisonResult(
            n_models=len(model_results),
            primary_metric=primary_metric.value,
            task_type=task_type,
            models=model_results,
            best_model=best_model.name,
            best_score=best_model.cv_mean,
            comparisons=comparisons,
            rankings=rankings,
            processing_time_sec=processing_time
        )
    
    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: ModelType
    ) -> Tuple[Dict[str, float], List[float], float]:
        """Evaluate a model with cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score, cross_validate
            from sklearn.metrics import (
                accuracy_score, f1_score, roc_auc_score,
                mean_squared_error, mean_absolute_error, r2_score
            )
            
            train_start = datetime.now()
            
            if task_type == ModelType.CLASSIFIER:
                scoring = {
                    'accuracy': 'accuracy',
                    'f1': 'f1_weighted'
                }
                
                cv_results = cross_validate(
                    model, X, y, cv=self.n_folds,
                    scoring=scoring, return_train_score=False
                )
                
                metrics = {
                    'accuracy': float(np.mean(cv_results['test_accuracy'])),
                    'f1': float(np.mean(cv_results['test_f1']))
                }
                cv_scores = cv_results['test_accuracy'].tolist()
                
            else:  # Regressor
                scoring = {
                    'r2': 'r2',
                    'neg_mse': 'neg_mean_squared_error',
                    'neg_mae': 'neg_mean_absolute_error'
                }
                
                cv_results = cross_validate(
                    model, X, y, cv=self.n_folds,
                    scoring=scoring, return_train_score=False
                )
                
                metrics = {
                    'r2': float(np.mean(cv_results['test_r2'])),
                    'rmse': float(np.sqrt(-np.mean(cv_results['test_neg_mse']))),
                    'mae': float(-np.mean(cv_results['test_neg_mae']))
                }
                cv_scores = cv_results['test_r2'].tolist()
            
            train_time = (datetime.now() - train_start).total_seconds()
            
            return metrics, cv_scores, train_time
            
        except ImportError:
            # Fallback without sklearn
            return self._simple_evaluation(model, X, y, task_type)
    
    def _simple_evaluation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: ModelType
    ) -> Tuple[Dict[str, float], List[float], float]:
        """Simple hold-out evaluation as fallback."""
        train_start = datetime.now()
        
        # Simple 80-20 split
        n = len(X)
        train_idx = list(range(int(n * 0.8)))
        test_idx = list(range(int(n * 0.8), n))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        if task_type == ModelType.CLASSIFIER:
            accuracy = np.mean(predictions == y_test)
            metrics = {'accuracy': accuracy}
            cv_scores = [accuracy]
        else:
            mse = np.mean((predictions - y_test) ** 2)
            metrics = {'rmse': np.sqrt(mse)}
            cv_scores = [-mse]
        
        train_time = (datetime.now() - train_start).total_seconds()
        
        return metrics, cv_scores, train_time
    
    def _statistical_test(
        self,
        model1: ModelMetrics,
        model2: ModelMetrics,
        higher_is_better: bool
    ) -> StatisticalComparison:
        """Statistical comparison between two models."""
        scores1 = model1.cv_scores
        scores2 = model2.cv_scores
        
        if len(scores1) < 2 or len(scores2) < 2:
            return StatisticalComparison(
                model1=model1.name,
                model2=model2.name,
                test_name="insufficient_data",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                winner=None
            )
        
        # Paired t-test
        try:
            stat, p_value = scipy_stats.ttest_rel(scores1, scores2)
        except:
            stat, p_value = 0, 1.0
        
        is_significant = p_value < self.alpha
        
        # Determine winner
        winner = None
        if is_significant:
            mean1, mean2 = np.mean(scores1), np.mean(scores2)
            if higher_is_better:
                winner = model1.name if mean1 > mean2 else model2.name
            else:
                winner = model1.name if mean1 < mean2 else model2.name
        
        return StatisticalComparison(
            model1=model1.name,
            model2=model2.name,
            test_name="paired_t_test",
            statistic=float(stat),
            p_value=float(p_value),
            is_significant=is_significant,
            winner=winner
        )
    
    def _detect_task_type(self, y: pd.Series) -> ModelType:
        """Detect if classification or regression."""
        unique = y.nunique()
        
        if unique <= 10 or y.dtype in ['object', 'category', 'bool']:
            return ModelType.CLASSIFIER
        
        return ModelType.REGRESSOR


# ============================================================================
# Factory Functions
# ============================================================================

def get_comparison_engine(n_folds: int = 5) -> ModelComparisonEngine:
    """Get model comparison engine."""
    return ModelComparisonEngine(n_folds=n_folds)


def quick_compare(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Quick model comparison."""
    engine = ModelComparisonEngine(verbose=False)
    result = engine.compare(models, X, y)
    return result.to_dict()


def get_default_classifiers() -> Dict[str, Any]:
    """Get default classifiers for comparison."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        
        return {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(max_depth=10),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'KNN': KNeighborsClassifier()
        }
    except ImportError:
        return {}


def get_default_regressors() -> Dict[str, Any]:
    """Get default regressors for comparison."""
    try:
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'DecisionTree': DecisionTreeRegressor(max_depth=10),
            'RandomForest': RandomForestRegressor(n_estimators=100)
        }
    except ImportError:
        return {}
