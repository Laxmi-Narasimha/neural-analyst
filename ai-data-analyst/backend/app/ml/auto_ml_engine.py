# AI Enterprise Data Analyst - Advanced AutoML Engine
# Production-grade ML with bulletproof data handling for ANY dataset
# Handles: nulls, outliers, mixed types, any columns, any size, any format

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID, uuid4
import pickle
import io
import traceback

import numpy as np
import pandas as pd

# Sklearn imports
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
    from sklearn.model_selection import (
        train_test_split, cross_val_score, cross_validate,
        StratifiedKFold, KFold, TimeSeriesSplit,
        GridSearchCV, RandomizedSearchCV
    )
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        LabelEncoder, OneHotEncoder, OrdinalEncoder,
        PowerTransformer, QuantileTransformer
    )
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression, mutual_info_classif,
        RFE, SelectFromModel, VarianceThreshold
    )
    from sklearn.metrics import (
        # Classification
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report, balanced_accuracy_score, log_loss,
        matthews_corrcoef, cohen_kappa_score,
        # Regression
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error, explained_variance_score,
        max_error, median_absolute_error
    )
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
        SGDClassifier, SGDRegressor, BayesianRidge, HuberRegressor
    )
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        AdaBoostClassifier, AdaBoostRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor,
        BaggingClassifier, BaggingRegressor,
        VotingClassifier, VotingRegressor,
        StackingClassifier, StackingRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        IsolationForest
    )
    from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.inspection import permutation_importance
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Gradient boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import MLException, DataProcessingException
except ImportError:
    class MLException(Exception):
        pass

    class DataProcessingException(Exception):
        pass

logger = get_logger(__name__)


# ============================================================================
# Enums and Types
# ============================================================================

class MLTask(str, Enum):
    """Machine learning task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"


class ModelStatus(str, Enum):
    """Model training status."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizationStrategy(str, Enum):
    """Hyperparameter optimization strategies."""
    NONE = "none"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"  # Optuna
    SUCCESSIVE_HALVING = "successive_halving"


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for an ML model."""
    name: str
    display_name: str
    estimator_class: Type[BaseEstimator]
    task: MLTask
    default_params: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    supports_feature_importance: bool = False
    supports_probability: bool = False
    handles_missing: bool = False
    handles_categorical: bool = False
    max_samples_limit: Optional[int] = None  # For slow models
    tags: List[str] = field(default_factory=list)


class ModelLibrary:
    """Library of available ML models with configurations."""
    
    _models: Dict[str, ModelConfig] = {}
    
    @classmethod
    def register(cls, config: ModelConfig) -> None:
        """Register a model configuration."""
        cls._models[config.name] = config
    
    @classmethod
    def get(cls, name: str) -> ModelConfig:
        """Get model configuration by name."""
        if name not in cls._models:
            raise MLException(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def get_for_task(cls, task: MLTask, max_count: int = 10) -> List[str]:
        """Get models suitable for a task."""
        # Map task to compatible tasks
        # Classifier configs are registered under BINARY_CLASSIFICATION but every
        # one of them also handles more than two classes.
        classifiers = [MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION]
        task_map = {
            MLTask.BINARY_CLASSIFICATION: classifiers,
            MLTask.MULTICLASS_CLASSIFICATION: classifiers,
            MLTask.REGRESSION: [MLTask.REGRESSION],
        }
        compatible = task_map.get(task, [task])
        
        models = [
            name for name, cfg in cls._models.items()
            if cfg.task in compatible
        ]
        return models[:max_count]
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())


# Register Classification Models
ModelLibrary.register(ModelConfig(
    name="logistic_regression",
    display_name="Logistic Regression",
    estimator_class=LogisticRegression,
    task=MLTask.BINARY_CLASSIFICATION,
    default_params={"max_iter": 2000, "random_state": 42},
    search_space={
        "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
    },
    supports_probability=True,
    tags=["fast", "interpretable", "baseline"]
))

ModelLibrary.register(ModelConfig(
    name="random_forest_classifier",
    display_name="Random Forest Classifier",
    estimator_class=RandomForestClassifier,
    task=MLTask.BINARY_CLASSIFICATION,
    default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1, "max_depth": 10},
    search_space={
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}
    },
    supports_feature_importance=True,
    supports_probability=True,
    tags=["ensemble", "robust", "parallel"]
))

ModelLibrary.register(ModelConfig(
    name="gradient_boosting_classifier",
    display_name="Gradient Boosting Classifier",
    estimator_class=GradientBoostingClassifier,
    task=MLTask.BINARY_CLASSIFICATION,
    default_params={"n_estimators": 100, "random_state": 42, "max_depth": 5},
    search_space={
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0}
    },
    supports_feature_importance=True,
    supports_probability=True,
    tags=["ensemble", "accurate"]
))

ModelLibrary.register(ModelConfig(
    name="hist_gradient_boosting_classifier",
    display_name="Hist Gradient Boosting Classifier",
    estimator_class=HistGradientBoostingClassifier,
    task=MLTask.BINARY_CLASSIFICATION,
    default_params={"max_iter": 100, "random_state": 42, "max_depth": 8},
    search_space={
        "max_iter": {"type": "int", "low": 50, "high": 300},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 15}
    },
    supports_feature_importance=True,
    supports_probability=True,
    handles_missing=True,  # Natively handles missing values
    handles_categorical=True,
    tags=["ensemble", "fast", "handles_missing"]
))

if HAS_XGBOOST:
    ModelLibrary.register(ModelConfig(
        name="xgboost_classifier",
        display_name="XGBoost Classifier",
        estimator_class=xgb.XGBClassifier,
        task=MLTask.BINARY_CLASSIFICATION,
        default_params={
            "n_estimators": 100, "random_state": 42, "n_jobs": -1,
            "use_label_encoder": False, "eval_metric": "logloss",
            "enable_categorical": True
        },
        search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10, "log": True}
        },
        supports_feature_importance=True,
        supports_probability=True,
        handles_missing=True,
        tags=["ensemble", "fast", "accurate", "handles_missing"]
    ))

if HAS_LIGHTGBM:
    ModelLibrary.register(ModelConfig(
        name="lightgbm_classifier",
        display_name="LightGBM Classifier",
        estimator_class=lgb.LGBMClassifier,
        task=MLTask.BINARY_CLASSIFICATION,
        default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1, "verbose": -1},
        search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 10, "high": 100},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}
        },
        supports_feature_importance=True,
        supports_probability=True,
        handles_missing=True,
        handles_categorical=True,
        tags=["ensemble", "very_fast", "accurate", "handles_missing"]
    ))

if HAS_CATBOOST:
    ModelLibrary.register(ModelConfig(
        name="catboost_classifier",
        display_name="CatBoost Classifier",
        estimator_class=cb.CatBoostClassifier,
        task=MLTask.BINARY_CLASSIFICATION,
        default_params={"iterations": 100, "random_state": 42, "verbose": 0, "allow_writing_files": False},
        search_space={
            "iterations": {"type": "int", "low": 50, "high": 500},
            "depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1, "high": 10}
        },
        supports_feature_importance=True,
        supports_probability=True,
        handles_missing=True,
        handles_categorical=True,
        tags=["ensemble", "handles_categorical", "accurate"]
    ))

# Register Regression Models
ModelLibrary.register(ModelConfig(
    name="linear_regression",
    display_name="Linear Regression",
    estimator_class=LinearRegression,
    task=MLTask.REGRESSION,
    default_params={},
    search_space={},
    tags=["fast", "interpretable", "baseline"]
))

ModelLibrary.register(ModelConfig(
    name="ridge_regression",
    display_name="Ridge Regression",
    estimator_class=Ridge,
    task=MLTask.REGRESSION,
    default_params={"random_state": 42},
    search_space={
        "alpha": {"type": "float", "low": 0.01, "high": 100, "log": True}
    },
    tags=["fast", "interpretable", "regularized"]
))

ModelLibrary.register(ModelConfig(
    name="elastic_net",
    display_name="Elastic Net",
    estimator_class=ElasticNet,
    task=MLTask.REGRESSION,
    default_params={"random_state": 42, "max_iter": 2000},
    search_space={
        "alpha": {"type": "float", "low": 0.01, "high": 10, "log": True},
        "l1_ratio": {"type": "float", "low": 0.1, "high": 0.9}
    },
    tags=["fast", "interpretable", "regularized"]
))

ModelLibrary.register(ModelConfig(
    name="random_forest_regressor",
    display_name="Random Forest Regressor",
    estimator_class=RandomForestRegressor,
    task=MLTask.REGRESSION,
    default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    search_space={
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20}
    },
    supports_feature_importance=True,
    tags=["ensemble", "robust"]
))

ModelLibrary.register(ModelConfig(
    name="hist_gradient_boosting_regressor",
    display_name="Hist Gradient Boosting Regressor",
    estimator_class=HistGradientBoostingRegressor,
    task=MLTask.REGRESSION,
    default_params={"max_iter": 100, "random_state": 42},
    search_space={
        "max_iter": {"type": "int", "low": 50, "high": 300},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 15}
    },
    supports_feature_importance=True,
    handles_missing=True,
    tags=["ensemble", "fast", "handles_missing"]
))

if HAS_XGBOOST:
    ModelLibrary.register(ModelConfig(
        name="xgboost_regressor",
        display_name="XGBoost Regressor",
        estimator_class=xgb.XGBRegressor,
        task=MLTask.REGRESSION,
        default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True}
        },
        supports_feature_importance=True,
        handles_missing=True,
        tags=["ensemble", "fast", "accurate"]
    ))


# ============================================================================
# Training Results
# ============================================================================

def _round_or_none(value: Any, ndigits: int = 4) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, ndigits)


@dataclass
class ModelTrainingResult:
    """Comprehensive result of model training."""

    model_id: UUID = field(default_factory=uuid4)
    model_name: str = ""
    display_name: str = ""
    task: MLTask = MLTask.BINARY_CLASSIFICATION
    status: ModelStatus = ModelStatus.COMPLETED
    is_baseline: bool = False

    # Data info
    n_samples_train: int = 0
    n_samples_test: int = 0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    target_column: str = ""

    # Metrics. cv_mean/cv_std are None when cross-validation could not be computed.
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    cv_scoring: str = ""

    # Feature importance (permutation importance on the held-out split, in
    # terms of the original input columns)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Hyperparameters
    best_params: Dict[str, Any] = field(default_factory=dict)
    optimization_trials: int = 0

    # Preprocessing info
    preprocessing_summary: Dict[str, Any] = field(default_factory=dict)

    # Timing
    preprocessing_time_sec: float = 0.0
    training_time_sec: float = 0.0
    optimization_time_sec: float = 0.0
    total_time_sec: float = 0.0

    # Errors/Warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "model_id": str(self.model_id),
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task": self.task.value,
            "status": self.status.value,
            "is_baseline": self.is_baseline,
            "data": {
                "train_samples": self.n_samples_train,
                "test_samples": self.n_samples_test,
                "n_features": self.n_features,
                "target": self.target_column
            },
            "metrics": {
                "train": {k: _round_or_none(v) for k, v in self.train_metrics.items()},
                "test": {k: _round_or_none(v) for k, v in self.test_metrics.items()},
                "cv_mean": _round_or_none(self.cv_mean),
                "cv_std": _round_or_none(self.cv_std),
                "cv_scoring": self.cv_scoring,
            },
            "feature_importance": {
                k: _round_or_none(v) for k, v in
                sorted(self.feature_importance.items(), key=lambda x: -x[1])[:20]
            },
            "best_params": self.best_params,
            "timing": {
                "preprocessing_sec": round(self.preprocessing_time_sec, 2),
                "training_sec": round(self.training_time_sec, 2),
                "optimization_sec": round(self.optimization_time_sec, 2),
                "total_sec": round(self.total_time_sec, 2)
            },
            "warnings": self.warnings[:10],
            "errors": self.errors[:5],
            "created_at": self.created_at.isoformat()
        }


# ============================================================================
# Advanced AutoML Engine
# ============================================================================

_CLASSIFICATION_TASKS = (MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION)
_BASELINE_MARGIN = 0.01


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class _FixedProba:
    """Stands in for a model whose probabilities were already computed (calibrated, thresholded)."""

    def __init__(self, proba: np.ndarray):
        self._proba = proba

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._proba


class AdvancedAutoMLEngine:
    """
    AutoML engine whose reported scores mean what they say.

    - All preprocessing (type handling, imputation, scaling, encoding, date
      features) is the first step of a scikit-learn Pipeline, so it is refitted
      on every training fold and never sees held-out rows.
    - Rows with a missing target are dropped, never imputed.
    - Columns that copy or deterministically encode the target are excluded
      (and reported) before training.
    - Model selection uses cross-validation on the training split only; the
      held-out split is scored once, after selection.
    - Every run includes a trivial baseline (DummyClassifier/DummyRegressor) and
      reports whether the chosen model beats it.
    - With ``time_column`` the held-out split is the latest period and
      cross-validation uses forward-chaining folds, so no future rows are used
      to evaluate past ones.
    - ``build_bundle`` exports the fitted pipeline as a ModelBundle that scores
      raw rows in a fresh process.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN,
        optimization_trials: int = 50,
        optimization_timeout: int = 300,  # seconds per model
        scoring: Optional[str] = None,  # auto-select if None
        n_jobs: int = 1,
        verbose: bool = True,
        max_onehot_categories: int = 20,
        exclude_leaky_columns: bool = True,
        balance_class_weights: bool = False,
        test_start: Optional[str] = None,
    ):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.optimization_strategy = optimization_strategy
        self.optimization_trials = optimization_trials
        self.optimization_timeout = optimization_timeout
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_onehot_categories = max_onehot_categories
        self.exclude_leaky_columns = exclude_leaky_columns
        # Off by default: reweighting distorts probabilities, while a tuned threshold on calibrated
        # probabilities reaches the same recall and keeps the probabilities meaningful.
        self.balance_class_weights = balance_class_weights
        # The prediction moment for a backtest: train on rows before this date, test on the rest.
        self.test_start = test_start
        self._class_balance: Optional[Dict[str, Any]] = None
        self._calibrator: Any = None
        self._decision_threshold: Optional[Dict[str, Any]] = None
        self._reference_profile: Optional[Dict[str, Any]] = None
        self._prediction_reference: Optional[Dict[str, Any]] = None

        # State
        self._trained_models: Dict[str, Tuple[Optional[Pipeline], ModelTrainingResult]] = {}
        self._best_model_name: Optional[str] = None
        self._task: Optional[MLTask] = None
        self._classes: Optional[List[Any]] = None
        self._target_column: str = ""
        self._input_columns: List[str] = []
        self._baseline: Optional[Tuple[Pipeline, ModelTrainingResult]] = None
        self.training_report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def auto_train(
        self,
        df: pd.DataFrame,
        target_column: str,
        task: Optional[MLTask] = None,
        models: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        optimize: bool = True,
        max_models: int = 5,
        time_column: Optional[str] = None,
    ) -> List[ModelTrainingResult]:
        """Train, cross-validate and compare models; return results best-first.

        The baseline is not part of the returned list; see ``training_report``.
        """
        total_start = datetime.utcnow()
        context = LogContext(component="AdvancedAutoML", operation="auto_train")
        run_warnings: List[str] = []

        if df is None or df.empty:
            raise MLException("Empty dataframe provided")
        if target_column not in df.columns:
            raise MLException(f"Target column '{target_column}' not found in dataframe")
        if time_column is not None and time_column not in df.columns:
            raise MLException(f"Time column '{time_column}' not found in dataframe")
        if time_column == target_column:
            raise MLException("The time column cannot also be the target")
        df, target_column, time_column, exclude_columns, name_notes = self._normalise_frame(
            df, target_column, time_column, exclude_columns
        )
        run_warnings.extend(name_notes)
        self._target_column = target_column
        self._class_balance = None
        self._calibrator = None
        self._decision_threshold = None
        self._reference_profile = None
        self._prediction_reference = None

        # Step 1: understand the target values, then never impute them.
        target_series, target_notes = self._clean_target(df[target_column], task)
        run_warnings.extend(target_notes)
        df = df.assign(**{target_column: target_series})
        known = df[target_column].notna()
        n_missing_target = int((~known).sum())
        data = df.loc[known]
        if n_missing_target:
            run_warnings.append(
                f"Dropped {n_missing_target:,} rows with no value for '{target_column}'; "
                "the target is never imputed."
            )

        # Step 2: settle the task.
        if task is None:
            task = self._detect_task(data[target_column])
        task = self._normalise_task(task, data[target_column])
        self._task = task
        is_classification = task in _CLASSIFICATION_TASKS

        # Step 3: time ordering (drop rows whose time is unknown).
        time_values: Optional[pd.Series] = None
        if time_column is not None:
            from app.ml.tabular_pipeline import _to_datetime

            time_values = _to_datetime(data[time_column])
            unknown_time = int(time_values.isna().sum())
            if unknown_time:
                run_warnings.append(
                    f"Dropped {unknown_time:,} rows with no usable '{time_column}' value for the time-ordered split."
                )
                keep = time_values.notna()
                data = data.loc[keep]
                time_values = time_values.loc[keep]

        min_rows = 20
        if len(data) < min_rows:
            raise MLException(
                f"Need at least {min_rows} rows with a known '{target_column}' to train and evaluate a model; "
                f"found {len(data):,}."
            )

        # Step 4: features, minus anything that leaks the target.
        exclude = set(exclude_columns or []) | {target_column}
        feature_cols = [c for c in data.columns if c not in exclude]
        leakage = self._screen_leakage(data[feature_cols], data[target_column], task)
        if time_column is not None:
            leakage.extend(
                self._probe_post_event_dates(
                    data, feature_cols, target_column, time_column, task,
                    already={i["column"] for i in leakage},
                )
            )
        flagged = [i for i in leakage if i.get("action") == "flag"]
        leakage = [i for i in leakage if i.get("action") != "flag"]
        if flagged:
            run_warnings.append(
                "Kept, but check these are known at prediction time: "
                + "; ".join(f"{i['column']} ({i['reason']})" for i in flagged)
            )
        if leakage:
            names = [item["column"] for item in leakage]
            if self.exclude_leaky_columns:
                feature_cols = [c for c in feature_cols if c not in set(names)]
                run_warnings.append(
                    "Excluded columns that give away the target: "
                    + "; ".join(f"{i['column']} ({i['reason']})" for i in leakage)
                )
            else:
                run_warnings.append(
                    "Possible target leakage (kept because exclusion is disabled): "
                    + "; ".join(f"{i['column']} ({i['reason']})" for i in leakage)
                )
        if not feature_cols:
            raise MLException("No feature columns remain after removing the target and excluded columns")
        self._input_columns = [str(c) for c in feature_cols]

        X = data[feature_cols].copy()
        X.columns = self._input_columns
        y = self._encode_target(data[target_column], task)

        # Step 5: split.
        split_info: Dict[str, Any]
        if time_values is not None:
            order = np.argsort(time_values.to_numpy(), kind="stable")
            X = X.iloc[order]
            y = y[order]
            ordered_time = time_values.iloc[order]
            n_test = max(1, int(round(len(X) * self.test_size)))
            cut = len(X) - n_test
            if self.test_start:
                boundary = pd.Timestamp(self.test_start)
                cut = int(np.searchsorted(ordered_time.to_numpy(), np.datetime64(boundary), side="left"))
                if cut < 20 or cut >= len(X):
                    raise MLException(
                        f"Testing from {boundary.date()} leaves {cut:,} training rows and {len(X) - cut:,} test rows; "
                        "choose a date with at least 20 rows before it and some after it."
                    )
            X_train, X_test = X.iloc[:cut], X.iloc[cut:]
            y_train, y_test = y[:cut], y[cut:]
            split_info = {
                "strategy": "time_ordered",
                "time_column": time_column,
                "test_start": str(pd.Timestamp(self.test_start).date()) if self.test_start else None,
                "train_period": [str(ordered_time.iloc[0]), str(ordered_time.iloc[cut - 1])],
                "test_period": [str(ordered_time.iloc[cut]), str(ordered_time.iloc[-1])],
            }
            if is_classification and len(np.unique(y_test)) < 2:
                run_warnings.append(
                    "The most recent period contains only one class, so ranking metrics such as ROC AUC "
                    "cannot be computed on the held-out split."
                )
        else:
            stratify = None
            if is_classification and np.bincount(y).min() >= 2:
                stratify = y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
            )
            split_info = {"strategy": "stratified_random" if stratify is not None else "random"}
        split_info.update({"train_rows": int(len(X_train)), "test_rows": int(len(X_test))})

        if is_classification and len(np.unique(y_train)) < 2:
            raise MLException("The training split contains a single class; there is nothing to learn")

        cv = self._get_cv_strategy(task, y_train, time_ordered=time_values is not None)
        scoring_reason = None
        scoring = self._get_scoring(task)
        if is_classification:
            from app.ml.evaluation import class_balance, selection_metric

            self._class_balance = class_balance(y_train, len(self._classes or []))
            if not self.scoring:
                scoring, scoring_reason = selection_metric(
                    len(self._classes or []), self._class_balance["imbalanced"]
                )
                if self._class_balance["imbalanced"]:
                    run_warnings.append(scoring_reason)

        # Step 6: baseline.
        baseline_pipe, baseline_result = self._train_single_model(
            model_name="baseline",
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            task=task, cv=cv, scoring=scoring, optimize=False, is_baseline=True,
        )
        self._baseline = (baseline_pipe, baseline_result)

        # Step 7: candidate models.
        if models is None:
            models = self._select_models(len(X_train), task, max_models)
        results: List[ModelTrainingResult] = []
        for model_name in models:
            try:
                if self.verbose:
                    logger.info(f"Training {model_name}...", context=context)
                pipe, result = self._train_single_model(
                    model_name=model_name,
                    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                    task=task, cv=cv, scoring=scoring, optimize=optimize, is_baseline=False,
                )
                self._trained_models[model_name] = (pipe, result)
                results.append(result)
            except Exception as e:
                logger.error(f"Training {model_name} failed: {e}", context=context)
                failed = ModelTrainingResult(
                    model_name=model_name,
                    task=task,
                    status=ModelStatus.FAILED,
                    target_column=target_column,
                    errors=[str(e)],
                )
                self._trained_models[model_name] = (None, failed)
                results.append(failed)

        # Step 8: select by cross-validation only.
        completed = [r for r in results if r.status == ModelStatus.COMPLETED]
        scored = sorted((r for r in completed if r.cv_mean is not None), key=lambda r: r.cv_mean, reverse=True)
        unscored = [r for r in completed if r.cv_mean is None]
        failed = [r for r in results if r.status == ModelStatus.FAILED]
        if scored:
            best = scored[0]
        elif unscored:
            best = unscored[0]
            run_warnings.append(
                "Cross-validation failed for every model, so the chosen model has not been validated; "
                "treat its scores with caution."
            )
        else:
            details = "; ".join(f"{r.model_name}: {r.errors[0] if r.errors else 'unknown error'}" for r in failed)
            raise MLException(f"Every model failed to train ({details})")
        self._best_model_name = best.model_name

        # Step 9: explain the chosen model on held-out rows.
        best_pipe = self._trained_models[best.model_name][0]
        importance, importance_note = self._permutation_importance(best_pipe, X_test, y_test, scoring)
        best.feature_importance = importance
        if importance_note:
            run_warnings.append(importance_note)

        # Step 9b: calibration, decision threshold, error by segment, drift reference.
        evaluation = self._evaluate_best(best, best_pipe, X_train, y_train, X_test, y_test, cv, task, run_warnings)

        # Step 10: compare with the baseline, fold by fold.
        beats_baseline: Optional[bool] = None
        improvement = self._paired_improvement(best, baseline_result)
        if improvement is not None:
            beats_baseline = bool(
                improvement["mean"] > _BASELINE_MARGIN and improvement["ci95_low"] > 0.0
            )
            if not beats_baseline:
                run_warnings.append(
                    f"The best model ({best.display_name}, CV {scoring} {best.cv_mean:.4f}) does not beat the "
                    f"naive baseline (CV {scoring} {baseline_result.cv_mean:.4f}) by a meaningful margin. "
                    f"The available columns may carry little signal about '{target_column}'."
                )

        prep_summary = best.preprocessing_summary
        total_time = (datetime.utcnow() - total_start).total_seconds()
        for r in results:
            r.total_time_sec = total_time
        best.warnings = list(dict.fromkeys(run_warnings + best.warnings))

        self.training_report = {
            "target_column": target_column,
            "task": task.value,
            "classes": list(self._classes) if self._classes is not None else None,
            "rows_used": int(len(X)),
            "rows_dropped_missing_target": n_missing_target,
            "split": split_info,
            "cross_validation": {
                "strategy": type(cv).__name__,
                "folds": int(cv.get_n_splits()),
                "scoring": scoring,
                "selection": "highest mean cross-validation score on the training split",
            },
            "baseline": {
                "model": baseline_result.display_name,
                "cv_mean": _round_or_none(baseline_result.cv_mean),
                "test_metrics": {k: _round_or_none(v) for k, v in baseline_result.test_metrics.items()},
            },
            "best_model": best.model_name,
            "best_cv_mean": _round_or_none(best.cv_mean),
            "beats_baseline": beats_baseline,
            "improvement_over_baseline": improvement,
            "excluded_leaky_columns": leakage if self.exclude_leaky_columns else [],
            "preprocessing": prep_summary,
            "importance_method": "permutation importance on the held-out split",
            "class_balance": self._class_balance,
            "selection_metric_reason": scoring_reason,
            **evaluation,
            "warnings": run_warnings,
        }
        if self.verbose:
            logger.info(f"Best model: {best.model_name}", context=context)
        return scored + unscored + failed

    def get_best_model(self) -> Tuple[str, Pipeline, ModelTrainingResult]:
        """Get the best performing model as (name, fitted pipeline, result)."""
        if self._best_model_name is None:
            raise MLException("No models have been trained")
        pipe, result = self._trained_models[self._best_model_name]
        return self._best_model_name, pipe, result

    def build_bundle(self, model_name: Optional[str] = None):
        """Package a trained pipeline with everything needed to score raw rows."""
        from app.ml.tabular_pipeline import ModelBundle

        name = model_name or self._best_model_name
        if name is None or name not in self._trained_models or self._trained_models[name][0] is None:
            raise MLException("No trained model available to package")
        pipe, result = self._trained_models[name]
        is_best = name == self._best_model_name
        info = dict(self.training_report)
        info.update(
            {
                "model_cv_mean": _round_or_none(result.cv_mean),
                "model_test_metrics": {k: _round_or_none(v) for k, v in result.test_metrics.items()},
                "model_feature_importance": {k: _round_or_none(v, 6) for k, v in result.feature_importance.items()},
                "best_params": result.best_params,
            }
        )
        return ModelBundle(
            pipeline=pipe,
            task=self._task.value if self._task else result.task.value,
            target_column=self._target_column,
            model_name=name,
            input_columns=list(self._input_columns),
            classes=list(self._classes) if self._classes is not None else None,
            used_columns=list(pipe.named_steps["prep"].column_roles_.keys()),
            training_info=info,
            calibrator=self._calibrator if is_best else None,
            decision_threshold=self._decision_threshold if is_best else None,
            reference_profile=self._reference_profile,
            prediction_reference=self._prediction_reference if is_best else None,
        )

    def _evaluate_best(
        self,
        best: "ModelTrainingResult",
        pipe: Pipeline,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        cv: Any,
        task: MLTask,
        run_warnings: List[str],
    ) -> Dict[str, Any]:
        """Decisions on out-of-fold training predictions; the held-out split is only scored."""
        from app.ml import drift, evaluation as ev

        out: Dict[str, Any] = {"calibration": None, "decision_threshold": None, "error_by_segment": None}
        is_classification = task in _CLASSIFICATION_TASKS
        n_classes = len(self._classes or [])
        y_pred_test = pipe.predict(X_test)

        if is_classification and hasattr(pipe, "predict_proba"):
            try:
                raw_test = np.asarray(pipe.predict_proba(X_test), dtype=float)
                if n_classes == 2:
                    pos = int((self._class_balance or {}).get("minority_class_index", 1))
                    oof, mask = ev.out_of_fold_proba(pipe, X_train, y_train, cv)
                    p_oof = oof[mask, pos]
                    y_oof = (y_train[mask] == pos).astype(int)
                    y_pos_test = (y_test == pos).astype(int)
                    calibrator, cal_info = ev.choose_calibration(p_oof, y_oof, self.random_state)
                    p_test = raw_test[:, pos]
                    cal_info["held_out_before"] = ev.reliability(y_pos_test, p_test)
                    if calibrator is not None:
                        p_oof = calibrator(p_oof)
                        p_test = calibrator(p_test)
                        cal_info["held_out_after"] = ev.reliability(y_pos_test, p_test)
                        self._calibrator = {"positive_index": pos, "map": calibrator}
                    cal_info["positive_class"] = str(self._classes[pos])
                    out["calibration"] = cal_info
                    if (self._class_balance or {}).get("imbalanced") and y_oof.sum() >= 5:
                        thr = ev.choose_threshold(y_oof, p_oof)
                        thr["positive_class"] = str(self._classes[pos])
                        thr["held_out_at_default_0_5"] = ev.scores_at_threshold(y_pos_test, p_test, 0.5)
                        thr["held_out_at_chosen"] = ev.scores_at_threshold(y_pos_test, p_test, thr["threshold"])
                        self._decision_threshold = {"positive_index": pos, "threshold": thr["threshold"]}
                        out["decision_threshold"] = thr
                        y_pred_test = np.where(p_test >= thr["threshold"], pos, 1 - pos)
                        proba_view = raw_test.copy()
                        proba_view[:, pos] = p_test
                        proba_view[:, 1 - pos] = 1 - p_test
                        best.test_metrics = self._classification_metrics(
                            y_test, y_pred_test, _FixedProba(proba_view), X_test, n_classes
                        )
                        run_warnings.append(
                            f"Predictions use a cut-off of {thr['threshold']:.3f} on the probability of "
                            f"'{thr['positive_class']}' (chosen on training folds); at 0.5 the model would "
                            f"find {thr['held_out_at_default_0_5']['recall']:.0%} of them on held-out rows, "
                            f"at the chosen cut-off {thr['held_out_at_chosen']['recall']:.0%}."
                        )
                else:
                    confident = raw_test.max(axis=1)
                    out["calibration"] = {
                        "applied": False, "method": None,
                        "reason": "calibration maps are learned for two-class targets only; shown: confidence of the predicted class",
                        "held_out_before": ev.reliability((raw_test.argmax(axis=1) == y_test).astype(int), confident),
                    }
            except Exception as e:  # never lose a trained model over a report
                run_warnings.append(f"Calibration and threshold checks were skipped ({e}).")

        try:
            out["error_by_segment"] = ev.error_by_segment(
                X_test.reset_index(drop=True),
                np.asarray(y_test),
                np.asarray(y_pred_test),
                task="regression" if not is_classification else "classification",
            )
            flagged = [s for s in out["error_by_segment"]["worst_segments"] if s["significant"]]
            if flagged:
                s = flagged[0]
                metric = out["error_by_segment"]["metric"].replace("_", " ")
                run_warnings.append(
                    f"The model is clearly worse for {s['column']} = {s['segment']} ({s['rows']} held-out rows, "
                    f"{metric} {s[out['error_by_segment']['metric']]} vs {s['overall']} overall)."
                )
        except Exception as e:
            run_warnings.append(f"Error-by-segment analysis was skipped ({e}).")

        try:
            used = list(pipe.named_steps["prep"].column_roles_.keys())
            self._reference_profile = drift.build_profile(X_train, used)
            if is_classification:
                labels = pd.Series([str(self._classes[int(i)]) for i in y_pred_test]).value_counts(normalize=True)
                self._prediction_reference = {"class_shares": {str(k): round(float(v), 4) for k, v in labels.items()}}
            else:
                vals = np.asarray(y_pred_test, dtype=float)
                self._prediction_reference = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}
        except Exception as e:
            run_warnings.append(f"No drift reference was stored ({e}); scoring new files will skip the drift check.")
        return out

    def predict(self, df: pd.DataFrame, model_name: Optional[str] = None) -> List[Any]:
        """Predict raw rows with a trained model (labels decoded for classification)."""
        return self.build_bundle(model_name).predict(df)

    @staticmethod
    def _paired_improvement(
        best: ModelTrainingResult, baseline: ModelTrainingResult
    ) -> Optional[Dict[str, Any]]:
        """Fold-by-fold gain over the baseline with a 95% t-interval.

        Both models are scored on identical folds, so the per-fold difference is
        the fair comparison; a gain whose interval includes zero is not counted
        as beating the baseline.
        """
        a = np.asarray(best.cv_scores.get("test") or [], dtype=float)
        b = np.asarray(baseline.cv_scores.get("test") or [], dtype=float)
        if a.size == 0 or a.size != b.size:
            return None
        diff = a - b
        diff = diff[np.isfinite(diff)]
        if diff.size < 2:
            return None
        mean = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(diff.size))
        from scipy import stats

        t_crit = float(stats.t.ppf(0.975, df=diff.size - 1))
        return {
            "metric": best.cv_scoring,
            "mean": round(mean, 4),
            "ci95_low": round(mean - t_crit * se, 4),
            "ci95_high": round(mean + t_crit * se, 4),
            "folds": int(diff.size),
        }

    # ------------------------------------------------------------------
    # Task and target handling
    # ------------------------------------------------------------------
    def _detect_task(self, target: pd.Series) -> MLTask:
        """Automatically detect the ML task from target variable."""
        target_clean = target.dropna()
        if len(target_clean) == 0:
            raise MLException("Target column has no valid values")
        if pd.api.types.is_bool_dtype(target_clean):
            return MLTask.BINARY_CLASSIFICATION
        if pd.api.types.is_numeric_dtype(target_clean):
            values = target_clean.astype(float)
            n_unique = int(values.nunique())
            if n_unique == 2:
                return MLTask.BINARY_CLASSIFICATION
            is_integer = bool(np.all(np.mod(values, 1) == 0))
            # Small sets of non-negative, near-consecutive integers are codes or
            # ratings (0-4, 1-5); spread-out integers (-500, -400 ... 500) are amounts.
            if is_integer and n_unique <= 15 and values.min() >= 0:
                if (values.max() - values.min() + 1) <= 1.5 * n_unique:
                    return MLTask.MULTICLASS_CLASSIFICATION
            return MLTask.REGRESSION
        n_unique = target_clean.nunique()
        if n_unique == 2:
            return MLTask.BINARY_CLASSIFICATION
        return MLTask.MULTICLASS_CLASSIFICATION

    @staticmethod
    def _normalise_frame(
        df: pd.DataFrame,
        target_column: str,
        time_column: Optional[str],
        exclude_columns: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, str, Optional[str], Optional[List[str]], List[str]]:
        """String, unique column names and flattened nested cells."""
        from app.ml.tabular_pipeline import dedupe_column_names, normalise_cells

        notes: List[str] = []
        original = list(df.columns)
        names = dedupe_column_names(original)
        renamed = [(str(o), n) for o, n in zip(original, names) if str(o) != n]
        if renamed:
            notes.append(
                "Renamed duplicate or blank column names: "
                + ", ".join(f"'{o}' -> '{n}'" for o, n in renamed[:10])
            )
        first_name = {}
        for o, n in zip(original, names):
            first_name.setdefault(o, n)
        out = df.copy()
        out.columns = names
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = normalise_cells(out[col])

        def mapped(c: Optional[str]) -> Optional[str]:
            return None if c is None else first_name.get(c, str(c))

        excl = [mapped(c) for c in exclude_columns] if exclude_columns else exclude_columns
        return out, mapped(target_column), mapped(time_column), excl, notes

    def _clean_target(self, target: pd.Series, task: Optional[MLTask]) -> Tuple[pd.Series, List[str]]:
        """Read numbers stored as text and merge label spellings that differ only in case/spacing."""
        from app.ml.tabular_pipeline import _clean_numeric_text

        notes: List[str] = []
        s = target
        if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype("object")
            text = s.where(s.notna(), None)
            stripped = text.map(lambda v: v.strip() if isinstance(v, str) else v)
            stripped = stripped.where(stripped != "", None)
            non_null = stripped.dropna()
            if task in (None, MLTask.REGRESSION) and len(non_null):
                as_numbers = _clean_numeric_text(non_null.astype(str))
                if float(as_numbers.notna().mean()) >= 0.98 and as_numbers.nunique() > 2:
                    numbers = _clean_numeric_text(stripped.astype("string"))
                    notes.append(
                        f"Read '{target.name}' as numbers (removed currency symbols, separators and % signs)."
                    )
                    return numbers, notes
            folded = non_null.astype(str).str.casefold()
            if folded.nunique() < non_null.astype(str).nunique():
                canonical = (
                    pd.DataFrame({"raw": non_null.astype(str), "key": folded})
                    .groupby("key")["raw"]
                    .agg(lambda v: v.value_counts().index[0])
                )
                merged = stripped.map(lambda v: canonical[str(v).casefold()] if isinstance(v, str) else v)
                groups = (
                    pd.DataFrame({"raw": target.dropna().astype(str), "key": target.dropna().astype(str).str.strip().str.casefold()})
                    .groupby("key")["raw"]
                    .agg(lambda v: sorted(set(v)))
                )
                variants = [f"{canonical[k]} <- {g}" for k, g in groups.items() if len(g) > 1]
                notes.append(
                    f"Merged '{target.name}' labels that differ only in case or spacing: " + "; ".join(variants[:5])
                )
                return merged, notes
            if not stripped.equals(text):
                return stripped, notes
            return text, notes
        if pd.api.types.is_numeric_dtype(s):
            values = s.astype(float)
            n_inf = int(np.isinf(values).sum())
            if n_inf:
                notes.append(f"Treated {n_inf:,} infinite target values as missing.")
                return s.where(np.isfinite(values), np.nan), notes
        return s, notes

    def _probe_post_event_dates(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_column: str,
        time_column: str,
        task: MLTask,
        already: set,
    ) -> List[Dict[str, str]]:
        """Exclude date columns recorded after the prediction moment that reveal the outcome.

        For each other date column D whose values fall after the time column T, the
        gap D - T is the kind of quantity that is only known once the outcome has
        happened (delivery took 9 days, so it was late). If that gap alone predicts
        the target almost perfectly in cross-validation, D is excluded.
        """
        from app.ml.tabular_pipeline import parse_datetime
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        findings: List[Dict[str, str]] = []
        t0 = parse_datetime(data[time_column])
        y_raw = data[target_column]
        for col in feature_cols:
            if col in already or col == time_column:
                continue
            s = data[col]
            if not (pd.api.types.is_datetime64_any_dtype(s) or s.dtype == object):
                continue
            parsed = parse_datetime(s) if not pd.api.types.is_datetime64_any_dtype(s) else s
            valid = parsed.notna() & t0.notna()
            if valid.mean() < 0.6 or int(valid.sum()) < 30:
                continue
            gap = (parsed[valid] - t0[valid]).dt.total_seconds() / 86400.0
            if float((gap >= 0).mean()) < 0.9:
                continue  # not recorded after the time column
            y = y_raw[valid]
            Xg = gap.to_numpy().reshape(-1, 1)
            try:
                if task in _CLASSIFICATION_TASKS:
                    yc = pd.factorize(y.astype(str))[0]
                    if np.bincount(yc).min() < 3:
                        continue
                    scores = cross_val_score(
                        DecisionTreeClassifier(max_depth=3, random_state=self.random_state), Xg, yc,
                        cv=StratifiedKFold(3, shuffle=True, random_state=self.random_state),
                        scoring="balanced_accuracy",
                    )
                else:
                    scores = cross_val_score(
                        DecisionTreeRegressor(max_depth=3, random_state=self.random_state), Xg,
                        pd.to_numeric(y, errors="coerce").to_numpy(),
                        cv=KFold(3, shuffle=True, random_state=self.random_state), scoring="r2",
                    )
            except Exception:
                continue
            score = float(np.nanmean(scores))
            if score >= 0.95:
                findings.append(
                    {
                        "column": str(col),
                        "reason": (
                            f"recorded after '{time_column}', and the gap between them alone predicts the target "
                            f"(cross-validated score {score:.2f}), so it is only known once the outcome has happened"
                        ),
                    }
                )
        return findings

    def _normalise_task(self, task: MLTask, target: pd.Series) -> MLTask:
        if task in _CLASSIFICATION_TASKS:
            n_classes = int(target.nunique(dropna=True))
            if n_classes < 2:
                raise MLException("The target has a single class; there is nothing to predict")
            return MLTask.BINARY_CLASSIFICATION if n_classes == 2 else MLTask.MULTICLASS_CLASSIFICATION
        if task == MLTask.REGRESSION:
            numeric = pd.to_numeric(target, errors="coerce")
            bad = int(numeric.isna().sum())
            if bad:
                raise MLException(
                    f"Regression needs a numeric target; {bad:,} values of '{target.name}' are not numbers"
                )
            return task
        raise MLException(f"Task '{task.value}' is not supported by the AutoML engine")

    def _encode_target(self, target: pd.Series, task: MLTask) -> np.ndarray:
        if task not in _CLASSIFICATION_TASKS:
            self._classes = None
            return pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        values = target
        if not (pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)):
            values = values.astype(str)
        classes = [_python_scalar(v) for v in sorted(pd.unique(values))]
        mapping = {c: i for i, c in enumerate(classes)}
        self._classes = classes
        return values.map(lambda v: mapping[_python_scalar(v)]).to_numpy(dtype=int)

    def _screen_leakage(self, X: pd.DataFrame, target: pd.Series, task: MLTask) -> List[Dict[str, str]]:
        """Find columns that copy or deterministically encode the target."""
        findings: List[Dict[str, str]] = []
        target_text = target.astype(str)
        n_target_classes = int(target.nunique(dropna=True))
        for col in X.columns:
            s = X[col]
            mask = s.notna()
            if int(mask.sum()) < 20:
                continue
            s_known, t_known = s[mask], target[mask]
            try:
                if float((s_known.astype(str) == target_text[mask]).mean()) >= 0.995:
                    findings.append({"column": str(col), "reason": "identical to the target"})
                    continue
                if task in _CLASSIFICATION_TASKS and n_target_classes >= 2:
                    n_values = int(s_known.nunique())
                    if 2 <= n_values <= max(20, int(0.05 * len(s_known))):
                        purity = (
                            pd.crosstab(s_known.astype(str), t_known.astype(str)).max(axis=1).sum()
                            / float(len(s_known))
                        )
                        if purity >= 0.995:
                            findings.append(
                                {"column": str(col), "reason": "each of its values maps to exactly one target class"}
                            )
                            continue
                if task == MLTask.REGRESSION and pd.api.types.is_numeric_dtype(s_known):
                    corr = float(pd.to_numeric(t_known, errors="coerce").corr(s_known.astype(float)))
                    if np.isfinite(corr) and abs(corr) >= 0.9999:
                        # An exact linear rescaling (unit or currency conversion, a copy).
                        findings.append(
                            {"column": str(col), "reason": f"it is a linear rescaling of the target (r = {corr:.5f})"}
                        )
                    elif np.isfinite(corr) and abs(corr) >= 0.99:
                        findings.append(
                            {
                                "column": str(col),
                                "reason": f"correlation with the target is {corr:.3f}; confirm it is known before the outcome",
                                "action": "flag",
                            }
                        )
            except Exception:
                continue
        return findings

    # ------------------------------------------------------------------
    # Model selection and training
    # ------------------------------------------------------------------
    def _select_models(self, n_samples: int, task: MLTask, max_models: int) -> List[str]:
        """Pick candidate models suited to the task and data size."""
        available = ModelLibrary.get_for_task(task, max_count=20)
        if task in _CLASSIFICATION_TASKS:
            if n_samples > 100_000:
                preferred = ["lightgbm_classifier", "hist_gradient_boosting_classifier", "xgboost_classifier",
                             "logistic_regression"]
            elif n_samples < 1000:
                preferred = ["logistic_regression", "random_forest_classifier",
                             "hist_gradient_boosting_classifier", "gradient_boosting_classifier"]
            else:
                preferred = ["hist_gradient_boosting_classifier", "logistic_regression",
                             "random_forest_classifier", "xgboost_classifier", "lightgbm_classifier"]
        else:
            if n_samples > 100_000:
                preferred = ["hist_gradient_boosting_regressor", "xgboost_regressor", "ridge_regression"]
            elif n_samples < 1000:
                preferred = ["ridge_regression", "random_forest_regressor",
                             "hist_gradient_boosting_regressor", "elastic_net"]
            else:
                preferred = ["hist_gradient_boosting_regressor", "ridge_regression",
                             "random_forest_regressor", "xgboost_regressor"]
        selected = [m for m in preferred if m in available]
        for m in available:
            if m not in selected:
                selected.append(m)
        return selected[: max(1, int(max_models))]

    def _make_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        from app.ml.tabular_pipeline import TabularPreprocessor

        return Pipeline(
            [
                ("prep", TabularPreprocessor(max_onehot_categories=self.max_onehot_categories)),
                ("model", estimator),
            ]
        )

    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        task: MLTask,
        cv: Any,
        scoring: str,
        optimize: bool,
        is_baseline: bool,
    ) -> Tuple[Pipeline, ModelTrainingResult]:
        """Cross-validate, fit and evaluate one model inside a leakage-safe pipeline."""
        train_start = datetime.utcnow()
        is_classification = task in _CLASSIFICATION_TASKS
        if is_baseline:
            if is_classification:
                estimator = DummyClassifier(strategy="prior")
                display = "Baseline (always predicts class frequencies)"
            else:
                estimator = DummyRegressor(strategy="mean")
                display = "Baseline (always predicts the training mean)"
            config = None
        else:
            config = ModelLibrary.get(model_name)
            estimator = config.estimator_class(**config.default_params)
            display = config.display_name
            if (
                self.balance_class_weights
                and is_classification
                and (self._class_balance or {}).get("imbalanced")
                and "class_weight" in estimator.get_params()
            ):
                estimator.set_params(class_weight="balanced")
                display = f"{display} (balanced class weights)"

        pipe = self._make_pipeline(estimator)
        result = ModelTrainingResult(
            model_name=model_name,
            display_name=display,
            task=task,
            status=ModelStatus.TRAINING,
            is_baseline=is_baseline,
            n_samples_train=len(X_train),
            n_samples_test=len(X_test),
            target_column=self._target_column,
            cv_scoring=scoring,
        )

        # Hyperparameter search (inside the same CV folds, on training rows only).
        opt_start = datetime.utcnow()
        if (
            optimize
            and config is not None
            and config.search_space
            and self.optimization_strategy != OptimizationStrategy.NONE
        ):
            try:
                best_params = self._optimize_hyperparameters(pipe, config.search_space, X_train, y_train, cv, scoring)
                pipe.set_params(**{f"model__{k}": v for k, v in best_params.items()})
                result.best_params = best_params
                result.optimization_trials = self.optimization_trials
            except Exception as e:
                result.warnings.append(f"Hyperparameter search failed ({e}); default settings were used")
        result.optimization_time_sec = (datetime.utcnow() - opt_start).total_seconds()

        # Cross-validation on the training split.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_results = cross_validate(
                    pipe, X_train, y_train,
                    cv=cv, scoring=scoring, n_jobs=self.n_jobs,
                    return_train_score=True, error_score=np.nan,
                )
            fold_scores = np.asarray(cv_results["test_score"], dtype=float)
            train_scores = np.asarray(cv_results["train_score"], dtype=float)
            finite = fold_scores[np.isfinite(fold_scores)]
            result.cv_scores = {"test": fold_scores.tolist(), "train": train_scores.tolist()}
            if finite.size == 0:
                result.warnings.append(f"Cross-validation could not compute {scoring} on any fold")
            else:
                if finite.size < fold_scores.size:
                    result.warnings.append(
                        f"Cross-validation failed on {fold_scores.size - finite.size} of {fold_scores.size} folds; "
                        "the score averages the remaining folds"
                    )
                result.cv_mean = float(finite.mean())
                result.cv_std = float(finite.std())
        except Exception as e:
            result.warnings.append(f"Cross-validation failed: {e}")

        # Final fit on the whole training split, then a single held-out evaluation.
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        if is_classification:
            n_classes = len(self._classes or [])
            result.train_metrics = self._classification_metrics(y_train, y_train_pred, pipe, X_train, n_classes)
            result.test_metrics = self._classification_metrics(y_test, y_test_pred, pipe, X_test, n_classes)
        else:
            result.train_metrics = self._regression_metrics(y_train, y_train_pred)
            result.test_metrics = self._regression_metrics(y_test, y_test_pred)

        prep = pipe.named_steps["prep"]
        result.feature_names = [str(n) for n in prep.get_feature_names_out()]
        result.n_features = len(result.feature_names)
        result.preprocessing_summary = prep.summary()
        result.training_time_sec = (datetime.utcnow() - train_start).total_seconds()
        result.status = ModelStatus.COMPLETED
        return pipe, result

    def _optimize_hyperparameters(
        self,
        pipe: Pipeline,
        search_space: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray,
        cv: Any,
        scoring: str,
    ) -> Dict[str, Any]:
        """Search hyperparameters with the same CV folds used for model selection."""
        if self.optimization_strategy == OptimizationStrategy.BAYESIAN and HAS_OPTUNA:
            return self._optuna_optimize(pipe, search_space, X, y, cv, scoring)
        return self._random_search_optimize(pipe, search_space, X, y, cv, scoring)

    def _optuna_optimize(self, pipe, search_space, X, y, cv, scoring) -> Dict[str, Any]:
        def objective(trial):
            params = {}
            for param_name, spec in search_space.items():
                if spec["type"] == "int":
                    params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"])
                elif spec["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, spec["low"], spec["high"], log=spec.get("log", False)
                    )
                elif spec["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
            candidate = clone(pipe).set_params(**{f"model__{k}": v for k, v in params.items()})
            scores = cross_val_score(candidate, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            return float(np.mean(scores))

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.optimization_trials,
            timeout=self.optimization_timeout,
            show_progress_bar=False,
            n_jobs=1,
            catch=(Exception,),
        )
        return study.best_params  # raises if every trial failed

    def _random_search_optimize(self, pipe, search_space, X, y, cv, scoring) -> Dict[str, Any]:
        distributions = {}
        for param_name, spec in search_space.items():
            if spec["type"] == "int":
                values = list(range(int(spec["low"]), int(spec["high"]) + 1))
            elif spec["type"] == "float":
                if spec.get("log"):
                    values = np.geomspace(spec["low"], spec["high"], 20).tolist()
                else:
                    values = np.linspace(spec["low"], spec["high"], 20).tolist()
            else:
                values = list(spec["choices"])
            distributions[f"model__{param_name}"] = values
        search = RandomizedSearchCV(
            pipe,
            distributions,
            n_iter=max(1, min(self.optimization_trials, 20)),
            cv=cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            error_score=np.nan,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)
        if not np.isfinite(search.best_score_):
            raise MLException("every hyperparameter candidate failed")
        return {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}

    def _get_cv_strategy(self, task: MLTask, y: np.ndarray, time_ordered: bool = False):
        """Cross-validation that matches how the model will be used."""
        n = len(y)
        if time_ordered:
            return TimeSeriesSplit(n_splits=max(2, min(self.cv_folds, n // 10)))
        if task in _CLASSIFICATION_TASKS:
            min_class = int(np.bincount(y).min()) if n else 0
            if min_class >= 2:
                return StratifiedKFold(
                    n_splits=max(2, min(self.cv_folds, min_class)),
                    shuffle=True,
                    random_state=self.random_state,
                )
        return KFold(n_splits=max(2, min(self.cv_folds, n // 2)), shuffle=True, random_state=self.random_state)

    def _get_scoring(self, task: MLTask) -> str:
        """Get scoring metric for task."""
        if self.scoring:
            return self.scoring
        if task == MLTask.BINARY_CLASSIFICATION:
            return "roc_auc"
        if task == MLTask.MULTICLASS_CLASSIFICATION:
            return "f1_weighted"
        if task == MLTask.REGRESSION:
            return "r2"
        return "accuracy"

    # ------------------------------------------------------------------
    # Metrics and explanations
    # ------------------------------------------------------------------
    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        X: pd.DataFrame,
        n_classes: int,
    ) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        present = np.unique(y_true)
        if hasattr(model, "predict_proba") and len(present) >= 2:
            try:
                proba = np.asarray(model.predict_proba(X), dtype=float)
                labels = list(range(proba.shape[1]))
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
                    metrics["average_precision"] = float(average_precision_score(y_true, proba[:, 1]))
                elif len(present) == proba.shape[1]:
                    metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr"))
                metrics["log_loss"] = float(log_loss(y_true, np.clip(proba, 1e-15, 1.0), labels=labels))
            except Exception:
                pass
        if n_classes == 2 and len(present) >= 2:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        return metrics

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "median_ae": float(median_absolute_error(y_true, y_pred)),
            "max_error": float(max_error(y_true, y_pred)),
        }
        if len(y_true) >= 2 and float(np.var(y_true)) > 0:
            metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))
        if (y_true != 0).all():
            metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        return metrics

    def _permutation_importance(
        self, pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, scoring: str
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """Drop in the selection metric when each input column is shuffled."""
        if len(X_test) < 10:
            return {}, "Feature importance was skipped because the held-out split has fewer than 10 rows."
        max_rows = 2000 if X_test.shape[1] <= 50 else 800
        n_repeats = 5 if X_test.shape[1] <= 30 else 2
        if len(X_test) > max_rows:
            idx = np.random.default_rng(self.random_state).choice(len(X_test), max_rows, replace=False)
            X_eval, y_eval = X_test.iloc[np.sort(idx)], y_test[np.sort(idx)]
        else:
            X_eval, y_eval = X_test, y_test
        metric = scoring
        if self._task in _CLASSIFICATION_TASKS and len(np.unique(y_eval)) < 2:
            metric = "accuracy"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp = permutation_importance(
                    pipe, X_eval, y_eval, scoring=metric, n_repeats=n_repeats,
                    random_state=self.random_state, n_jobs=self.n_jobs,
                )
        except Exception as e:
            return {}, f"Feature importance could not be computed: {e}"
        scores = {str(c): float(v) for c, v in zip(X_eval.columns, imp.importances_mean)}
        return dict(sorted(scores.items(), key=lambda kv: -kv[1])), None

    def compare_models(self, results: List[ModelTrainingResult]) -> Dict[str, Any]:
        """Compare multiple trained models."""
        successful = [r for r in results if r.status == ModelStatus.COMPLETED]
        if not successful:
            return {"error": "No successful models to compare", "models": []}

        def fmt(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.4f}"

        comparison: Dict[str, Any] = {
            "n_models": len(successful),
            "best_model": successful[0].model_name,
            "summary": [
                {
                    "name": r.model_name,
                    "display_name": r.display_name,
                    "cv_mean": r.cv_mean,
                    "cv_std": r.cv_std,
                    "test_metrics": {k: _round_or_none(v) for k, v in r.test_metrics.items()},
                    "training_time_sec": round(r.training_time_sec, 2),
                }
                for r in successful
            ],
            "baseline": self.training_report.get("baseline"),
            "beats_baseline": self.training_report.get("beats_baseline"),
        }
        best = successful[0]
        if len(successful) > 1:
            comparison["recommendation"] = (
                f"Best: {best.display_name} (CV={fmt(best.cv_mean)}). "
                f"Consider {successful[1].display_name} as alternative."
            )
        else:
            comparison["recommendation"] = f"Best: {best.display_name} (CV={fmt(best.cv_mean)})"
        return comparison


# ============================================================================
# Factory Functions
# ============================================================================

def get_auto_ml_engine(**kwargs) -> AdvancedAutoMLEngine:
    """Get an AutoML engine instance."""
    return AdvancedAutoMLEngine(**kwargs)


def quick_train(
    df: pd.DataFrame,
    target: str,
    **kwargs
) -> List[ModelTrainingResult]:
    """
    Quick one-shot training on ANY data.
    
    Example:
        results = quick_train(df, target="price")
        print(results[0].to_dict())
    """
    engine = AdvancedAutoMLEngine(verbose=kwargs.pop('verbose', False))
    return engine.auto_train(df, target, **kwargs)


# ----------------------------------------------------------------------------
# Backwards-compatible API expected by repo tests
# ----------------------------------------------------------------------------

@dataclass
class AutoMLResult:
    best_model: Any
    task: str
    metrics: Dict[str, float] = field(default_factory=dict)
    best_score: Optional[float] = None


class AutoMLEngine:
    def __init__(self, random_state: int = 42, verbose: bool = True) -> None:
        self.random_state = random_state
        self.verbose = verbose

        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for AutoMLEngine")

    def train(self, X: pd.DataFrame, y: pd.Series, task: str = "classification") -> AutoMLResult:
        if X is None or y is None:
            raise ValueError("X and y are required")
        if not isinstance(X, (pd.DataFrame,)):
            X = pd.DataFrame(X)

        y_series = y if isinstance(y, pd.Series) else pd.Series(y)

        # Minimal, reliable baselines for test coverage.
        if task == "classification":
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            try:
                scores = cross_val_score(model, X, y_series, cv=splitter, scoring="accuracy")
                best_score = float(np.mean(scores)) if len(scores) else None
            except Exception:
                best_score = None

            model.fit(X, y_series)
            metrics: Dict[str, float] = {}
            try:
                preds = model.predict(X)
                metrics["accuracy"] = float(accuracy_score(y_series, preds))
            except Exception:
                pass

            return AutoMLResult(best_model=model, task=task, metrics=metrics, best_score=best_score)

        if task == "regression":
            model = Ridge()
            splitter = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            try:
                scores = cross_val_score(model, X, y_series, cv=splitter, scoring="r2")
                best_score = float(np.mean(scores)) if len(scores) else None
            except Exception:
                best_score = None

            model.fit(X, y_series)
            metrics = {}
            try:
                preds = model.predict(X)
                metrics["r2"] = float(r2_score(y_series, preds))
            except Exception:
                pass

            return AutoMLResult(best_model=model, task=task, metrics=metrics, best_score=best_score)

        raise ValueError(f"Unknown task: {task}")
