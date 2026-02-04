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

# Import universal data handler
from app.ml.universal_data_handler import (
    UniversalDataPreprocessor, IntelligentTypeDetector,
    SemanticType, ColumnMetadata, PreprocessingResult,
    preprocess_for_ml, analyze_dataset
)

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


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
        task_map = {
            MLTask.BINARY_CLASSIFICATION: [MLTask.BINARY_CLASSIFICATION, 
                                           MLTask.MULTICLASS_CLASSIFICATION],
            MLTask.MULTICLASS_CLASSIFICATION: [MLTask.MULTICLASS_CLASSIFICATION],
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
    default_params={"max_iter": 1000, "random_state": 42, "n_jobs": -1},
    search_space={
        "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
        "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]},
        "solver": {"type": "categorical", "choices": ["saga"]}
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
    default_params={"n_jobs": -1},
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

@dataclass
class ModelTrainingResult:
    """Comprehensive result of model training."""
    
    model_id: UUID = field(default_factory=uuid4)
    model_name: str = ""
    display_name: str = ""
    task: MLTask = MLTask.BINARY_CLASSIFICATION
    status: ModelStatus = ModelStatus.COMPLETED
    
    # Data info
    n_samples_train: int = 0
    n_samples_test: int = 0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    target_column: str = ""
    
    # Metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Feature importance
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
            "data": {
                "train_samples": self.n_samples_train,
                "test_samples": self.n_samples_test,
                "n_features": self.n_features,
                "target": self.target_column
            },
            "metrics": {
                "train": {k: round(v, 4) for k, v in self.train_metrics.items()},
                "test": {k: round(v, 4) for k, v in self.test_metrics.items()},
                "cv_mean": round(self.cv_mean, 4),
                "cv_std": round(self.cv_std, 4)
            },
            "feature_importance": {
                k: round(v, 4) for k, v in 
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

class AdvancedAutoMLEngine:
    """
    Advanced AutoML engine with bulletproof data handling.
    
    Features:
    - Handles ANY data: nulls, outliers, mixed types, any columns
    - Automatic task detection (classification vs regression)
    - Intelligent model selection based on data characteristics
    - Bayesian hyperparameter optimization (Optuna)
    - Cross-validation with appropriate strategy (stratified, time series, etc.)
    - Feature importance and model explanations
    - Ensemble model creation
    - Full preprocessing transparency
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
        n_jobs: int = -1,
        verbose: bool = True
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
        
        # State
        self._preprocessor: Optional[UniversalDataPreprocessor] = None
        self._trained_models: Dict[str, Tuple[BaseEstimator, ModelTrainingResult]] = {}
        self._best_model_name: Optional[str] = None
        self._task: Optional[MLTask] = None
        self._label_encoder: Optional[LabelEncoder] = None
    
    def auto_train(
        self,
        df: pd.DataFrame,
        target_column: str,
        task: Optional[MLTask] = None,
        models: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        optimize: bool = True,
        max_models: int = 5
    ) -> List[ModelTrainingResult]:
        """
        Automatically train models on ANY data.
        
        Handles:
        - Missing values at any percentage
        - Outliers in any column
        - Mixed data types
        - Any number of columns/rows
        - Categorical and numerical features
        - Date/time columns
        - Text columns
        
        Args:
            df: Input dataframe (any format)
            target_column: Target column name
            task: ML task (auto-detected if None)
            models: List of model names (auto-selected if None)
            exclude_columns: Columns to exclude from features
            optimize: Whether to optimize hyperparameters
            max_models: Maximum models to train
        
        Returns:
            List of training results sorted by performance
        """
        total_start = datetime.utcnow()
        context = LogContext(component="AdvancedAutoML", operation="auto_train")
        
        if self.verbose:
            logger.info(f"Starting AutoML training on {df.shape[0]} rows, {df.shape[1]} columns", 
                       context=context)
        
        # Validate input
        if target_column not in df.columns:
            raise MLException(f"Target column '{target_column}' not found in dataframe")
        
        if df.empty:
            raise MLException("Empty dataframe provided")
        
        # Step 1: Detect task type
        if task is None:
            task = self._detect_task(df[target_column])
            if self.verbose:
                logger.info(f"Auto-detected task: {task.value}", context=context)
        self._task = task
        
        # Step 2: Preprocess data
        if self.verbose:
            logger.info("Preprocessing data...", context=context)
        
        preprocess_start = datetime.utcnow()
        try:
            prep_result = self._preprocess_data(df, target_column, exclude_columns)
            X, y, feature_names = prep_result.X, prep_result.y, prep_result.feature_names
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", context=context)
            raise MLException(f"Data preprocessing failed: {e}")
        
        preprocess_time = (datetime.utcnow() - preprocess_start).total_seconds()
        
        if self.verbose:
            logger.info(f"Preprocessed: {X.shape[0]} samples, {X.shape[1]} features", context=context)
        
        # Handle case where preprocessing removed all data
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise MLException("No valid data remaining after preprocessing")
        
        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = self._smart_split(X, y, task)
        
        if self.verbose:
            logger.info(f"Split: {len(X_train)} train, {len(X_test)} test", context=context)
        
        # Step 4: Select models
        if models is None:
            models = self._select_models(X_train, task, max_models)
            if self.verbose:
                logger.info(f"Selected models: {models}", context=context)
        
        # Step 5: Train each model
        results = []
        for model_name in models:
            try:
                if self.verbose:
                    logger.info(f"Training {model_name}...", context=context)
                
                result = self._train_single_model(
                    model_name=model_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_names=feature_names,
                    target_column=target_column,
                    task=task,
                    optimize=optimize,
                    preprocess_time=preprocess_time,
                    prep_summary=prep_result.__dict__
                )
                
                results.append(result)
                self._trained_models[model_name] = (
                    self._trained_models.get(model_name, (None, None))[0],
                    result
                )
                
                if self.verbose:
                    logger.info(
                        f"{model_name}: CV={result.cv_mean:.4f} (+/- {result.cv_std:.4f})",
                        context=context
                    )
                    
            except Exception as e:
                logger.error(f"Training {model_name} failed: {e}", context=context)
                logger.error(traceback.format_exc())
                
                # Create failure result
                results.append(ModelTrainingResult(
                    model_name=model_name,
                    task=task,
                    status=ModelStatus.FAILED,
                    errors=[str(e)]
                ))
        
        # Sort by performance
        successful = [r for r in results if r.status == ModelStatus.COMPLETED]
        if successful:
            if task == MLTask.REGRESSION:
                successful.sort(key=lambda r: r.test_metrics.get('r2', 0), reverse=True)
            else:
                successful.sort(key=lambda r: r.cv_mean, reverse=True)
            
            self._best_model_name = successful[0].model_name
            
            if self.verbose:
                logger.info(f"Best model: {self._best_model_name}", context=context)
        
        # Add failed results at the end
        failed = [r for r in results if r.status == ModelStatus.FAILED]
        results = successful + failed
        
        total_time = (datetime.utcnow() - total_start).total_seconds()
        for r in results:
            r.total_time_sec = total_time
        
        return results
    
    def _detect_task(self, target: pd.Series) -> MLTask:
        """Automatically detect the ML task from target variable."""
        # Remove nulls for analysis
        target_clean = target.dropna()
        
        if len(target_clean) == 0:
            raise MLException("Target column has no valid values")
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(target_clean):
            n_unique = target_clean.nunique()
            
            # Binary classification
            if n_unique == 2:
                return MLTask.BINARY_CLASSIFICATION
            
            # Few unique values - likely classification
            if n_unique <= 10 and n_unique / len(target_clean) < 0.05:
                return MLTask.MULTICLASS_CLASSIFICATION
            
            # Check if values are actually discrete
            if all(float(x).is_integer() for x in target_clean.head(100)):
                if n_unique <= 20:
                    return MLTask.MULTICLASS_CLASSIFICATION
            
            return MLTask.REGRESSION
        
        # Categorical/string target
        n_unique = target_clean.nunique()
        if n_unique == 2:
            return MLTask.BINARY_CLASSIFICATION
        return MLTask.MULTICLASS_CLASSIFICATION
    
    def _preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        exclude_columns: Optional[List[str]]
    ) -> PreprocessingResult:
        """Preprocess data using universal handler."""
        self._preprocessor = UniversalDataPreprocessor(
            max_categories_onehot=10,
            outlier_handling='clip',
            null_threshold_drop=0.7,
            verbose=self.verbose
        )
        
        result = self._preprocessor.fit_transform(
            df=df,
            target_column=target_column,
            exclude_columns=exclude_columns
        )
        
        return result
    
    def _smart_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: MLTask
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Smart train/test split based on task."""
        if task in [MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION]:
            # Stratified split for classification
            try:
                return train_test_split(
                    X, y,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=y
                )
            except ValueError:
                # Fall back to non-stratified if stratification fails
                return train_test_split(
                    X, y,
                    test_size=self.test_size,
                    random_state=self.random_state
                )
        else:
            return train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
    
    def _select_models(
        self,
        X: np.ndarray,
        task: MLTask,
        max_models: int
    ) -> List[str]:
        """Intelligently select models based on data characteristics."""
        n_samples, n_features = X.shape
        
        # Get all models for the task
        available = ModelLibrary.get_for_task(task, max_count=20)
        
        # Prioritize based on data size
        if n_samples > 100000:
            # Prefer fast models for large data
            preferred = ['lightgbm_classifier', 'lightgbm_regressor',
                        'hist_gradient_boosting_classifier', 'hist_gradient_boosting_regressor',
                        'xgboost_classifier', 'xgboost_regressor']
        elif n_samples < 1000:
            # Prefer regularized models for small data
            preferred = ['logistic_regression', 'ridge_regression', 'elastic_net',
                        'random_forest_classifier', 'random_forest_regressor']
        else:
            # Mix for medium data
            preferred = ['xgboost_classifier', 'xgboost_regressor',
                        'lightgbm_classifier', 'lightgbm_regressor',
                        'random_forest_classifier', 'random_forest_regressor',
                        'gradient_boosting_classifier']
        
        # Filter to available and add any extras
        selected = [m for m in preferred if m in available]
        
        # Add remaining models if needed
        for m in available:
            if m not in selected and len(selected) < max_models:
                selected.append(m)
        
        return selected[:max_models]
    
    def _train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        target_column: str,
        task: MLTask,
        optimize: bool,
        preprocess_time: float,
        prep_summary: Dict
    ) -> ModelTrainingResult:
        """Train a single model with optional optimization."""
        train_start = datetime.utcnow()
        
        config = ModelLibrary.get(model_name)
        result = ModelTrainingResult(
            model_name=model_name,
            display_name=config.display_name,
            task=task,
            status=ModelStatus.TRAINING,
            n_samples_train=len(X_train),
            n_samples_test=len(X_test),
            n_features=X_train.shape[1],
            feature_names=feature_names,
            target_column=target_column,
            preprocessing_time_sec=preprocess_time,
            preprocessing_summary={"dropped_columns": prep_summary.get("dropped_columns", []),
                                  "warnings": prep_summary.get("warnings", [])}
        )
        
        # Create base estimator
        params = config.default_params.copy()
        estimator = config.estimator_class(**params)
        
        # Hyperparameter optimization
        opt_start = datetime.utcnow()
        if optimize and config.search_space and self.optimization_strategy != OptimizationStrategy.NONE:
            try:
                best_params = self._optimize_hyperparameters(
                    estimator=estimator,
                    search_space=config.search_space,
                    X=X_train,
                    y=y_train,
                    task=task
                )
                params.update(best_params)
                result.best_params = best_params
                result.optimization_trials = self.optimization_trials
                estimator = config.estimator_class(**params)
            except Exception as e:
                result.warnings.append(f"Optimization failed: {e}, using defaults")
        
        result.optimization_time_sec = (datetime.utcnow() - opt_start).total_seconds()
        
        # Cross-validation
        cv = self._get_cv_strategy(task, y_train)
        scoring = self._get_scoring(task)
        
        try:
            cv_results = cross_validate(
                estimator, X_train, y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                return_train_score=True
            )
            
            result.cv_scores = {
                'test': cv_results[f'test_{scoring}'].tolist(),
                'train': cv_results[f'train_{scoring}'].tolist()
            }
            result.cv_mean = float(cv_results[f'test_{scoring}'].mean())
            result.cv_std = float(cv_results[f'test_{scoring}'].std())
        except Exception as e:
            result.warnings.append(f"CV failed: {e}")
            result.cv_mean = 0.0
            result.cv_std = 0.0
        
        # Fit on full training data
        estimator.fit(X_train, y_train)
        self._trained_models[model_name] = (estimator, result)
        
        # Predictions
        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test)
        
        # Calculate metrics
        if task == MLTask.REGRESSION:
            result.train_metrics = self._regression_metrics(y_train, y_train_pred)
            result.test_metrics = self._regression_metrics(y_test, y_test_pred)
        else:
            result.train_metrics = self._classification_metrics(
                y_train, y_train_pred, estimator, X_train
            )
            result.test_metrics = self._classification_metrics(
                y_test, y_test_pred, estimator, X_test
            )
        
        # Feature importance
        if config.supports_feature_importance:
            result.feature_importance = self._get_feature_importance(
                estimator, feature_names
            )
        
        result.training_time_sec = (datetime.utcnow() - train_start).total_seconds()
        result.status = ModelStatus.COMPLETED
        
        return result
    
    def _optimize_hyperparameters(
        self,
        estimator: BaseEstimator,
        search_space: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        task: MLTask
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna or fallback."""
        if self.optimization_strategy == OptimizationStrategy.BAYESIAN and HAS_OPTUNA:
            return self._optuna_optimize(estimator, search_space, X, y, task)
        else:
            return self._random_search_optimize(estimator, search_space, X, y, task)
    
    def _optuna_optimize(
        self,
        estimator: BaseEstimator,
        search_space: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        task: MLTask
    ) -> Dict[str, Any]:
        """Bayesian optimization using Optuna."""
        def objective(trial):
            params = {}
            for param_name, spec in search_space.items():
                if spec["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, spec["low"], spec["high"]
                    )
                elif spec["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, spec["low"], spec["high"],
                        log=spec.get("log", False)
                    )
                elif spec["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, spec["choices"]
                    )
            
            # Create model with suggested params
            model = clone(estimator)
            model.set_params(**params)
            
            # Quick CV
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) \
                if task != MLTask.REGRESSION else KFold(n_splits=3, shuffle=True, random_state=42)
            
            scoring = self._get_scoring(task)
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                return scores.mean()
            except:
                return 0.0
        
        # Run optimization
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.optimization_trials,
            timeout=self.optimization_timeout,
            show_progress_bar=False,
            n_jobs=1
        )
        
        return study.best_params
    
    def _random_search_optimize(
        self,
        estimator: BaseEstimator,
        search_space: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        task: MLTask
    ) -> Dict[str, Any]:
        """Random search optimization."""
        # Convert search space to sklearn format
        param_distributions = {}
        for param_name, spec in search_space.items():
            if spec["type"] == "int":
                param_distributions[param_name] = list(range(spec["low"], spec["high"] + 1))
            elif spec["type"] == "float":
                param_distributions[param_name] = np.linspace(spec["low"], spec["high"], 20).tolist()
            elif spec["type"] == "categorical":
                param_distributions[param_name] = spec["choices"]
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) \
            if task != MLTask.REGRESSION else KFold(n_splits=3, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=min(self.optimization_trials, 20),
            cv=cv,
            scoring=self._get_scoring(task),
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        search.fit(X, y)
        
        return search.best_params_
    
    def _get_cv_strategy(self, task: MLTask, y: np.ndarray):
        """Get appropriate cross-validation strategy."""
        if task in [MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION]:
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        elif task == MLTask.TIME_SERIES:
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            return KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
    
    def _get_scoring(self, task: MLTask) -> str:
        """Get scoring metric for task."""
        if self.scoring:
            return self.scoring
        
        if task == MLTask.BINARY_CLASSIFICATION:
            return 'roc_auc'
        elif task == MLTask.MULTICLASS_CLASSIFICATION:
            return 'f1_weighted'
        elif task == MLTask.REGRESSION:
            return 'r2'
        else:
            return 'accuracy'
    
    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: BaseEstimator,
        X: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Binary classification extras
        n_classes = len(np.unique(y_true))
        if n_classes == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
                metrics['average_precision'] = float(average_precision_score(y_true, y_proba))
                metrics['log_loss'] = float(log_loss(y_true, y_proba))
            except:
                pass
        
        # Matthews correlation for binary
        if n_classes == 2:
            try:
                metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
            except:
                pass
        
        return metrics
    
    def _regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        # Handle potential issues
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        metrics = {
            'r2': float(r2_score(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'explained_variance': float(explained_variance_score(y_true, y_pred)),
            'max_error': float(max_error(y_true, y_pred)),
            'median_ae': float(median_absolute_error(y_true, y_pred))
        }
        
        # MAPE (handle zeros)
        try:
            if (y_true != 0).all():
                metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
            else:
                # Use SMAPE for data with zeros
                denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
                denominator = np.where(denominator == 0, 1, denominator)
                metrics['smape'] = float(np.mean(np.abs(y_true - y_pred) / denominator))
        except:
            pass
        
        return metrics
    
    def _get_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract feature importance from model."""
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            importance = dict(zip(feature_names, imp.tolist()))
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_).flatten()
            if len(coef) == len(feature_names):
                importance = dict(zip(feature_names, coef.tolist()))
        
        # Normalize
        if importance:
            max_imp = max(importance.values()) or 1
            importance = {k: v / max_imp for k, v in importance.items()}
        
        return importance
    
    def predict(
        self,
        df: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """Make predictions using trained model."""
        if model_name is None:
            model_name = self._best_model_name
        
        if model_name is None or model_name not in self._trained_models:
            raise MLException("No trained model available for prediction")
        
        model, _ = self._trained_models[model_name]
        
        if self._preprocessor is None:
            raise MLException("Preprocessor not available")
        
        # Preprocess new data
        X = self._preprocessor.transform(df)
        
        return model.predict(X)
    
    def get_best_model(self) -> Tuple[str, BaseEstimator, ModelTrainingResult]:
        """Get the best performing model."""
        if self._best_model_name is None:
            raise MLException("No models have been trained")
        
        model, result = self._trained_models[self._best_model_name]
        return self._best_model_name, model, result
    
    def compare_models(
        self,
        results: List[ModelTrainingResult]
    ) -> Dict[str, Any]:
        """Compare multiple trained models."""
        successful = [r for r in results if r.status == ModelStatus.COMPLETED]
        
        if not successful:
            return {"error": "No successful models to compare", "models": []}
        
        comparison = {
            "n_models": len(successful),
            "best_model": successful[0].model_name if successful else None,
            "summary": [],
            "recommendation": ""
        }
        
        for r in successful:
            summary = {
                "name": r.model_name,
                "display_name": r.display_name,
                "cv_mean": r.cv_mean,
                "cv_std": r.cv_std,
                "test_metrics": {k: round(v, 4) for k, v in r.test_metrics.items()},
                "training_time_sec": round(r.training_time_sec, 2)
            }
            comparison["summary"].append(summary)
        
        if successful:
            best = successful[0]
            if len(successful) > 1:
                runner_up = successful[1]
                comparison["recommendation"] = (
                    f"Best: {best.display_name} (CV={best.cv_mean:.4f}). "
                    f"Consider {runner_up.display_name} as alternative."
                )
            else:
                comparison["recommendation"] = f"Best: {best.display_name} (CV={best.cv_mean:.4f})"
        
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
