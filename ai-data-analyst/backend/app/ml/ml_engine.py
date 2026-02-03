# AI Enterprise Data Analyst - ML Engine
# Production-grade AutoML with classical algorithms, hyperparameter optimization, and model evaluation

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Type, Union
from uuid import UUID, uuid4
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
    # Regression
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Classical ML algorithms
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

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
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import MLException
except ImportError:
    class MLException(Exception): pass

logger = get_logger(__name__)

warnings.filterwarnings('ignore')


# ============================================================================
# Types and Enums
# ============================================================================

class MLTask(str, Enum):
    """Machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelStatus(str, Enum):
    """Model training status."""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class ScalerType(str, Enum):
    """Feature scaler types."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class ImputerType(str, Enum):
    """Imputer types for missing values."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    KNN = "knn"
    NONE = "none"


# ============================================================================
# Model Registry
# ============================================================================

@dataclass
class AlgorithmConfig:
    """Configuration for an ML algorithm."""
    
    name: str
    estimator_class: Type[BaseEstimator]
    task: MLTask
    default_params: dict[str, Any] = field(default_factory=dict)
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    supports_feature_importance: bool = False
    supports_probability: bool = False
    description: str = ""


class ModelRegistry:
    """Registry of available ML algorithms."""
    
    _algorithms: dict[str, AlgorithmConfig] = {}
    
    @classmethod
    def register(cls, config: AlgorithmConfig) -> None:
        """Register an algorithm."""
        cls._algorithms[config.name] = config
    
    @classmethod
    def get(cls, name: str) -> AlgorithmConfig:
        """Get algorithm config by name."""
        if name not in cls._algorithms:
            raise MLException(f"Unknown algorithm: {name}")
        return cls._algorithms[name]
    
    @classmethod
    def list_algorithms(cls, task: Optional[MLTask] = None) -> list[str]:
        """List available algorithms, optionally filtered by task."""
        if task:
            return [name for name, cfg in cls._algorithms.items() if cfg.task == task]
        return list(cls._algorithms.keys())
    
    @classmethod
    def get_default_for_task(cls, task: MLTask) -> list[str]:
        """Get default algorithms for a task."""
        return [name for name, cfg in cls._algorithms.items() if cfg.task == task][:5]


# Register classification algorithms
ModelRegistry.register(AlgorithmConfig(
    name="logistic_regression",
    estimator_class=LogisticRegression,
    task=MLTask.CLASSIFICATION,
    default_params={"max_iter": 1000, "random_state": 42},
    param_grid={
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["saga"]
    },
    supports_probability=True,
    description="Fast linear model for baseline classification"
))

ModelRegistry.register(AlgorithmConfig(
    name="random_forest_classifier",
    estimator_class=RandomForestClassifier,
    task=MLTask.CLASSIFICATION,
    default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    supports_feature_importance=True,
    supports_probability=True,
    description="Ensemble of decision trees"
))

ModelRegistry.register(AlgorithmConfig(
    name="gradient_boosting_classifier",
    estimator_class=GradientBoostingClassifier,
    task=MLTask.CLASSIFICATION,
    default_params={"n_estimators": 100, "random_state": 42},
    param_grid={
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    supports_feature_importance=True,
    supports_probability=True,
    description="Gradient boosting for classification"
))

ModelRegistry.register(AlgorithmConfig(
    name="svm_classifier",
    estimator_class=SVC,
    task=MLTask.CLASSIFICATION,
    default_params={"probability": True, "random_state": 42},
    param_grid={
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    supports_probability=True,
    description="Support Vector Machine"
))

# Register XGBoost if available
if HAS_XGBOOST:
    ModelRegistry.register(AlgorithmConfig(
        name="xgboost_classifier",
        estimator_class=xgb.XGBClassifier,
        task=MLTask.CLASSIFICATION,
        default_params={
            "n_estimators": 100,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        },
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3]
        },
        supports_feature_importance=True,
        supports_probability=True,
        description="Extreme Gradient Boosting (fast, accurate)"
    ))

# Register LightGBM if available
if HAS_LIGHTGBM:
    ModelRegistry.register(AlgorithmConfig(
        name="lightgbm_classifier",
        estimator_class=lgb.LGBMClassifier,
        task=MLTask.CLASSIFICATION,
        default_params={"n_estimators": 100, "random_state": 42, "verbose": -1},
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.1, 0.3]
        },
        supports_feature_importance=True,
        supports_probability=True,
        description="Light Gradient Boosting (very fast)"
    ))

# Register regression algorithms
ModelRegistry.register(AlgorithmConfig(
    name="linear_regression",
    estimator_class=LinearRegression,
    task=MLTask.REGRESSION,
    default_params={},
    param_grid={},
    description="Simple linear regression baseline"
))

ModelRegistry.register(AlgorithmConfig(
    name="ridge_regression",
    estimator_class=Ridge,
    task=MLTask.REGRESSION,
    default_params={"random_state": 42},
    param_grid={"alpha": [0.01, 0.1, 1, 10, 100]},
    description="L2 regularized linear regression"
))

ModelRegistry.register(AlgorithmConfig(
    name="random_forest_regressor",
    estimator_class=RandomForestRegressor,
    task=MLTask.REGRESSION,
    default_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    supports_feature_importance=True,
    description="Ensemble for regression"
))

if HAS_XGBOOST:
    ModelRegistry.register(AlgorithmConfig(
        name="xgboost_regressor",
        estimator_class=xgb.XGBRegressor,
        task=MLTask.REGRESSION,
        default_params={"n_estimators": 100, "random_state": 42},
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3]
        },
        supports_feature_importance=True,
        description="XGBoost for regression"
    ))


# ============================================================================
# Training Results
# ============================================================================

@dataclass
class TrainingResult:
    """Result of model training."""
    
    model_id: UUID = field(default_factory=uuid4)
    algorithm: str = ""
    task: MLTask = MLTask.CLASSIFICATION
    status: ModelStatus = ModelStatus.COMPLETED
    
    # Data info
    training_samples: int = 0
    test_samples: int = 0
    features_used: list[str] = field(default_factory=list)
    target_column: str = ""
    
    # Metrics
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Feature importance
    feature_importance: dict[str, float] = field(default_factory=dict)
    
    # Best hyperparameters
    best_params: dict[str, Any] = field(default_factory=dict)
    
    # Model artifact
    model_path: Optional[str] = None
    
    # Timing
    training_time_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": str(self.model_id),
            "algorithm": self.algorithm,
            "task": self.task.value,
            "status": self.status.value,
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "features_used": self.features_used,
            "target_column": self.target_column,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "feature_importance": {
                k: round(v, 4) for k, v in 
                sorted(self.feature_importance.items(), key=lambda x: -x[1])[:20]
            },
            "best_params": self.best_params,
            "training_time_seconds": round(self.training_time_seconds, 2),
            "created_at": self.created_at.isoformat()
        }


# ============================================================================
# Feature Engineering
# ============================================================================

class FeatureEngineer:
    """
    Automated feature engineering pipeline.
    
    Handles:
    - Missing value imputation
    - Numeric scaling
    - Categorical encoding
    - Feature selection
    """
    
    def __init__(
        self,
        numeric_imputer: ImputerType = ImputerType.MEDIAN,
        categorical_imputer: ImputerType = ImputerType.MODE,
        scaler: ScalerType = ScalerType.STANDARD,
        encode_categoricals: bool = True,
        max_categories: int = 20
    ) -> None:
        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer
        self.scaler = scaler
        self.encode_categoricals = encode_categoricals
        self.max_categories = max_categories
        
        self._preprocessor: Optional[ColumnTransformer] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._numeric_columns: list[str] = []
        self._categorical_columns: list[str] = []
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        exclude_columns: Optional[list[str]] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[str]]:
        """
        Fit and transform the dataframe.
        
        Returns:
            (X transformed, y encoded, feature names)
        """
        exclude = set(exclude_columns or [])
        if target_column:
            exclude.add(target_column)
        
        # Identify column types
        self._numeric_columns = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]
        
        self._categorical_columns = [
            col for col in df.select_dtypes(include=['object', 'category']).columns
            if col not in exclude and df[col].nunique() <= self.max_categories
        ]
        
        # Build preprocessing pipeline
        transformers = []
        
        # Numeric pipeline
        if self._numeric_columns:
            numeric_steps = []
            
            # Imputer
            if self.numeric_imputer == ImputerType.KNN:
                numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            elif self.numeric_imputer != ImputerType.NONE:
                numeric_steps.append(('imputer', SimpleImputer(strategy=self.numeric_imputer.value)))
            
            # Scaler
            if self.scaler == ScalerType.STANDARD:
                numeric_steps.append(('scaler', StandardScaler()))
            elif self.scaler == ScalerType.MINMAX:
                numeric_steps.append(('scaler', MinMaxScaler()))
            elif self.scaler == ScalerType.ROBUST:
                numeric_steps.append(('scaler', RobustScaler()))
            
            if numeric_steps:
                transformers.append(('num', Pipeline(numeric_steps), self._numeric_columns))
        
        # Categorical pipeline
        if self._categorical_columns and self.encode_categoricals:
            cat_steps = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]
            transformers.append(('cat', Pipeline(cat_steps), self._categorical_columns))
        
        # Create column transformer
        self._preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Transform features
        X = self._preprocessor.fit_transform(df)
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        # Handle target
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column].values
            
            # Encode if categorical
            if df[target_column].dtype == 'object':
                self._label_encoder = LabelEncoder()
                y = self._label_encoder.fit_transform(y)
        
        return X, y, feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if self._preprocessor is None:
            raise MLException("Preprocessor not fitted. Call fit_transform first.")
        return self._preprocessor.transform(df)
    
    def _get_feature_names(self) -> list[str]:
        """Get feature names after transformation."""
        feature_names = []
        
        for name, transformer, columns in self._preprocessor.transformers_:
            if name == 'remainder':
                continue
            
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # Get one-hot encoded names
                encoder = transformer.named_steps.get('encoder')
                if encoder and hasattr(encoder, 'get_feature_names_out'):
                    cat_names = encoder.get_feature_names_out(columns)
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend(columns)
        
        return feature_names


# ============================================================================
# ML Engine
# ============================================================================

class MLEngine:
    """
    Production-grade ML engine with AutoML capabilities.
    
    Features:
    - Automatic algorithm selection
    - Hyperparameter optimization (Optuna/GridSearch)
    - Cross-validation
    - Feature importance
    - Model comparison
    - Model persistence
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        use_optuna: bool = True
    ) -> None:
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_optuna = use_optuna and HAS_OPTUNA
        
        self._feature_engineer = FeatureEngineer()
        self._trained_models: dict[str, tuple[BaseEstimator, TrainingResult]] = {}
    
    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        task: MLTask = MLTask.CLASSIFICATION,
        algorithms: Optional[list[str]] = None,
        optimize_hyperparams: bool = True,
        n_trials: int = 50
    ) -> list[TrainingResult]:
        """
        Train models on the dataset.
        
        Args:
            df: Input dataframe
            target_column: Target column name
            task: ML task type
            algorithms: List of algorithm names (auto-selected if None)
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials (for Optuna)
        
        Returns:
            List of training results sorted by performance
        """
        context = LogContext(component="MLEngine", operation="train")
        
        logger.info(
            f"Starting training: {task.value}",
            context=context,
            target=target_column,
            rows=len(df)
        )
        
        # Get algorithms for task
        if algorithms is None:
            algorithms = ModelRegistry.get_default_for_task(task)
        
        # Preprocess data
        X, y, feature_names = self._feature_engineer.fit_transform(
            df, target_column=target_column
        )
        
        # Split data
        stratify = y if task == MLTask.CLASSIFICATION else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        results = []
        
        # Train each algorithm
        for algo_name in algorithms:
            try:
                logger.info(f"Training {algo_name}", context=context)
                
                result = self._train_single_model(
                    algo_name=algo_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_names=feature_names,
                    target_column=target_column,
                    task=task,
                    optimize=optimize_hyperparams,
                    n_trials=n_trials
                )
                
                results.append(result)
                
                logger.info(
                    f"Completed {algo_name}: {result.cv_mean:.4f} (+/- {result.cv_std:.4f})",
                    context=context
                )
                
            except Exception as e:
                logger.error(f"Failed training {algo_name}: {e}", context=context)
        
        # Sort by CV score
        results.sort(key=lambda r: r.cv_mean, reverse=True)
        
        return results
    
    def _train_single_model(
        self,
        algo_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        target_column: str,
        task: MLTask,
        optimize: bool,
        n_trials: int
    ) -> TrainingResult:
        """Train a single model with optional hyperparameter optimization."""
        start_time = datetime.utcnow()
        
        config = ModelRegistry.get(algo_name)
        
        # Create estimator
        estimator = config.estimator_class(**config.default_params)
        
        # Hyperparameter optimization
        best_params = config.default_params.copy()
        
        if optimize and config.param_grid:
            if self.use_optuna:
                best_params = self._optimize_with_optuna(
                    estimator, config.param_grid, X_train, y_train, task, n_trials
                )
            else:
                best_params = self._optimize_with_grid_search(
                    estimator, config.param_grid, X_train, y_train, task
                )
            
            # Recreate with best params
            estimator = config.estimator_class(**{**config.default_params, **best_params})
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state) \
             if task == MLTask.CLASSIFICATION else \
             KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring = 'accuracy' if task == MLTask.CLASSIFICATION else 'r2'
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring)
        
        # Fit on full training data
        estimator.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test)
        
        # Calculate metrics
        if task == MLTask.CLASSIFICATION:
            train_metrics = self._classification_metrics(y_train, y_train_pred, estimator, X_train)
            test_metrics = self._classification_metrics(y_test, y_test_pred, estimator, X_test)
        else:
            train_metrics = self._regression_metrics(y_train, y_train_pred)
            test_metrics = self._regression_metrics(y_test, y_test_pred)
        
        # Feature importance
        feature_importance = {}
        if config.supports_feature_importance:
            if hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
                feature_importance = dict(zip(feature_names, importance.tolist()))
            elif hasattr(estimator, 'coef_'):
                importance = np.abs(estimator.coef_).flatten()
                if len(importance) == len(feature_names):
                    feature_importance = dict(zip(feature_names, importance.tolist()))
        
        # Store model
        self._trained_models[algo_name] = (estimator, None)
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        return TrainingResult(
            algorithm=algo_name,
            task=task,
            status=ModelStatus.COMPLETED,
            training_samples=len(X_train),
            test_samples=len(X_test),
            features_used=feature_names,
            target_column=target_column,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            feature_importance=feature_importance,
            best_params=best_params,
            training_time_seconds=training_time
        )
    
    def _optimize_with_optuna(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list],
        X: np.ndarray,
        y: np.ndarray,
        task: MLTask,
        n_trials: int
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = {}
            for param_name, values in param_grid.items():
                if all(isinstance(v, int) for v in values):
                    params[param_name] = trial.suggest_int(param_name, min(values), max(values))
                elif all(isinstance(v, float) for v in values):
                    params[param_name] = trial.suggest_float(param_name, min(values), max(values))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, values)
            
            model = estimator.__class__(**{**estimator.get_params(), **params})
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) \
                 if task == MLTask.CLASSIFICATION else KFold(n_splits=3, shuffle=True, random_state=42)
            
            scoring = 'accuracy' if task == MLTask.CLASSIFICATION else 'r2'
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _optimize_with_grid_search(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list],
        X: np.ndarray,
        y: np.ndarray,
        task: MLTask
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Grid Search."""
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) \
             if task == MLTask.CLASSIFICATION else KFold(n_splits=3, shuffle=True, random_state=42)
        
        scoring = 'accuracy' if task == MLTask.CLASSIFICATION else 'r2'
        
        search = RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring=scoring,
            random_state=42
        )
        search.fit(X, y)
        
        return search.best_params_
    
    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: BaseEstimator,
        X: np.ndarray
    ) -> dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # AUC if binary and model supports probability
        if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                pass
        
        return metrics
    
    def _regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate regression metrics."""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def predict(
        self,
        algorithm: str,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions using a trained model."""
        if algorithm not in self._trained_models:
            raise MLException(f"Model '{algorithm}' not trained")
        
        model, _ = self._trained_models[algorithm]
        
        if isinstance(X, pd.DataFrame):
            X = self._feature_engineer.transform(X)
        
        return model.predict(X)
    
    def get_best_model(self) -> tuple[str, BaseEstimator]:
        """Get the best performing model."""
        if not self._trained_models:
            raise MLException("No models trained")
        
        # Return first (best) model
        algo_name = list(self._trained_models.keys())[0]
        return algo_name, self._trained_models[algo_name][0]
    
    def compare_models(self, results: list[TrainingResult]) -> dict[str, Any]:
        """Compare multiple trained models."""
        comparison = {
            "models": [],
            "best_model": None,
            "recommendation": ""
        }
        
        for result in results:
            comparison["models"].append({
                "algorithm": result.algorithm,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "test_metrics": result.test_metrics,
                "training_time": result.training_time_seconds
            })
        
        if results:
            best = results[0]
            comparison["best_model"] = best.algorithm
            comparison["recommendation"] = (
                f"Best model: {best.algorithm} with CV score {best.cv_mean:.4f}. "
                f"Consider {results[1].algorithm if len(results) > 1 else best.algorithm} as alternative."
            )
        
        return comparison


# Factory function
def get_ml_engine() -> MLEngine:
    """Get ML engine instance."""
    return MLEngine()
