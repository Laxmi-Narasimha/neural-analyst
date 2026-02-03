# AI Enterprise Data Analyst - Ensemble Methods
# Advanced ensemble learning techniques

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Ensemble Types
# ============================================================================

class EnsembleMethod(str, Enum):
    """Ensemble methods."""
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    BLENDING = "blending"


class VotingType(str, Enum):
    """Voting types."""
    HARD = "hard"
    SOFT = "soft"
    WEIGHTED = "weighted"


@dataclass 
class EnsembleResult:
    """Ensemble prediction result."""
    
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    
    individual_predictions: dict[str, np.ndarray] = field(default_factory=dict)
    model_weights: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions.tolist()[:10],
            "model_weights": self.model_weights
        }


# ============================================================================
# Voting Ensemble
# ============================================================================

class VotingEnsemble:
    """Voting ensemble for classification/regression."""
    
    def __init__(
        self,
        voting: VotingType = VotingType.SOFT,
        weights: dict[str, float] = None
    ):
        self.voting = voting
        self.weights = weights or {}
        self._models: dict[str, Any] = {}
    
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add model to ensemble."""
        self._models[name] = model
        self.weights[name] = weight
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        """Fit all models."""
        for name, model in self._models.items():
            model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Make ensemble predictions."""
        predictions = {}
        probabilities = {}
        
        for name, model in self._models.items():
            predictions[name] = model.predict(X)
            if hasattr(model, 'predict_proba'):
                probabilities[name] = model.predict_proba(X)
        
        # Combine predictions
        if self.voting == VotingType.HARD:
            # Majority vote
            all_preds = np.array(list(predictions.values()))
            from scipy import stats
            final_pred, _ = stats.mode(all_preds, axis=0)
            final_pred = final_pred.flatten()
        
        elif self.voting == VotingType.SOFT:
            # Average probabilities
            if probabilities:
                all_probs = np.array(list(probabilities.values()))
                avg_probs = np.mean(all_probs, axis=0)
                final_pred = np.argmax(avg_probs, axis=1)
            else:
                all_preds = np.array(list(predictions.values()))
                final_pred = np.mean(all_preds, axis=0)
        
        else:  # WEIGHTED
            # Weighted average
            if probabilities:
                weighted_probs = None
                total_weight = 0
                
                for name, probs in probabilities.items():
                    w = self.weights.get(name, 1.0)
                    if weighted_probs is None:
                        weighted_probs = probs * w
                    else:
                        weighted_probs += probs * w
                    total_weight += w
                
                weighted_probs /= total_weight
                final_pred = np.argmax(weighted_probs, axis=1)
            else:
                weighted_sum = None
                total_weight = 0
                
                for name, pred in predictions.items():
                    w = self.weights.get(name, 1.0)
                    if weighted_sum is None:
                        weighted_sum = pred * w
                    else:
                        weighted_sum += pred * w
                    total_weight += w
                
                final_pred = weighted_sum / total_weight
        
        return EnsembleResult(
            predictions=final_pred,
            individual_predictions=predictions,
            model_weights=self.weights
        )


# ============================================================================
# Stacking Ensemble
# ============================================================================

class StackingEnsemble:
    """Stacking ensemble with meta-learner."""
    
    def __init__(
        self,
        meta_learner: Any = None,
        use_probabilities: bool = True
    ):
        self._base_models: dict[str, Any] = {}
        self._meta_learner = meta_learner
        self.use_probabilities = use_probabilities
    
    def add_base_model(self, name: str, model: Any) -> None:
        """Add base model."""
        self._base_models[name] = model
    
    def set_meta_learner(self, model: Any) -> None:
        """Set meta-learner."""
        self._meta_learner = model
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> "StackingEnsemble":
        """Fit stacking ensemble."""
        from sklearn.model_selection import cross_val_predict
        
        # Fit base models and get meta-features
        meta_features = []
        
        for name, model in self._base_models.items():
            model.fit(X, y)
            
            # Get out-of-fold predictions
            if self.use_probabilities and hasattr(model, 'predict_proba'):
                oof_pred = cross_val_predict(
                    model, X, y, cv=5, method='predict_proba'
                )
            else:
                oof_pred = cross_val_predict(model, X, y, cv=5)
                oof_pred = oof_pred.reshape(-1, 1)
            
            meta_features.append(oof_pred)
        
        # Stack meta-features
        X_meta = np.hstack(meta_features)
        
        # Fit meta-learner
        if self._meta_learner is None:
            from sklearn.linear_model import LogisticRegression
            self._meta_learner = LogisticRegression()
        
        self._meta_learner.fit(X_meta, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Make stacking predictions."""
        # Get base model predictions
        meta_features = []
        individual_preds = {}
        
        for name, model in self._base_models.items():
            if self.use_probabilities and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            
            meta_features.append(pred)
            individual_preds[name] = model.predict(X)
        
        X_meta = np.hstack(meta_features)
        
        # Meta-learner prediction
        final_pred = self._meta_learner.predict(X_meta)
        
        probs = None
        if hasattr(self._meta_learner, 'predict_proba'):
            probs = self._meta_learner.predict_proba(X_meta)
        
        return EnsembleResult(
            predictions=final_pred,
            probabilities=probs,
            individual_predictions=individual_preds
        )


# ============================================================================
# Blending Ensemble
# ============================================================================

class BlendingEnsemble:
    """Blending ensemble using holdout set."""
    
    def __init__(
        self,
        holdout_fraction: float = 0.2,
        meta_learner: Any = None
    ):
        self.holdout_fraction = holdout_fraction
        self._base_models: dict[str, Any] = {}
        self._meta_learner = meta_learner
    
    def add_base_model(self, name: str, model: Any) -> None:
        """Add base model."""
        self._base_models[name] = model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlendingEnsemble":
        """Fit blending ensemble."""
        from sklearn.model_selection import train_test_split
        
        # Split into train and holdout
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=self.holdout_fraction, random_state=42
        )
        
        # Train base models
        blend_features = []
        
        for name, model in self._base_models.items():
            model.fit(X_train, y_train)
            
            # Predict on holdout
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_holdout)
            else:
                pred = model.predict(X_holdout).reshape(-1, 1)
            
            blend_features.append(pred)
        
        X_blend = np.hstack(blend_features)
        
        # Train meta-learner
        if self._meta_learner is None:
            from sklearn.linear_model import LogisticRegression
            self._meta_learner = LogisticRegression()
        
        self._meta_learner.fit(X_blend, y_holdout)
        
        return self
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Make blending predictions."""
        blend_features = []
        individual_preds = {}
        
        for name, model in self._base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            
            blend_features.append(pred)
            individual_preds[name] = model.predict(X)
        
        X_blend = np.hstack(blend_features)
        final_pred = self._meta_learner.predict(X_blend)
        
        return EnsembleResult(
            predictions=final_pred,
            individual_predictions=individual_preds
        )


# ============================================================================
# Ensemble Engine
# ============================================================================

class EnsembleEngine:
    """
    Ensemble methods engine.
    
    Features:
    - Voting (hard, soft, weighted)
    - Stacking with meta-learner
    - Blending
    - Model selection
    - Weight optimization
    """
    
    def __init__(self):
        self._ensembles: dict[str, Any] = {}
    
    def create_voting_ensemble(
        self,
        models: dict[str, Any],
        voting: VotingType = VotingType.SOFT,
        weights: dict[str, float] = None
    ) -> VotingEnsemble:
        """Create voting ensemble."""
        ensemble = VotingEnsemble(voting, weights)
        
        for name, model in models.items():
            w = weights.get(name, 1.0) if weights else 1.0
            ensemble.add_model(name, model, w)
        
        return ensemble
    
    def create_stacking_ensemble(
        self,
        base_models: dict[str, Any],
        meta_learner: Any = None
    ) -> StackingEnsemble:
        """Create stacking ensemble."""
        ensemble = StackingEnsemble(meta_learner)
        
        for name, model in base_models.items():
            ensemble.add_base_model(name, model)
        
        return ensemble
    
    def create_blending_ensemble(
        self,
        base_models: dict[str, Any],
        holdout_fraction: float = 0.2,
        meta_learner: Any = None
    ) -> BlendingEnsemble:
        """Create blending ensemble."""
        ensemble = BlendingEnsemble(holdout_fraction, meta_learner)
        
        for name, model in base_models.items():
            ensemble.add_base_model(name, model)
        
        return ensemble
    
    def optimize_weights(
        self,
        models: dict[str, Any],
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: Callable = None
    ) -> dict[str, float]:
        """Optimize ensemble weights using validation set."""
        from scipy.optimize import minimize
        
        if metric is None:
            from sklearn.metrics import accuracy_score
            metric = accuracy_score
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X_val)
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            combined = np.zeros_like(y_val, dtype=float)
            for i, name in enumerate(model_names):
                combined += weights[i] * predictions[name]
            
            combined = np.round(combined).astype(int)
            return -metric(y_val, combined)  # Negative for minimization
        
        # Initial weights
        initial = np.ones(n_models) / n_models
        
        # Bounds (weights between 0 and 1)
        bounds = [(0, 1)] * n_models
        
        # Optimize
        result = minimize(objective, initial, bounds=bounds, method='SLSQP')
        
        optimal_weights = result.x / result.x.sum()
        
        return dict(zip(model_names, optimal_weights.tolist()))


# Factory function
def get_ensemble_engine() -> EnsembleEngine:
    """Get ensemble engine instance."""
    return EnsembleEngine()
