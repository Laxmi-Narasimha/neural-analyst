# AI Enterprise Data Analyst - Explainability Module
# SHAP, LIME, and model interpretation (responsible AI patterns)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Explainability Types
# ============================================================================

class ExplainerType(str, Enum):
    """Types of model explainers."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION = "permutation"
    PARTIAL_DEPENDENCE = "pdp"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class FeatureContribution:
    """Feature contribution to prediction."""
    
    feature: str
    value: Any
    contribution: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "contribution": round(self.contribution, 4)
        }


@dataclass
class ExplanationResult:
    """Model explanation result."""
    
    prediction: Any
    base_value: float
    contributions: list[FeatureContribution]
    
    # Confidence
    confidence: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "base_value": round(self.base_value, 4),
            "confidence": round(self.confidence, 4),
            "contributions": [c.to_dict() for c in self.contributions],
            "top_positive": [c.to_dict() for c in sorted(
                self.contributions, key=lambda x: x.contribution, reverse=True
            )[:5]],
            "top_negative": [c.to_dict() for c in sorted(
                self.contributions, key=lambda x: x.contribution
            )[:5]]
        }


@dataclass
class GlobalExplanation:
    """Global model explanation."""
    
    feature_importance: dict[str, float]
    partial_dependence: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_importance": {k: round(v, 4) for k, v in sorted(
                self.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:20]},
            "partial_dependence": {k: v[:10] for k, v in self.partial_dependence.items()}
        }


# ============================================================================
# SHAP Explainer
# ============================================================================

class SHAPExplainer:
    """SHAP-based model explanation."""
    
    def __init__(self, model: Any, X_background: pd.DataFrame = None):
        self._model = model
        self._X_background = X_background
        self._explainer = None
        self._shap_available = self._check_shap()
    
    def _check_shap(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap
            return True
        except ImportError:
            return False
    
    def fit(self, X: pd.DataFrame):
        """Fit the explainer."""
        if not self._shap_available:
            logger.warning("SHAP not installed, using fallback")
            return
        
        import shap
        
        # Choose appropriate explainer
        try:
            # Tree explainer for tree-based models
            self._explainer = shap.TreeExplainer(self._model)
        except:
            try:
                # Kernel explainer as fallback
                background = X.sample(min(100, len(X)), random_state=42)
                if hasattr(self._model, 'predict_proba'):
                    self._explainer = shap.KernelExplainer(
                        self._model.predict_proba, background
                    )
                else:
                    self._explainer = shap.KernelExplainer(
                        self._model.predict, background
                    )
            except:
                logger.warning("Could not create SHAP explainer")
    
    def explain(self, X: pd.DataFrame) -> list[ExplanationResult]:
        """Generate SHAP explanations."""
        if not self._shap_available or self._explainer is None:
            return self._fallback_explain(X)
        
        import shap
        
        shap_values = self._explainer.shap_values(X)
        
        # Handle multiclass
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        results = []
        for i in range(len(X)):
            contributions = []
            for j, col in enumerate(X.columns):
                contributions.append(FeatureContribution(
                    feature=col,
                    value=X.iloc[i, j],
                    contribution=float(shap_values[i, j])
                ))
            
            # Get prediction
            if hasattr(self._model, 'predict_proba'):
                pred = self._model.predict_proba(X.iloc[[i]])[0]
                prediction = float(pred[1]) if len(pred) > 1 else float(pred[0])
            else:
                prediction = float(self._model.predict(X.iloc[[i]])[0])
            
            results.append(ExplanationResult(
                prediction=prediction,
                base_value=float(self._explainer.expected_value) if hasattr(self._explainer, 'expected_value') else 0,
                contributions=contributions,
                confidence=abs(prediction - 0.5) * 2 if 0 <= prediction <= 1 else 0
            ))
        
        return results
    
    def get_global_importance(self, X: pd.DataFrame) -> dict[str, float]:
        """Get global feature importance."""
        if not self._shap_available or self._explainer is None:
            return self._fallback_importance(X)
        
        import shap
        
        shap_values = self._explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(X.columns, importance.tolist()))
    
    def _fallback_explain(self, X: pd.DataFrame) -> list[ExplanationResult]:
        """Fallback explanation using feature importance."""
        results = []
        
        for i in range(len(X)):
            contributions = []
            for col in X.columns:
                contributions.append(FeatureContribution(
                    feature=col,
                    value=X.iloc[i][col],
                    contribution=0.0  # No SHAP values without library
                ))
            
            if hasattr(self._model, 'predict_proba'):
                pred = self._model.predict_proba(X.iloc[[i]])[0]
                prediction = float(pred[1]) if len(pred) > 1 else float(pred[0])
            else:
                prediction = float(self._model.predict(X.iloc[[i]])[0])
            
            results.append(ExplanationResult(
                prediction=prediction,
                base_value=0,
                contributions=contributions
            ))
        
        return results
    
    def _fallback_importance(self, X: pd.DataFrame) -> dict[str, float]:
        """Fallback feature importance."""
        if hasattr(self._model, 'feature_importances_'):
            return dict(zip(X.columns, self._model.feature_importances_.tolist()))
        elif hasattr(self._model, 'coef_'):
            coef = np.abs(self._model.coef_).flatten()
            return dict(zip(X.columns, coef.tolist()))
        return {col: 0.0 for col in X.columns}


# ============================================================================
# Permutation Importance
# ============================================================================

class PermutationExplainer:
    """Permutation-based feature importance."""
    
    def __init__(self, model: Any, n_repeats: int = 10):
        self._model = model
        self.n_repeats = n_repeats
    
    def explain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = 'accuracy'
    ) -> dict[str, float]:
        """Calculate permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
            
            result = permutation_importance(
                self._model, X, y,
                n_repeats=self.n_repeats,
                random_state=42,
                scoring=scoring
            )
            
            return dict(zip(X.columns, result.importances_mean.tolist()))
            
        except ImportError:
            # Manual implementation
            return self._manual_permutation(X, y)
    
    def _manual_permutation(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict[str, float]:
        """Manual permutation importance."""
        X_array = X.values
        baseline_score = self._score(X_array, y)
        
        importances = {}
        for i, col in enumerate(X.columns):
            scores = []
            for _ in range(self.n_repeats):
                X_permuted = X_array.copy()
                np.random.shuffle(X_permuted[:, i])
                scores.append(baseline_score - self._score(X_permuted, y))
            importances[col] = float(np.mean(scores))
        
        return importances
    
    def _score(self, X: np.ndarray, y: pd.Series) -> float:
        """Calculate model score."""
        predictions = self._model.predict(X)
        return float((predictions == y.values).mean())


# ============================================================================
# Partial Dependence
# ============================================================================

class PartialDependenceExplainer:
    """Partial Dependence Plots computation."""
    
    def __init__(self, model: Any):
        self._model = model
    
    def compute(
        self,
        X: pd.DataFrame,
        features: list[str],
        grid_resolution: int = 20
    ) -> dict[str, list[tuple[float, float]]]:
        """Compute partial dependence for features."""
        results = {}
        
        for feature in features:
            if feature not in X.columns:
                continue
            
            # Create grid
            values = X[feature].values
            grid = np.linspace(values.min(), values.max(), grid_resolution)
            
            pdp_values = []
            for val in grid:
                X_modified = X.copy()
                X_modified[feature] = val
                
                if hasattr(self._model, 'predict_proba'):
                    preds = self._model.predict_proba(X_modified)[:, 1]
                else:
                    preds = self._model.predict(X_modified)
                
                pdp_values.append((float(val), float(preds.mean())))
            
            results[feature] = pdp_values
        
        return results


# ============================================================================
# Explainability Engine
# ============================================================================

class ExplainabilityEngine:
    """
    Unified model explainability engine.
    
    Features:
    - SHAP values
    - Permutation importance
    - Partial dependence
    - Local explanations
    - Global explanations
    """
    
    def __init__(self):
        self._explainers: dict[str, Any] = {}
    
    def explain_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> GlobalExplanation:
        """Get global model explanation."""
        shap_exp = SHAPExplainer(model)
        shap_exp.fit(X)
        
        # Feature importance
        importance = shap_exp.get_global_importance(X)
        
        # Partial dependence for top features
        top_features = sorted(importance, key=importance.get, reverse=True)[:5]
        pdp_exp = PartialDependenceExplainer(model)
        pdp = pdp_exp.compute(X, top_features)
        
        return GlobalExplanation(
            feature_importance=importance,
            partial_dependence=pdp
        )
    
    def explain_prediction(
        self,
        model: Any,
        X: pd.DataFrame,
        row_index: int = 0
    ) -> ExplanationResult:
        """Explain a single prediction."""
        shap_exp = SHAPExplainer(model)
        shap_exp.fit(X)
        
        explanations = shap_exp.explain(X.iloc[[row_index]])
        return explanations[0] if explanations else ExplanationResult(
            prediction=None, base_value=0, contributions=[]
        )
    
    def explain_batch(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> list[ExplanationResult]:
        """Explain multiple predictions."""
        shap_exp = SHAPExplainer(model)
        shap_exp.fit(X)
        return shap_exp.explain(X)
    
    def get_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series = None,
        method: ExplainerType = ExplainerType.SHAP
    ) -> dict[str, float]:
        """Get feature importance using specified method."""
        if method == ExplainerType.SHAP:
            shap_exp = SHAPExplainer(model)
            shap_exp.fit(X)
            return shap_exp.get_global_importance(X)
        
        elif method == ExplainerType.PERMUTATION:
            if y is None:
                raise ValidationException("Target required for permutation importance")
            perm_exp = PermutationExplainer(model)
            return perm_exp.explain(X, y)
        
        elif method == ExplainerType.FEATURE_IMPORTANCE:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(X.columns, model.feature_importances_.tolist()))
            elif hasattr(model, 'coef_'):
                return dict(zip(X.columns, np.abs(model.coef_).flatten().tolist()))
            return {}
        
        return {}


# Factory function
def get_explainability_engine() -> ExplainabilityEngine:
    """Get explainability engine instance."""
    return ExplainabilityEngine()
