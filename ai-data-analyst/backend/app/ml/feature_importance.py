# AI Enterprise Data Analyst - Feature Importance Engine
# Production-grade feature importance analysis
# Handles: any ML model, multiple importance methods

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class ImportanceMethod(str, Enum):
    """Feature importance methods."""
    PERMUTATION = "permutation"
    TREE_BASED = "tree_based"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    COEFFICIENT = "coefficient"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FeatureImportance:
    """Single feature importance."""
    feature: str
    importance: float
    rank: int
    importance_pct: float
    std: Optional[float] = None


@dataclass
class FeatureImportanceResult:
    """Complete feature importance result."""
    method: ImportanceMethod
    n_features: int = 0
    
    # Rankings
    importances: List[FeatureImportance] = field(default_factory=list)
    
    # Top features
    top_features: List[str] = field(default_factory=list)
    
    # Cumulative importance
    top_n_for_80pct: int = 0
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "n_features": self.n_features,
            "top_features": self.top_features[:10],
            "top_n_for_80pct": self.top_n_for_80pct,
            "importances": [
                {
                    "feature": f.feature,
                    "importance": round(f.importance, 4),
                    "importance_pct": round(f.importance_pct, 2),
                    "rank": f.rank
                }
                for f in self.importances[:20]
            ]
        }


# ============================================================================
# Feature Importance Engine
# ============================================================================

class FeatureImportanceEngine:
    """
    Feature Importance engine.
    
    Features:
    - Multiple importance methods
    - Correlation-based importance
    - Permutation importance
    - Model-based importance
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        model: Any = None,
        method: ImportanceMethod = ImportanceMethod.CORRELATION
    ) -> FeatureImportanceResult:
        """Calculate feature importances."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Calculating feature importance using {method.value}")
        
        if method == ImportanceMethod.TREE_BASED and model is not None:
            importances = self._tree_based_importance(model, X.columns.tolist())
        elif method == ImportanceMethod.PERMUTATION and model is not None and y is not None:
            importances = self._permutation_importance(model, X, y)
        elif method == ImportanceMethod.MUTUAL_INFO and y is not None:
            importances = self._mutual_info_importance(X, y)
        elif method == ImportanceMethod.COEFFICIENT and model is not None:
            importances = self._coefficient_importance(model, X.columns.tolist())
        else:
            importances = self._correlation_importance(X, y)
        
        # Sort and rank
        importances.sort(key=lambda x: -x.importance)
        for i, f in enumerate(importances):
            f.rank = i + 1
        
        # Calculate percentage
        total = sum(f.importance for f in importances)
        for f in importances:
            f.importance_pct = f.importance / total * 100 if total > 0 else 0
        
        # Top features for 80%
        cumulative = 0
        top_n = 0
        for f in importances:
            cumulative += f.importance_pct
            top_n += 1
            if cumulative >= 80:
                break
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FeatureImportanceResult(
            method=method,
            n_features=len(importances),
            importances=importances,
            top_features=[f.feature for f in importances[:10]],
            top_n_for_80pct=top_n,
            processing_time_sec=processing_time
        )
    
    def _correlation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[FeatureImportance]:
        """Calculate correlation-based importance."""
        importances = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                continue
            
            corr = np.abs(X[col].corr(y)) if y is not None else 0
            
            if np.isnan(corr):
                corr = 0
            
            importances.append(FeatureImportance(
                feature=col,
                importance=corr,
                rank=0,
                importance_pct=0
            ))
        
        return importances
    
    def _tree_based_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Get tree-based model importance."""
        importances = []
        
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            
            for name, imp in zip(feature_names, fi):
                importances.append(FeatureImportance(
                    feature=name,
                    importance=float(imp),
                    rank=0,
                    importance_pct=0
                ))
        
        return importances
    
    def _coefficient_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Get coefficient-based importance (linear models)."""
        importances = []
        
        if hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_).flatten()
            
            for name, coef in zip(feature_names, coefs):
                importances.append(FeatureImportance(
                    feature=name,
                    importance=float(coef),
                    rank=0,
                    importance_pct=0
                ))
        
        return importances
    
    def _permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> List[FeatureImportance]:
        """Calculate permutation importance."""
        importances = []
        
        try:
            from sklearn.inspection import permutation_importance as perm_imp
            
            result = perm_imp(model, X, y, n_repeats=n_repeats, random_state=42)
            
            for name, mean, std in zip(X.columns, result.importances_mean, result.importances_std):
                importances.append(FeatureImportance(
                    feature=name,
                    importance=float(max(0, mean)),
                    rank=0,
                    importance_pct=0,
                    std=float(std)
                ))
        except:
            # Fall back to correlation
            importances = self._correlation_importance(X, y)
        
        return importances
    
    def _mutual_info_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[FeatureImportance]:
        """Calculate mutual information importance."""
        importances = []
        
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # Determine if classification or regression
            if y.nunique() <= 10:
                mi = mutual_info_classif(X.select_dtypes(include=[np.number]), y)
            else:
                mi = mutual_info_regression(X.select_dtypes(include=[np.number]), y)
            
            num_cols = X.select_dtypes(include=[np.number]).columns
            
            for name, score in zip(num_cols, mi):
                importances.append(FeatureImportance(
                    feature=name,
                    importance=float(score),
                    rank=0,
                    importance_pct=0
                ))
        except:
            importances = self._correlation_importance(X, y)
        
        return importances


# ============================================================================
# Factory Functions
# ============================================================================

def get_feature_importance_engine() -> FeatureImportanceEngine:
    """Get feature importance engine."""
    return FeatureImportanceEngine()


def quick_feature_importance(
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, Any]:
    """Quick feature importance analysis."""
    engine = FeatureImportanceEngine(verbose=False)
    result = engine.analyze(X, y)
    return result.to_dict()
