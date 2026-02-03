# AI Enterprise Data Analyst - Dimensionality Reduction
# PCA, t-SNE, UMAP, and feature selection techniques

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Dimensionality Reduction Types
# ============================================================================

class ReductionMethod(str, Enum):
    """Dimensionality reduction methods."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    LDA = "lda"
    SVD = "svd"
    FACTOR_ANALYSIS = "factor_analysis"
    ICA = "ica"


@dataclass
class ReductionResult:
    """Dimensionality reduction result."""
    
    method: ReductionMethod
    n_components: int
    
    # Transformed data
    transformed: np.ndarray
    
    # Explained variance (for PCA, SVD)
    explained_variance: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    cumulative_variance: Optional[np.ndarray] = None
    
    # Components (for interpretability)
    components: Optional[np.ndarray] = None
    feature_importance: dict[str, list[float]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "method": self.method.value,
            "n_components": self.n_components,
            "shape": list(self.transformed.shape)
        }
        
        if self.explained_variance_ratio is not None:
            result["explained_variance_ratio"] = [
                round(v, 4) for v in self.explained_variance_ratio
            ]
            result["total_variance_explained"] = round(
                sum(self.explained_variance_ratio), 4
            )
        
        if self.feature_importance:
            result["feature_importance"] = {
                k: [round(v, 4) for v in vals[:5]]
                for k, vals in self.feature_importance.items()
            }
        
        return result


# ============================================================================
# PCA
# ============================================================================

class PCAReducer:
    """Principal Component Analysis."""
    
    def __init__(self, n_components: int = None, variance_threshold: float = 0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self._model = None
        self._scaler = None
        self._feature_names = []
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        scale: bool = True
    ) -> ReductionResult:
        """Fit PCA and transform data."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        self._feature_names = X.columns.tolist()
        X_array = X.values
        
        # Scale data
        if scale:
            self._scaler = StandardScaler()
            X_array = self._scaler.fit_transform(X_array)
        
        # Determine n_components
        if self.n_components is None:
            # Use variance threshold
            pca_full = PCA()
            pca_full.fit(X_array)
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1
        
        # Fit PCA
        self._model = PCA(n_components=self.n_components)
        transformed = self._model.fit_transform(X_array)
        
        # Feature importance per component
        importance = {}
        for i in range(self.n_components):
            importance[f"PC{i+1}"] = [
                abs(self._model.components_[i, j])
                for j in range(len(self._feature_names))
            ]
        
        return ReductionResult(
            method=ReductionMethod.PCA,
            n_components=self.n_components,
            transformed=transformed,
            explained_variance=self._model.explained_variance_,
            explained_variance_ratio=self._model.explained_variance_ratio_,
            cumulative_variance=np.cumsum(self._model.explained_variance_ratio_),
            components=self._model.components_,
            feature_importance=importance
        )
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data."""
        X_array = X.values
        if self._scaler:
            X_array = self._scaler.transform(X_array)
        return self._model.transform(X_array)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Reconstruct original data."""
        X_original = self._model.inverse_transform(X_reduced)
        if self._scaler:
            X_original = self._scaler.inverse_transform(X_original)
        return X_original


# ============================================================================
# t-SNE
# ============================================================================

class TSNEReducer:
    """t-Distributed Stochastic Neighbor Embedding."""
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        scale: bool = True
    ) -> ReductionResult:
        """Fit t-SNE and transform data."""
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        X_array = X.values
        
        if scale:
            scaler = StandardScaler()
            X_array = scaler.fit_transform(X_array)
        
        # Adjust perplexity if needed
        perplexity = min(self.perplexity, len(X) - 1)
        
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            learning_rate=self.learning_rate,
            random_state=42,
            n_iter=1000
        )
        
        transformed = tsne.fit_transform(X_array)
        
        return ReductionResult(
            method=ReductionMethod.TSNE,
            n_components=self.n_components,
            transformed=transformed
        )


# ============================================================================
# UMAP
# ============================================================================

class UMAPReducer:
    """Uniform Manifold Approximation and Projection."""
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self._model = None
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        scale: bool = True
    ) -> ReductionResult:
        """Fit UMAP and transform data."""
        try:
            import umap
        except ImportError:
            logger.warning("UMAP not installed, falling back to t-SNE")
            return TSNEReducer(self.n_components).fit_transform(X, scale)
        
        from sklearn.preprocessing import StandardScaler
        
        X_array = X.values
        
        if scale:
            scaler = StandardScaler()
            X_array = scaler.fit_transform(X_array)
        
        self._model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            min_dist=self.min_dist,
            random_state=42
        )
        
        transformed = self._model.fit_transform(X_array)
        
        return ReductionResult(
            method=ReductionMethod.UMAP,
            n_components=self.n_components,
            transformed=transformed
        )
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data."""
        return self._model.transform(X.values)


# ============================================================================
# SVD
# ============================================================================

class SVDReducer:
    """Truncated Singular Value Decomposition."""
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self._model = None
    
    def fit_transform(self, X: pd.DataFrame) -> ReductionResult:
        """Fit SVD and transform data."""
        from sklearn.decomposition import TruncatedSVD
        
        self._model = TruncatedSVD(
            n_components=min(self.n_components, min(X.shape) - 1),
            random_state=42
        )
        
        transformed = self._model.fit_transform(X.values)
        
        return ReductionResult(
            method=ReductionMethod.SVD,
            n_components=self._model.n_components,
            transformed=transformed,
            explained_variance=self._model.explained_variance_,
            explained_variance_ratio=self._model.explained_variance_ratio_,
            components=self._model.components_
        )


# ============================================================================
# Factor Analysis
# ============================================================================

class FactorAnalysisReducer:
    """Factor Analysis for latent factor discovery."""
    
    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self._model = None
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        scale: bool = True
    ) -> ReductionResult:
        """Fit Factor Analysis."""
        from sklearn.decomposition import FactorAnalysis
        from sklearn.preprocessing import StandardScaler
        
        X_array = X.values
        
        if scale:
            scaler = StandardScaler()
            X_array = scaler.fit_transform(X_array)
        
        self._model = FactorAnalysis(
            n_components=min(self.n_factors, X.shape[1]),
            random_state=42
        )
        
        transformed = self._model.fit_transform(X_array)
        
        # Loadings
        loadings = self._model.components_
        
        return ReductionResult(
            method=ReductionMethod.FACTOR_ANALYSIS,
            n_components=self._model.n_components,
            transformed=transformed,
            components=loadings
        )


# ============================================================================
# Dimensionality Reduction Engine
# ============================================================================

class DimensionalityReductionEngine:
    """
    Unified dimensionality reduction engine.
    
    Features:
    - PCA with automatic component selection
    - t-SNE for visualization
    - UMAP for large datasets
    - SVD for sparse data
    - Factor Analysis
    - Optimal dimension recommendation
    """
    
    def __init__(self):
        self._reducers = {
            ReductionMethod.PCA: PCAReducer,
            ReductionMethod.TSNE: TSNEReducer,
            ReductionMethod.UMAP: UMAPReducer,
            ReductionMethod.SVD: SVDReducer,
            ReductionMethod.FACTOR_ANALYSIS: FactorAnalysisReducer
        }
    
    def reduce(
        self,
        df: pd.DataFrame,
        method: ReductionMethod = ReductionMethod.PCA,
        n_components: int = None,
        **kwargs
    ) -> ReductionResult:
        """Reduce dimensionality using specified method."""
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns for dimensionality reduction")
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Get reducer
        reducer_class = self._reducers.get(method)
        if not reducer_class:
            raise ValueError(f"Unknown method: {method}")
        
        if n_components:
            kwargs['n_components'] = n_components
        
        reducer = reducer_class(**kwargs)
        return reducer.fit_transform(numeric_df)
    
    def find_optimal_dimensions(
        self,
        df: pd.DataFrame,
        variance_threshold: float = 0.95
    ) -> dict[str, Any]:
        """Find optimal number of dimensions using PCA."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df.values)
        
        pca = PCA()
        pca.fit(X_scaled)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        # Find components for different thresholds
        thresholds = [0.80, 0.90, 0.95, 0.99]
        recommendations = {}
        
        for thresh in thresholds:
            n = np.argmax(cumsum >= thresh) + 1
            recommendations[f"{int(thresh*100)}%_variance"] = int(n)
        
        return {
            "original_dimensions": numeric_df.shape[1],
            "recommendations": recommendations,
            "optimal_for_threshold": int(np.argmax(cumsum >= variance_threshold) + 1),
            "variance_by_component": [
                round(v, 4) for v in pca.explained_variance_ratio_[:20]
            ],
            "cumulative_variance": [round(v, 4) for v in cumsum[:20]]
        }
    
    def visualize_2d(
        self,
        df: pd.DataFrame,
        method: ReductionMethod = ReductionMethod.TSNE,
        labels: pd.Series = None
    ) -> dict[str, Any]:
        """Reduce to 2D for visualization."""
        result = self.reduce(df, method, n_components=2)
        
        viz_data = {
            "x": result.transformed[:, 0].tolist(),
            "y": result.transformed[:, 1].tolist(),
            "method": method.value
        }
        
        if labels is not None:
            viz_data["labels"] = labels.tolist()
        
        return viz_data


# Factory function
def get_dimensionality_reduction_engine() -> DimensionalityReductionEngine:
    """Get dimensionality reduction engine instance."""
    return DimensionalityReductionEngine()
