# AI Enterprise Data Analyst - Advanced Segmentation Engine
# Production-grade clustering/segmentation for ANY data
# Handles: missing values, mixed types, auto k selection, multiple algorithms

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist

try:
    from sklearn.cluster import (
        KMeans, MiniBatchKMeans, AgglomerativeClustering, 
        DBSCAN, OPTICS, SpectralClustering, Birch, MeanShift
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
        adjusted_rand_score, normalized_mutual_info_score
    )
    from sklearn.neighbors import NearestNeighbors
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

class ClusteringMethod(str, Enum):
    """Clustering algorithms."""
    KMEANS = "kmeans"
    KMEANS_MINIBATCH = "kmeans_minibatch"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    OPTICS = "optics"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    SPECTRAL = "spectral"
    BIRCH = "birch"
    MEAN_SHIFT = "mean_shift"
    AUTO = "auto"


class KSelectionMethod(str, Enum):
    """Methods to select optimal K."""
    ELBOW = "elbow"
    SILHOUETTE = "silhouette"
    GAP_STATISTIC = "gap_statistic"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClusterProfile:
    """Profile of a single cluster."""
    cluster_id: int
    n_samples: int
    percentage: float
    
    # Centroid / representative values
    centroid: Dict[str, float] = field(default_factory=dict)
    
    # Statistics per feature
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Distinguishing features
    distinguishing_features: List[Tuple[str, float, str]] = field(default_factory=list)
    
    # Label (auto-generated or user-defined)
    label: str = ""


@dataclass
class SegmentationResult:
    """Complete segmentation result."""
    method: ClusteringMethod
    n_clusters: int
    n_samples: int
    
    # Cluster assignments
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Cluster profiles
    cluster_profiles: List[ClusterProfile] = field(default_factory=list)
    
    # Quality metrics
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    inertia: Optional[float] = None
    
    # K selection info
    k_selection_method: Optional[KSelectionMethod] = None
    k_selection_scores: Dict[int, float] = field(default_factory=dict)
    
    # Feature importance for clustering
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Dimensionality reduction for visualization
    reduced_2d: Optional[np.ndarray] = None
    
    # Timing
    preprocessing_time_sec: float = 0.0
    clustering_time_sec: float = 0.0
    total_time_sec: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "n_samples": self.n_samples,
            "quality_metrics": {
                "silhouette": round(self.silhouette_score, 4) if self.silhouette_score else None,
                "calinski_harabasz": round(self.calinski_harabasz_score, 2) if self.calinski_harabasz_score else None,
                "davies_bouldin": round(self.davies_bouldin_score, 4) if self.davies_bouldin_score else None
            },
            "cluster_profiles": [
                {
                    "id": cp.cluster_id,
                    "n_samples": cp.n_samples,
                    "percentage": round(cp.percentage, 1),
                    "label": cp.label,
                    "distinguishing_features": cp.distinguishing_features[:5],
                    "centroid": {k: round(v, 4) for k, v in cp.centroid.items()}
                }
                for cp in self.cluster_profiles
            ],
            "k_selection": {
                "method": self.k_selection_method.value if self.k_selection_method else None,
                "scores": {str(k): round(v, 4) for k, v in self.k_selection_scores.items()}
            },
            "feature_importance": {k: round(v, 4) for k, v in 
                                  sorted(self.feature_importance.items(), key=lambda x: -x[1])[:10]},
            "timing": {
                "total_sec": round(self.total_time_sec, 2)
            },
            "warnings": self.warnings[:5]
        }


@dataclass
class SegmentationConfig:
    """Configuration for segmentation."""
    method: ClusteringMethod = ClusteringMethod.AUTO
    n_clusters: Optional[int] = None  # Auto-select if None
    
    # K selection
    k_selection_method: KSelectionMethod = KSelectionMethod.SILHOUETTE
    k_min: int = 2
    k_max: int = 10
    
    # Preprocessing
    scale_features: bool = True
    handle_missing: bool = True
    
    # DBSCAN specific
    eps: Optional[float] = None  # Auto-estimate if None
    min_samples: int = 5
    
    # Gaussian Mixture specific
    covariance_type: str = "full"
    
    # Output
    generate_profiles: bool = True
    reduce_dimensions: bool = True
    
    # Performance
    random_state: int = 42
    n_jobs: int = -1


# ============================================================================  
# Preprocessing
# ============================================================================

class SegmentationPreprocessor:
    """Preprocess data for clustering."""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self._scaler = None
        self._feature_names = []
        self._original_columns = []
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for clustering."""
        # Select columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self._original_columns = columns
        
        if not columns:
            raise DataProcessingException("No numeric columns for clustering")
        
        # Extract data
        data = df[columns].copy()
        
        # Handle missing values
        if self.config.handle_missing:
            for col in columns:
                if data[col].isna().any():
                    # Use median for imputation
                    data[col] = data[col].fillna(data[col].median())
        
        # Convert to numpy
        X = data.values
        self._feature_names = columns
        
        # Scale
        if self.config.scale_features:
            self._scaler = RobustScaler()
            X = self._scaler.fit_transform(X)
        
        return X, columns
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data."""
        data = df[self._original_columns].copy()
        
        for col in self._original_columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].median())
        
        X = data.values
        
        if self._scaler:
            X = self._scaler.transform(X)
        
        return X


# ============================================================================
# K Selection
# ============================================================================

class OptimalKSelector:
    """Select optimal number of clusters."""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
    
    def find_optimal_k(
        self,
        X: np.ndarray,
        method: KSelectionMethod = None
    ) -> Tuple[int, Dict[int, float]]:
        """Find optimal K using specified method."""
        method = method or self.config.k_selection_method
        k_range = range(self.config.k_min, min(self.config.k_max + 1, len(X)))
        
        if method == KSelectionMethod.SILHOUETTE:
            return self._silhouette_method(X, k_range)
        elif method == KSelectionMethod.ELBOW:
            return self._elbow_method(X, k_range)
        elif method == KSelectionMethod.CALINSKI_HARABASZ:
            return self._calinski_harabasz_method(X, k_range)
        elif method == KSelectionMethod.DAVIES_BOULDIN:
            return self._davies_bouldin_method(X, k_range)
        else:
            return self._silhouette_method(X, k_range)
    
    def _silhouette_method(
        self,
        X: np.ndarray,
        k_range: range
    ) -> Tuple[int, Dict[int, float]]:
        """Select K using silhouette score."""
        scores = {}
        
        for k in k_range:
            if k >= len(X):
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                scores[k] = score
            except:
                continue
        
        if not scores:
            return self.config.k_min, {}
        
        optimal_k = max(scores, key=scores.get)
        return optimal_k, scores
    
    def _elbow_method(
        self,
        X: np.ndarray,
        k_range: range
    ) -> Tuple[int, Dict[int, float]]:
        """Select K using elbow method."""
        inertias = {}
        
        for k in k_range:
            if k >= len(X):
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                kmeans.fit(X)
                inertias[k] = kmeans.inertia_
            except:
                continue
        
        if len(inertias) < 3:
            return self.config.k_min, inertias
        
        # Find elbow using second derivative
        k_vals = sorted(inertias.keys())
        inertia_vals = [inertias[k] for k in k_vals]
        
        # Calculate second derivative
        first_deriv = np.diff(inertia_vals)
        second_deriv = np.diff(first_deriv)
        
        # Elbow is where second derivative is maximum
        if len(second_deriv) > 0:
            elbow_idx = np.argmax(second_deriv) + 1
            optimal_k = k_vals[elbow_idx]
        else:
            optimal_k = k_vals[len(k_vals) // 2]
        
        return optimal_k, inertias
    
    def _calinski_harabasz_method(
        self,
        X: np.ndarray,
        k_range: range
    ) -> Tuple[int, Dict[int, float]]:
        """Select K using Calinski-Harabasz index."""
        scores = {}
        
        for k in k_range:
            if k >= len(X):
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                score = calinski_harabasz_score(X, labels)
                scores[k] = score
            except:
                continue
        
        if not scores:
            return self.config.k_min, {}
        
        optimal_k = max(scores, key=scores.get)
        return optimal_k, scores
    
    def _davies_bouldin_method(
        self,
        X: np.ndarray,
        k_range: range
    ) -> Tuple[int, Dict[int, float]]:
        """Select K using Davies-Bouldin index (lower is better)."""
        scores = {}
        
        for k in k_range:
            if k >= len(X):
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                score = davies_bouldin_score(X, labels)
                scores[k] = score
            except:
                continue
        
        if not scores:
            return self.config.k_min, {}
        
        optimal_k = min(scores, key=scores.get)
        return optimal_k, scores


# ============================================================================
# Clustering Algorithms
# ============================================================================

class ClusteringAlgorithms:
    """Collection of clustering algorithms."""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
    
    def kmeans(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
        """K-Means clustering."""
        model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=10,
            max_iter=300
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def kmeans_minibatch(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
        """Mini-batch K-Means for large datasets."""
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            batch_size=min(1000, len(X))
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def hierarchical(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
        """Hierarchical/Agglomerative clustering."""
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def dbscan(self, X: np.ndarray) -> Tuple[np.ndarray, Any]:
        """DBSCAN density-based clustering."""
        eps = self.config.eps
        
        # Auto-estimate eps if not provided
        if eps is None:
            eps = self._estimate_eps(X)
        
        model = DBSCAN(
            eps=eps,
            min_samples=self.config.min_samples,
            n_jobs=self.config.n_jobs
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def optics(self, X: np.ndarray) -> Tuple[np.ndarray, Any]:
        """OPTICS clustering."""
        model = OPTICS(
            min_samples=self.config.min_samples,
            n_jobs=self.config.n_jobs
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def gaussian_mixture(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
        """Gaussian Mixture Model."""
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=self.config.covariance_type,
            random_state=self.config.random_state
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def spectral(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
        """Spectral clustering."""
        # Limit samples for spectral (memory intensive)
        if len(X) > 5000:
            logger.warning("Spectral clustering on >5000 samples may be slow")
        
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            affinity='nearest_neighbors'
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def birch(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
        """BIRCH clustering."""
        model = Birch(
            n_clusters=n_clusters,
            threshold=0.5
        )
        labels = model.fit_predict(X)
        return labels, model
    
    def _estimate_eps(self, X: np.ndarray, k: int = 4) -> float:
        """Estimate eps for DBSCAN using k-distance graph."""
        k = min(k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        
        # Use the "elbow" of sorted k-distances
        k_distances = np.sort(distances[:, -1])
        
        # Simple heuristic: use 90th percentile
        eps = np.percentile(k_distances, 90)
        
        return eps


# ============================================================================
# Cluster Profiler
# ============================================================================

class ClusterProfiler:
    """Generate detailed cluster profiles."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def generate_profiles(
        self,
        X_original: np.ndarray,
        labels: np.ndarray
    ) -> List[ClusterProfile]:
        """Generate profile for each cluster."""
        unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
        profiles = []
        
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_data = X_original[mask]
            
            profile = ClusterProfile(
                cluster_id=int(cluster_id),
                n_samples=int(mask.sum()),
                percentage=mask.sum() / len(labels) * 100
            )
            
            # Calculate centroid and stats per feature
            for i, feat in enumerate(self.feature_names):
                feat_data = cluster_data[:, i]
                
                profile.centroid[feat] = float(np.mean(feat_data))
                profile.feature_stats[feat] = {
                    "mean": float(np.mean(feat_data)),
                    "std": float(np.std(feat_data)),
                    "median": float(np.median(feat_data)),
                    "min": float(np.min(feat_data)),
                    "max": float(np.max(feat_data))
                }
            
            # Find distinguishing features
            profile.distinguishing_features = self._find_distinguishing(
                X_original, labels, cluster_id
            )
            
            # Auto-generate label
            profile.label = self._generate_label(profile)
            
            profiles.append(profile)
        
        return profiles
    
    def _find_distinguishing(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        cluster_id: int
    ) -> List[Tuple[str, float, str]]:
        """Find features that distinguish this cluster from others."""
        mask = labels == cluster_id
        cluster_data = X[mask]
        other_data = X[~mask & (labels >= 0)]
        
        if len(other_data) == 0:
            return []
        
        distinguishing = []
        
        for i, feat in enumerate(self.feature_names):
            cluster_mean = np.mean(cluster_data[:, i])
            other_mean = np.mean(other_data[:, i])
            other_std = np.std(other_data[:, i])
            
            if other_std > 0:
                z_diff = (cluster_mean - other_mean) / other_std
                
                if abs(z_diff) > 0.5:
                    direction = "higher" if z_diff > 0 else "lower"
                    distinguishing.append((feat, abs(z_diff), direction))
        
        # Sort by importance
        distinguishing.sort(key=lambda x: -x[1])
        
        return distinguishing[:5]
    
    def _generate_label(self, profile: ClusterProfile) -> str:
        """Generate descriptive label for cluster."""
        if not profile.distinguishing_features:
            return f"Cluster {profile.cluster_id}"
        
        top_feat, score, direction = profile.distinguishing_features[0]
        return f"{direction.capitalize()} {top_feat} Group"


# ============================================================================
# Advanced Segmentation Engine
# ============================================================================

class AdvancedSegmentationEngine:
    """
    Advanced segmentation engine for ANY data.
    
    Features:
    - Auto-selects optimal K
    - Handles missing values
    - Multiple clustering algorithms
    - Detailed cluster profiles
    - Quality metrics
    - Dimensionality reduction for visualization
    """
    
    def __init__(self, config: SegmentationConfig = None, verbose: bool = True):
        self.config = config or SegmentationConfig()
        self.verbose = verbose
        self.preprocessor = SegmentationPreprocessor(self.config)
        self.k_selector = OptimalKSelector(self.config)
        self.algorithms = ClusteringAlgorithms(self.config)
    
    def segment(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        n_clusters: int = None,
        method: ClusteringMethod = None,
        auto_k: bool = False,
        max_k: Optional[int] = None,
    ) -> SegmentationResult:
        """
        Segment data into clusters.
        
        Handles ANY numeric data automatically.
        """
        total_start = datetime.now()
        method = method or self.config.method
        
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for clustering")
        
        warnings_list = []
        
        # Preprocess
        preprocess_start = datetime.now()
        X, feature_names = self.preprocessor.fit_transform(df, columns)
        X_original = df[feature_names].values
        preprocess_time = (datetime.now() - preprocess_start).total_seconds()
        
        if self.verbose:
            logger.info(f"Preprocessed {X.shape[0]} samples, {X.shape[1]} features")
        
        # Handle small datasets
        if len(X) < 10:
            warnings_list.append("Very small dataset may yield unreliable clusters")
        
        # Auto-select method if needed
        if method == ClusteringMethod.AUTO:
            method = self._select_method(X)
            if self.verbose:
                logger.info(f"Auto-selected method: {method.value}")
        
        # Determine K
        k_scores = {}
        needs_k = method not in [ClusteringMethod.DBSCAN, ClusteringMethod.OPTICS, ClusteringMethod.MEAN_SHIFT]

        if max_k is not None:
            try:
                max_k_int = int(max_k)
            except Exception:
                max_k_int = None
        else:
            max_k_int = None

        if (auto_k or n_clusters is None) and needs_k:
            if self.verbose:
                logger.info("Finding optimal K...")

            original_k_max = self.config.k_max
            if max_k_int is not None:
                self.config.k_max = max_k_int
            try:
                n_clusters, k_scores = self.k_selector.find_optimal_k(X)
            finally:
                self.config.k_max = original_k_max

            if self.verbose:
                logger.info(f"Optimal K: {n_clusters}")
        elif n_clusters is None:
            n_clusters = 0  # Will be determined by algorithm
        
        # Run clustering
        cluster_start = datetime.now()
        labels, model = self._run_clustering(X, n_clusters, method)
        cluster_time = (datetime.now() - cluster_start).total_seconds()
        
        # Calculate metrics
        n_clusters_actual = len(np.unique(labels[labels >= 0]))
        
        silhouette = None
        calinski = None
        davies = None
        
        if n_clusters_actual >= 2:
            try:
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
                davies = davies_bouldin_score(X, labels)
            except:
                pass
        
        # Generate profiles
        profiles = []
        if self.config.generate_profiles:
            profiler = ClusterProfiler(feature_names)
            profiles = profiler.generate_profiles(X_original, labels)
        
        # Feature importance
        feature_importance = self._calculate_feature_importance(X, labels, feature_names)
        
        # Dimensionality reduction
        reduced_2d = None
        if self.config.reduce_dimensions and X.shape[1] > 2:
            try:
                if len(X) > 10000:
                    # Use PCA for large datasets
                    reducer = PCA(n_components=2, random_state=self.config.random_state)
                else:
                    reducer = PCA(n_components=2, random_state=self.config.random_state)
                reduced_2d = reducer.fit_transform(X)
            except:
                pass
        
        total_time = (datetime.now() - total_start).total_seconds()
        
        return SegmentationResult(
            method=method,
            n_clusters=n_clusters_actual,
            n_samples=len(X),
            labels=labels,
            cluster_profiles=profiles,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski,
            davies_bouldin_score=davies,
            inertia=getattr(model, 'inertia_', None),
            k_selection_method=self.config.k_selection_method if k_scores else None,
            k_selection_scores=k_scores,
            feature_importance=feature_importance,
            reduced_2d=reduced_2d,
            preprocessing_time_sec=preprocess_time,
            clustering_time_sec=cluster_time,
            total_time_sec=total_time,
            warnings=warnings_list
        )
    
    def _select_method(self, X: np.ndarray) -> ClusteringMethod:
        """Auto-select clustering method based on data."""
        n_samples, n_features = X.shape
        
        # Large datasets
        if n_samples > 50000:
            return ClusteringMethod.KMEANS_MINIBATCH
        
        # Medium-large
        if n_samples > 10000:
            return ClusteringMethod.KMEANS
        
        # High dimensional
        if n_features > 50:
            return ClusteringMethod.GAUSSIAN_MIXTURE
        
        # Default
        return ClusteringMethod.KMEANS
    
    def _run_clustering(
        self,
        X: np.ndarray,
        n_clusters: int,
        method: ClusteringMethod
    ) -> Tuple[np.ndarray, Any]:
        """Run the selected clustering algorithm."""
        if method == ClusteringMethod.KMEANS:
            return self.algorithms.kmeans(X, n_clusters)
        elif method == ClusteringMethod.KMEANS_MINIBATCH:
            return self.algorithms.kmeans_minibatch(X, n_clusters)
        elif method == ClusteringMethod.HIERARCHICAL:
            return self.algorithms.hierarchical(X, n_clusters)
        elif method == ClusteringMethod.DBSCAN:
            return self.algorithms.dbscan(X)
        elif method == ClusteringMethod.OPTICS:
            return self.algorithms.optics(X)
        elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
            return self.algorithms.gaussian_mixture(X, n_clusters)
        elif method == ClusteringMethod.SPECTRAL:
            return self.algorithms.spectral(X, n_clusters)
        elif method == ClusteringMethod.BIRCH:
            return self.algorithms.birch(X, n_clusters)
        else:
            return self.algorithms.kmeans(X, n_clusters)
    
    def _calculate_feature_importance(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance for clustering."""
        importance = {}
        unique_labels = np.unique(labels[labels >= 0])
        
        if len(unique_labels) < 2:
            return {f: 1.0 / len(feature_names) for f in feature_names}
        
        for i, feat in enumerate(feature_names):
            # F-statistic between clusters
            groups = [X[labels == k, i] for k in unique_labels]
            try:
                f_stat, p_val = scipy_stats.f_oneway(*groups)
                importance[feat] = f_stat if not np.isnan(f_stat) else 0
            except:
                importance[feat] = 0
        
        # Normalize
        max_imp = max(importance.values()) or 1
        importance = {k: v / max_imp for k, v in importance.items()}
        
        return importance


# ============================================================================
# Factory Functions
# ============================================================================

def get_segmentation_engine(config: SegmentationConfig = None) -> AdvancedSegmentationEngine:
    """Get a segmentation engine."""
    return AdvancedSegmentationEngine(config=config)


def quick_segment(
    df: pd.DataFrame,
    n_clusters: int = None,
    columns: List[str] = None
) -> Dict[str, Any]:
    """
    Quick segmentation on any data.
    
    Example:
        result = quick_segment(df, n_clusters=5)
        print(result['cluster_profiles'])
    """
    engine = AdvancedSegmentationEngine(verbose=False)
    result = engine.segment(df, columns=columns, n_clusters=n_clusters)
    return result.to_dict()
