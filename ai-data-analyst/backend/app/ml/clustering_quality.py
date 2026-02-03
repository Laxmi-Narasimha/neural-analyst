# AI Enterprise Data Analyst - Clustering Quality Engine
# Production-grade clustering quality assessment
# Handles: silhouette, calinski-harabasz, davies-bouldin, elbow method

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClusterMetric:
    """Single clustering quality metric."""
    name: str
    value: float
    interpretation: str
    optimal_direction: str  # higher, lower


@dataclass
class ElbowPoint:
    """Elbow analysis result."""
    k: int
    inertia: float
    is_elbow: bool


@dataclass
class ClusterQualityResult:
    """Complete clustering quality result."""
    n_clusters: int = 0
    n_samples: int = 0
    
    # Quality metrics
    silhouette_score: float = 0.0
    calinski_harabasz: float = 0.0
    davies_bouldin: float = 0.0
    
    # Cluster sizes
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    size_balance: float = 0.0  # 0-1, 1 = perfectly balanced
    
    # Elbow analysis
    elbow_data: List[ElbowPoint] = field(default_factory=list)
    suggested_k: int = 0
    
    # Overall quality
    quality_rating: str = ""  # excellent, good, fair, poor
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_clusters": self.n_clusters,
                "n_samples": self.n_samples,
                "quality_rating": self.quality_rating
            },
            "metrics": {
                "silhouette_score": round(self.silhouette_score, 4),
                "calinski_harabasz": round(self.calinski_harabasz, 2),
                "davies_bouldin": round(self.davies_bouldin, 4)
            },
            "cluster_sizes": self.cluster_sizes,
            "balance": round(self.size_balance, 4),
            "suggested_k": self.suggested_k
        }


# ============================================================================
# Clustering Quality Engine
# ============================================================================

class ClusteringQualityEngine:
    """
    Production-grade Clustering Quality engine.
    
    Features:
    - Multiple quality metrics
    - Elbow method
    - Cluster balance analysis
    - Optimal K suggestion
    - Overall quality rating
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def assess(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        include_elbow: bool = True,
        max_k: int = 10
    ) -> ClusterQualityResult:
        """Assess clustering quality."""
        start_time = datetime.now()
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        n_samples = len(labels)
        n_clusters = len(np.unique(labels))
        
        if self.verbose:
            logger.info(f"Assessing clustering: {n_samples} samples, {n_clusters} clusters")
        
        # Calculate metrics
        silhouette = self._silhouette_score(X_array, labels)
        calinski = self._calinski_harabasz(X_array, labels)
        davies = self._davies_bouldin(X_array, labels)
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}
        
        # Balance score
        size_balance = min(counts) / max(counts) if max(counts) > 0 else 0
        
        # Elbow analysis
        elbow_data = []
        suggested_k = n_clusters
        if include_elbow:
            elbow_data, suggested_k = self._elbow_analysis(X_array, max_k)
        
        # Quality rating
        quality_rating = self._rate_quality(silhouette, calinski, davies, size_balance)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ClusterQualityResult(
            n_clusters=n_clusters,
            n_samples=n_samples,
            silhouette_score=silhouette,
            calinski_harabasz=calinski,
            davies_bouldin=davies,
            cluster_sizes=cluster_sizes,
            size_balance=size_balance,
            elbow_data=elbow_data,
            suggested_k=suggested_k,
            quality_rating=quality_rating,
            processing_time_sec=processing_time
        )
    
    def _silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score."""
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(silhouette_score(X, labels))
        except ImportError:
            return self._silhouette_simple(X, labels)
    
    def _silhouette_simple(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Simple silhouette approximation."""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        # Sample for efficiency
        n_sample = min(1000, len(X))
        indices = np.random.choice(len(X), n_sample, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
        
        silhouettes = []
        
        for i, (x, label) in enumerate(zip(X_sample, labels_sample)):
            # Intra-cluster distance (a)
            same_cluster = X_sample[labels_sample == label]
            if len(same_cluster) > 1:
                a = np.mean([np.linalg.norm(x - c) for c in same_cluster if not np.array_equal(x, c)])
            else:
                a = 0
            
            # Nearest cluster distance (b)
            b = float('inf')
            for other_label in unique_labels:
                if other_label != label:
                    other_cluster = X_sample[labels_sample == other_label]
                    if len(other_cluster) > 0:
                        dist = np.mean([np.linalg.norm(x - c) for c in other_cluster])
                        b = min(b, dist)
            
            if b == float('inf'):
                b = 0
            
            # Silhouette
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouettes.append(s)
        
        return float(np.mean(silhouettes))
    
    def _calinski_harabasz(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz index."""
        try:
            from sklearn.metrics import calinski_harabasz_score
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(calinski_harabasz_score(X, labels))
        except ImportError:
            return self._calinski_simple(X, labels)
    
    def _calinski_simple(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Simple Calinski-Harabasz approximation."""
        n = len(X)
        k = len(np.unique(labels))
        
        if k < 2:
            return 0.0
        
        # Overall centroid
        overall_centroid = np.mean(X, axis=0)
        
        # Between-cluster dispersion
        b = 0
        for label in np.unique(labels):
            cluster = X[labels == label]
            n_k = len(cluster)
            centroid = np.mean(cluster, axis=0)
            b += n_k * np.sum((centroid - overall_centroid) ** 2)
        
        # Within-cluster dispersion
        w = 0
        for label in np.unique(labels):
            cluster = X[labels == label]
            centroid = np.mean(cluster, axis=0)
            w += np.sum((cluster - centroid) ** 2)
        
        if w == 0:
            return 0.0
        
        return float((b / (k - 1)) / (w / (n - k)))
    
    def _davies_bouldin(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin index."""
        try:
            from sklearn.metrics import davies_bouldin_score
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(davies_bouldin_score(X, labels))
        except ImportError:
            return 0.0  # Complex to implement from scratch
    
    def _elbow_analysis(
        self,
        X: np.ndarray,
        max_k: int
    ) -> Tuple[List[ElbowPoint], int]:
        """Perform elbow analysis."""
        try:
            from sklearn.cluster import KMeans
            
            inertias = []
            k_range = range(2, min(max_k + 1, len(X)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append((k, kmeans.inertia_))
            
            # Find elbow point
            elbow_data = []
            suggested_k = 2
            
            if len(inertias) > 2:
                # Calculate rate of change
                changes = []
                for i in range(1, len(inertias)):
                    change = (inertias[i-1][1] - inertias[i][1]) / inertias[i-1][1]
                    changes.append(change)
                
                # Elbow is where change slows significantly
                for i, (k, inertia) in enumerate(inertias):
                    is_elbow = False
                    if i > 0 and i < len(changes):
                        if changes[i-1] > 0.1 and (i >= len(changes) or changes[i] < changes[i-1] * 0.5):
                            is_elbow = True
                            suggested_k = k
                    
                    elbow_data.append(ElbowPoint(
                        k=k,
                        inertia=inertia,
                        is_elbow=is_elbow
                    ))
            
            return elbow_data, suggested_k
            
        except ImportError:
            return [], 3
    
    def _rate_quality(
        self,
        silhouette: float,
        calinski: float,
        davies: float,
        balance: float
    ) -> str:
        """Rate overall clustering quality."""
        score = 0
        
        # Silhouette: -1 to 1, higher is better
        if silhouette > 0.7:
            score += 3
        elif silhouette > 0.5:
            score += 2
        elif silhouette > 0.25:
            score += 1
        
        # Balance: 0 to 1, higher is better
        if balance > 0.7:
            score += 2
        elif balance > 0.3:
            score += 1
        
        # Davies-Bouldin: lower is better
        if davies < 0.5:
            score += 2
        elif davies < 1.0:
            score += 1
        
        if score >= 6:
            return "excellent"
        elif score >= 4:
            return "good"
        elif score >= 2:
            return "fair"
        return "poor"


# ============================================================================
# Factory Functions
# ============================================================================

def get_clustering_quality_engine() -> ClusteringQualityEngine:
    """Get clustering quality engine."""
    return ClusteringQualityEngine()


def assess_clustering(
    X: pd.DataFrame,
    labels: np.ndarray
) -> Dict[str, Any]:
    """Quick clustering assessment."""
    engine = ClusteringQualityEngine(verbose=False)
    result = engine.assess(X, labels)
    return result.to_dict()


def find_optimal_k(
    X: pd.DataFrame,
    max_k: int = 10
) -> int:
    """Find optimal number of clusters using elbow method."""
    engine = ClusteringQualityEngine(verbose=False)
    # Create dummy labels for elbow analysis
    dummy_labels = np.zeros(len(X))
    result = engine.assess(X, dummy_labels, include_elbow=True, max_k=max_k)
    return result.suggested_k
