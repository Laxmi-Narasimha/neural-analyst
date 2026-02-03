# AI Enterprise Data Analyst - Segmentation Engine
# Customer/data segmentation using clustering and rule-based approaches

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Segmentation Types
# ============================================================================

class SegmentationMethod(str, Enum):
    """Segmentation methods."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    RULE_BASED = "rule_based"


@dataclass
class Segment:
    """Single segment definition."""
    
    segment_id: int
    name: str
    size: int
    percentage: float
    
    centroid: Optional[dict[str, float]] = None
    profile: dict[str, Any] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "size": self.size,
            "percentage": round(self.percentage, 2),
            "profile": self.profile,
            "rules": self.rules
        }


@dataclass
class SegmentationResult:
    """Segmentation analysis result."""
    
    method: SegmentationMethod
    n_segments: int
    segments: list[Segment]
    labels: np.ndarray
    
    # Quality metrics
    silhouette_score: float = 0.0
    davies_bouldin: float = 0.0
    calinski_harabasz: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "n_segments": self.n_segments,
            "segments": [s.to_dict() for s in self.segments],
            "quality_metrics": {
                "silhouette_score": round(self.silhouette_score, 4),
                "davies_bouldin": round(self.davies_bouldin, 4),
                "calinski_harabasz": round(self.calinski_harabasz, 4)
            }
        }


# ============================================================================
# Clustering Segmentation
# ============================================================================

class ClusteringSegmentation:
    """Clustering-based segmentation."""
    
    def __init__(self, method: SegmentationMethod = SegmentationMethod.KMEANS):
        self.method = method
        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []
    
    def fit_predict(
        self,
        df: pd.DataFrame,
        n_segments: int = 5,
        features: list[str] = None
    ) -> SegmentationResult:
        """Perform segmentation."""
        # Select features
        features = features or df.select_dtypes(include=[np.number]).columns.tolist()
        self._feature_names = features
        
        X = df[features].fillna(0).values
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Fit clustering
        labels = self._fit_clustering(X_scaled, n_segments)
        
        # Calculate quality metrics
        metrics = self._calculate_metrics(X_scaled, labels)
        
        # Build segments
        segments = self._build_segments(df, features, labels)
        
        return SegmentationResult(
            method=self.method,
            n_segments=len(set(labels)),
            segments=segments,
            labels=labels,
            silhouette_score=metrics.get("silhouette", 0),
            davies_bouldin=metrics.get("davies_bouldin", 0),
            calinski_harabasz=metrics.get("calinski_harabasz", 0)
        )
    
    def _fit_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Fit clustering algorithm."""
        if self.method == SegmentationMethod.KMEANS:
            from sklearn.cluster import KMeans
            self._model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return self._model.fit_predict(X)
        
        elif self.method == SegmentationMethod.DBSCAN:
            from sklearn.cluster import DBSCAN
            self._model = DBSCAN(eps=0.5, min_samples=5)
            return self._model.fit_predict(X)
        
        elif self.method == SegmentationMethod.HIERARCHICAL:
            from sklearn.cluster import AgglomerativeClustering
            self._model = AgglomerativeClustering(n_clusters=n_clusters)
            return self._model.fit_predict(X)
        
        elif self.method == SegmentationMethod.GAUSSIAN_MIXTURE:
            from sklearn.mixture import GaussianMixture
            self._model = GaussianMixture(n_components=n_clusters, random_state=42)
            return self._model.fit_predict(X)
        
        else:
            from sklearn.cluster import KMeans
            self._model = KMeans(n_clusters=n_clusters, random_state=42)
            return self._model.fit_predict(X)
    
    def _calculate_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> dict[str, float]:
        """Calculate clustering quality metrics."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        
        n_unique = len(set(labels)) - (1 if -1 in labels else 0)
        if n_unique > 1:
            try:
                metrics["silhouette"] = float(silhouette_score(X, labels))
                metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            except:
                pass
        
        return metrics
    
    def _build_segments(
        self,
        df: pd.DataFrame,
        features: list[str],
        labels: np.ndarray
    ) -> list[Segment]:
        """Build segment profiles."""
        segments = []
        total = len(df)
        
        for label in sorted(set(labels)):
            if label == -1:  # Noise in DBSCAN
                continue
            
            mask = labels == label
            segment_df = df[mask]
            
            # Calculate profile (mean of each feature)
            profile = {}
            for feat in features:
                profile[feat] = {
                    "mean": float(segment_df[feat].mean()),
                    "std": float(segment_df[feat].std()),
                    "min": float(segment_df[feat].min()),
                    "max": float(segment_df[feat].max())
                }
            
            segments.append(Segment(
                segment_id=int(label),
                name=f"Segment {label + 1}",
                size=int(mask.sum()),
                percentage=mask.sum() / total * 100,
                centroid={f: profile[f]["mean"] for f in features},
                profile=profile
            ))
        
        return segments
    
    def find_optimal_k(
        self,
        df: pd.DataFrame,
        features: list[str] = None,
        max_k: int = 10
    ) -> dict[str, Any]:
        """Find optimal number of clusters using elbow method."""
        features = features or df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[features].fillna(0).values
        
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        
        from sklearn.cluster import KMeans
        
        inertias = []
        silhouettes = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            
            from sklearn.metrics import silhouette_score
            silhouettes.append(silhouette_score(X_scaled, labels))
        
        # Find elbow (max silhouette)
        optimal_k = silhouettes.index(max(silhouettes)) + 2
        
        return {
            "optimal_k": optimal_k,
            "k_values": list(range(2, max_k + 1)),
            "inertias": inertias,
            "silhouettes": silhouettes
        }


# ============================================================================
# Rule-Based Segmentation
# ============================================================================

class RuleBasedSegmentation:
    """Rule-based segmentation."""
    
    def __init__(self):
        self._rules: list[dict] = []
    
    def add_rule(
        self,
        name: str,
        condition: str,
        segment_id: int
    ) -> None:
        """Add segmentation rule."""
        self._rules.append({
            "name": name,
            "condition": condition,
            "segment_id": segment_id
        })
    
    def segment(self, df: pd.DataFrame) -> SegmentationResult:
        """Apply rules to segment data."""
        labels = np.full(len(df), -1)  # Unassigned
        
        for rule in self._rules:
            try:
                mask = df.eval(rule["condition"])
                labels[mask & (labels == -1)] = rule["segment_id"]
            except Exception as e:
                logger.warning(f"Rule {rule['name']} failed: {e}")
        
        # Build segments
        segments = []
        total = len(df)
        
        for rule in self._rules:
            mask = labels == rule["segment_id"]
            if mask.sum() > 0:
                segments.append(Segment(
                    segment_id=rule["segment_id"],
                    name=rule["name"],
                    size=int(mask.sum()),
                    percentage=mask.sum() / total * 100,
                    rules=[rule["condition"]]
                ))
        
        # Unassigned segment
        unassigned = labels == -1
        if unassigned.sum() > 0:
            segments.append(Segment(
                segment_id=-1,
                name="Unassigned",
                size=int(unassigned.sum()),
                percentage=unassigned.sum() / total * 100
            ))
        
        return SegmentationResult(
            method=SegmentationMethod.RULE_BASED,
            n_segments=len(segments),
            segments=segments,
            labels=labels
        )


# ============================================================================
# Segmentation Engine
# ============================================================================

class SegmentationEngine:
    """
    Unified segmentation engine.
    
    Features:
    - K-Means, DBSCAN, Hierarchical clustering
    - Gaussian Mixture Models
    - Rule-based segmentation
    - Optimal k selection
    - Segment profiling
    """
    
    def __init__(self):
        self._clustering = ClusteringSegmentation()
        self._rule_based = RuleBasedSegmentation()
    
    def segment(
        self,
        df: pd.DataFrame,
        method: SegmentationMethod = SegmentationMethod.KMEANS,
        n_segments: int = 5,
        features: list[str] = None
    ) -> SegmentationResult:
        """Perform segmentation."""
        if method == SegmentationMethod.RULE_BASED:
            return self._rule_based.segment(df)
        
        self._clustering.method = method
        return self._clustering.fit_predict(df, n_segments, features)
    
    def add_segmentation_rule(
        self,
        name: str,
        condition: str,
        segment_id: int
    ) -> None:
        """Add rule for rule-based segmentation."""
        self._rule_based.add_rule(name, condition, segment_id)
    
    def find_optimal_segments(
        self,
        df: pd.DataFrame,
        features: list[str] = None,
        max_k: int = 10
    ) -> dict[str, Any]:
        """Find optimal number of segments."""
        return self._clustering.find_optimal_k(df, features, max_k)
    
    def profile_segment(
        self,
        df: pd.DataFrame,
        segment_labels: np.ndarray,
        segment_id: int
    ) -> dict[str, Any]:
        """Get detailed profile for a segment."""
        mask = segment_labels == segment_id
        segment_df = df[mask]
        
        profile = {
            "size": int(mask.sum()),
            "percentage": mask.sum() / len(df) * 100
        }
        
        # Numeric features
        numeric_cols = segment_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile[col] = {
                "mean": float(segment_df[col].mean()),
                "median": float(segment_df[col].median()),
                "std": float(segment_df[col].std()),
                "min": float(segment_df[col].min()),
                "max": float(segment_df[col].max())
            }
        
        # Categorical features
        cat_cols = segment_df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            value_counts = segment_df[col].value_counts(normalize=True).head(5)
            profile[col] = value_counts.to_dict()
        
        return profile


# Factory function
def get_segmentation_engine() -> SegmentationEngine:
    """Get segmentation engine instance."""
    return SegmentationEngine()
