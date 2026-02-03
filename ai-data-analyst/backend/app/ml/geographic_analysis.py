# AI Enterprise Data Analyst - Geographic Analysis Engine
# Production-grade geographic and location analysis
# Handles: coordinates, distances, clustering

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LocationCluster:
    """Geographic cluster."""
    cluster_id: int
    centroid_lat: float
    centroid_lon: float
    n_points: int
    radius_km: float


@dataclass
class GeographicResult:
    """Complete geographic analysis result."""
    n_locations: int = 0
    
    # Bounding box
    bounds: Dict[str, float] = field(default_factory=dict)
    
    # Centroid
    centroid: Tuple[float, float] = (0, 0)
    
    # Clusters
    clusters: List[LocationCluster] = field(default_factory=list)
    
    # Distances
    avg_distance_km: float = 0.0
    max_distance_km: float = 0.0
    
    # Distribution by region
    region_distribution: Dict[str, int] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_locations": self.n_locations,
            "bounds": self.bounds,
            "centroid": {"lat": round(self.centroid[0], 6), "lon": round(self.centroid[1], 6)},
            "clusters": [
                {
                    "id": c.cluster_id,
                    "centroid": {"lat": round(c.centroid_lat, 6), "lon": round(c.centroid_lon, 6)},
                    "n_points": c.n_points,
                    "radius_km": round(c.radius_km, 2)
                }
                for c in self.clusters
            ],
            "distances": {
                "avg_km": round(self.avg_distance_km, 2),
                "max_km": round(self.max_distance_km, 2)
            }
        }


# ============================================================================
# Geographic Utilities
# ============================================================================

class GeoUtils:
    """Geographic utility functions."""
    
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in km."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    @staticmethod
    def centroid(lats: List[float], lons: List[float]) -> Tuple[float, float]:
        """Calculate geographic centroid."""
        return float(np.mean(lats)), float(np.mean(lons))
    
    @staticmethod
    def bounding_box(lats: List[float], lons: List[float]) -> Dict[str, float]:
        """Calculate bounding box."""
        return {
            "min_lat": float(min(lats)),
            "max_lat": float(max(lats)),
            "min_lon": float(min(lons)),
            "max_lon": float(max(lons))
        }


# ============================================================================
# Geographic Analysis Engine
# ============================================================================

class GeographicAnalysisEngine:
    """
    Geographic Analysis engine.
    
    Features:
    - Distance calculations
    - Spatial clustering
    - Centroid and bounds
    - Region distribution
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.utils = GeoUtils()
    
    def analyze(
        self,
        df: pd.DataFrame,
        lat_col: str = None,
        lon_col: str = None,
        n_clusters: int = 5
    ) -> GeographicResult:
        """Analyze geographic data."""
        start_time = datetime.now()
        
        # Auto-detect columns
        if lat_col is None:
            lat_col = self._detect_lat_col(df)
        if lon_col is None:
            lon_col = self._detect_lon_col(df)
        
        lats = df[lat_col].dropna().values
        lons = df[lon_col].dropna().values
        
        if len(lats) != len(lons):
            min_len = min(len(lats), len(lons))
            lats = lats[:min_len]
            lons = lons[:min_len]
        
        if self.verbose:
            logger.info(f"Analyzing {len(lats)} geographic points")
        
        # Basic metrics
        centroid = self.utils.centroid(lats.tolist(), lons.tolist())
        bounds = self.utils.bounding_box(lats.tolist(), lons.tolist())
        
        # Distances
        distances = []
        sample_size = min(100, len(lats))
        sample_idx = np.random.choice(len(lats), sample_size, replace=False)
        
        for i in sample_idx:
            for j in sample_idx:
                if i < j:
                    d = self.utils.haversine(lats[i], lons[i], lats[j], lons[j])
                    distances.append(d)
        
        avg_distance = np.mean(distances) if distances else 0
        max_distance = max(distances) if distances else 0
        
        # Clustering
        clusters = self._cluster_locations(lats, lons, n_clusters)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GeographicResult(
            n_locations=len(lats),
            bounds=bounds,
            centroid=centroid,
            clusters=clusters,
            avg_distance_km=avg_distance,
            max_distance_km=max_distance,
            processing_time_sec=processing_time
        )
    
    def _cluster_locations(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        n_clusters: int
    ) -> List[LocationCluster]:
        """Cluster geographic locations using K-means."""
        try:
            from sklearn.cluster import KMeans
            
            coords = np.column_stack([lats, lons])
            n_clusters = min(n_clusters, len(coords))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)
            
            clusters = []
            for i in range(n_clusters):
                mask = labels == i
                cluster_lats = lats[mask]
                cluster_lons = lons[mask]
                
                cent_lat = float(np.mean(cluster_lats))
                cent_lon = float(np.mean(cluster_lons))
                
                # Calculate radius
                max_dist = 0
                for lat, lon in zip(cluster_lats, cluster_lons):
                    d = self.utils.haversine(cent_lat, cent_lon, lat, lon)
                    max_dist = max(max_dist, d)
                
                clusters.append(LocationCluster(
                    cluster_id=i,
                    centroid_lat=cent_lat,
                    centroid_lon=cent_lon,
                    n_points=int(mask.sum()),
                    radius_km=max_dist
                ))
            
            return clusters
            
        except ImportError:
            # Simple grid-based clustering
            return self._simple_cluster(lats, lons, n_clusters)
    
    def _simple_cluster(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        n_clusters: int
    ) -> List[LocationCluster]:
        """Simple grid-based clustering fallback."""
        lat_bins = np.linspace(lats.min(), lats.max(), int(np.sqrt(n_clusters)) + 1)
        lon_bins = np.linspace(lons.min(), lons.max(), int(np.sqrt(n_clusters)) + 1)
        
        clusters = []
        cluster_id = 0
        
        for i in range(len(lat_bins) - 1):
            for j in range(len(lon_bins) - 1):
                mask = (
                    (lats >= lat_bins[i]) & (lats < lat_bins[i+1]) &
                    (lons >= lon_bins[j]) & (lons < lon_bins[j+1])
                )
                
                if mask.sum() > 0:
                    clusters.append(LocationCluster(
                        cluster_id=cluster_id,
                        centroid_lat=float(np.mean(lats[mask])),
                        centroid_lon=float(np.mean(lons[mask])),
                        n_points=int(mask.sum()),
                        radius_km=0
                    ))
                    cluster_id += 1
        
        return clusters[:n_clusters]
    
    def _detect_lat_col(self, df: pd.DataFrame) -> str:
        patterns = ['lat', 'latitude', 'y']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.select_dtypes(include=[np.number]).columns[0]
    
    def _detect_lon_col(self, df: pd.DataFrame) -> str:
        patterns = ['lon', 'lng', 'longitude', 'x']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        num_cols = df.select_dtypes(include=[np.number]).columns
        return num_cols[1] if len(num_cols) > 1 else num_cols[0]
    
    def calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points."""
        return self.utils.haversine(lat1, lon1, lat2, lon2)


# ============================================================================
# Factory Functions
# ============================================================================

def get_geographic_engine() -> GeographicAnalysisEngine:
    """Get geographic analysis engine."""
    return GeographicAnalysisEngine()


def quick_geo_analysis(
    df: pd.DataFrame,
    lat_col: str = None,
    lon_col: str = None
) -> Dict[str, Any]:
    """Quick geographic analysis."""
    engine = GeographicAnalysisEngine(verbose=False)
    result = engine.analyze(df, lat_col, lon_col)
    return result.to_dict()
