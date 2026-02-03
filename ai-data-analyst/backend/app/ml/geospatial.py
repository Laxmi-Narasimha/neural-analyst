# AI Enterprise Data Analyst - Geospatial Analytics Module
# H3 hexagonal indexing, clustering, and location intelligence (Uber pattern)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import math

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Geospatial Types
# ============================================================================

class GeoResolution(int, Enum):
    """H3 resolution levels (0-15, higher = smaller hexagons)."""
    CONTINENT = 0
    COUNTRY = 3
    STATE = 5
    CITY = 7
    NEIGHBORHOOD = 9
    BLOCK = 11
    BUILDING = 13


@dataclass
class GeoPoint:
    """Geographic point."""
    latitude: float
    longitude: float
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)
    
    def to_dict(self) -> dict[str, float]:
        return {"lat": self.latitude, "lng": self.longitude}


@dataclass
class GeoBounds:
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float
    
    def contains(self, point: GeoPoint) -> bool:
        return (self.min_lat <= point.latitude <= self.max_lat and
                self.min_lng <= point.longitude <= self.max_lng)


@dataclass
class GeoCluster:
    """Geographic cluster result."""
    cluster_id: int
    center: GeoPoint
    points: list[GeoPoint] = field(default_factory=list)
    radius_km: float = 0.0
    density: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "center": self.center.to_dict(),
            "point_count": len(self.points),
            "radius_km": round(self.radius_km, 3),
            "density": round(self.density, 4)
        }


# ============================================================================
# Distance Calculations
# ============================================================================

class GeoDistance:
    """Geographic distance calculations."""
    
    EARTH_RADIUS_KM = 6371.0
    
    @staticmethod
    def haversine(p1: GeoPoint, p2: GeoPoint) -> float:
        """
        Calculate great-circle distance using Haversine formula.
        
        Returns distance in kilometers.
        """
        lat1, lon1 = math.radians(p1.latitude), math.radians(p1.longitude)
        lat2, lon2 = math.radians(p2.latitude), math.radians(p2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        return GeoDistance.EARTH_RADIUS_KM * c
    
    @staticmethod
    def bearing(p1: GeoPoint, p2: GeoPoint) -> float:
        """Calculate bearing from p1 to p2 in degrees."""
        lat1, lon1 = math.radians(p1.latitude), math.radians(p1.longitude)
        lat2, lon2 = math.radians(p2.latitude), math.radians(p2.longitude)
        
        dlon = lon2 - lon1
        
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(x, y)
        return (math.degrees(bearing) + 360) % 360
    
    @staticmethod
    def destination_point(start: GeoPoint, bearing: float, distance_km: float) -> GeoPoint:
        """Calculate destination point given start, bearing, and distance."""
        lat1 = math.radians(start.latitude)
        lon1 = math.radians(start.longitude)
        bearing_rad = math.radians(bearing)
        
        d = distance_km / GeoDistance.EARTH_RADIUS_KM
        
        lat2 = math.asin(math.sin(lat1) * math.cos(d) + 
                         math.cos(lat1) * math.sin(d) * math.cos(bearing_rad))
        
        lon2 = lon1 + math.atan2(
            math.sin(bearing_rad) * math.sin(d) * math.cos(lat1),
            math.cos(d) - math.sin(lat1) * math.sin(lat2)
        )
        
        return GeoPoint(math.degrees(lat2), math.degrees(lon2))


# ============================================================================
# H3 Hexagonal Indexing (Uber Pattern)
# ============================================================================

class H3Indexer:
    """
    H3 hexagonal spatial indexing.
    
    Provides uniform area hexagonal cells for spatial aggregation.
    Uses pure Python implementation as fallback if h3 not installed.
    """
    
    def __init__(self, resolution: int = 9):
        """
        Initialize H3 indexer.
        
        Args:
            resolution: H3 resolution (0-15). Default 9 ~ 0.1 km² cells
        """
        self.resolution = resolution
        self._h3_available = self._check_h3()
    
    def _check_h3(self) -> bool:
        """Check if h3 library is available."""
        try:
            import h3
            return True
        except ImportError:
            logger.warning("h3 library not installed. Using fallback indexing.")
            return False
    
    def geo_to_h3(self, lat: float, lng: float) -> str:
        """Convert lat/lng to H3 index."""
        if self._h3_available:
            import h3
            return h3.geo_to_h3(lat, lng, self.resolution)
        else:
            # Fallback: Simple grid-based indexing
            return self._fallback_index(lat, lng)
    
    def h3_to_geo(self, h3_index: str) -> GeoPoint:
        """Convert H3 index to center lat/lng."""
        if self._h3_available:
            import h3
            lat, lng = h3.h3_to_geo(h3_index)
            return GeoPoint(lat, lng)
        else:
            # Fallback: Decode grid index
            return self._fallback_decode(h3_index)
    
    def get_neighbors(self, h3_index: str, k: int = 1) -> list[str]:
        """Get k-ring neighbors of H3 cell."""
        if self._h3_available:
            import h3
            return list(h3.k_ring(h3_index, k))
        else:
            return [h3_index]  # Fallback
    
    def polyfill(self, polygon: list[tuple[float, float]]) -> list[str]:
        """Fill polygon with H3 cells."""
        if self._h3_available:
            import h3
            geojson = {
                "type": "Polygon",
                "coordinates": [polygon]
            }
            return list(h3.polyfill(geojson, self.resolution))
        else:
            # Fallback: Grid cells in bounding box
            lats = [p[0] for p in polygon]
            lngs = [p[1] for p in polygon]
            cells = []
            for lat in np.arange(min(lats), max(lats), 0.01):
                for lng in np.arange(min(lngs), max(lngs), 0.01):
                    cells.append(self._fallback_index(lat, lng))
            return list(set(cells))
    
    def _fallback_index(self, lat: float, lng: float) -> str:
        """Fallback grid-based indexing."""
        # Create grid at specified resolution
        grid_size = 0.1 / (2 ** (self.resolution - 9))  # Approximate H3 size
        lat_idx = int((lat + 90) / grid_size)
        lng_idx = int((lng + 180) / grid_size)
        return f"g{self.resolution}_{lat_idx}_{lng_idx}"
    
    def _fallback_decode(self, index: str) -> GeoPoint:
        """Decode fallback grid index."""
        parts = index.split('_')
        if len(parts) == 3:
            res = int(parts[0][1:])
            grid_size = 0.1 / (2 ** (res - 9))
            lat = int(parts[1]) * grid_size - 90 + grid_size / 2
            lng = int(parts[2]) * grid_size - 180 + grid_size / 2
            return GeoPoint(lat, lng)
        return GeoPoint(0, 0)
    
    def index_dataframe(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lng_col: str = "longitude"
    ) -> pd.DataFrame:
        """Add H3 index column to dataframe."""
        result = df.copy()
        result['h3_index'] = df.apply(
            lambda row: self.geo_to_h3(row[lat_col], row[lng_col]),
            axis=1
        )
        return result
    
    def aggregate_by_h3(
        self,
        df: pd.DataFrame,
        value_col: str,
        agg_func: str = "mean",
        lat_col: str = "latitude",
        lng_col: str = "longitude"
    ) -> pd.DataFrame:
        """Aggregate values by H3 cells."""
        indexed = self.index_dataframe(df, lat_col, lng_col)
        
        agg = indexed.groupby('h3_index').agg({
            value_col: agg_func,
            lat_col: 'mean',
            lng_col: 'mean'
        }).reset_index()
        
        agg.columns = ['h3_index', f'{value_col}_{agg_func}', 'center_lat', 'center_lng']
        return agg


# ============================================================================
# Geospatial Clustering
# ============================================================================

class GeoClustering:
    """Geographic clustering algorithms."""
    
    @staticmethod
    def dbscan(
        points: list[GeoPoint],
        eps_km: float = 1.0,
        min_samples: int = 5
    ) -> list[GeoCluster]:
        """
        DBSCAN clustering on geographic points.
        
        Args:
            points: List of geographic points
            eps_km: Maximum distance (km) for neighbors
            min_samples: Minimum points for a cluster
        """
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import haversine_distances
            
            # Convert to radians
            coords = np.array([[math.radians(p.latitude), math.radians(p.longitude)] 
                              for p in points])
            
            # DBSCAN with haversine distance
            eps_rad = eps_km / GeoDistance.EARTH_RADIUS_KM
            
            db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
            labels = db.fit_predict(coords)
            
            # Build clusters
            clusters = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # Noise
                    continue
                
                cluster_points = [p for i, p in enumerate(points) if labels[i] == label]
                
                # Calculate center
                center_lat = np.mean([p.latitude for p in cluster_points])
                center_lng = np.mean([p.longitude for p in cluster_points])
                center = GeoPoint(center_lat, center_lng)
                
                # Calculate radius
                distances = [GeoDistance.haversine(center, p) for p in cluster_points]
                radius = max(distances) if distances else 0
                
                clusters.append(GeoCluster(
                    cluster_id=int(label),
                    center=center,
                    points=cluster_points,
                    radius_km=radius,
                    density=len(cluster_points) / (math.pi * radius ** 2) if radius > 0 else 0
                ))
            
            return clusters
            
        except ImportError:
            logger.warning("sklearn not available for DBSCAN")
            return []
    
    @staticmethod
    def kmeans(
        points: list[GeoPoint],
        n_clusters: int = 5
    ) -> list[GeoCluster]:
        """K-means clustering on geographic points."""
        try:
            from sklearn.cluster import KMeans
            
            coords = np.array([[p.latitude, p.longitude] for p in points])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)
            
            clusters = []
            for i in range(n_clusters):
                cluster_points = [p for j, p in enumerate(points) if labels[j] == i]
                center = GeoPoint(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1])
                
                distances = [GeoDistance.haversine(center, p) for p in cluster_points]
                radius = max(distances) if distances else 0
                
                clusters.append(GeoCluster(
                    cluster_id=i,
                    center=center,
                    points=cluster_points,
                    radius_km=radius
                ))
            
            return clusters
            
        except ImportError:
            return []


# ============================================================================
# Location Intelligence
# ============================================================================

class LocationIntelligence:
    """Location-based analytics and insights."""
    
    def __init__(self):
        self.h3_indexer = H3Indexer()
    
    def hotspot_detection(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lng_col: str = "longitude",
        resolution: int = 9
    ) -> pd.DataFrame:
        """Detect geographic hotspots."""
        self.h3_indexer.resolution = resolution
        
        # Index and count
        indexed = self.h3_indexer.index_dataframe(df, lat_col, lng_col)
        counts = indexed.groupby('h3_index').size().reset_index(name='count')
        
        # Calculate statistics
        mean_count = counts['count'].mean()
        std_count = counts['count'].std()
        
        # Z-score for hotspot detection
        counts['z_score'] = (counts['count'] - mean_count) / (std_count + 1e-10)
        counts['is_hotspot'] = counts['z_score'] > 2.0
        counts['is_coldspot'] = counts['z_score'] < -2.0
        
        # Add center coordinates
        counts['center'] = counts['h3_index'].apply(
            lambda x: self.h3_indexer.h3_to_geo(x).to_dict()
        )
        
        return counts
    
    def spatial_autocorrelation(
        self,
        df: pd.DataFrame,
        value_col: str,
        lat_col: str = "latitude",
        lng_col: str = "longitude"
    ) -> dict[str, float]:
        """Calculate Moran's I spatial autocorrelation."""
        n = len(df)
        if n < 10:
            return {"morans_i": 0, "p_value": 1.0}
        
        values = df[value_col].values
        mean_val = values.mean()
        
        # Build distance matrix
        points = [GeoPoint(row[lat_col], row[lng_col]) for _, row in df.iterrows()]
        
        # Weight matrix (inverse distance)
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = GeoDistance.haversine(points[i], points[j])
                    W[i, j] = 1 / (d + 0.001)
        
        # Row-standardize
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]
        
        # Moran's I
        z = values - mean_val
        numerator = n * np.sum(W * np.outer(z, z))
        denominator = W.sum() * np.sum(z ** 2)
        
        morans_i = numerator / denominator if denominator != 0 else 0
        
        return {
            "morans_i": float(morans_i),
            "interpretation": "positive" if morans_i > 0 else "negative" if morans_i < 0 else "random"
        }
    
    def create_heatmap_data(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lng_col: str = "longitude",
        value_col: str = None,
        resolution: int = 9
    ) -> list[dict]:
        """Create heatmap data for visualization."""
        self.h3_indexer.resolution = resolution
        indexed = self.h3_indexer.index_dataframe(df, lat_col, lng_col)
        
        if value_col:
            agg = indexed.groupby('h3_index').agg({
                value_col: 'mean',
                lat_col: 'mean',
                lng_col: 'mean'
            })
        else:
            agg = indexed.groupby('h3_index').agg({
                lat_col: ['mean', 'count'],
                lng_col: 'mean'
            })
            agg.columns = [value_col or 'count', 'lat', 'lng']
        
        agg = agg.reset_index()
        
        return [
            {
                "lat": row['lat'] if 'lat' in row else row[lat_col],
                "lng": row['lng'] if 'lng' in row else row[lng_col],
                "value": float(row[value_col] if value_col else row.get('count', 1))
            }
            for _, row in agg.iterrows()
        ]


# ============================================================================
# Geospatial Engine
# ============================================================================

class GeospatialEngine:
    """
    Unified geospatial analytics engine.
    
    Features:
    - H3 hexagonal indexing (Uber pattern)
    - Geographic clustering
    - Hotspot detection
    - Spatial autocorrelation
    - Distance calculations
    """
    
    def __init__(self, default_resolution: int = 9):
        self.h3 = H3Indexer(default_resolution)
        self.clustering = GeoClustering()
        self.intelligence = LocationIntelligence()
        self.distance = GeoDistance()
    
    def index_locations(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lng_col: str = "longitude",
        resolution: int = None
    ) -> pd.DataFrame:
        """Index locations with H3."""
        if resolution:
            self.h3.resolution = resolution
        return self.h3.index_dataframe(df, lat_col, lng_col)
    
    def cluster_locations(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lng_col: str = "longitude",
        method: str = "dbscan",
        **kwargs
    ) -> list[GeoCluster]:
        """Cluster geographic locations."""
        points = [
            GeoPoint(row[lat_col], row[lng_col])
            for _, row in df.iterrows()
        ]
        
        if method == "dbscan":
            return self.clustering.dbscan(points, **kwargs)
        else:
            return self.clustering.kmeans(points, **kwargs)
    
    def detect_hotspots(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lng_col: str = "longitude"
    ) -> pd.DataFrame:
        """Detect geographic hotspots."""
        return self.intelligence.hotspot_detection(df, lat_col, lng_col)
    
    def calculate_distance_matrix(
        self,
        points: list[GeoPoint]
    ) -> np.ndarray:
        """Calculate pairwise distance matrix."""
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance.haversine(points[i], points[j])
                distances[i, j] = d
                distances[j, i] = d
        
        return distances


# Factory function
def get_geospatial_engine() -> GeospatialEngine:
    """Get geospatial engine instance."""
    return GeospatialEngine()
