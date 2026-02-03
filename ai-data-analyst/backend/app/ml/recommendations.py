# AI Enterprise Data Analyst - Recommendation Engine
# Netflix/Spotify-inspired collaborative and content-based recommendations

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine

from app.core.logging import get_logger, LogContext
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Recommendation Types
# ============================================================================

class RecommendationType(str, Enum):
    """Types of recommendation strategies."""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    POPULARITY = "popularity"
    KNOWLEDGE = "knowledge"


class SimilarityMetric(str, Enum):
    """Similarity calculation methods."""
    COSINE = "cosine"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    EUCLIDEAN = "euclidean"


@dataclass
class Recommendation:
    """Single recommendation item."""
    
    item_id: Any
    score: float
    rank: int
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": round(self.score, 4),
            "rank": self.rank,
            "reason": self.reason,
            "metadata": self.metadata
        }


@dataclass
class RecommendationResult:
    """Complete recommendation result."""
    
    user_id: Any
    strategy: RecommendationType
    recommendations: list[Recommendation] = field(default_factory=list)
    
    # Metrics
    coverage: float = 0.0  # % of items that can be recommended
    diversity: float = 0.0  # Diversity of recommendations
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "strategy": self.strategy.value,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "metrics": {
                "coverage": round(self.coverage, 4),
                "diversity": round(self.diversity, 4)
            }
        }


# ============================================================================
# Collaborative Filtering
# ============================================================================

class CollaborativeFiltering:
    """
    User-based and Item-based collaborative filtering.
    
    Based on Netflix prize competition patterns.
    """
    
    def __init__(
        self,
        method: str = "item",  # 'user' or 'item'
        k_neighbors: int = 50,
        similarity: SimilarityMetric = SimilarityMetric.COSINE
    ):
        self.method = method
        self.k_neighbors = k_neighbors
        self.similarity = similarity
        
        self._user_item_matrix: Optional[pd.DataFrame] = None
        self._similarity_matrix: Optional[np.ndarray] = None
        self._users: list = []
        self._items: list = []
        self._fitted = False
    
    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating"
    ) -> "CollaborativeFiltering":
        """
        Fit collaborative filtering model.
        
        Args:
            interactions: DataFrame with user-item interactions
            user_col: Column name for users
            item_col: Column name for items
            rating_col: Column name for ratings/scores
        """
        # Create user-item matrix
        self._user_item_matrix = interactions.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            aggfunc='mean'
        ).fillna(0)
        
        self._users = self._user_item_matrix.index.tolist()
        self._items = self._user_item_matrix.columns.tolist()
        
        # Compute similarity matrix
        if self.method == "item":
            matrix = self._user_item_matrix.T.values
        else:
            matrix = self._user_item_matrix.values
        
        self._similarity_matrix = self._compute_similarity(matrix)
        self._fitted = True
        
        return self
    
    def _compute_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        n = matrix.shape[0]
        similarity = np.zeros((n, n))
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1
        matrix_normalized = matrix / norms[:, np.newaxis]
        
        if self.similarity == SimilarityMetric.COSINE:
            similarity = matrix_normalized @ matrix_normalized.T
        
        elif self.similarity == SimilarityMetric.PEARSON:
            # Center the data
            means = matrix.mean(axis=1)
            centered = matrix - means[:, np.newaxis]
            stds = np.linalg.norm(centered, axis=1)
            stds[stds == 0] = 1
            normalized = centered / stds[:, np.newaxis]
            similarity = normalized @ normalized.T
        
        elif self.similarity == SimilarityMetric.JACCARD:
            binary = (matrix > 0).astype(float)
            intersection = binary @ binary.T
            union = np.outer(binary.sum(axis=1), np.ones(n)) + \
                    np.outer(np.ones(n), binary.sum(axis=1)) - intersection
            union[union == 0] = 1
            similarity = intersection / union
        
        # Remove self-similarity
        np.fill_diagonal(similarity, 0)
        
        return similarity
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_known: bool = True
    ) -> list[Recommendation]:
        """Generate recommendations for a user."""
        if not self._fitted:
            raise ValidationException("Model not fitted")
        
        if user_id not in self._users:
            # Cold start - return popular items
            return self._popular_fallback(n_recommendations)
        
        user_idx = self._users.index(user_id)
        user_ratings = self._user_item_matrix.iloc[user_idx].values
        
        if self.method == "item":
            # Item-based: predict ratings using similar items
            predictions = self._predict_item_based(user_ratings)
        else:
            # User-based: predict using similar users
            predictions = self._predict_user_based(user_idx)
        
        # Filter already rated items
        if exclude_known:
            predictions[user_ratings > 0] = -np.inf
        
        # Get top recommendations
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        
        recommendations = []
        for rank, idx in enumerate(top_indices):
            if predictions[idx] > -np.inf:
                recommendations.append(Recommendation(
                    item_id=self._items[idx],
                    score=float(predictions[idx]),
                    rank=rank + 1,
                    reason=f"Based on similar {'users' if self.method == 'user' else 'items'}"
                ))
        
        return recommendations
    
    def _predict_item_based(self, user_ratings: np.ndarray) -> np.ndarray:
        """Predict ratings using item-based CF."""
        predictions = np.zeros(len(self._items))
        
        for i in range(len(self._items)):
            # Find most similar items that user has rated
            similarities = self._similarity_matrix[i]
            rated_mask = user_ratings > 0
            
            if rated_mask.sum() == 0:
                continue
            
            # Top-k similar items
            similar_items = similarities * rated_mask
            top_k_idx = np.argsort(similar_items)[::-1][:self.k_neighbors]
            
            numerator = np.sum(similarities[top_k_idx] * user_ratings[top_k_idx])
            denominator = np.sum(np.abs(similarities[top_k_idx])) + 1e-10
            
            predictions[i] = numerator / denominator
        
        return predictions
    
    def _predict_user_based(self, user_idx: int) -> np.ndarray:
        """Predict ratings using user-based CF."""
        similarities = self._similarity_matrix[user_idx]
        
        # Top-k similar users
        top_k_idx = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        # Weighted average of similar users' ratings
        similar_ratings = self._user_item_matrix.iloc[top_k_idx].values
        weights = similarities[top_k_idx]
        
        numerator = weights @ similar_ratings
        denominator = np.sum(np.abs(weights)) + 1e-10
        
        return numerator / denominator
    
    def _popular_fallback(self, n: int) -> list[Recommendation]:
        """Fallback to popularity-based recommendations."""
        if self._user_item_matrix is None:
            return []
        
        popularity = (self._user_item_matrix > 0).sum(axis=0)
        top_items = popularity.nlargest(n)
        
        return [
            Recommendation(
                item_id=item,
                score=float(count),
                rank=i + 1,
                reason="Popular item"
            )
            for i, (item, count) in enumerate(top_items.items())
        ]


# ============================================================================
# Content-Based Filtering
# ============================================================================

class ContentBasedFiltering:
    """
    Content-based recommendations using item features.
    
    Spotify-inspired approach using item embeddings.
    """
    
    def __init__(self, similarity: SimilarityMetric = SimilarityMetric.COSINE):
        self.similarity = similarity
        
        self._item_features: Optional[pd.DataFrame] = None
        self._feature_matrix: Optional[np.ndarray] = None
        self._items: list = []
        self._user_profiles: dict[Any, np.ndarray] = {}
        self._fitted = False
    
    def fit(
        self,
        item_features: pd.DataFrame,
        item_col: str = "item_id"
    ) -> "ContentBasedFiltering":
        """
        Fit content-based model with item features.
        
        Args:
            item_features: DataFrame with item_id and feature columns
            item_col: Column name for item identifier
        """
        self._item_features = item_features.set_index(item_col)
        self._items = self._item_features.index.tolist()
        
        # Normalize feature matrix
        feature_cols = self._item_features.select_dtypes(include=[np.number]).columns
        self._feature_matrix = self._item_features[feature_cols].fillna(0).values
        
        # Normalize
        norms = np.linalg.norm(self._feature_matrix, axis=1)
        norms[norms == 0] = 1
        self._feature_matrix = self._feature_matrix / norms[:, np.newaxis]
        
        self._fitted = True
        return self
    
    def build_user_profile(
        self,
        user_id: Any,
        interactions: pd.DataFrame,
        item_col: str = "item_id",
        rating_col: str = "rating"
    ) -> None:
        """Build user profile from interaction history."""
        if not self._fitted:
            raise ValidationException("Model not fitted")
        
        user_items = interactions[item_col].tolist()
        user_ratings = interactions[rating_col].values if rating_col in interactions else None
        
        # Get feature vectors for user's items
        profile = np.zeros(self._feature_matrix.shape[1])
        total_weight = 0
        
        for i, item in enumerate(user_items):
            if item in self._items:
                idx = self._items.index(item)
                weight = user_ratings[i] if user_ratings is not None else 1
                profile += weight * self._feature_matrix[idx]
                total_weight += weight
        
        if total_weight > 0:
            profile /= total_weight
        
        self._user_profiles[user_id] = profile
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_items: list = None
    ) -> list[Recommendation]:
        """Generate content-based recommendations."""
        if user_id not in self._user_profiles:
            return []
        
        user_profile = self._user_profiles[user_id]
        
        # Compute similarity to all items
        similarities = self._feature_matrix @ user_profile
        
        # Exclude items
        if exclude_items:
            for item in exclude_items:
                if item in self._items:
                    idx = self._items.index(item)
                    similarities[idx] = -np.inf
        
        # Top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > -np.inf:
                recommendations.append(Recommendation(
                    item_id=self._items[idx],
                    score=float(similarities[idx]),
                    rank=rank + 1,
                    reason="Matches your preferences"
                ))
        
        return recommendations
    
    def find_similar_items(
        self,
        item_id: Any,
        n_similar: int = 10
    ) -> list[Recommendation]:
        """Find items similar to a given item."""
        if item_id not in self._items:
            return []
        
        idx = self._items.index(item_id)
        item_vector = self._feature_matrix[idx]
        
        # Compute similarities
        similarities = self._feature_matrix @ item_vector
        similarities[idx] = -np.inf  # Exclude self
        
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        
        return [
            Recommendation(
                item_id=self._items[i],
                score=float(similarities[i]),
                rank=rank + 1,
                reason=f"Similar to {item_id}"
            )
            for rank, i in enumerate(top_indices)
            if similarities[i] > -np.inf
        ]


# ============================================================================
# Hybrid Recommender
# ============================================================================

class HybridRecommender:
    """
    Hybrid recommendation system combining multiple strategies.
    
    Weights can be adjusted based on data availability.
    """
    
    def __init__(
        self,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
        diversity_weight: float = 0.1
    ):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.diversity_weight = diversity_weight
        
        self.cf_model = CollaborativeFiltering(method="item")
        self.cb_model = ContentBasedFiltering()
        self._fitted = False
    
    def fit(
        self,
        interactions: pd.DataFrame,
        item_features: Optional[pd.DataFrame] = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating"
    ) -> "HybridRecommender":
        """Fit hybrid model."""
        # Fit collaborative filtering
        self.cf_model.fit(interactions, user_col, item_col, rating_col)
        
        # Fit content-based if features available
        if item_features is not None:
            self.cb_model.fit(item_features, item_col)
            
            # Build user profiles
            for user_id in interactions[user_col].unique():
                user_data = interactions[interactions[user_col] == user_id]
                self.cb_model.build_user_profile(user_id, user_data, item_col, rating_col)
        
        self._fitted = True
        return self
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_known: bool = True
    ) -> RecommendationResult:
        """Generate hybrid recommendations."""
        # Get CF recommendations
        cf_recs = self.cf_model.recommend(user_id, n_recommendations * 2, exclude_known)
        
        # Get CB recommendations
        cb_recs = self.cb_model.recommend(user_id, n_recommendations * 2)
        
        # Combine scores
        item_scores = {}
        
        for rec in cf_recs:
            item_scores[rec.item_id] = self.cf_weight * rec.score
        
        for rec in cb_recs:
            if rec.item_id in item_scores:
                item_scores[rec.item_id] += self.cb_weight * rec.score
            else:
                item_scores[rec.item_id] = self.cb_weight * rec.score
        
        # Sort by combined score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply diversity (MMR-like)
        if self.diversity_weight > 0:
            sorted_items = self._apply_diversity(
                sorted_items, n_recommendations
            )
        else:
            sorted_items = sorted_items[:n_recommendations]
        
        recommendations = [
            Recommendation(
                item_id=item,
                score=score,
                rank=i + 1,
                reason="Hybrid recommendation"
            )
            for i, (item, score) in enumerate(sorted_items)
        ]
        
        return RecommendationResult(
            user_id=user_id,
            strategy=RecommendationType.HYBRID,
            recommendations=recommendations,
            coverage=len(item_scores) / len(self.cf_model._items) if self.cf_model._items else 0,
            diversity=self._compute_diversity(recommendations)
        )
    
    def _apply_diversity(
        self,
        item_scores: list[tuple],
        n: int
    ) -> list[tuple]:
        """Apply Maximal Marginal Relevance for diversity."""
        if not self.cb_model._fitted:
            return item_scores[:n]
        
        selected = []
        remaining = list(item_scores)
        
        while len(selected) < n and remaining:
            if not selected:
                # First item - highest score
                selected.append(remaining.pop(0))
            else:
                # Balance relevance and diversity
                best_mmr = -np.inf
                best_idx = 0
                
                for i, (item, score) in enumerate(remaining):
                    # Compute max similarity to selected items
                    max_sim = 0
                    for sel_item, _ in selected:
                        sim = self._item_similarity(item, sel_item)
                        max_sim = max(max_sim, sim)
                    
                    # MMR score
                    mmr = (1 - self.diversity_weight) * score - self.diversity_weight * max_sim
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
                
                selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _item_similarity(self, item1: Any, item2: Any) -> float:
        """Compute similarity between two items."""
        if not self.cb_model._fitted:
            return 0
        
        if item1 not in self.cb_model._items or item2 not in self.cb_model._items:
            return 0
        
        idx1 = self.cb_model._items.index(item1)
        idx2 = self.cb_model._items.index(item2)
        
        return float(self.cb_model._feature_matrix[idx1] @ self.cb_model._feature_matrix[idx2])
    
    def _compute_diversity(self, recommendations: list[Recommendation]) -> float:
        """Compute diversity of recommendations."""
        if len(recommendations) < 2:
            return 1.0
        
        similarities = []
        for i, rec1 in enumerate(recommendations):
            for rec2 in recommendations[i + 1:]:
                sim = self._item_similarity(rec1.item_id, rec2.item_id)
                similarities.append(sim)
        
        return 1 - np.mean(similarities) if similarities else 1.0


# ============================================================================
# Recommendation Engine
# ============================================================================

class RecommendationEngine:
    """
    Production recommendation engine with multiple strategies.
    
    Features:
    - Collaborative filtering (user/item-based)
    - Content-based filtering
    - Hybrid approach
    - A/B testing support
    - Cold start handling
    """
    
    def __init__(self):
        self.hybrid = HybridRecommender()
        self.cf = CollaborativeFiltering()
        self.cb = ContentBasedFiltering()
        self._fitted = False
    
    def fit(
        self,
        interactions: pd.DataFrame,
        item_features: Optional[pd.DataFrame] = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating"
    ) -> "RecommendationEngine":
        """Fit all recommendation models."""
        self.hybrid.fit(interactions, item_features, user_col, item_col, rating_col)
        self._fitted = True
        return self
    
    def recommend(
        self,
        user_id: Any,
        strategy: RecommendationType = RecommendationType.HYBRID,
        n_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations using specified strategy."""
        if not self._fitted:
            raise ValidationException("Engine not fitted")
        
        if strategy == RecommendationType.HYBRID:
            return self.hybrid.recommend(user_id, n_recommendations)
        
        elif strategy == RecommendationType.COLLABORATIVE:
            recs = self.hybrid.cf_model.recommend(user_id, n_recommendations)
            return RecommendationResult(
                user_id=user_id,
                strategy=strategy,
                recommendations=recs
            )
        
        elif strategy == RecommendationType.CONTENT_BASED:
            recs = self.hybrid.cb_model.recommend(user_id, n_recommendations)
            return RecommendationResult(
                user_id=user_id,
                strategy=strategy,
                recommendations=recs
            )
        
        elif strategy == RecommendationType.POPULARITY:
            recs = self.hybrid.cf_model._popular_fallback(n_recommendations)
            return RecommendationResult(
                user_id=user_id,
                strategy=strategy,
                recommendations=recs
            )
        
        else:
            return self.hybrid.recommend(user_id, n_recommendations)
    
    def similar_items(
        self,
        item_id: Any,
        n_similar: int = 10
    ) -> list[Recommendation]:
        """Find similar items."""
        return self.hybrid.cb_model.find_similar_items(item_id, n_similar)


# Factory function
def get_recommendation_engine() -> RecommendationEngine:
    """Get recommendation engine instance."""
    return RecommendationEngine()
