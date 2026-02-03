# AI Enterprise Data Analyst - Text Similarity Engine
# Production-grade text comparison and similarity
# Handles: any text pairs, multiple similarity methods

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

class SimilarityMethod(str, Enum):
    """Similarity calculation methods."""
    COSINE = "cosine"
    JACCARD = "jaccard"
    LEVENSHTEIN = "levenshtein"
    TFIDF = "tfidf"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TextPairSimilarity:
    """Similarity between two texts."""
    text1_id: int
    text2_id: int
    similarity: float
    method: SimilarityMethod


@dataclass
class TextSimilarityResult:
    """Complete text similarity result."""
    n_texts: int = 0
    n_pairs: int = 0
    method: SimilarityMethod = SimilarityMethod.COSINE
    
    # Pairwise similarities
    similarities: List[TextPairSimilarity] = field(default_factory=list)
    
    # Most similar pairs
    most_similar: List[TextPairSimilarity] = field(default_factory=list)
    
    # Similarity matrix
    similarity_matrix: List[List[float]] = field(default_factory=list)
    
    # Statistics
    avg_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 0.0
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_texts": self.n_texts,
            "n_pairs": self.n_pairs,
            "method": self.method.value,
            "statistics": {
                "avg_similarity": round(self.avg_similarity, 4),
                "max_similarity": round(self.max_similarity, 4),
                "min_similarity": round(self.min_similarity, 4)
            },
            "most_similar_pairs": [
                {
                    "text1_id": p.text1_id,
                    "text2_id": p.text2_id,
                    "similarity": round(p.similarity, 4)
                }
                for p in self.most_similar[:10]
            ]
        }


# ============================================================================
# Similarity Calculators
# ============================================================================

class SimilarityCalculator:
    """Text similarity calculation."""
    
    @staticmethod
    def cosine(text1: str, text2: str) -> float:
        """Calculate cosine similarity."""
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        all_words = words1 | words2
        
        if not all_words:
            return 0.0
        
        # Create vectors
        vec1 = [1 if w in words1 else 0 for w in all_words]
        vec2 = [1 if w in words2 else 0 for w in all_words]
        
        # Cosine similarity
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a ** 2 for a in vec1) ** 0.5
        mag2 = sum(b ** 2 for b in vec2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
    
    @staticmethod
    def jaccard(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def levenshtein(text1: str, text2: str) -> float:
        """Calculate Levenshtein similarity (1 - normalized distance)."""
        s1 = text1.lower()
        s2 = text2.lower()
        
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return 0.0 if s1 else 1.0
        
        prev_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        distance = prev_row[-1]
        max_len = max(len(s1), len(s2))
        
        return 1 - (distance / max_len) if max_len > 0 else 1.0


# ============================================================================
# Text Similarity Engine
# ============================================================================

class TextSimilarityEngine:
    """
    Text Similarity engine.
    
    Features:
    - Multiple similarity methods
    - Pairwise comparison
    - Most similar finding
    - Similarity matrix
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.calc = SimilarityCalculator()
    
    def compare_all(
        self,
        texts: List[str],
        method: SimilarityMethod = SimilarityMethod.COSINE
    ) -> TextSimilarityResult:
        """Compare all pairs of texts."""
        start_time = datetime.now()
        
        n = len(texts)
        
        if self.verbose:
            logger.info(f"Comparing {n} texts using {method.value}")
        
        # Get similarity function
        if method == SimilarityMethod.COSINE:
            sim_func = self.calc.cosine
        elif method == SimilarityMethod.JACCARD:
            sim_func = self.calc.jaccard
        elif method == SimilarityMethod.LEVENSHTEIN:
            sim_func = self.calc.levenshtein
        else:
            sim_func = self.calc.cosine
        
        # Calculate pairwise similarities
        similarities = []
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = sim_func(texts[i], texts[j])
                
                matrix[i][j] = sim
                matrix[j][i] = sim
                
                similarities.append(TextPairSimilarity(
                    text1_id=i,
                    text2_id=j,
                    similarity=sim,
                    method=method
                ))
        
        # Sort for most similar
        similarities.sort(key=lambda x: -x.similarity)
        
        # Statistics
        sim_values = [s.similarity for s in similarities]
        avg_sim = np.mean(sim_values) if sim_values else 0
        max_sim = max(sim_values) if sim_values else 0
        min_sim = min(sim_values) if sim_values else 0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TextSimilarityResult(
            n_texts=n,
            n_pairs=len(similarities),
            method=method,
            similarities=similarities,
            most_similar=similarities[:10],
            similarity_matrix=matrix,
            avg_similarity=float(avg_sim),
            max_similarity=float(max_sim),
            min_similarity=float(min_sim),
            processing_time_sec=processing_time
        )
    
    def find_similar(
        self,
        query: str,
        corpus: List[str],
        method: SimilarityMethod = SimilarityMethod.COSINE,
        top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """Find most similar texts to query."""
        if method == SimilarityMethod.COSINE:
            sim_func = self.calc.cosine
        elif method == SimilarityMethod.JACCARD:
            sim_func = self.calc.jaccard
        else:
            sim_func = self.calc.levenshtein
        
        similarities = [(i, sim_func(query, text)) for i, text in enumerate(corpus)]
        similarities.sort(key=lambda x: -x[1])
        
        return similarities[:top_n]


# ============================================================================
# Factory Functions
# ============================================================================

def get_similarity_engine() -> TextSimilarityEngine:
    """Get text similarity engine."""
    return TextSimilarityEngine()


def quick_similarity(
    texts: List[str],
    method: str = "cosine"
) -> Dict[str, Any]:
    """Quick text similarity."""
    engine = TextSimilarityEngine(verbose=False)
    result = engine.compare_all(texts, SimilarityMethod(method))
    return result.to_dict()
