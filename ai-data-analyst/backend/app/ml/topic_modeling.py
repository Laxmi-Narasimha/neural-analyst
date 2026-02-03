# AI Enterprise Data Analyst - Topic Modeling Engine
# Production-grade topic extraction from text
# Handles: any text corpus, LDA, NMF methods

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class TopicMethod(str, Enum):
    """Topic modeling methods."""
    LDA = "lda"
    NMF = "nmf"
    LSA = "lsa"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Topic:
    """Single topic."""
    topic_id: int
    label: str
    top_words: List[str]
    word_weights: Dict[str, float]
    coherence: float
    document_count: int


@dataclass
class DocumentTopic:
    """Topic assignment for a document."""
    doc_id: int
    dominant_topic: int
    topic_distribution: Dict[int, float]


@dataclass
class TopicModelResult:
    """Complete topic modeling result."""
    n_documents: int = 0
    n_topics: int = 0
    method: TopicMethod = TopicMethod.LDA
    
    # Topics
    topics: List[Topic] = field(default_factory=list)
    
    # Document assignments
    doc_topics: List[DocumentTopic] = field(default_factory=list)
    
    # Model metrics
    coherence_score: float = 0.0
    perplexity: Optional[float] = None
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_documents": self.n_documents,
            "n_topics": self.n_topics,
            "method": self.method.value,
            "coherence_score": round(self.coherence_score, 4),
            "topics": [
                {
                    "id": t.topic_id,
                    "label": t.label,
                    "top_words": t.top_words[:10],
                    "document_count": t.document_count
                }
                for t in self.topics
            ],
            "topic_distribution": {
                t.topic_id: t.document_count / self.n_documents
                for t in self.topics
            }
        }


# ============================================================================
# Topic Modeling Engine
# ============================================================================

class TopicModelingEngine:
    """
    Topic Modeling engine.
    
    Features:
    - LDA and NMF methods
    - Automatic topic labeling
    - Document-topic assignment
    - Coherence scoring
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def fit(
        self,
        documents: List[str],
        n_topics: int = 5,
        method: TopicMethod = TopicMethod.NMF,
        n_top_words: int = 10
    ) -> TopicModelResult:
        """Extract topics from documents."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Topic modeling: {len(documents)} docs, {n_topics} topics")
        
        # Vectorize
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
            
            if method == TopicMethod.LDA:
                vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
                doc_term = vectorizer.fit_transform(documents)
                model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            else:  # NMF or LSA
                vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
                doc_term = vectorizer.fit_transform(documents)
                
                if method == TopicMethod.NMF:
                    model = NMF(n_components=n_topics, random_state=42)
                else:
                    model = TruncatedSVD(n_components=n_topics, random_state=42)
            
            doc_topic_matrix = model.fit_transform(doc_term)
            
            feature_names = vectorizer.get_feature_names_out()
            
        except ImportError:
            # Fallback: simple TF-based extraction
            return self._simple_topics(documents, n_topics, n_top_words, method)
        
        # Extract topics
        topics = []
        for topic_idx, topic_weights in enumerate(model.components_):
            top_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_indices]
            word_weights = {feature_names[i]: float(topic_weights[i]) for i in top_indices}
            
            # Count documents for this topic
            doc_count = sum(1 for row in doc_topic_matrix if row.argmax() == topic_idx)
            
            # Generate label
            label = f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"
            
            topics.append(Topic(
                topic_id=topic_idx,
                label=label,
                top_words=top_words,
                word_weights=word_weights,
                coherence=0.0,
                document_count=doc_count
            ))
        
        # Document assignments
        doc_topics = []
        for doc_id, row in enumerate(doc_topic_matrix):
            total = row.sum()
            distribution = {i: float(row[i] / total) if total > 0 else 0 for i in range(n_topics)}
            
            doc_topics.append(DocumentTopic(
                doc_id=doc_id,
                dominant_topic=int(row.argmax()),
                topic_distribution=distribution
            ))
        
        # Calculate coherence (simplified)
        coherence = self._calculate_coherence(topics, documents)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TopicModelResult(
            n_documents=len(documents),
            n_topics=n_topics,
            method=method,
            topics=topics,
            doc_topics=doc_topics,
            coherence_score=coherence,
            processing_time_sec=processing_time
        )
    
    def _simple_topics(
        self,
        documents: List[str],
        n_topics: int,
        n_top_words: int,
        method: TopicMethod
    ) -> TopicModelResult:
        """Simple frequency-based topic extraction."""
        # Word frequency
        word_freq = {}
        for doc in documents:
            words = doc.lower().split()
            for word in words:
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Remove common words
        stop_words = {'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'were', 'been'}
        word_freq = {k: v for k, v in word_freq.items() if k not in stop_words}
        
        # Get top words
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        
        # Create pseudo-topics
        topics = []
        words_per_topic = len(sorted_words) // n_topics
        
        for i in range(n_topics):
            start = i * words_per_topic
            topic_words = sorted_words[start:start + n_top_words]
            
            topics.append(Topic(
                topic_id=i,
                label=f"Topic {i + 1}: {', '.join([w[0] for w in topic_words[:3]])}",
                top_words=[w[0] for w in topic_words],
                word_weights={w[0]: float(w[1]) for w in topic_words},
                coherence=0.0,
                document_count=len(documents) // n_topics
            ))
        
        return TopicModelResult(
            n_documents=len(documents),
            n_topics=n_topics,
            method=method,
            topics=topics,
            coherence_score=0.0
        )
    
    def _calculate_coherence(self, topics: List[Topic], documents: List[str]) -> float:
        """Calculate topic coherence (simplified UMass)."""
        # Count co-occurrences
        doc_words = [set(doc.lower().split()) for doc in documents]
        
        coherences = []
        for topic in topics:
            top_words = topic.top_words[:5]
            pairs_coherence = []
            
            for i, w1 in enumerate(top_words):
                for w2 in top_words[i+1:]:
                    co_occur = sum(1 for dw in doc_words if w1 in dw and w2 in dw)
                    w1_occur = sum(1 for dw in doc_words if w1 in dw)
                    
                    if w1_occur > 0:
                        pairs_coherence.append(np.log((co_occur + 1) / w1_occur))
            
            if pairs_coherence:
                coherences.append(np.mean(pairs_coherence))
        
        return float(np.mean(coherences)) if coherences else 0.0


# ============================================================================
# Factory Functions
# ============================================================================

def get_topic_engine() -> TopicModelingEngine:
    """Get topic modeling engine."""
    return TopicModelingEngine()


def quick_topics(
    documents: List[str],
    n_topics: int = 5
) -> Dict[str, Any]:
    """Quick topic modeling."""
    engine = TopicModelingEngine(verbose=False)
    result = engine.fit(documents, n_topics)
    return result.to_dict()
