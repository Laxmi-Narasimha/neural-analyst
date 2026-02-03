# AI Enterprise Data Analyst - Text Summarization Engine
# Production-grade text summarization
# Handles: extractive summarization, key sentence selection

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

class SummarizationMethod(str, Enum):
    """Summarization methods."""
    EXTRACTIVE = "extractive"
    FREQUENCY = "frequency"
    POSITION = "position"
    TEXTRANK = "textrank"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Sentence:
    """Sentence with score."""
    text: str
    index: int
    score: float
    word_count: int


@dataclass
class SummaryResult:
    """Complete summarization result."""
    original_text: str = ""
    summary: str = ""
    
    n_original_sentences: int = 0
    n_summary_sentences: int = 0
    
    compression_ratio: float = 0.0
    
    key_sentences: List[Sentence] = field(default_factory=list)
    
    # Keywords found
    top_keywords: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "stats": {
                "original_sentences": self.n_original_sentences,
                "summary_sentences": self.n_summary_sentences,
                "compression_ratio": round(self.compression_ratio, 2)
            },
            "key_sentences": [
                {"text": s.text[:100], "score": round(s.score, 4)}
                for s in self.key_sentences[:5]
            ],
            "top_keywords": self.top_keywords[:10]
        }


# ============================================================================
# Text Summarization Engine
# ============================================================================

class TextSummarizationEngine:
    """
    Production-grade Text Summarization engine.
    
    Features:
    - Extractive summarization
    - Multiple scoring methods
    - Sentence ranking
    - Keyword extraction
    - Compression control
    """
    
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'then', 'once', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'would', 'could', 'should', 'will', 'shall', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some'
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def summarize(
        self,
        text: str,
        method: SummarizationMethod = SummarizationMethod.TEXTRANK,
        ratio: float = 0.3,
        max_sentences: int = None
    ) -> SummaryResult:
        """Summarize text."""
        start_time = datetime.now()
        
        if not text or len(text.strip()) == 0:
            return SummaryResult(original_text="", summary="")
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 2:
            return SummaryResult(
                original_text=text,
                summary=text,
                n_original_sentences=len(sentences),
                n_summary_sentences=len(sentences),
                compression_ratio=1.0
            )
        
        if self.verbose:
            logger.info(f"Summarizing {len(sentences)} sentences using {method.value}")
        
        # Score sentences
        if method == SummarizationMethod.TEXTRANK:
            scored = self._textrank_score(sentences)
        elif method == SummarizationMethod.FREQUENCY:
            scored = self._frequency_score(sentences)
        elif method == SummarizationMethod.POSITION:
            scored = self._position_score(sentences)
        else:
            scored = self._combined_score(sentences)
        
        # Select top sentences
        n_select = max_sentences or max(1, int(len(sentences) * ratio))
        n_select = min(n_select, len(scored))
        
        # Sort by score and take top
        top_sentences = sorted(scored, key=lambda x: -x.score)[:n_select]
        
        # Reorder by original position
        top_sentences.sort(key=lambda x: x.index)
        
        # Generate summary
        summary = ' '.join(s.text for s in top_sentences)
        
        # Extract keywords
        all_words = []
        for s in sentences:
            words = self._tokenize(s)
            all_words.extend(words)
        
        word_freq = {}
        for w in all_words:
            if w not in self.STOP_WORDS and len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1
        
        top_keywords = sorted(word_freq.keys(), key=lambda x: -word_freq[x])[:10]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            n_original_sentences=len(sentences),
            n_summary_sentences=len(top_sentences),
            compression_ratio=len(summary) / len(text) if len(text) > 0 else 1,
            key_sentences=top_sentences,
            top_keywords=top_keywords,
            processing_time_sec=processing_time
        )
    
    def summarize_column(
        self,
        df: pd.DataFrame,
        text_col: str,
        ratio: float = 0.3
    ) -> pd.DataFrame:
        """Summarize each text in a column."""
        summaries = []
        
        for text in df[text_col].fillna(''):
            result = self.summarize(str(text), ratio=ratio)
            summaries.append(result.summary)
        
        result_df = df.copy()
        result_df[f'{text_col}_summary'] = summaries
        
        return result_df
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return words
    
    def _frequency_score(self, sentences: List[str]) -> List[Sentence]:
        """Score sentences by word frequency."""
        # Calculate word frequencies
        all_words = []
        for s in sentences:
            all_words.extend(self._tokenize(s))
        
        word_freq = {}
        for w in all_words:
            if w not in self.STOP_WORDS:
                word_freq[w] = word_freq.get(w, 0) + 1
        
        # Normalize
        max_freq = max(word_freq.values()) if word_freq else 1
        word_freq = {k: v / max_freq for k, v in word_freq.items()}
        
        # Score sentences
        scored = []
        for i, sent in enumerate(sentences):
            words = self._tokenize(sent)
            score = sum(word_freq.get(w, 0) for w in words) / len(words) if words else 0
            
            scored.append(Sentence(
                text=sent,
                index=i,
                score=score,
                word_count=len(words)
            ))
        
        return scored
    
    def _position_score(self, sentences: List[str]) -> List[Sentence]:
        """Score sentences by position (first and last are more important)."""
        n = len(sentences)
        scored = []
        
        for i, sent in enumerate(sentences):
            # Higher score for beginning and end
            position_weight = max(1 - (i / n), (i / n)) * 0.5 + 0.5
            
            # Boost first sentence
            if i == 0:
                position_weight = 1.0
            
            words = self._tokenize(sent)
            
            scored.append(Sentence(
                text=sent,
                index=i,
                score=position_weight,
                word_count=len(words)
            ))
        
        return scored
    
    def _textrank_score(self, sentences: List[str]) -> List[Sentence]:
        """Score sentences using TextRank algorithm."""
        n = len(sentences)
        
        # Build similarity matrix
        similarity = np.zeros((n, n))
        
        for i in range(n):
            words_i = set(self._tokenize(sentences[i])) - self.STOP_WORDS
            if not words_i:
                continue
                
            for j in range(i + 1, n):
                words_j = set(self._tokenize(sentences[j])) - self.STOP_WORDS
                if not words_j:
                    continue
                
                # Jaccard-like similarity
                overlap = len(words_i & words_j)
                if overlap > 0:
                    sim = overlap / (np.log(len(words_i)) + np.log(len(words_j)) + 1)
                    similarity[i][j] = sim
                    similarity[j][i] = sim
        
        # Normalize
        for i in range(n):
            row_sum = similarity[i].sum()
            if row_sum > 0:
                similarity[i] /= row_sum
        
        # PageRank iteration
        scores = np.ones(n) / n
        damping = 0.85
        
        for _ in range(30):
            new_scores = (1 - damping) / n + damping * similarity.T.dot(scores)
            if np.allclose(scores, new_scores):
                break
            scores = new_scores
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        scored = []
        for i, sent in enumerate(sentences):
            words = self._tokenize(sent)
            scored.append(Sentence(
                text=sent,
                index=i,
                score=float(scores[i]),
                word_count=len(words)
            ))
        
        return scored
    
    def _combined_score(self, sentences: List[str]) -> List[Sentence]:
        """Combine multiple scoring methods."""
        freq_scored = self._frequency_score(sentences)
        pos_scored = self._position_score(sentences)
        textrank_scored = self._textrank_score(sentences)
        
        combined = []
        for i in range(len(sentences)):
            combined_score = (
                0.4 * textrank_scored[i].score +
                0.4 * freq_scored[i].score +
                0.2 * pos_scored[i].score
            )
            
            combined.append(Sentence(
                text=sentences[i],
                index=i,
                score=combined_score,
                word_count=textrank_scored[i].word_count
            ))
        
        return combined


# ============================================================================
# Factory Functions
# ============================================================================

def get_summarization_engine() -> TextSummarizationEngine:
    """Get text summarization engine."""
    return TextSummarizationEngine()


def quick_summarize(
    text: str,
    ratio: float = 0.3
) -> str:
    """Quick text summarization."""
    engine = TextSummarizationEngine(verbose=False)
    result = engine.summarize(text, ratio=ratio)
    return result.summary


def summarize_documents(
    documents: List[str],
    ratio: float = 0.3
) -> List[str]:
    """Summarize multiple documents."""
    engine = TextSummarizationEngine(verbose=False)
    return [engine.summarize(doc, ratio=ratio).summary for doc in documents]
