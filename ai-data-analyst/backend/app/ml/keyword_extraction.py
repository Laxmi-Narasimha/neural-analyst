# AI Enterprise Data Analyst - Keyword Extraction Engine
# Production-grade keyword and keyphrase extraction
# Handles: TF-IDF, TextRank, YAKE methods

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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

class ExtractionMethod(str, Enum):
    """Keyword extraction methods."""
    TFIDF = "tfidf"
    TEXTRANK = "textrank"
    RAKE = "rake"
    FREQUENCY = "frequency"
    YAKE = "yake"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Keyword:
    """Extracted keyword."""
    keyword: str
    score: float
    frequency: int
    rank: int
    is_phrase: bool = False
    word_count: int = 1


@dataclass
class DocumentKeywords:
    """Keywords for a single document."""
    doc_id: int
    text_preview: str
    keywords: List[Keyword]


@dataclass
class KeywordResult:
    """Complete keyword extraction result."""
    n_documents: int = 0
    method: ExtractionMethod = ExtractionMethod.TFIDF
    
    # Global keywords (across all documents)
    global_keywords: List[Keyword] = field(default_factory=list)
    
    # Per-document keywords
    document_keywords: List[DocumentKeywords] = field(default_factory=list)
    
    # Keyword cloud data
    keyword_cloud: Dict[str, float] = field(default_factory=dict)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_documents": self.n_documents,
            "method": self.method.value,
            "global_keywords": [
                {
                    "keyword": k.keyword,
                    "score": round(k.score, 4),
                    "frequency": k.frequency,
                    "rank": k.rank
                }
                for k in self.global_keywords[:30]
            ],
            "keyword_cloud": {k: round(v, 4) for k, v in list(self.keyword_cloud.items())[:50]},
            "top_phrases": [
                k.keyword for k in self.global_keywords 
                if k.is_phrase
            ][:10]
        }


# ============================================================================
# Text Preprocessor
# ============================================================================

class TextPreprocessor:
    """Lightweight text preprocessing."""
    
    # Expanded stop words
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'me',
        'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'would', 'could', 'might', 'may', 'shall', 'must', 'also', 'however',
        'therefore', 'thus', 'hence', 'although', 'though', 'while', 'whereas'
    }
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        return words
    
    @staticmethod
    def remove_stopwords(words: List[str]) -> List[str]:
        """Remove stop words."""
        return [w for w in words if w not in TextPreprocessor.STOP_WORDS]
    
    @staticmethod
    def extract_ngrams(words: List[str], n: int) -> List[str]:
        """Extract n-grams."""
        if len(words) < n:
            return []
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


# ============================================================================
# Keyword Extraction Engine
# ============================================================================

class KeywordExtractionEngine:
    """
    Production-grade Keyword Extraction engine.
    
    Features:
    - Multiple extraction methods
    - Phrase extraction
    - Document-level and corpus-level keywords
    - Keyword cloud generation
    - Edge case handling
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.preprocessor = TextPreprocessor()
    
    def extract(
        self,
        documents: List[str],
        method: ExtractionMethod = ExtractionMethod.TFIDF,
        top_n: int = 20,
        include_phrases: bool = True,
        max_phrase_length: int = 3
    ) -> KeywordResult:
        """Extract keywords from documents."""
        start_time = datetime.now()
        
        # Filter empty documents
        documents = [d for d in documents if d and len(d.strip()) > 0]
        
        if len(documents) == 0:
            return KeywordResult(n_documents=0, method=method)
        
        if self.verbose:
            logger.info(f"Extracting keywords from {len(documents)} documents using {method.value}")
        
        if method == ExtractionMethod.TFIDF:
            global_keywords, doc_keywords = self._tfidf_extraction(
                documents, top_n, include_phrases, max_phrase_length
            )
        elif method == ExtractionMethod.TEXTRANK:
            global_keywords, doc_keywords = self._textrank_extraction(
                documents, top_n
            )
        elif method == ExtractionMethod.RAKE:
            global_keywords, doc_keywords = self._rake_extraction(
                documents, top_n
            )
        elif method == ExtractionMethod.YAKE:
            global_keywords, doc_keywords = self._yake_extraction(
                documents, top_n
            )
        else:  # FREQUENCY
            global_keywords, doc_keywords = self._frequency_extraction(
                documents, top_n, include_phrases, max_phrase_length
            )
        
        # Generate keyword cloud
        max_score = max((k.score for k in global_keywords), default=1)
        keyword_cloud = {
            k.keyword: k.score / max_score if max_score > 0 else 0
            for k in global_keywords[:50]
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return KeywordResult(
            n_documents=len(documents),
            method=method,
            global_keywords=global_keywords,
            document_keywords=doc_keywords,
            keyword_cloud=keyword_cloud,
            processing_time_sec=processing_time
        )
    
    def _tfidf_extraction(
        self,
        documents: List[str],
        top_n: int,
        include_phrases: bool,
        max_phrase_length: int
    ) -> Tuple[List[Keyword], List[DocumentKeywords]]:
        """TF-IDF based extraction."""
        # Tokenize all documents
        doc_tokens = []
        for doc in documents:
            words = self.preprocessor.tokenize(doc)
            words = self.preprocessor.remove_stopwords(words)
            
            if include_phrases:
                for n in range(2, max_phrase_length + 1):
                    phrases = self.preprocessor.extract_ngrams(words, n)
                    # Filter phrases with stopwords at start/end
                    phrases = [p for p in phrases 
                              if p.split()[0] not in TextPreprocessor.STOP_WORDS
                              and p.split()[-1] not in TextPreprocessor.STOP_WORDS]
                    words.extend(phrases)
            
            doc_tokens.append(words)
        
        # Calculate document frequency
        all_terms = set()
        for tokens in doc_tokens:
            all_terms.update(set(tokens))
        
        df = {}
        for term in all_terms:
            df[term] = sum(1 for tokens in doc_tokens if term in tokens)
        
        n_docs = len(documents)
        
        # Calculate TF-IDF for each document
        doc_keywords = []
        global_scores = Counter()
        global_freq = Counter()
        
        for doc_id, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            doc_scores = {}
            
            for term, count in tf.items():
                idf = math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1
                tfidf = count * idf
                doc_scores[term] = tfidf
                global_scores[term] += tfidf
                global_freq[term] += count
            
            # Top keywords for document
            sorted_terms = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_n]
            keywords = [
                Keyword(
                    keyword=term,
                    score=score,
                    frequency=tf.get(term, 0),
                    rank=i + 1,
                    is_phrase=' ' in term,
                    word_count=len(term.split())
                )
                for i, (term, score) in enumerate(sorted_terms)
            ]
            
            doc_keywords.append(DocumentKeywords(
                doc_id=doc_id,
                text_preview=documents[doc_id][:100],
                keywords=keywords
            ))
        
        # Global keywords
        sorted_global = sorted(global_scores.items(), key=lambda x: -x[1])[:top_n * 2]
        global_keywords = [
            Keyword(
                keyword=term,
                score=score,
                frequency=global_freq.get(term, 0),
                rank=i + 1,
                is_phrase=' ' in term,
                word_count=len(term.split())
            )
            for i, (term, score) in enumerate(sorted_global)
        ]
        
        return global_keywords[:top_n], doc_keywords
    
    def _frequency_extraction(
        self,
        documents: List[str],
        top_n: int,
        include_phrases: bool,
        max_phrase_length: int
    ) -> Tuple[List[Keyword], List[DocumentKeywords]]:
        """Simple frequency-based extraction."""
        global_freq = Counter()
        doc_keywords = []
        
        for doc_id, doc in enumerate(documents):
            words = self.preprocessor.tokenize(doc)
            words = self.preprocessor.remove_stopwords(words)
            
            all_terms = list(words)
            if include_phrases:
                for n in range(2, max_phrase_length + 1):
                    all_terms.extend(self.preprocessor.extract_ngrams(words, n))
            
            doc_freq = Counter(all_terms)
            global_freq.update(doc_freq)
            
            sorted_terms = doc_freq.most_common(top_n)
            keywords = [
                Keyword(
                    keyword=term,
                    score=float(count),
                    frequency=count,
                    rank=i + 1,
                    is_phrase=' ' in term
                )
                for i, (term, count) in enumerate(sorted_terms)
            ]
            
            doc_keywords.append(DocumentKeywords(
                doc_id=doc_id,
                text_preview=doc[:100],
                keywords=keywords
            ))
        
        # Global
        sorted_global = global_freq.most_common(top_n)
        global_keywords = [
            Keyword(
                keyword=term,
                score=float(count),
                frequency=count,
                rank=i + 1,
                is_phrase=' ' in term
            )
            for i, (term, count) in enumerate(sorted_global)
        ]
        
        return global_keywords, doc_keywords
    
    def _textrank_extraction(
        self,
        documents: List[str],
        top_n: int
    ) -> Tuple[List[Keyword], List[DocumentKeywords]]:
        """TextRank-based extraction (simplified)."""
        # Combine all documents for graph
        all_text = ' '.join(documents)
        words = self.preprocessor.tokenize(all_text)
        words = self.preprocessor.remove_stopwords(words)
        
        # Build co-occurrence graph
        window_size = 4
        word_set = set(words)
        graph = {w: Counter() for w in word_set}
        
        for i, word in enumerate(words):
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    graph[word][words[j]] += 1
        
        # PageRank-style scoring
        scores = {w: 1.0 for w in word_set}
        damping = 0.85
        iterations = 30
        
        for _ in range(iterations):
            new_scores = {}
            for word in word_set:
                incoming = 0
                for other, weight in graph[word].items():
                    total_out = sum(graph[other].values())
                    if total_out > 0:
                        incoming += scores[other] * weight / total_out
                new_scores[word] = (1 - damping) + damping * incoming
            scores = new_scores
        
        # Rank
        sorted_words = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        freq = Counter(words)
        
        global_keywords = [
            Keyword(
                keyword=word,
                score=score,
                frequency=freq.get(word, 0),
                rank=i + 1,
                is_phrase=False
            )
            for i, (word, score) in enumerate(sorted_words)
        ]
        
        return global_keywords, []
    
    def _rake_extraction(
        self,
        documents: List[str],
        top_n: int
    ) -> Tuple[List[Keyword], List[DocumentKeywords]]:
        """RAKE-style extraction (simplified)."""
        all_text = ' '.join(documents)
        
        # Split by stopwords to get candidate phrases
        stopword_pattern = '|'.join(TextPreprocessor.STOP_WORDS)
        phrases = re.split(r'\b(' + stopword_pattern + r')\b', all_text.lower())
        phrases = [p.strip() for p in phrases if p.strip() and p not in TextPreprocessor.STOP_WORDS]
        
        # Clean phrases
        clean_phrases = []
        for phrase in phrases:
            words = re.findall(r'[a-zA-Z]+', phrase)
            if 1 <= len(words) <= 4:
                clean_phrases.append(' '.join(words))
        
        # Score by word frequency and degree
        word_freq = Counter()
        word_degree = Counter()
        
        for phrase in clean_phrases:
            words = phrase.split()
            for word in words:
                word_freq[word] += 1
                word_degree[word] += len(words) - 1
        
        word_score = {w: (word_degree[w] + word_freq[w]) / word_freq[w] 
                      for w in word_freq}
        
        # Phrase scores
        phrase_scores = {}
        for phrase in set(clean_phrases):
            words = phrase.split()
            score = sum(word_score.get(w, 0) for w in words)
            phrase_scores[phrase] = score
        
        phrase_freq = Counter(clean_phrases)
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: -x[1])[:top_n]
        
        global_keywords = [
            Keyword(
                keyword=phrase,
                score=score,
                frequency=phrase_freq.get(phrase, 0),
                rank=i + 1,
                is_phrase=' ' in phrase,
                word_count=len(phrase.split())
            )
            for i, (phrase, score) in enumerate(sorted_phrases)
        ]
        
        return global_keywords, []
    
    def _yake_extraction(
        self,
        documents: List[str],
        top_n: int
    ) -> Tuple[List[Keyword], List[DocumentKeywords]]:
        """YAKE-style extraction (simplified)."""
        all_text = ' '.join(documents)
        words = self.preprocessor.tokenize(all_text)
        words = self.preprocessor.remove_stopwords(words)
        
        # Word statistics
        n_words = len(words)
        freq = Counter(words)
        
        # Position-weighted frequency
        position_scores = {}
        for i, word in enumerate(words):
            position = (i + 1) / n_words  # 0 to 1
            if word not in position_scores:
                position_scores[word] = []
            position_scores[word].append(position)
        
        # Calculate YAKE-like scores (lower is better, so we invert)
        scores = {}
        for word in set(words):
            tf = freq[word] / n_words
            positions = position_scores.get(word, [0.5])
            mean_pos = np.mean(positions)
            
            # Combine frequency and position
            score = tf * (1 - mean_pos)  # Favor early, frequent words
            scores[word] = score
        
        sorted_words = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        
        global_keywords = [
            Keyword(
                keyword=word,
                score=score,
                frequency=freq.get(word, 0),
                rank=i + 1,
                is_phrase=False
            )
            for i, (word, score) in enumerate(sorted_words)
        ]
        
        return global_keywords, []


# ============================================================================
# Factory Functions
# ============================================================================

def get_keyword_engine() -> KeywordExtractionEngine:
    """Get keyword extraction engine."""
    return KeywordExtractionEngine()


def quick_keywords(
    documents: List[str],
    top_n: int = 20,
    method: str = "tfidf"
) -> Dict[str, Any]:
    """Quick keyword extraction."""
    engine = KeywordExtractionEngine(verbose=False)
    result = engine.extract(documents, ExtractionMethod(method), top_n)
    return result.to_dict()


def extract_from_column(
    df: pd.DataFrame,
    text_column: str,
    top_n: int = 20
) -> Dict[str, Any]:
    """Extract keywords from DataFrame text column."""
    documents = df[text_column].dropna().astype(str).tolist()
    return quick_keywords(documents, top_n)
