# AI Enterprise Data Analyst - Text Analytics Engine
# Advanced NLP: sentiment, NER, topic modeling, summarization

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import re
from collections import Counter

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Text Analytics Types
# ============================================================================

class SentimentLabel(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EntityType(str, Enum):
    """Named entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    MONEY = "MONEY"
    PRODUCT = "PRODUCT"
    EMAIL = "EMAIL"
    URL = "URL"
    PHONE = "PHONE"


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    
    text: str
    label: SentimentLabel
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:100],
            "label": self.label.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4)
        }


@dataclass
class Entity:
    """Named entity."""
    
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4)
        }


@dataclass
class Topic:
    """Topic extracted from text."""
    
    topic_id: int
    keywords: list[str]
    weight: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "keywords": self.keywords[:10],
            "weight": round(self.weight, 4)
        }


@dataclass
class TextProfile:
    """Complete text analysis profile."""
    
    text: str
    word_count: int
    char_count: int
    sentence_count: int
    
    sentiment: Optional[SentimentResult] = None
    entities: list[Entity] = field(default_factory=list)
    topics: list[Topic] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    
    # Readability
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics": {
                "word_count": self.word_count,
                "char_count": self.char_count,
                "sentence_count": self.sentence_count,
                "avg_word_length": round(self.avg_word_length, 2),
                "avg_sentence_length": round(self.avg_sentence_length, 2),
                "vocabulary_richness": round(self.vocabulary_richness, 4)
            },
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "entities": [e.to_dict() for e in self.entities[:20]],
            "topics": [t.to_dict() for t in self.topics],
            "keywords": self.keywords[:20]
        }


# ============================================================================
# Text Preprocessor
# ============================================================================

class TextPreprocessor:
    """Text preprocessing and cleaning."""
    
    def __init__(self):
        self._stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'can', 'now'
        }
    
    def clean(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True
    ) -> str:
        """Clean text."""
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        if remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        if lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> list[str]:
        """Simple word tokenization."""
        return text.split()
    
    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords."""
        return [t for t in tokens if t.lower() not in self._stopwords]
    
    def get_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# Sentiment Analyzer
# ============================================================================

class SentimentAnalyzer:
    """
    Sentiment analysis using lexicon and ML approaches.
    """
    
    def __init__(self):
        # Simple sentiment lexicon
        self._positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'happy', 'joy', 'best', 'beautiful', 'perfect', 'awesome',
            'brilliant', 'outstanding', 'superb', 'nice', 'positive', 'success',
            'win', 'winning', 'pleased', 'delighted', 'satisfied', 'recommend'
        }
        
        self._negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'sad',
            'angry', 'disappointed', 'poor', 'wrong', 'fail', 'failure',
            'negative', 'problem', 'issue', 'broken', 'useless', 'waste',
            'annoying', 'frustrating', 'upset', 'unhappy', 'regret', 'avoid'
        }
        
        self._intensifiers = {'very', 'really', 'extremely', 'absolutely', 'totally'}
        self._negators = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing'}
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_count = 0
        neg_count = 0
        intensity = 1.0
        negation = False
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self._intensifiers:
                intensity = 1.5
                continue
            
            # Check for negators
            if word in self._negators or word.endswith("n't"):
                negation = True
                continue
            
            # Check sentiment
            if word in self._positive_words:
                if negation:
                    neg_count += intensity
                else:
                    pos_count += intensity
                negation = False
                intensity = 1.0
            elif word in self._negative_words:
                if negation:
                    pos_count += intensity
                else:
                    neg_count += intensity
                negation = False
                intensity = 1.0
        
        # Calculate score
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
            confidence = 0.3
        else:
            score = (pos_count - neg_count) / total
            confidence = min(total / len(words) * 2, 1.0) if words else 0.5
        
        # Determine label
        if score > 0.1:
            label = SentimentLabel.POSITIVE
        elif score < -0.1:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL
        
        return SentimentResult(
            text=text,
            label=label,
            score=score,
            confidence=confidence
        )
    
    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze(t) for t in texts]


# ============================================================================
# Named Entity Recognizer
# ============================================================================

class EntityRecognizer:
    """
    Named Entity Recognition using patterns and heuristics.
    """
    
    def __init__(self):
        self._patterns = {
            EntityType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            EntityType.URL: r'https?://[^\s]+|www\.[^\s]+',
            EntityType.PHONE: r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            EntityType.MONEY: r'\$[\d,]+\.?\d*|\d+\s*(?:dollars?|USD|EUR|GBP)',
            EntityType.DATE: r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        }
    
    def extract(self, text: str) -> list[Entity]:
        """Extract named entities from text."""
        entities = []
        
        # Pattern-based extraction
        for entity_type, pattern in self._patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end()
                ))
        
        # Capitalized phrases (potential names/organizations)
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(cap_pattern, text):
            phrase = match.group()
            words = phrase.split()
            
            if len(words) >= 2:
                # Skip if already matched
                if not any(e.start == match.start() for e in entities):
                    entities.append(Entity(
                        text=phrase,
                        entity_type=EntityType.PERSON if len(words) <= 3 else EntityType.ORGANIZATION,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7
                    ))
        
        return entities


# ============================================================================
# Topic Modeler
# ============================================================================

class TopicModeler:
    """
    Topic modeling using TF-IDF and clustering.
    """
    
    def __init__(self, n_topics: int = 5):
        self.n_topics = n_topics
        self._vectorizer = None
        self._fitted = False
    
    def fit(self, documents: list[str]) -> dict[str, Any]:
        """Fit topic model on documents."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import NMF
            
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
            
            tfidf = self._vectorizer.fit_transform(documents)
            
            self._nmf = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200
            )
            
            self._doc_topics = self._nmf.fit_transform(tfidf)
            self._fitted = True
            
            # Extract topic keywords
            feature_names = self._vectorizer.get_feature_names_out()
            topics = []
            
            for idx, topic in enumerate(self._nmf.components_):
                top_indices = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                topics.append({
                    "topic_id": idx,
                    "keywords": keywords,
                    "weight": float(topic[top_indices].mean())
                })
            
            return {"topics": topics, "document_count": len(documents)}
            
        except ImportError:
            logger.warning("sklearn not available for topic modeling")
            return {}
    
    def get_document_topics(self, document: str) -> list[Topic]:
        """Get topics for a single document."""
        if not self._fitted:
            return []
        
        tfidf = self._vectorizer.transform([document])
        topic_dist = self._nmf.transform(tfidf)[0]
        
        feature_names = self._vectorizer.get_feature_names_out()
        topics = []
        
        for idx, weight in enumerate(topic_dist):
            if weight > 0.1:
                top_indices = self._nmf.components_[idx].argsort()[-5:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                topics.append(Topic(
                    topic_id=idx,
                    keywords=keywords,
                    weight=float(weight)
                ))
        
        return sorted(topics, key=lambda t: t.weight, reverse=True)


# ============================================================================
# Keyword Extractor
# ============================================================================

class KeywordExtractor:
    """Extract keywords using TF-IDF and statistical methods."""
    
    def extract(
        self,
        text: str,
        n_keywords: int = 10,
        min_word_length: int = 3
    ) -> list[str]:
        """Extract keywords from text."""
        preprocessor = TextPreprocessor()
        
        # Clean and tokenize
        clean_text = preprocessor.clean(text)
        tokens = preprocessor.tokenize(clean_text)
        tokens = preprocessor.remove_stopwords(tokens)
        
        # Filter by length
        tokens = [t for t in tokens if len(t) >= min_word_length]
        
        # Count frequencies
        counter = Counter(tokens)
        
        # Score by frequency and position
        scored = []
        for word, freq in counter.items():
            # Position bonus (words appearing early are more important)
            first_pos = tokens.index(word) if word in tokens else len(tokens)
            position_score = 1 - (first_pos / len(tokens))
            
            # Combined score
            score = freq * (1 + position_score * 0.5)
            scored.append((word, score))
        
        # Sort and return top keywords
        scored.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored[:n_keywords]]


# ============================================================================
# Text Summarizer
# ============================================================================

class TextSummarizer:
    """Extractive text summarization."""
    
    def summarize(
        self,
        text: str,
        n_sentences: int = 3
    ) -> str:
        """Generate extractive summary."""
        preprocessor = TextPreprocessor()
        sentences = preprocessor.get_sentences(text)
        
        if len(sentences) <= n_sentences:
            return text
        
        # Score sentences
        word_freq = Counter(preprocessor.remove_stopwords(
            preprocessor.tokenize(preprocessor.clean(text))
        ))
        
        sentence_scores = []
        for i, sent in enumerate(sentences):
            words = preprocessor.tokenize(preprocessor.clean(sent))
            score = sum(word_freq.get(w, 0) for w in words)
            
            # Normalize by length
            if words:
                score /= len(words)
            
            # Position bonus (first and last sentences often important)
            if i == 0:
                score *= 1.2
            elif i == len(sentences) - 1:
                score *= 1.1
            
            sentence_scores.append((i, sent, score))
        
        # Select top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)[:n_sentences]
        
        # Restore original order
        top_sentences.sort(key=lambda x: x[0])
        
        return ' '.join(s[1] for s in top_sentences)


# ============================================================================
# Text Analytics Engine
# ============================================================================

class TextAnalyticsEngine:
    """
    Unified text analytics engine.
    
    Features:
    - Preprocessing and cleaning
    - Sentiment analysis
    - Named entity recognition
    - Topic modeling
    - Keyword extraction
    - Text summarization
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment = SentimentAnalyzer()
        self.ner = EntityRecognizer()
        self.topics = TopicModeler()
        self.keywords = KeywordExtractor()
        self.summarizer = TextSummarizer()
    
    def analyze(self, text: str) -> TextProfile:
        """Perform comprehensive text analysis."""
        # Basic statistics
        words = self.preprocessor.tokenize(text)
        sentences = self.preprocessor.get_sentences(text)
        unique_words = set(w.lower() for w in words)
        
        profile = TextProfile(
            text=text,
            word_count=len(words),
            char_count=len(text),
            sentence_count=len(sentences),
            avg_word_length=np.mean([len(w) for w in words]) if words else 0,
            avg_sentence_length=len(words) / len(sentences) if sentences else 0,
            vocabulary_richness=len(unique_words) / len(words) if words else 0
        )
        
        # Sentiment
        profile.sentiment = self.sentiment.analyze(text)
        
        # Entities
        profile.entities = self.ner.extract(text)
        
        # Keywords
        profile.keywords = self.keywords.extract(text)
        
        return profile
    
    def analyze_batch(
        self,
        texts: list[str],
        include_topics: bool = True
    ) -> dict[str, Any]:
        """Analyze batch of texts."""
        profiles = [self.analyze(t) for t in texts]
        
        result = {
            "count": len(texts),
            "profiles": [p.to_dict() for p in profiles],
            "aggregate": {
                "avg_word_count": np.mean([p.word_count for p in profiles]),
                "sentiment_distribution": {
                    "positive": sum(1 for p in profiles if p.sentiment and p.sentiment.label == SentimentLabel.POSITIVE),
                    "negative": sum(1 for p in profiles if p.sentiment and p.sentiment.label == SentimentLabel.NEGATIVE),
                    "neutral": sum(1 for p in profiles if p.sentiment and p.sentiment.label == SentimentLabel.NEUTRAL)
                }
            }
        }
        
        # Topic modeling
        if include_topics and len(texts) >= 10:
            topic_result = self.topics.fit(texts)
            result["topics"] = topic_result.get("topics", [])
        
        return result
    
    def summarize(self, text: str, n_sentences: int = 3) -> str:
        """Generate text summary."""
        return self.summarizer.summarize(text, n_sentences)


# Factory function
def get_text_analytics_engine() -> TextAnalyticsEngine:
    """Get text analytics engine instance."""
    return TextAnalyticsEngine()
