# AI Enterprise Data Analyst - Text Preprocessing Engine
# Production-grade text cleaning and preprocessing
# Handles: any text data, multiple languages, cleaning strategies

from __future__ import annotations

import re
import string
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums and Types
# ============================================================================

class TextCleaningStep(str, Enum):
    """Text cleaning operations."""
    LOWERCASE = "lowercase"
    REMOVE_URLS = "remove_urls"
    REMOVE_EMAILS = "remove_emails"
    REMOVE_MENTIONS = "remove_mentions"
    REMOVE_HASHTAGS = "remove_hashtags"
    REMOVE_NUMBERS = "remove_numbers"
    REMOVE_PUNCTUATION = "remove_punctuation"
    REMOVE_WHITESPACE = "remove_whitespace"
    REMOVE_STOPWORDS = "remove_stopwords"
    STEMMING = "stemming"
    LEMMATIZATION = "lemmatization"
    REMOVE_HTML = "remove_html"
    REMOVE_SPECIAL_CHARS = "remove_special_chars"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TextCleaningConfig:
    """Configuration for text preprocessing."""
    steps: List[TextCleaningStep] = field(default_factory=lambda: [
        TextCleaningStep.LOWERCASE,
        TextCleaningStep.REMOVE_URLS,
        TextCleaningStep.REMOVE_HTML,
        TextCleaningStep.REMOVE_PUNCTUATION,
        TextCleaningStep.REMOVE_WHITESPACE
    ])
    
    language: str = "english"
    custom_stopwords: List[str] = field(default_factory=list)
    min_word_length: int = 2
    max_word_length: int = 50


@dataclass
class TextStats:
    """Statistics about text data."""
    n_documents: int = 0
    avg_length: float = 0.0
    avg_word_count: float = 0.0
    unique_words: int = 0
    most_common_words: List[tuple] = field(default_factory=list)
    language_detected: str = ""


@dataclass
class TextPreprocessingResult:
    """Result of text preprocessing."""
    cleaned_texts: pd.Series = None
    original_texts: pd.Series = None
    stats_before: TextStats = None
    stats_after: TextStats = None
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats_before": {
                "n_documents": self.stats_before.n_documents,
                "avg_length": round(self.stats_before.avg_length, 1),
                "avg_word_count": round(self.stats_before.avg_word_count, 1)
            } if self.stats_before else {},
            "stats_after": {
                "n_documents": self.stats_after.n_documents,
                "avg_length": round(self.stats_after.avg_length, 1),
                "avg_word_count": round(self.stats_after.avg_word_count, 1),
                "unique_words": self.stats_after.unique_words
            } if self.stats_after else {},
            "sample_cleaned": self.cleaned_texts.head(5).tolist() if self.cleaned_texts is not None else []
        }


# ============================================================================
# Text Cleaning Functions
# ============================================================================

class TextCleaners:
    """Individual text cleaning functions."""
    
    URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#\w+')
    HTML_PATTERN = re.compile(r'<[^>]+>')
    SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s]')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    NUMBER_PATTERN = re.compile(r'\d+')
    
    @staticmethod
    def lowercase(text: str) -> str:
        return text.lower()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        return TextCleaners.URL_PATTERN.sub(' ', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        return TextCleaners.EMAIL_PATTERN.sub(' ', text)
    
    @staticmethod
    def remove_mentions(text: str) -> str:
        return TextCleaners.MENTION_PATTERN.sub(' ', text)
    
    @staticmethod
    def remove_hashtags(text: str) -> str:
        return TextCleaners.HASHTAG_PATTERN.sub(' ', text)
    
    @staticmethod
    def remove_html(text: str) -> str:
        return TextCleaners.HTML_PATTERN.sub(' ', text)
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        return TextCleaners.NUMBER_PATTERN.sub(' ', text)
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))
    
    @staticmethod
    def remove_special_chars(text: str) -> str:
        return TextCleaners.SPECIAL_CHARS_PATTERN.sub(' ', text)
    
    @staticmethod
    def remove_whitespace(text: str) -> str:
        return TextCleaners.WHITESPACE_PATTERN.sub(' ', text).strip()


# ============================================================================
# Text Preprocessing Engine
# ============================================================================

class TextPreprocessingEngine:
    """
    Complete Text Preprocessing engine.
    
    Features:
    - Multiple cleaning operations
    - Stopword removal
    - Stemming and lemmatization
    - Statistics calculation
    """
    
    def __init__(self, config: TextCleaningConfig = None, verbose: bool = True):
        self.config = config or TextCleaningConfig()
        self.verbose = verbose
        self._stopwords = set()
        self._stemmer = None
        self._lemmatizer = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP components."""
        if HAS_NLTK:
            try:
                self._stopwords = set(stopwords.words(self.config.language))
            except:
                self._stopwords = set()
            
            self._stopwords.update(self.config.custom_stopwords)
            self._stemmer = PorterStemmer()
            
            try:
                self._lemmatizer = WordNetLemmatizer()
            except:
                self._lemmatizer = None
    
    def preprocess(
        self,
        df: pd.DataFrame,
        text_col: str = None
    ) -> TextPreprocessingResult:
        """Preprocess text data."""
        start_time = datetime.now()
        
        # Auto-detect text column
        if text_col is None:
            text_col = self._detect_text_col(df)
        
        if self.verbose:
            logger.info(f"Preprocessing column: {text_col}")
        
        original = df[text_col].astype(str).fillna('')
        
        # Stats before
        stats_before = self._calculate_stats(original)
        
        # Apply cleaning steps
        cleaned = original.copy()
        
        for step in self.config.steps:
            cleaned = self._apply_step(cleaned, step)
        
        # Filter by word length
        if self.config.min_word_length > 1 or self.config.max_word_length < 100:
            cleaned = cleaned.apply(lambda x: self._filter_words(x))
        
        # Stats after
        stats_after = self._calculate_stats(cleaned)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TextPreprocessingResult(
            cleaned_texts=cleaned,
            original_texts=original,
            stats_before=stats_before,
            stats_after=stats_after,
            processing_time_sec=processing_time
        )
    
    def _apply_step(self, series: pd.Series, step: TextCleaningStep) -> pd.Series:
        """Apply a single cleaning step."""
        if step == TextCleaningStep.LOWERCASE:
            return series.apply(TextCleaners.lowercase)
        elif step == TextCleaningStep.REMOVE_URLS:
            return series.apply(TextCleaners.remove_urls)
        elif step == TextCleaningStep.REMOVE_EMAILS:
            return series.apply(TextCleaners.remove_emails)
        elif step == TextCleaningStep.REMOVE_MENTIONS:
            return series.apply(TextCleaners.remove_mentions)
        elif step == TextCleaningStep.REMOVE_HASHTAGS:
            return series.apply(TextCleaners.remove_hashtags)
        elif step == TextCleaningStep.REMOVE_HTML:
            return series.apply(TextCleaners.remove_html)
        elif step == TextCleaningStep.REMOVE_NUMBERS:
            return series.apply(TextCleaners.remove_numbers)
        elif step == TextCleaningStep.REMOVE_PUNCTUATION:
            return series.apply(TextCleaners.remove_punctuation)
        elif step == TextCleaningStep.REMOVE_SPECIAL_CHARS:
            return series.apply(TextCleaners.remove_special_chars)
        elif step == TextCleaningStep.REMOVE_WHITESPACE:
            return series.apply(TextCleaners.remove_whitespace)
        elif step == TextCleaningStep.REMOVE_STOPWORDS:
            return series.apply(self._remove_stopwords)
        elif step == TextCleaningStep.STEMMING:
            return series.apply(self._stem_text)
        elif step == TextCleaningStep.LEMMATIZATION:
            return series.apply(self._lemmatize_text)
        return series
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        return ' '.join(w for w in words if w.lower() not in self._stopwords)
    
    def _stem_text(self, text: str) -> str:
        """Apply stemming."""
        if self._stemmer is None:
            return text
        words = text.split()
        return ' '.join(self._stemmer.stem(w) for w in words)
    
    def _lemmatize_text(self, text: str) -> str:
        """Apply lemmatization."""
        if self._lemmatizer is None:
            return text
        words = text.split()
        return ' '.join(self._lemmatizer.lemmatize(w) for w in words)
    
    def _filter_words(self, text: str) -> str:
        """Filter words by length."""
        words = text.split()
        return ' '.join(
            w for w in words 
            if self.config.min_word_length <= len(w) <= self.config.max_word_length
        )
    
    def _calculate_stats(self, series: pd.Series) -> TextStats:
        """Calculate text statistics."""
        lengths = series.str.len()
        word_counts = series.str.split().str.len()
        
        # Most common words
        all_words = ' '.join(series).split()
        word_freq = pd.Series(all_words).value_counts()
        
        return TextStats(
            n_documents=len(series),
            avg_length=float(lengths.mean()),
            avg_word_count=float(word_counts.mean()),
            unique_words=len(set(all_words)),
            most_common_words=word_freq.head(20).items()
        )
    
    def _detect_text_col(self, df: pd.DataFrame) -> str:
        """Auto-detect text column."""
        patterns = ['text', 'content', 'message', 'comment', 'review', 'description', 'body']
        
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        
        # Find longest string column
        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            avg_lengths = {col: df[col].astype(str).str.len().mean() for col in string_cols}
            return max(avg_lengths, key=avg_lengths.get)
        
        return df.columns[0]


# ============================================================================
# Factory Functions
# ============================================================================

def get_text_preprocessing_engine(config: TextCleaningConfig = None) -> TextPreprocessingEngine:
    """Get text preprocessing engine."""
    return TextPreprocessingEngine(config=config)


def quick_clean_text(
    df: pd.DataFrame,
    text_col: str = None
) -> Dict[str, Any]:
    """Quick text cleaning."""
    engine = TextPreprocessingEngine(verbose=False)
    result = engine.preprocess(df, text_col)
    return result.to_dict()
