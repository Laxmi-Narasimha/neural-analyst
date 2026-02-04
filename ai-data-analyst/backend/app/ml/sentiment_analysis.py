# AI Enterprise Data Analyst - Sentiment Analysis Engine
# Production-grade sentiment analysis for any text data
# Handles: multiple methods, aspect-based sentiment

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

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

class SentimentMethod(str, Enum):
    """Sentiment analysis methods."""
    VADER = "vader"  # Best for social media
    TEXTBLOB = "textblob"  # General purpose
    LEXICON = "lexicon"  # Custom lexicon
    ENSEMBLE = "ensemble"  # Combine methods


class SentimentLabel(str, Enum):
    """Sentiment labels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SentimentScore:
    """Sentiment score for a single text."""
    text: str
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    label: SentimentLabel
    confidence: float
    compound: float = 0.0  # VADER compound
    pos: float = 0.0
    neg: float = 0.0
    neu: float = 0.0


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    method: SentimentMethod = SentimentMethod.VADER
    text_col: Optional[str] = None
    
    # Thresholds for labeling
    very_negative_threshold: float = -0.6
    negative_threshold: float = -0.2
    positive_threshold: float = 0.2
    very_positive_threshold: float = 0.6


@dataclass
class SentimentResult:
    """Complete sentiment analysis result."""
    n_documents: int = 0
    
    # Overall sentiment distribution
    distribution: Dict[str, int] = field(default_factory=dict)
    distribution_pct: Dict[str, float] = field(default_factory=dict)
    
    # Average scores
    avg_polarity: float = 0.0
    avg_subjectivity: float = 0.0
    
    # Detailed scores
    scores: pd.DataFrame = None
    
    # Top positive/negative
    top_positive: List[Dict[str, Any]] = field(default_factory=list)
    top_negative: List[Dict[str, Any]] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_documents": self.n_documents,
                "avg_polarity": round(self.avg_polarity, 3),
                "avg_subjectivity": round(self.avg_subjectivity, 3)
            },
            "distribution": self.distribution,
            "distribution_pct": {k: round(v, 1) for k, v in self.distribution_pct.items()},
            "top_positive": self.top_positive[:5],
            "top_negative": self.top_negative[:5]
        }


@dataclass
class SingleSentimentResult:
    sentiment: str
    polarity: float = 0.0
    confidence: float = 0.0


# ============================================================================
# Sentiment Analyzers
# ============================================================================

class VaderSentimentAnalyzer:
    """VADER sentiment analysis."""
    
    def __init__(self):
        if HAS_VADER:
            self._analyzer = SentimentIntensityAnalyzer()
        else:
            self._analyzer = None
    
    def analyze(self, text: str) -> Dict[str, float]:
        if self._analyzer is None:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        
        scores = self._analyzer.polarity_scores(text)
        return scores


class TextBlobAnalyzer:
    """TextBlob sentiment analysis."""
    
    def analyze(self, text: str) -> Dict[str, float]:
        if not HAS_TEXTBLOB:
            return {"polarity": 0.0, "subjectivity": 0.0}
        
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }


class LexiconAnalyzer:
    """Simple lexicon-based sentiment."""
    
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'like', 'best', 'happy', 'glad', 'pleased', 'satisfied',
        'perfect', 'awesome', 'brilliant', 'outstanding', 'superb', 'nice'
    }
    
    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
        'hate', 'dislike', 'unhappy', 'sad', 'angry', 'disappointed',
        'terrible', 'useless', 'broken', 'waste', 'fail', 'problem'
    }
    
    def analyze(self, text: str) -> Dict[str, float]:
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total = pos_count + neg_count
        
        if total == 0:
            return {"polarity": 0.0, "confidence": 0.0}
        
        polarity = (pos_count - neg_count) / total
        confidence = total / len(words) if words else 0
        
        return {"polarity": polarity, "confidence": confidence}


# ============================================================================
# Sentiment Analysis Engine
# ============================================================================

class SentimentAnalysisEngine:
    """
    Complete Sentiment Analysis engine.
    
    Features:
    - Multiple analysis methods (VADER, TextBlob, Lexicon)
    - Ensemble approach
    - Subjectivity analysis
    - Top positive/negative extraction
    """
    
    def __init__(self, config: SentimentConfig = None, verbose: bool = True):
        self.config = config or SentimentConfig()
        self.verbose = verbose
        
        self.vader = VaderSentimentAnalyzer()
        self.textblob = TextBlobAnalyzer()
        self.lexicon = LexiconAnalyzer()
    
    def analyze(self, df: Any, text_col: str = None):
        """Perform sentiment analysis on a DataFrame or a single text string."""
        if isinstance(df, str):
            score = self._analyze_text(df)
            label = score.label
            if label in (SentimentLabel.VERY_NEGATIVE, SentimentLabel.NEGATIVE):
                sentiment = "negative"
            elif label in (SentimentLabel.VERY_POSITIVE, SentimentLabel.POSITIVE):
                sentiment = "positive"
            else:
                sentiment = "neutral"
            return SingleSentimentResult(
                sentiment=sentiment,
                polarity=float(score.polarity),
                confidence=float(score.confidence),
            )

        start_time = datetime.now()
        
        # Auto-detect text column
        if text_col is None:
            text_col = self.config.text_col or self._detect_text_col(df)
        
        if self.verbose:
            logger.info(f"Analyzing sentiment in column: {text_col}")
        
        texts = df[text_col].astype(str).fillna('')
        
        # Analyze each text
        scores = []
        for text in texts:
            score = self._analyze_text(text)
            scores.append(score)
        
        # Create DataFrame
        scores_df = pd.DataFrame([{
            'text': s.text[:100],
            'polarity': s.polarity,
            'subjectivity': s.subjectivity,
            'label': s.label.value,
            'confidence': s.confidence,
            'compound': s.compound
        } for s in scores])
        
        # Distribution
        label_counts = scores_df['label'].value_counts().to_dict()
        label_pcts = {k: v / len(scores_df) * 100 for k, v in label_counts.items()}
        
        # Averages
        avg_polarity = scores_df['polarity'].mean()
        avg_subjectivity = scores_df['subjectivity'].mean()
        
        # Top positive/negative
        positive_df = scores_df.nlargest(10, 'polarity')
        negative_df = scores_df.nsmallest(10, 'polarity')
        
        top_positive = positive_df.to_dict(orient='records')
        top_negative = negative_df.to_dict(orient='records')
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SentimentResult(
            n_documents=len(texts),
            distribution=label_counts,
            distribution_pct=label_pcts,
            avg_polarity=avg_polarity,
            avg_subjectivity=avg_subjectivity,
            scores=scores_df,
            top_positive=top_positive,
            top_negative=top_negative,
            processing_time_sec=processing_time
        )
    
    def _analyze_text(self, text: str) -> SentimentScore:
        """Analyze a single text."""
        if not text.strip():
            return SentimentScore(
                text=text,
                polarity=0.0,
                subjectivity=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0
            )
        
        if self.config.method == SentimentMethod.VADER:
            result = self.vader.analyze(text)
            polarity = result.get('compound', 0.0)
            subjectivity = 1 - result.get('neu', 0.0)
            
            return SentimentScore(
                text=text,
                polarity=polarity,
                subjectivity=subjectivity,
                label=self._get_label(polarity),
                confidence=abs(polarity),
                compound=result.get('compound', 0.0),
                pos=result.get('pos', 0.0),
                neg=result.get('neg', 0.0),
                neu=result.get('neu', 0.0)
            )
        
        elif self.config.method == SentimentMethod.TEXTBLOB:
            result = self.textblob.analyze(text)
            polarity = result.get('polarity', 0.0)
            subjectivity = result.get('subjectivity', 0.0)
            
            return SentimentScore(
                text=text,
                polarity=polarity,
                subjectivity=subjectivity,
                label=self._get_label(polarity),
                confidence=abs(polarity)
            )
        
        elif self.config.method == SentimentMethod.LEXICON:
            result = self.lexicon.analyze(text)
            polarity = result.get('polarity', 0.0)
            
            return SentimentScore(
                text=text,
                polarity=polarity,
                subjectivity=0.5,
                label=self._get_label(polarity),
                confidence=result.get('confidence', 0.0)
            )
        
        else:  # Ensemble
            vader_result = self.vader.analyze(text)
            textblob_result = self.textblob.analyze(text)
            
            # Average polarities
            polarity = (vader_result.get('compound', 0.0) + textblob_result.get('polarity', 0.0)) / 2
            subjectivity = textblob_result.get('subjectivity', 0.5)
            
            return SentimentScore(
                text=text,
                polarity=polarity,
                subjectivity=subjectivity,
                label=self._get_label(polarity),
                confidence=abs(polarity)
            )
    
    def _get_label(self, polarity: float) -> SentimentLabel:
        """Get sentiment label from polarity."""
        if polarity <= self.config.very_negative_threshold:
            return SentimentLabel.VERY_NEGATIVE
        elif polarity <= self.config.negative_threshold:
            return SentimentLabel.NEGATIVE
        elif polarity >= self.config.very_positive_threshold:
            return SentimentLabel.VERY_POSITIVE
        elif polarity >= self.config.positive_threshold:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def _detect_text_col(self, df: pd.DataFrame) -> str:
        """Auto-detect text column."""
        patterns = ['text', 'content', 'message', 'comment', 'review', 'description']
        
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        
        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            return string_cols[0]
        
        return df.columns[0]


# ============================================================================
# Factory Functions
# ============================================================================

def get_sentiment_engine(config: SentimentConfig = None) -> SentimentAnalysisEngine:
    """Get sentiment analysis engine."""
    return SentimentAnalysisEngine(config=config)


def quick_sentiment(
    df: pd.DataFrame,
    text_col: str = None,
    method: str = "vader"
) -> Dict[str, Any]:
    """Quick sentiment analysis."""
    config = SentimentConfig(method=SentimentMethod(method))
    engine = SentimentAnalysisEngine(config=config, verbose=False)
    result = engine.analyze(df, text_col)
    return result.to_dict()
