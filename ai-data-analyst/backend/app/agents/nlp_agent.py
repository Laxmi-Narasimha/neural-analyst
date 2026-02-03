# AI Enterprise Data Analyst - NLP Agent
# Natural Language Processing for text analysis

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import UUID
import re
from collections import Counter

import numpy as np
import pandas as pd

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
)
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


class NLPTask(str, Enum):
    """NLP task types."""
    SENTIMENT = "sentiment"
    CLASSIFICATION = "classification"
    NER = "ner"
    SUMMARIZATION = "summarization"
    KEYWORD_EXTRACTION = "keyword_extraction"
    TOPIC_MODELING = "topic_modeling"
    SIMILARITY = "similarity"
    TRANSLATION = "translation"


@dataclass
class NLPResult:
    """Result from NLP analysis."""
    
    task: NLPTask
    input_text: str
    output: Any
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "input_text": self.input_text[:100] + "..." if len(self.input_text) > 100 else self.input_text,
            "output": self.output,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class NLPEngine:
    """
    NLP Engine for text analysis.
    
    Uses Hugging Face transformers for:
    - Sentiment analysis
    - Named entity recognition
    - Text classification
    - Summarization
    - Keyword extraction
    """
    
    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}
    
    def _get_pipeline(self, task: str, model: str = None) -> Any:
        """Get or create a pipeline for a task."""
        if not HAS_TRANSFORMERS:
            return None
        
        cache_key = f"{task}:{model or 'default'}"
        if cache_key not in self._pipelines:
            try:
                self._pipelines[cache_key] = pipeline(task, model=model)
            except Exception as e:
                logger.warning(f"Failed to load pipeline for {task}: {e}")
                return None
        return self._pipelines[cache_key]
    
    def analyze_sentiment(self, texts: list[str]) -> list[NLPResult]:
        """Analyze sentiment of texts."""
        results = []
        
        if HAS_TRANSFORMERS:
            pipe = self._get_pipeline("sentiment-analysis")
            if pipe:
                for text in texts:
                    try:
                        output = pipe(text[:512])[0]  # Limit input length
                        results.append(NLPResult(
                            task=NLPTask.SENTIMENT,
                            input_text=text,
                            output=output['label'],
                            confidence=output['score'],
                            metadata={"raw_output": output}
                        ))
                    except Exception as e:
                        results.append(NLPResult(
                            task=NLPTask.SENTIMENT,
                            input_text=text,
                            output="ERROR",
                            metadata={"error": str(e)}
                        ))
        else:
            # Fallback: Simple rule-based sentiment
            positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'happy'}
            negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'sad', 'poor'}
            
            for text in texts:
                words = set(text.lower().split())
                pos_count = len(words & positive_words)
                neg_count = len(words & negative_words)
                
                if pos_count > neg_count:
                    sentiment = "POSITIVE"
                    conf = min(0.5 + 0.1 * (pos_count - neg_count), 0.95)
                elif neg_count > pos_count:
                    sentiment = "NEGATIVE"
                    conf = min(0.5 + 0.1 * (neg_count - pos_count), 0.95)
                else:
                    sentiment = "NEUTRAL"
                    conf = 0.5
                
                results.append(NLPResult(
                    task=NLPTask.SENTIMENT,
                    input_text=text,
                    output=sentiment,
                    confidence=conf
                ))
        
        return results
    
    def extract_entities(self, texts: list[str]) -> list[NLPResult]:
        """Extract named entities from texts."""
        results = []
        
        if HAS_TRANSFORMERS:
            pipe = self._get_pipeline("ner")
            if pipe:
                for text in texts:
                    try:
                        entities = pipe(text[:512])
                        # Group entities
                        grouped = {}
                        for ent in entities:
                            ent_type = ent['entity_group'] if 'entity_group' in ent else ent['entity']
                            if ent_type not in grouped:
                                grouped[ent_type] = []
                            grouped[ent_type].append(ent['word'])
                        
                        results.append(NLPResult(
                            task=NLPTask.NER,
                            input_text=text,
                            output=grouped,
                            metadata={"entity_count": len(entities)}
                        ))
                    except Exception as e:
                        results.append(NLPResult(
                            task=NLPTask.NER,
                            input_text=text,
                            output={},
                            metadata={"error": str(e)}
                        ))
        else:
            # Fallback: Simple pattern-based extraction
            patterns = {
                "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "URL": r'https?://\S+',
                "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            }
            
            for text in texts:
                entities = {}
                for ent_type, pattern in patterns.items():
                    matches = re.findall(pattern, text)
                    if matches:
                        entities[ent_type] = matches
                
                results.append(NLPResult(
                    task=NLPTask.NER,
                    input_text=text,
                    output=entities
                ))
        
        return results
    
    def extract_keywords(self, texts: list[str], top_n: int = 10) -> list[NLPResult]:
        """Extract keywords from texts using TF-IDF like approach."""
        results = []
        
        # Simple word frequency with stopwords removal
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which',
            'who', 'how', 'when', 'where', 'why', 'all', 'each', 'every', 'both'
        }
        
        for text in texts:
            # Tokenize and clean
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            words = [w for w in words if w not in stopwords]
            
            # Count frequencies
            word_freq = Counter(words)
            top_keywords = word_freq.most_common(top_n)
            
            results.append(NLPResult(
                task=NLPTask.KEYWORD_EXTRACTION,
                input_text=text,
                output=[{"keyword": kw, "frequency": freq} for kw, freq in top_keywords],
                metadata={"total_words": len(words), "unique_words": len(set(words))}
            ))
        
        return results
    
    def summarize(self, texts: list[str], max_length: int = 150) -> list[NLPResult]:
        """Summarize texts."""
        results = []
        
        if HAS_TRANSFORMERS:
            pipe = self._get_pipeline("summarization", model="facebook/bart-large-cnn")
            if pipe:
                for text in texts:
                    try:
                        if len(text.split()) < 50:
                            # Text too short to summarize
                            results.append(NLPResult(
                                task=NLPTask.SUMMARIZATION,
                                input_text=text,
                                output=text,
                                metadata={"note": "Text too short to summarize"}
                            ))
                        else:
                            summary = pipe(text[:1024], max_length=max_length, min_length=30)
                            results.append(NLPResult(
                                task=NLPTask.SUMMARIZATION,
                                input_text=text,
                                output=summary[0]['summary_text']
                            ))
                    except Exception as e:
                        results.append(NLPResult(
                            task=NLPTask.SUMMARIZATION,
                            input_text=text,
                            output="",
                            metadata={"error": str(e)}
                        ))
        else:
            # Fallback: Simple extractive summary (first 2 sentences)
            for text in texts:
                sentences = re.split(r'[.!?]+', text)
                summary = '. '.join(sentences[:2]).strip()
                if summary and not summary.endswith('.'):
                    summary += '.'
                
                results.append(NLPResult(
                    task=NLPTask.SUMMARIZATION,
                    input_text=text,
                    output=summary,
                    metadata={"method": "extractive"}
                ))
        
        return results
    
    def calculate_similarity(
        self,
        text1: str,
        text2: str
    ) -> NLPResult:
        """Calculate semantic similarity between texts."""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        
        return NLPResult(
            task=NLPTask.SIMILARITY,
            input_text=f"{text1[:50]}... vs {text2[:50]}...",
            output=similarity,
            confidence=similarity,
            metadata={
                "method": "jaccard",
                "common_words": list(words1 & words2)[:10]
            }
        )


class NLPAgent(BaseAgent[dict[str, Any]]):
    """
    NLP Agent for text analysis tasks.
    
    Capabilities:
    - Sentiment analysis
    - Named entity recognition
    - Keyword extraction
    - Text summarization
    - Semantic similarity
    - Topic modeling
    """
    
    name: str = "NLPAgent"
    description: str = "Natural language processing for text analysis"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.engine = NLPEngine()
    
    def _register_tools(self) -> None:
        """Register NLP tools."""
        
        self.register_tool(AgentTool(
            name="analyze_sentiment",
            description="Analyze sentiment of text(s)",
            function=self._analyze_sentiment,
            parameters={
                "texts": {"type": "array", "items": {"type": "string"}}
            },
            required_params=["texts"]
        ))
        
        self.register_tool(AgentTool(
            name="extract_entities",
            description="Extract named entities (people, places, organizations) from text",
            function=self._extract_entities,
            parameters={
                "texts": {"type": "array", "items": {"type": "string"}}
            },
            required_params=["texts"]
        ))
        
        self.register_tool(AgentTool(
            name="extract_keywords",
            description="Extract important keywords from text",
            function=self._extract_keywords,
            parameters={
                "texts": {"type": "array", "items": {"type": "string"}},
                "top_n": {"type": "integer", "default": 10}
            },
            required_params=["texts"]
        ))
        
        self.register_tool(AgentTool(
            name="summarize_text",
            description="Create summaries of long texts",
            function=self._summarize_text,
            parameters={
                "texts": {"type": "array", "items": {"type": "string"}},
                "max_length": {"type": "integer", "default": 150}
            },
            required_params=["texts"]
        ))
        
        self.register_tool(AgentTool(
            name="analyze_text_column",
            description="Analyze a text column in a dataframe",
            function=self._analyze_text_column,
            parameters={
                "data": {"type": "object"},
                "column": {"type": "string"},
                "analysis_type": {"type": "string", "enum": ["sentiment", "keywords", "entities"]}
            },
            required_params=["data", "column"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute NLP analysis."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are an NLP expert. Analyze the request and determine the best approach."
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "available_tasks": [t.value for t in NLPTask]
        }
    
    async def _analyze_sentiment(
        self,
        texts: list[str]
    ) -> dict[str, Any]:
        """Analyze sentiment of texts."""
        results = self.engine.analyze_sentiment(texts)
        
        # Aggregate statistics
        sentiments = [r.output for r in results]
        sentiment_counts = Counter(sentiments)
        
        return {
            "results": [r.to_dict() for r in results],
            "summary": {
                "total": len(results),
                "sentiment_distribution": dict(sentiment_counts),
                "average_confidence": np.mean([r.confidence or 0 for r in results])
            }
        }
    
    async def _extract_entities(
        self,
        texts: list[str]
    ) -> dict[str, Any]:
        """Extract entities from texts."""
        results = self.engine.extract_entities(texts)
        
        # Aggregate all entities
        all_entities = {}
        for r in results:
            for ent_type, entities in (r.output or {}).items():
                if ent_type not in all_entities:
                    all_entities[ent_type] = []
                all_entities[ent_type].extend(entities)
        
        return {
            "results": [r.to_dict() for r in results],
            "aggregated_entities": {k: list(set(v)) for k, v in all_entities.items()}
        }
    
    async def _extract_keywords(
        self,
        texts: list[str],
        top_n: int = 10
    ) -> dict[str, Any]:
        """Extract keywords from texts."""
        results = self.engine.extract_keywords(texts, top_n)
        
        return {
            "results": [r.to_dict() for r in results]
        }
    
    async def _summarize_text(
        self,
        texts: list[str],
        max_length: int = 150
    ) -> dict[str, Any]:
        """Summarize texts."""
        results = self.engine.summarize(texts, max_length)
        
        return {
            "results": [r.to_dict() for r in results]
        }
    
    async def _analyze_text_column(
        self,
        data: dict,
        column: str,
        analysis_type: str = "sentiment"
    ) -> dict[str, Any]:
        """Analyze a text column in a dataframe."""
        df = pd.DataFrame(data)
        
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        texts = df[column].dropna().astype(str).tolist()
        
        if analysis_type == "sentiment":
            results = self.engine.analyze_sentiment(texts[:100])  # Limit for performance
        elif analysis_type == "keywords":
            results = self.engine.extract_keywords([" ".join(texts[:100])])
        elif analysis_type == "entities":
            results = self.engine.extract_entities(texts[:50])
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
        
        return {
            "column": column,
            "analysis_type": analysis_type,
            "sample_size": len(results),
            "results": [r.to_dict() for r in results[:10]]
        }


# Factory function
def get_nlp_agent() -> NLPAgent:
    """Get NLP agent instance."""
    return NLPAgent()
