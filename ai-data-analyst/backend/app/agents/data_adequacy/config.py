"""Configuration adapter for Data Adequacy Agent.

This module bridges the main application's settings with the configuration expected
by the ported validator agents.
"""

import os
from typing import Dict, Any, List

from app.core.config import settings

class Config:
    """Application configuration adapter."""
    
    # API Keys - mapped from main settings
    OPENAI_API_KEY: str = settings.openai.api_key.get_secret_value()
    # Handle optional Pinecone key
    PINECONE_API_KEY: str = settings.pinecone.api_key.get_secret_value() if settings.pinecone.api_key else ""
    PINECONE_ENVIRONMENT: str = settings.pinecone.environment
    PINECONE_INDEX_NAME: str = settings.pinecone.index_name
    
    # File processing limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    MAX_LLM_CALLS_PER_SESSION: int = int(os.getenv("MAX_LLM_CALLS_PER_SESSION", 50))
    
    # Chunking configuration
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 800))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 120))
    
    # Retrieval configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 5))
    
    # Database - Not used directly as main app uses SQLAlchemy, but kept for compatibility
    DATABASE_URL: str = settings.database.async_url
    
    # Server configuration
    HOST: str = settings.host
    PORT: int = settings.port
    DEBUG: bool = settings.debug
    
    # OpenAI model configuration
    MODELS: Dict[str, str] = {
        "research": "gpt-4-turbo-preview",  # For deep research tasks
        "chat": settings.openai.default_model,  # For general chat and answers
        "eval": "gpt-3.5-turbo",  # For evaluation and scoring
        "embedding": settings.openai.embedding_model  # For embeddings
    }
    
    # Quality thresholds
    QUALITY_THRESHOLDS: Dict[str, float] = {
        "ready": 0.8,
        "partially_ready": 0.6,
        "unsafe": 0.4,
        "duplicate_similarity": 0.92,
        "near_duplicate_similarity": 0.85,
        "coverage_target": 0.85,
        "retrieval_relevance_target": 0.8
    }
    
    # Scoring weights (configurable per domain)
    SCORING_WEIGHTS: Dict[str, float] = {
        "coverage": 0.2,
        "accuracy": 0.15,
        "consistency": 0.15,
        "timeliness": 0.1,
        "uniqueness": 0.1,
        "retrieval": 0.15,
        "formatting": 0.15
    }
    
    # Domain-specific configurations
    DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
        "automotive": {
            "staleness_threshold_days": 90,  # 3 months for car inventory
            "pricing_staleness_days": 30,  # 1 month for pricing
            "required_fields": ["make", "model", "year", "price"]
        },
        "manufacturing": {
            "staleness_threshold_days": 180,  # 6 months for specs
            "pricing_staleness_days": 60,  # 2 months for pricing
            "required_fields": ["product_id", "specifications", "compliance"]
        },
        "real_estate": {
            "staleness_threshold_days": 30,  # 1 month for listings
            "pricing_staleness_days": 7,  # 1 week for pricing
            "required_fields": ["address", "price", "square_feet", "bedrooms"]
        },
        "general": {
            "staleness_threshold_days": 365,  # 1 year default
            "pricing_staleness_days": 90,  # 3 months default
            "required_fields": []
        }
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate required configuration values."""
        # Main settings should already be validated at startup
        if not cls.OPENAI_API_KEY:
             raise ValueError("Missing OpenAI API Key")
        return True
    
    @classmethod
    def get_domain_config(cls, domain: str) -> Dict[str, Any]:
        """Get domain-specific configuration."""
        return cls.DOMAIN_CONFIGS.get(domain, cls.DOMAIN_CONFIGS["general"])


# Global config instance
config = Config()
