"""Configuration module for AI Data Adequacy Agent."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ai-kb")
    
    # File processing limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    MAX_LLM_CALLS_PER_SESSION: int = int(os.getenv("MAX_LLM_CALLS_PER_SESSION", 50))
    
    # Chunking configuration
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 800))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 120))
    
    # Retrieval configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 5))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sessions.db")
    
    # Server configuration
    HOST: str = os.getenv("HOST", "localhost")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # OpenAI model configuration
    MODELS: Dict[str, str] = {
        "research": "gpt-4-turbo-preview",  # For deep research tasks
        "chat": "gpt-4",  # For general chat and answers
        "eval": "gpt-3.5-turbo",  # For evaluation and scoring
        "embedding": "text-embedding-3-large"  # For embeddings
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
        required_keys = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")
        
        return True
    
    @classmethod
    def get_domain_config(cls, domain: str) -> Dict[str, Any]:
        """Get domain-specific configuration."""
        return cls.DOMAIN_CONFIGS.get(domain, cls.DOMAIN_CONFIGS["general"])


# Global config instance
config = Config()
