# AI Enterprise Data Analyst - Core Configuration
# Production-grade configuration with type safety, validation, and environment management

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseSettings):
    """Database configuration with connection pooling settings."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")

    # Optional full URL (useful for cloud providers like Render/Neon)
    # Examples:
    # - DATABASE_URL=postgresql://user:pass@host:5432/dbname
    # - DB_URL=postgresql://...
    url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL"),
        description="Optional database URL override"
    )
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="ai_data_analyst", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")

    # Operational behavior
    auto_create_tables: bool = Field(
        default=False,
        description="If true, create tables on startup (useful for first deploys)"
    )
    
    # Connection pool settings (enterprise-grade)
    pool_size: int = Field(default=20, ge=5, le=100, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=50, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=10, le=120, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=1800, ge=300, le=7200, description="Connection recycle time")
    echo: bool = Field(default=False, description="Echo SQL statements")
    
    @property
    def async_url(self) -> str:
        """Generate async database URL."""
        if self.url:
            # Convert plain postgres URL to SQLAlchemy async driver URL.
            # Keep other schemes untouched in case user already provides an async URL.
            if self.url.startswith("postgresql://"):
                return self.url.replace("postgresql://", "postgresql+asyncpg://", 1)
            if self.url.startswith("postgres://"):
                return self.url.replace("postgres://", "postgresql+asyncpg://", 1)
            return self.url

        password = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """Generate sync database URL."""
        password = self.password.get_secret_value()
        return f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseSettings):
    """Redis configuration for caching and message queuing."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[SecretStr] = Field(default=None, description="Redis password")
    
    # Connection settings
    max_connections: int = Field(default=100, ge=10, le=1000)
    socket_timeout: int = Field(default=5, ge=1, le=30)
    socket_connect_timeout: int = Field(default=5, ge=1, le=30)
    
    @property
    def url(self) -> str:
        """Generate Redis URL."""
        if self.password:
            return f"redis://:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    
    api_key: SecretStr = Field(..., description="OpenAI API key")
    organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    
    # Model settings
    default_model: str = Field(default="gpt-4o", description="Default model for general tasks")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")
    
    # Rate limiting
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=60, ge=10, le=300)
    
    # Token limits
    max_tokens_default: int = Field(default=4096, ge=100, le=128000)
    temperature_default: float = Field(default=0.0, ge=0.0, le=2.0)


class PineconeConfig(BaseSettings):
    """Pinecone vector store configuration."""
    
    model_config = SettingsConfigDict(env_prefix="PINECONE_")
    
    api_key: SecretStr = Field(..., description="Pinecone API key")
    environment: str = Field(default="us-east-1", description="Pinecone environment")
    index_name: str = Field(default="ai-data-analyst", description="Default index name")
    
    # Vector settings
    dimension: int = Field(default=3072, description="Vector dimension (matches embedding model)")
    metric: str = Field(default="cosine", description="Distance metric")
    
    # Namespace settings
    default_namespace: str = Field(default="default", description="Default namespace")


class CeleryConfig(BaseSettings):
    """Celery task queue configuration."""
    
    model_config = SettingsConfigDict(env_prefix="CELERY_")
    
    broker_url: str = Field(default="redis://localhost:6379/1", description="Celery broker URL")
    result_backend: str = Field(default="redis://localhost:6379/2", description="Result backend")
    
    # Task settings
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: list[str] = Field(default=["json"])
    timezone: str = Field(default="UTC")
    enable_utc: bool = Field(default=True)
    
    # Concurrency
    worker_concurrency: int = Field(default=4, ge=1, le=32)
    task_soft_time_limit: int = Field(default=300, ge=60, le=3600)
    task_time_limit: int = Field(default=600, ge=120, le=7200)


class SecurityConfig(BaseSettings):
    """Security configuration for enterprise-grade protection."""
    
    model_config = SettingsConfigDict(env_prefix="SECURITY_")
    
    secret_key: SecretStr = Field(..., description="Application secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    
    # CORS settings
    cors_origins: list[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: list[str] = Field(default=["*"])
    cors_allow_headers: list[str] = Field(default=["*"])
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=10, le=10000)
    rate_limit_period: int = Field(default=60, ge=1, le=3600)


class MLConfig(BaseSettings):
    """Machine Learning configuration."""
    
    model_config = SettingsConfigDict(env_prefix="ML_")
    
    # Model storage
    model_storage_path: Path = Field(default=Path("./models"), description="Model storage path")

    @field_validator("model_storage_path")
    @classmethod
    def ensure_model_storage_path_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    # Training settings
    default_test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    default_random_state: int = Field(default=42)
    cross_validation_folds: int = Field(default=5, ge=3, le=10)
    
    # AutoML settings
    automl_time_budget: int = Field(default=3600, ge=60, le=86400)
    automl_max_models: int = Field(default=20, ge=5, le=100)
    
    # Feature engineering
    max_features_auto_select: int = Field(default=50, ge=10, le=500)
    
    # GPU settings
    use_gpu: bool = Field(default=False)
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0)


class Settings(BaseSettings):
    """Main application settings - Singleton pattern with caching."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application metadata
    app_name: str = Field(default="AI Enterprise Data Analyst", description="Application name")
    app_version: str = Field(default="3.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1000, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(default="json")  # json or text
    
    # File upload settings
    max_upload_size_mb: int = Field(default=100, ge=1, le=1000)
    allowed_extensions: list[str] = Field(
        default=["csv", "xlsx", "xls", "json", "parquet", "pdf", "docx", "txt"]
    )
    upload_directory: Path = Field(default=Path("./uploads"))
    artifact_directory: Path = Field(default=Path("./artifacts"))
    
    # Nested configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    openai: OpenAIConfig = Field(default_factory=lambda: OpenAIConfig(api_key=SecretStr(os.getenv("OPENAI_API_KEY", ""))))
    pinecone: PineconeConfig = Field(default_factory=lambda: PineconeConfig(api_key=SecretStr(os.getenv("PINECONE_API_KEY", ""))))
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig(secret_key=SecretStr(os.getenv("SECRET_KEY", "dev-secret-key"))))
    ml: MLConfig = Field(default_factory=MLConfig)
    
    @field_validator("upload_directory", "artifact_directory")
    @classmethod
    def ensure_directory_exists(cls, v: Any) -> Any:
        """Ensure directories exist on initialization."""
        if isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance (Singleton pattern).
    
    Uses LRU cache to ensure only one Settings instance exists,
    following the Singleton design pattern for configuration management.
    """
    return Settings()


# Export for easy access
settings = get_settings()
