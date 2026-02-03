# AI Enterprise Data Analyst - Database Models
# Production-grade SQLAlchemy models with proper relationships, mixins, and audit trails

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Index,
    UniqueConstraint,
    event
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    declared_attr
)
from enum import Enum


# ============================================================================
# Enums for Database Models
# ============================================================================

class DatasetStatus(str, Enum):
    """Status of a dataset in the system."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    EDA = "eda"
    STATISTICAL = "statistical"
    ML_CLASSIFICATION = "ml_classification"
    ML_REGRESSION = "ml_regression"
    ML_CLUSTERING = "ml_clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    DEEP_LEARNING = "deep_learning"
    CUSTOM = "custom"


class AnalysisStatus(str, Enum):
    """Status of an analysis job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    """Types of ML models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"
    CUSTOM = "custom"


class AgentType(str, Enum):
    """Types of AI agents."""
    ORCHESTRATOR = "orchestrator"
    DATA_INGESTION = "data_ingestion"
    QUALITY = "quality"
    EDA = "eda"
    STATISTICAL = "statistical"
    ML = "ml"
    DEEP_LEARNING = "deep_learning"
    NLP = "nlp"
    VISUALIZATION = "visualization"
    REPORT = "report"
    SQL_GENERATION = "sql_generation"
    RESEARCH = "research"


# ============================================================================
# Base Model with Mixins
# ============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """
    Base model class with async support.
    
    All models inherit from this base to ensure consistent
    behavior and async compatibility.
    """
    
    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[str]: ARRAY(String),
        UUID: PGUUID(as_uuid=True)
    }


class TimestampMixin:
    """
    Mixin for automatic timestamp management.
    
    Provides created_at and updated_at columns with automatic
    population on insert and update.
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


class SoftDeleteMixin:
    """
    Mixin for soft delete functionality.
    
    Instead of permanently deleting records, marks them as deleted
    while maintaining data integrity and audit trail.
    """
    
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True
    )
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    def soft_delete(self) -> None:
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """
    Mixin for audit trail tracking.
    
    Tracks who created and last modified a record.
    """
    
    created_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True
    )
    
    updated_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True
    )


class UUIDMixin:
    """
    Mixin for UUID primary key.
    
    Uses UUID4 for globally unique identifiers that are
    safe for distributed systems.
    """
    
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )


# ============================================================================
# Core Domain Models
# ============================================================================

class User(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email_active", "email", "is_deleted"),
    )
    
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Profile settings
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Relationships
    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset",
        back_populates="owner",
        lazy="selectin"
    )
    analyses: Mapped[list["Analysis"]] = relationship(
        "Analysis",
        back_populates="owner",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"


class Dataset(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Dataset model for storing uploaded data information."""
    
    __tablename__ = "datasets"
    __table_args__ = (
        Index("ix_datasets_owner_status", "owner_id", "status"),
        Index("ix_datasets_created", "created_at"),
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # File info
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    file_format: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Status
    status: Mapped[DatasetStatus] = mapped_column(
        SQLEnum(DatasetStatus),
        default=DatasetStatus.PENDING,
        nullable=False,
        index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Schema info (populated after processing)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schema_info: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Data quality metrics
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_report: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Profiling results
    profile_report: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Vector store info
    vector_namespace: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    embedding_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Tags and metadata
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    extra_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Relationships
    owner_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    owner: Mapped["User"] = relationship("User", back_populates="datasets")
    
    columns: Mapped[list["DatasetColumn"]] = relationship(
        "DatasetColumn",
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    analyses: Mapped[list["Analysis"]] = relationship(
        "Analysis",
        back_populates="dataset",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}', status={self.status.value})>"


class DatasetColumn(Base, UUIDMixin, TimestampMixin):
    """Column metadata for datasets."""
    
    __tablename__ = "dataset_columns"
    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_dataset_column_name"),
        Index("ix_dataset_columns_dataset", "dataset_id"),
    )
    
    # Column info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Type info
    inferred_type: Mapped[str] = mapped_column(String(50), nullable=False)
    semantic_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Statistics
    null_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    null_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unique_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Numeric stats (if applicable)
    min_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    median_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    std_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Distribution info
    distribution_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    value_distribution: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Quality flags
    has_outliers: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Full statistics
    statistics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Relationship
    dataset_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False
    )
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="columns")
    
    def __repr__(self) -> str:
        return f"<DatasetColumn(id={self.id}, name='{self.name}', type='{self.inferred_type}')>"


class Analysis(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Analysis job model for tracking analysis executions."""
    
    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_owner_status", "owner_id", "status"),
        Index("ix_analyses_dataset_type", "dataset_id", "analysis_type"),
        Index("ix_analyses_created", "created_at"),
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Analysis configuration
    analysis_type: Mapped[AnalysisType] = mapped_column(
        SQLEnum(AnalysisType),
        nullable=False,
        index=True
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Status tracking
    status: Mapped[AnalysisStatus] = mapped_column(
        SQLEnum(AnalysisStatus),
        default=AnalysisStatus.QUEUED,
        nullable=False,
        index=True
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Results
    results: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    insights: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    visualizations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    
    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Agent execution trace
    agent_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    
    # Relationships
    owner_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    owner: Mapped["User"] = relationship("User", back_populates="analyses")
    
    dataset_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="analyses")
    
    models: Mapped[list["MLModel"]] = relationship(
        "MLModel",
        back_populates="analysis",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<Analysis(id={self.id}, name='{self.name}', status={self.status.value})>"


class MLModel(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """ML Model storage and versioning."""
    
    __tablename__ = "ml_models"
    __table_args__ = (
        Index("ix_ml_models_analysis", "analysis_id"),
        Index("ix_ml_models_type_version", "model_type", "version"),
    )
    
    # Model info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0", nullable=False)
    
    # Model type and algorithm
    model_type: Mapped[ModelType] = mapped_column(
        SQLEnum(ModelType),
        nullable=False,
        index=True
    )
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False)  # sklearn, pytorch, etc.
    
    # Model file storage
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    model_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    model_format: Mapped[str] = mapped_column(String(50), nullable=False)  # pickle, onnx, joblib, etc.
    
    # Training configuration
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    feature_names: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    target_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Performance metrics
    training_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    validation_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    test_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Feature importance
    feature_importance: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Model state
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Metadata
    training_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Relationships
    analysis_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False
    )
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="models")
    
    def __repr__(self) -> str:
        return f"<MLModel(id={self.id}, name='{self.name}', algorithm='{self.algorithm}')>"


class Conversation(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """Chat conversation history for the AI assistant."""
    
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_user", "user_id"),
        Index("ix_conversations_created", "created_at"),
    )
    
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Context
    context: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    active_dataset_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    
    # Relationships
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class Message(Base, UUIDMixin, TimestampMixin):
    """Individual messages in a conversation."""
    
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation", "conversation_id"),
    )
    
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Token tracking
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Agent trace for assistant messages
    agent_actions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    
    # Relationship
    conversation_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}')>"


# ============================================================================
# Event Listeners for Audit Trail
# ============================================================================

@event.listens_for(Base, "before_update", propagate=True)
def receive_before_update(mapper, connection, target):
    """Automatically update updated_at timestamp on any update."""
    if hasattr(target, "updated_at"):
        target.updated_at = datetime.utcnow()
