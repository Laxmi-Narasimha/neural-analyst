# AI Enterprise Data Analyst - API Schemas
# Pydantic schemas for API request/response validation with full typing

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, EmailStr, field_validator


# ============================================================================
# Generic Response Schemas
# ============================================================================

DataT = TypeVar("DataT")


class ResponseStatus(str, Enum):
    """API response status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class PaginationMeta(BaseModel):
    """Pagination metadata."""
    
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class APIResponse(BaseModel, Generic[DataT]):
    """
    Standard API response wrapper.
    
    Provides consistent response structure across all endpoints.
    """
    
    model_config = ConfigDict(from_attributes=True)
    
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: Optional[str] = None
    data: Optional[DataT] = None
    errors: Optional[list[dict[str, Any]]] = None
    meta: Optional[dict[str, Any]] = None
    
    @classmethod
    def success(
        cls,
        data: DataT,
        message: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None
    ) -> "APIResponse[DataT]":
        """Create success response."""
        return cls(
            status=ResponseStatus.SUCCESS,
            data=data,
            message=message,
            meta=meta
        )
    
    @classmethod
    def error(
        cls,
        message: str,
        errors: Optional[list[dict[str, Any]]] = None
    ) -> "APIResponse[None]":
        """Create error response."""
        return cls(
            status=ResponseStatus.ERROR,
            message=message,
            errors=errors
        )


class PaginatedResponse(APIResponse[list[DataT]], Generic[DataT]):
    """Paginated API response."""
    
    pagination: Optional[PaginationMeta] = None


# ============================================================================
# User Schemas
# ============================================================================

class UserBase(BaseModel):
    """Base user schema."""
    
    email: EmailStr = Field(..., description="User email address")
    full_name: str = Field(..., min_length=1, max_length=255, description="Full name")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """Schema for updating user."""
    
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    settings: Optional[dict[str, Any]] = None


class UserResponse(UserBase):
    """Schema for user response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    is_active: bool
    is_verified: bool
    created_at: datetime
    settings: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Authentication Schemas
# ============================================================================

class TokenResponse(BaseModel):
    """Schema for authentication token response."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiry in seconds")


class LoginRequest(BaseModel):
    """Schema for login request."""
    
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh request."""
    
    refresh_token: str


# ============================================================================
# Dataset Schemas
# ============================================================================

class DatasetStatus(str, Enum):
    """Dataset processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class ColumnInfo(BaseModel):
    """Schema for column information."""
    
    name: str
    original_name: str
    position: int
    inferred_type: str
    semantic_type: Optional[str] = None
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    statistics: dict[str, Any] = Field(default_factory=dict)


class DatasetCreate(BaseModel):
    """Schema for dataset creation metadata."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class DatasetUpdate(BaseModel):
    """Schema for updating dataset."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    tags: Optional[list[str]] = Field(None, max_length=20)


class DatasetResponse(BaseModel):
    """Schema for dataset response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    description: Optional[str]
    original_filename: str
    file_size_bytes: int
    file_format: str
    status: DatasetStatus
    error_message: Optional[str]
    row_count: Optional[int]
    column_count: Optional[int]
    quality_score: Optional[float]
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class DatasetDetailResponse(DatasetResponse):
    """Detailed dataset response with columns and metadata."""
    
    columns: list[ColumnInfo] = Field(default_factory=list)
    schema_info: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    profile_report: dict[str, Any] = Field(default_factory=dict)


class DatasetUploadResponse(BaseModel):
    """Response after file upload."""
    
    dataset_id: UUID
    filename: str
    file_size_bytes: int
    status: DatasetStatus
    message: str


# ============================================================================
# Analysis Schemas
# ============================================================================

class AnalysisType(str, Enum):
    """Types of analysis."""
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
    """Analysis job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisCreate(BaseModel):
    """Schema for creating analysis."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    dataset_id: UUID
    analysis_type: AnalysisType
    config: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Schema for analysis response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    description: Optional[str]
    dataset_id: UUID
    analysis_type: AnalysisType
    status: AnalysisStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    status_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    created_at: datetime


class AnalysisDetailResponse(AnalysisResponse):
    """Detailed analysis response with results."""
    
    config: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    insights: list[dict[str, Any]] = Field(default_factory=list)
    visualizations: list[dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# Chat/Conversation Schemas
# ============================================================================

class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Schema for a chat message."""
    
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=32000)


class ChatRequest(BaseModel):
    """Schema for chat request."""
    
    message: str = Field(..., min_length=1, max_length=32000)
    conversation_id: Optional[UUID] = None
    dataset_id: Optional[UUID] = None
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Schema for chat response."""
    
    conversation_id: UUID
    message_id: UUID
    content: str
    role: MessageRole = MessageRole.ASSISTANT
    agent_actions: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    visualizations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """Schema for conversation response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    active_dataset_id: Optional[UUID] = None


# ============================================================================
# ML Model Schemas
# ============================================================================

class ModelType(str, Enum):
    """ML model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"


class MLModelResponse(BaseModel):
    """Schema for ML model response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    description: Optional[str]
    version: str
    model_type: ModelType
    algorithm: str
    framework: str
    is_deployed: bool
    training_metrics: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    created_at: datetime


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    
    model_id: UUID
    data: list[dict[str, Any]] = Field(..., min_length=1, max_length=10000)


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    model_id: UUID
    predictions: list[Any]
    probabilities: Optional[list[list[float]]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Query/NL2SQL Schemas
# ============================================================================

class NLQueryRequest(BaseModel):
    """Schema for natural language query request."""
    
    query: str = Field(..., min_length=1, max_length=2000)
    dataset_id: UUID
    include_visualization: bool = True
    include_explanation: bool = True


class NLQueryResponse(BaseModel):
    """Schema for natural language query response."""
    
    query: str
    generated_sql: Optional[str] = None
    results: Optional[list[dict[str, Any]]] = None
    row_count: Optional[int] = None
    columns: Optional[list[str]] = None
    explanation: Optional[str] = None
    visualization: Optional[dict[str, Any]] = None
    suggestions: list[str] = Field(default_factory=list)


# ============================================================================
# Health Check Schemas
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = "healthy"
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: dict[str, dict[str, Any]] = Field(default_factory=dict)
