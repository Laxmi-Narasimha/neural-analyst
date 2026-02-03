# AI Enterprise Data Analyst - API Schemas
"""Pydantic schemas for API validation."""

from app.api.schemas.base import (
    # Generic
    APIResponse,
    PaginatedResponse,
    ResponseStatus,
    PaginationMeta,
    # User
    UserBase,
    UserCreate,
    UserUpdate,
    UserResponse,
    # Auth
    TokenResponse,
    LoginRequest,
    RefreshTokenRequest,
    # Dataset
    DatasetStatus,
    ColumnInfo,
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetDetailResponse,
    DatasetUploadResponse,
    # Analysis
    AnalysisType,
    AnalysisStatus,
    AnalysisCreate,
    AnalysisResponse,
    AnalysisDetailResponse,
    # Chat
    MessageRole,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    # ML Model
    ModelType,
    MLModelResponse,
    PredictionRequest,
    PredictionResponse,
    # NL2SQL
    NLQueryRequest,
    NLQueryResponse,
    # Health
    HealthCheckResponse,
)

__all__ = [
    # Generic
    "APIResponse",
    "PaginatedResponse",
    "ResponseStatus",
    "PaginationMeta",
    # User
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    # Auth
    "TokenResponse",
    "LoginRequest",
    "RefreshTokenRequest",
    # Dataset
    "DatasetStatus",
    "ColumnInfo",
    "DatasetCreate",
    "DatasetUpdate",
    "DatasetResponse",
    "DatasetDetailResponse",
    "DatasetUploadResponse",
    # Analysis
    "AnalysisType",
    "AnalysisStatus",
    "AnalysisCreate",
    "AnalysisResponse",
    "AnalysisDetailResponse",
    # Chat
    "MessageRole",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConversationResponse",
    # ML Model
    "ModelType",
    "MLModelResponse",
    "PredictionRequest",
    "PredictionResponse",
    # NL2SQL
    "NLQueryRequest",
    "NLQueryResponse",
    # Health
    "HealthCheckResponse",
]
