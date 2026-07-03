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
    DatasetVersionResponse,
    DatasetTransformStep,
    DatasetTransformPreviewRequest,
    DatasetTransformPreviewResponse,
    DatasetTransformApplyRequest,
    DatasetTransformApplyResponse,
    DatasetTransformSuggestRequest,
    DatasetTransformSuggestion,
    DatasetTransformSuggestResponse,
    DatasetQueryRequest,
    DatasetQueryResponse,
    # Jobs
    JobType,
    JobStatus,
    JobResponse,
    # Artifacts
    ArtifactType,
    ArtifactResponse,
    ArtifactRowsResponse,
    ReportShareCreateRequest,
    ReportShareResponse,
    PublicReportResponse,
    StorageGcRequest,
    StorageGcResponse,
    CachePruneResponse,
    # Analysis
    AnalysisType,
    AnalysisStatus,
    AnalysisCreate,
    AnalysisResponse,
    AnalysisDetailResponse,
    AnalysisActionKind,
    SuggestedAction,
    AnalysisActionRunRequest,
    AnalysisActionFeedItem,
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
    # Dashboard
    DashboardSummary,
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
    "DatasetVersionResponse",
    "DatasetTransformStep",
    "DatasetTransformPreviewRequest",
    "DatasetTransformPreviewResponse",
    "DatasetTransformApplyRequest",
    "DatasetTransformApplyResponse",
    "DatasetTransformSuggestRequest",
    "DatasetTransformSuggestion",
    "DatasetTransformSuggestResponse",
    "DatasetQueryRequest",
    "DatasetQueryResponse",
    # Jobs
    "JobType",
    "JobStatus",
    "JobResponse",
    # Artifacts
    "ArtifactType",
    "ArtifactResponse",
    "ArtifactRowsResponse",
    "ReportShareCreateRequest",
    "ReportShareResponse",
    "PublicReportResponse",
    "StorageGcRequest",
    "StorageGcResponse",
    "CachePruneResponse",
    # Analysis
    "AnalysisType",
    "AnalysisStatus",
    "AnalysisCreate",
    "AnalysisResponse",
    "AnalysisDetailResponse",
    "AnalysisActionKind",
    "SuggestedAction",
    "AnalysisActionRunRequest",
    "AnalysisActionFeedItem",
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
    # Dashboard
    "DashboardSummary",
]
