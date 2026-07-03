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
    job_id: Optional[UUID] = None
    filename: str
    file_size_bytes: int
    status: DatasetStatus
    message: str


# ============================================================================
# Dataset Versioning / Transformations
# ============================================================================

class DatasetVersionResponse(BaseModel):
    """Immutable dataset version record (file + metadata snapshot)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_id: UUID
    version_hash: str
    label: Optional[str] = None
    parent_version_hash: Optional[str] = None
    transform_spec: dict[str, Any] = Field(default_factory=dict)

    file_format: str
    file_size_bytes: int

    row_count: Optional[int] = None
    column_count: Optional[int] = None
    quality_score: Optional[float] = None

    created_at: datetime
    updated_at: datetime

    is_active: bool = False


class DatasetTransformStep(BaseModel):
    """Single dataset transformation step (validated server-side)."""

    op: str = Field(..., min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetTransformPreviewRequest(BaseModel):
    """Preview a transformation pipeline on a bounded sample."""

    steps: list[DatasetTransformStep] = Field(..., min_length=1, max_length=50)
    sample_rows: int = Field(50_000, ge=100, le=1_000_000)
    preview_rows: int = Field(25, ge=1, le=200)


class DatasetTransformPreviewResponse(BaseModel):
    """Preview response (diff + sample output)."""

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int

    added_columns: list[str] = Field(default_factory=list)
    removed_columns: list[str] = Field(default_factory=list)
    changed_dtypes: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class DatasetTransformApplyRequest(BaseModel):
    """Apply a transformation pipeline to create a new dataset version."""

    steps: list[DatasetTransformStep] = Field(..., min_length=1, max_length=50)
    label: Optional[str] = Field(None, max_length=255)
    set_as_current: bool = True


class DatasetTransformApplyResponse(BaseModel):
    dataset_id: UUID
    job_id: UUID
    message: str


class DatasetTransformSuggestRequest(BaseModel):
    """Generate a deterministic transformation plan from dataset metadata."""

    max_steps: int = Field(8, ge=1, le=25)
    include_drop_columns: bool = True
    include_string_normalization: bool = True


class DatasetTransformSuggestion(BaseModel):
    """Single suggested transform step with rationale."""

    step: DatasetTransformStep
    reason: str
    impact: Optional[str] = None


class DatasetTransformSuggestResponse(BaseModel):
    """Suggested transform plan for quick no-code cleaning."""

    dataset_id: UUID
    suggestions: list[DatasetTransformSuggestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Dataset SQL Query (Uploaded Datasets)
# ============================================================================

class DatasetQueryRequest(BaseModel):
    """Execute a read-only SQL query against an uploaded dataset (server-side, guarded)."""

    query: str = Field(..., min_length=1)
    max_rows: Optional[int] = Field(default=None, ge=1, le=50_000)
    timeout_seconds: Optional[int] = Field(default=None, ge=1, le=120)


class DatasetQueryResponse(BaseModel):
    """Query result as an evidence artifact."""

    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    artifact: ArtifactResponse


# ============================================================================
# Job Schemas
# ============================================================================

class JobType(str, Enum):
    """Types of background jobs."""

    DATASET_PROCESSING = "dataset_processing"
    DATASET_TRANSFORM = "dataset_transform"
    COMPUTE_PLAN = "compute_plan"
    DATASET_PURGE = "dataset_purge"


class JobStatus(str, Enum):
    """Job status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResponse(BaseModel):
    """Schema for a background job record."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_type: JobType
    status: JobStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    status_message: Optional[str] = None

    dataset_id: Optional[UUID] = None

    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Artifact Schemas
# ============================================================================

class ArtifactType(str, Enum):
    """Compute artifact types."""

    METRIC = "metric"
    TABLE = "table"
    CHART = "chart"
    REPORT = "report"


class ArtifactResponse(BaseModel):
    """Schema for indexed artifacts."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    artifact_type: ArtifactType
    name: str

    dataset_id: Optional[UUID] = None
    dataset_version: Optional[str] = None
    operator_name: Optional[str] = None
    operator_params: dict[str, Any] = Field(default_factory=dict)
    preview: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime


class ArtifactRowsResponse(BaseModel):
    """Bounded page of rows for a table artifact (server-side paging)."""

    artifact_id: UUID
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)

    offset: int = 0
    limit: int = 0
    total_rows: Optional[int] = None
    execution_time_ms: float = 0.0


class ReportShareCreateRequest(BaseModel):
    """Create a share link for a report artifact (token returned once)."""

    expires_days: Optional[int] = Field(default=None, ge=1, le=3650, description="Optional expiration in days")


class ReportShareResponse(BaseModel):
    """Share link token + metadata."""

    share_id: UUID
    share_token: str
    share_path: str
    expires_at: Optional[datetime] = None


class PublicReportResponse(BaseModel):
    """Public read-only report payload (resolved by share token)."""

    artifact_id: UUID
    name: str
    format: str
    created_at: datetime
    content: str


class StorageGcRequest(BaseModel):
    """Request payload for storage garbage collection."""

    dry_run: bool = True
    min_age_days: int = Field(default=7, ge=0, le=3650)
    include_cache: bool = False
    include_s3: bool = True
    s3_max_scan: int = Field(default=200_000, ge=1, le=1_000_000)


class StorageGcResponse(BaseModel):
    """Response payload for storage garbage collection."""

    dry_run: bool
    min_age_days: int
    include_s3: bool = False
    roots: list[str] = Field(default_factory=list)
    scanned_files: int = 0
    deleted_files: int = 0
    skipped_active: int = 0
    skipped_recent: int = 0
    failed: int = 0
    scanned_local_files: int = 0
    scanned_s3_objects: int = 0
    deleted_local_files: int = 0
    deleted_s3_objects: int = 0
    skipped_active_local: int = 0
    skipped_active_s3: int = 0
    skipped_recent_local: int = 0
    skipped_recent_s3: int = 0
    failed_local: int = 0
    failed_s3: int = 0
    deleted_examples: list[str] = Field(default_factory=list)
    skipped_examples: list[str] = Field(default_factory=list)
    skipped_recent_examples: list[str] = Field(default_factory=list)


class CachePruneResponse(BaseModel):
    """Response payload for cache pruning."""

    deleted_files: int = 0
    deleted_bytes: int = 0


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
# Analysis Action Schemas (Data Speaks / One-click actions)
# ============================================================================

class AnalysisActionKind(str, Enum):
    """Kinds of suggested actions."""

    ANALYSIS = "analysis"
    NAVIGATE = "navigate"
    REPORT = "report"


class SuggestedAction(BaseModel):
    """Suggested next action surfaced to the user (evidence-first)."""

    action_id: str = Field(..., min_length=1, max_length=64)
    kind: AnalysisActionKind = AnalysisActionKind.ANALYSIS
    title: str = Field(..., min_length=1, max_length=200)
    detail: Optional[str] = Field(None, max_length=2000)
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisActionRunRequest(BaseModel):
    """Request to run a one-click action as a child analysis/job."""

    action_id: str = Field(..., min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisActionFeedItem(BaseModel):
    """A logged action run linked to a parent analysis session."""

    action_id: str = Field(..., min_length=1, max_length=64)
    kind: AnalysisActionKind = AnalysisActionKind.ANALYSIS
    title: str = Field(..., min_length=1, max_length=200)
    detail: Optional[str] = Field(None, max_length=2000)
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    analysis_id: UUID
    status: AnalysisStatus
    status_message: Optional[str] = None
    error_message: Optional[str] = None
    takeaways: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


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
    clarification: Optional[dict[str, Any]] = None
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


# ============================================================================
# Dashboard Schemas
# ============================================================================


class DashboardSummary(BaseModel):
    """Aggregated, computed metrics for the dashboard UI."""

    datasets_total: int = Field(..., ge=0)
    datasets_ready: int = Field(..., ge=0)

    analyses_total: int = Field(..., ge=0)
    analyses_running: int = Field(..., ge=0)
    analyses_this_month: int = Field(..., ge=0)

    rows_processed: int = Field(..., ge=0)
    compute_seconds: float = Field(..., ge=0.0)

    jobs_running: int = Field(..., ge=0)
    month_start: datetime
