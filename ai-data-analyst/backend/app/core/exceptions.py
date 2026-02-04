# AI Enterprise Data Analyst - Custom Exceptions
# Enterprise-grade exception hierarchy with error codes, context, and recovery hints

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class ErrorCode(str, Enum):
    """Standardized error codes for API responses and logging."""
    
    # General errors (1xxx)
    UNKNOWN_ERROR = "E1000"
    VALIDATION_ERROR = "E1001"
    CONFIGURATION_ERROR = "E1002"
    INITIALIZATION_ERROR = "E1003"
    
    # Authentication & Authorization (2xxx)
    AUTHENTICATION_FAILED = "E2000"
    AUTHORIZATION_FAILED = "E2001"
    TOKEN_EXPIRED = "E2002"
    TOKEN_INVALID = "E2003"
    INSUFFICIENT_PERMISSIONS = "E2004"
    
    # Data errors (3xxx)
    DATA_NOT_FOUND = "E3000"
    DATA_ALREADY_EXISTS = "E3001"
    DATA_VALIDATION_FAILED = "E3002"
    DATA_INTEGRITY_ERROR = "E3003"
    DATA_FORMAT_ERROR = "E3004"
    DATA_TYPE_MISMATCH = "E3005"
    DATA_SIZE_EXCEEDED = "E3006"
    DATA_QUALITY_ERROR = "E3007"
    
    # File errors (4xxx)
    FILE_NOT_FOUND = "E4000"
    FILE_UPLOAD_FAILED = "E4001"
    FILE_FORMAT_UNSUPPORTED = "E4002"
    FILE_SIZE_EXCEEDED = "E4003"
    FILE_PARSE_ERROR = "E4004"
    FILE_ENCODING_ERROR = "E4005"
    
    # Database errors (5xxx)
    DATABASE_CONNECTION_ERROR = "E5000"
    DATABASE_QUERY_ERROR = "E5001"
    DATABASE_TRANSACTION_ERROR = "E5002"
    DATABASE_CONSTRAINT_VIOLATION = "E5003"
    DATABASE_TIMEOUT = "E5004"
    
    # External service errors (6xxx)
    EXTERNAL_SERVICE_ERROR = "E6000"
    OPENAI_API_ERROR = "E6001"
    OPENAI_RATE_LIMIT = "E6002"
    PINECONE_ERROR = "E6003"
    REDIS_ERROR = "E6004"
    
    # ML/AI errors (7xxx)
    MODEL_NOT_FOUND = "E7000"
    MODEL_TRAINING_FAILED = "E7001"
    MODEL_PREDICTION_FAILED = "E7002"
    MODEL_VALIDATION_FAILED = "E7003"
    FEATURE_ENGINEERING_ERROR = "E7004"
    INSUFFICIENT_DATA = "E7005"
    
    # Agent errors (8xxx)
    AGENT_EXECUTION_ERROR = "E8000"
    AGENT_TIMEOUT = "E8001"
    AGENT_INVALID_ACTION = "E8002"
    AGENT_MEMORY_ERROR = "E8003"
    
    # Rate limiting & resource errors (9xxx)
    RATE_LIMIT_EXCEEDED = "E9000"
    RESOURCE_EXHAUSTED = "E9001"
    QUOTA_EXCEEDED = "E9002"


@dataclass(frozen=True)
class ErrorContext:
    """Immutable context information for error tracking and debugging."""
    
    error_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    component: str = ""
    operation: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_id": str(self.error_id),
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "additional_data": self.additional_data
        }


class BaseApplicationException(Exception):
    """
    Base exception class for all application exceptions.
    
    Implements the Template Method pattern for consistent error handling
    across the entire application.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recovery_hint: Optional[str] = None,
        is_retryable: bool = False,
        http_status_code: int = 500
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext()
        self.cause = cause
        self.recovery_hint = recovery_hint
        self.is_retryable = is_retryable
        self.http_status_code = http_status_code
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": True,
            "error_code": self.error_code.value,
            "error_type": self.__class__.__name__,
            "message": self.message,
            "recovery_hint": self.recovery_hint,
            "is_retryable": self.is_retryable,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code={self.error_code}, "
            f"error_id={self.context.error_id})"
        )


# ============================================================================
# Validation Exceptions
# ============================================================================

class ValidationException(BaseApplicationException):
    """Exception for data validation failures."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[dict[str, list[str]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            http_status_code=422,
            **kwargs
        )
        self.field_errors = field_errors or {}
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["field_errors"] = self.field_errors
        return result


# ============================================================================
# Authentication Exceptions
# ============================================================================

class AuthenticationException(BaseApplicationException):
    """Exception for authentication failures."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            http_status_code=401,
            recovery_hint="Please log in again",
            **kwargs
        )


class AuthorizationException(BaseApplicationException):
    """Exception for authorization failures."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            http_status_code=403,
            recovery_hint="Request necessary permissions from administrator",
            **kwargs
        )


# ============================================================================
# Data Exceptions
# ============================================================================

class DataException(BaseApplicationException):
    """Base exception for data-related errors."""
    pass


class DataNotFoundException(DataException):
    """Exception when requested data is not found."""
    
    def __init__(self, resource_type: str, resource_id: Any, **kwargs: Any) -> None:
        super().__init__(
            message=f"{resource_type} with ID '{resource_id}' not found",
            error_code=ErrorCode.DATA_NOT_FOUND,
            http_status_code=404,
            recovery_hint=f"Verify the {resource_type.lower()} ID and try again",
            **kwargs
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class DataAlreadyExistsException(DataException):
    """Exception when attempting to create duplicate data."""
    
    def __init__(self, resource_type: str, identifier: Any, **kwargs: Any) -> None:
        super().__init__(
            message=f"{resource_type} with identifier '{identifier}' already exists",
            error_code=ErrorCode.DATA_ALREADY_EXISTS,
            http_status_code=409,
            recovery_hint=f"Use a different identifier or update the existing {resource_type.lower()}",
            **kwargs
        )


class DataQualityException(DataException):
    """Exception for data quality issues."""
    
    def __init__(
        self,
        message: str,
        quality_issues: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DATA_QUALITY_ERROR,
            http_status_code=422,
            **kwargs
        )
        self.quality_issues = quality_issues or []


class DataProcessingException(DataException):
    """Exception for data processing errors."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DATA_FORMAT_ERROR,
            http_status_code=500,
            **kwargs
        )


# ============================================================================
# File Exceptions
# ============================================================================

class FileException(BaseApplicationException):
    """Base exception for file-related errors."""
    pass


class FileUploadException(FileException):
    """Exception for file upload failures."""
    
    def __init__(self, filename: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Failed to upload file '{filename}': {reason}",
            error_code=ErrorCode.FILE_UPLOAD_FAILED,
            http_status_code=400,
            is_retryable=True,
            **kwargs
        )
        self.filename = filename
        self.reason = reason


class FileFormatException(FileException):
    """Exception for unsupported file formats."""
    
    def __init__(
        self,
        filename: str,
        actual_format: str,
        supported_formats: list[str],
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Unsupported file format '{actual_format}' for file '{filename}'",
            error_code=ErrorCode.FILE_FORMAT_UNSUPPORTED,
            http_status_code=415,
            recovery_hint=f"Supported formats: {', '.join(supported_formats)}",
            **kwargs
        )
        self.filename = filename
        self.actual_format = actual_format
        self.supported_formats = supported_formats


class FileParseException(FileException):
    """Exception for file parsing errors."""
    
    def __init__(
        self,
        filename: str,
        parse_errors: Optional[list[str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Failed to parse file '{filename}'",
            error_code=ErrorCode.FILE_PARSE_ERROR,
            http_status_code=422,
            **kwargs
        )
        self.filename = filename
        self.parse_errors = parse_errors or []


class FileNotFoundException(FileException):
    """Exception when a referenced file is missing on disk/object storage."""

    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"File not found: '{filename}'",
            error_code=ErrorCode.FILE_NOT_FOUND,
            http_status_code=404,
            recovery_hint="Re-upload the file or verify the dataset asset location",
            **kwargs,
        )
        self.filename = filename


# ============================================================================
# Database Exceptions
# ============================================================================

class DatabaseException(BaseApplicationException):
    """Base exception for database errors."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            error_code=kwargs.pop("error_code", ErrorCode.DATABASE_QUERY_ERROR),
            http_status_code=500,
            is_retryable=True,
            **kwargs
        )


class DatabaseConnectionException(DatabaseException):
    """Exception for database connection failures."""
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            message="Failed to connect to database",
            error_code=ErrorCode.DATABASE_CONNECTION_ERROR,
            recovery_hint="Check database configuration and connectivity",
            **kwargs
        )


# ============================================================================
# External Service Exceptions
# ============================================================================

class ExternalServiceException(BaseApplicationException):
    """Base exception for external service errors."""
    
    def __init__(self, service_name: str, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"{service_name} error: {message}",
            http_status_code=502,
            is_retryable=True,
            **kwargs
        )
        self.service_name = service_name


class OpenAIException(ExternalServiceException):
    """Exception for OpenAI API errors."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            service_name="OpenAI",
            message=message,
            error_code=ErrorCode.OPENAI_API_ERROR,
            **kwargs
        )


class OpenAIRateLimitException(OpenAIException):
    """Exception for OpenAI rate limiting."""
    
    def __init__(self, retry_after: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(
            message="Rate limit exceeded",
            error_code=ErrorCode.OPENAI_RATE_LIMIT,
            recovery_hint=f"Retry after {retry_after} seconds" if retry_after else "Wait and retry",
            **kwargs
        )
        self.retry_after = retry_after


class PineconeException(ExternalServiceException):
    """Exception for Pinecone errors."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            service_name="Pinecone",
            message=message,
            error_code=ErrorCode.PINECONE_ERROR,
            **kwargs
        )


# ============================================================================
# ML/AI Exceptions
# ============================================================================

class MLException(BaseApplicationException):
    """Base exception for ML/AI errors."""
    pass


class ModelNotFoundException(MLException):
    """Exception when ML model is not found."""
    
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Model '{model_name}' not found",
            error_code=ErrorCode.MODEL_NOT_FOUND,
            http_status_code=404,
            **kwargs
        )
        self.model_name = model_name


class ModelTrainingException(MLException):
    """Exception for model training failures."""
    
    def __init__(
        self,
        message: str,
        training_metrics: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Model training failed: {message}",
            error_code=ErrorCode.MODEL_TRAINING_FAILED,
            http_status_code=500,
            is_retryable=True,
            **kwargs
        )
        self.training_metrics = training_metrics or {}


class InsufficientDataException(MLException):
    """Exception when data is insufficient for ML operations."""
    
    def __init__(
        self,
        required_samples: int,
        actual_samples: int,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Insufficient data: required {required_samples}, got {actual_samples}",
            error_code=ErrorCode.INSUFFICIENT_DATA,
            http_status_code=422,
            recovery_hint=f"Provide at least {required_samples} samples",
            **kwargs
        )
        self.required_samples = required_samples
        self.actual_samples = actual_samples


# ============================================================================
# Agent Exceptions
# ============================================================================

class AgentException(BaseApplicationException):
    """Base exception for agent errors."""
    pass


class AgentExecutionException(AgentException):
    """Exception for agent execution failures."""
    
    def __init__(
        self,
        agent_name: str,
        step: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        message = f"Agent '{agent_name}' execution failed"
        if step:
            message += f" at step '{step}'"
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_EXECUTION_ERROR,
            http_status_code=500,
            is_retryable=True,
            **kwargs
        )
        self.agent_name = agent_name
        self.step = step


class AgentTimeoutException(AgentException):
    """Exception for agent timeout."""
    
    def __init__(
        self,
        agent_name: str,
        timeout_seconds: int,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Agent '{agent_name}' timed out after {timeout_seconds} seconds",
            error_code=ErrorCode.AGENT_TIMEOUT,
            http_status_code=504,
            is_retryable=True,
            recovery_hint="Try reducing the complexity of the request",
            **kwargs
        )
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds


# ============================================================================
# Rate Limiting Exceptions
# ============================================================================

class RateLimitException(BaseApplicationException):
    """Exception for rate limiting."""
    
    def __init__(
        self,
        limit: int,
        period_seconds: int,
        retry_after: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {period_seconds} seconds",
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            http_status_code=429,
            is_retryable=True,
            recovery_hint=f"Retry after {retry_after} seconds" if retry_after else "Wait and retry",
            **kwargs
        )
        self.limit = limit
        self.period_seconds = period_seconds
        self.retry_after = retry_after
