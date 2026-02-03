# AI Enterprise Data Analyst - Structured Logging
# Production-grade logging with JSON formatting, context propagation, and performance tracking

from __future__ import annotations

import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec
from uuid import UUID, uuid4
import json
import traceback

from pydantic import BaseModel

# Type variables for decorator typing
P = ParamSpec("P")
T = TypeVar("T")

# Context variables for request tracking (thread-safe)
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


class LogContext(BaseModel):
    """Structured log context for correlation and debugging."""
    
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    extra: dict[str, Any] = {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None and v != {}}


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for production environments.
    
    Produces structured logs compatible with log aggregation systems
    like ELK Stack, Datadog, Splunk, etc.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context from context variables
        if request_id := request_id_ctx.get():
            log_data["request_id"] = request_id
        if user_id := user_id_ctx.get():
            log_data["user_id"] = user_id
        if session_id := session_id_ctx.get():
            log_data["session_id"] = session_id
        
        # Add extra fields from LogRecord
        if hasattr(record, "context") and isinstance(record.context, LogContext):
            log_data["context"] = record.context.to_dict()
        
        if hasattr(record, "extra_data") and record.extra_data:
            log_data["extra"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }
        
        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development environments."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m"
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Build context string
        context_parts = []
        if request_id := request_id_ctx.get():
            context_parts.append(f"req:{request_id[:8]}")
        if user_id := user_id_ctx.get():
            context_parts.append(f"user:{user_id}")
        
        context_str = f" [{' '.join(context_parts)}]" if context_parts else ""
        
        formatted = (
            f"{timestamp} | "
            f"{color}{record.levelname:8}{reset} | "
            f"{record.name}:{record.funcName}:{record.lineno}"
            f"{context_str} | "
            f"{record.getMessage()}"
        )
        
        if record.exc_info:
            formatted += "\n" + "".join(traceback.format_exception(*record.exc_info))
        
        return formatted


class StructuredLogger:
    """
    Enterprise-grade structured logger with context propagation.
    
    Implements the Decorator and Facade patterns for clean logging API.
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        use_json: bool = False
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False
        
        # Remove existing handlers
        self._logger.handlers.clear()
        
        # Add appropriate handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter() if use_json else TextFormatter())
        self._logger.addHandler(handler)
    
    def _log(
        self,
        level: int,
        message: str,
        context: Optional[LogContext] = None,
        **extra: Any
    ) -> None:
        """Internal logging method with context support."""
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        if context:
            record.context = context
        if extra:
            record.extra_data = extra
        
        self._logger.handle(record)
    
    def debug(self, message: str, context: Optional[LogContext] = None, **extra: Any) -> None:
        self._log(logging.DEBUG, message, context, **extra)
    
    def info(self, message: str, context: Optional[LogContext] = None, **extra: Any) -> None:
        self._log(logging.INFO, message, context, **extra)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **extra: Any) -> None:
        self._log(logging.WARNING, message, context, **extra)
    
    def error(
        self,
        message: str,
        context: Optional[LogContext] = None,
        exc_info: bool = False,
        **extra: Any
    ) -> None:
        if exc_info:
            self._logger.error(message, exc_info=True, extra={"context": context, "extra_data": extra})
        else:
            self._log(logging.ERROR, message, context, **extra)
    
    def critical(
        self,
        message: str,
        context: Optional[LogContext] = None,
        exc_info: bool = True,
        **extra: Any
    ) -> None:
        if exc_info:
            self._logger.critical(message, exc_info=True, extra={"context": context, "extra_data": extra})
        else:
            self._log(logging.CRITICAL, message, context, **extra)
    
    def exception(self, message: str, context: Optional[LogContext] = None, **extra: Any) -> None:
        """Log exception with full traceback."""
        self._logger.exception(message, extra={"context": context, "extra_data": extra})


def log_execution_time(
    logger: Optional[StructuredLogger] = None,
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    warn_threshold_ms: float = 1000.0
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for logging function execution time and performance metrics.
    
    Implements the Decorator pattern for cross-cutting performance monitoring.
    
    Args:
        logger: Logger instance (uses default if None)
        operation_name: Custom operation name (uses function name if None)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        warn_threshold_ms: Threshold in ms for warning log level
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        _logger = logger or StructuredLogger(func.__module__)
        _operation = operation_name or func.__name__
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()
            
            context = LogContext(
                operation=_operation,
                request_id=request_id_ctx.get()
            )
            
            extra: dict[str, Any] = {}
            if log_args:
                extra["args"] = str(args)[:200]
                extra["kwargs"] = str(kwargs)[:200]
            
            _logger.debug(f"Starting {_operation}", context=context, **extra)
            
            try:
                result = func(*args, **kwargs)
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                context.duration_ms = round(duration_ms, 2)
                
                if log_result:
                    extra["result"] = str(result)[:200]
                
                log_method = _logger.warning if duration_ms > warn_threshold_ms else _logger.info
                log_method(
                    f"Completed {_operation} in {duration_ms:.2f}ms",
                    context=context,
                    **extra
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                context.duration_ms = round(duration_ms, 2)
                
                _logger.error(
                    f"Failed {_operation} after {duration_ms:.2f}ms: {str(e)}",
                    context=context,
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


async def log_execution_time_async(
    logger: Optional[StructuredLogger] = None,
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    warn_threshold_ms: float = 1000.0
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Async version of log_execution_time decorator."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        _logger = logger or StructuredLogger(func.__module__)
        _operation = operation_name or func.__name__
        
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()
            
            context = LogContext(operation=_operation, request_id=request_id_ctx.get())
            
            try:
                result = await func(*args, **kwargs)
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                context.duration_ms = round(duration_ms, 2)
                
                log_method = _logger.warning if duration_ms > warn_threshold_ms else _logger.info
                log_method(f"Completed {_operation} in {duration_ms:.2f}ms", context=context)
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                context.duration_ms = round(duration_ms, 2)
                _logger.error(
                    f"Failed {_operation} after {duration_ms:.2f}ms: {str(e)}",
                    context=context,
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> None:
    """Set request context for all logs in the current context."""
    if request_id:
        request_id_ctx.set(request_id)
    if user_id:
        user_id_ctx.set(user_id)
    if session_id:
        session_id_ctx.set(session_id)


def clear_request_context() -> None:
    """Clear request context after request completion."""
    request_id_ctx.set(None)
    user_id_ctx.set(None)
    session_id_ctx.set(None)


def generate_request_id() -> str:
    """Generate unique request ID for correlation."""
    return str(uuid4())


# Factory function for creating loggers
def get_logger(
    name: str,
    use_json: Optional[bool] = None,
    level: int = logging.INFO
) -> StructuredLogger:
    """
    Factory function for creating structured loggers.
    
    Args:
        name: Logger name (typically __name__)
        use_json: Use JSON format (auto-detected from environment if None)
        level: Logging level
    
    Returns:
        Configured StructuredLogger instance
    """
    import os
    if use_json is None:
        use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"
    
    return StructuredLogger(name=name, level=level, use_json=use_json)
