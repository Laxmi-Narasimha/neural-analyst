# AI Enterprise Data Analyst - FastAPI Main Application
# Production-grade FastAPI application with middleware, lifecycle, and error handling

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import BaseApplicationException
from app.core.logging import (
    get_logger,
    set_request_context,
    clear_request_context,
    generate_request_id,
    LogContext
)
from app.services.database import init_database, close_database

logger = get_logger(__name__)


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for:
    - Database connections
    - Redis connections
    - Background task workers
    - ML model loading
    """
    logger.info(
        "Starting application",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value
    )
    
    try:
        # Startup
        await init_database()
        
        # TODO: Initialize Redis
        # TODO: Initialize Celery workers
        # TODO: Load ML models
        
        logger.info("Application startup complete")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down application")
        
        await close_database()
        
        # TODO: Close Redis connections
        # TODO: Stop Celery workers
        
        logger.info("Application shutdown complete")


# ============================================================================
# Middleware Classes
# ============================================================================

class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request context management.
    
    Adds request ID tracking, timing, and logging context
    to all requests for observability.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", generate_request_id())
        
        # Extract user info from auth (if available)
        user_id = getattr(request.state, "user_id", None)
        
        # Set context for logging
        set_request_context(
            request_id=request_id,
            user_id=str(user_id) if user_id else None
        )
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Track timing
        start_time = datetime.utcnow()
        
        try:
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log request completion
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            log_context = LogContext(
                request_id=request_id,
                operation="http_request",
                duration_ms=duration_ms
            )
            
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code}",
                context=log_context,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2)
            )
            
            return response
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                exc_info=True,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2)
            )
            raise
            
        finally:
            clear_request_context()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    
    In production, use Redis-based rate limiting for distributed systems.
    """
    
    # In-memory store (replace with Redis in production)
    _requests: dict[str, list[float]] = {}
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP or user ID)
        client_id = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Check rate limit
        now = datetime.utcnow().timestamp()
        window_start = now - settings.security.rate_limit_period
        
        # Clean old requests
        if client_id in self._requests:
            self._requests[client_id] = [
                ts for ts in self._requests[client_id]
                if ts > window_start
            ]
        else:
            self._requests[client_id] = []
        
        # Check limit
        if len(self._requests[client_id]) >= settings.security.rate_limit_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "message": "Rate limit exceeded",
                    "retry_after": settings.security.rate_limit_period
                },
                headers={"Retry-After": str(settings.security.rate_limit_period)}
            )
        
        # Record request
        self._requests[client_id].append(now)
        
        return await call_next(request)


# ============================================================================
# Exception Handlers
# ============================================================================

async def application_exception_handler(
    request: Request,
    exc: BaseApplicationException
) -> JSONResponse:
    """Handle application-specific exceptions."""
    logger.error(
        f"Application error: {exc.message}",
        error_code=exc.error_code.value,
        error_id=str(exc.context.error_id)
    )
    
    return JSONResponse(
        status_code=exc.http_status_code,
        content=exc.to_dict()
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": errors
        }
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        request_id=request_id
    )
    
    # Don't expose internal errors in production
    if settings.is_production:
        message = "An internal error occurred"
    else:
        message = str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": message,
            "request_id": request_id
        }
    )


# ============================================================================
# Application Factory
# ============================================================================

def create_application() -> FastAPI:
    """
    Application factory function.
    
    Creates and configures the FastAPI application with all
    middleware, exception handlers, and routers.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI Enterprise Data Analyst - Comprehensive AI-powered data analytics platform",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan
    )
    
    # Add middleware (order matters - last added runs first)
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.cors_allow_credentials,
        allow_methods=settings.security.cors_allow_methods,
        allow_headers=settings.security.cors_allow_headers,
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware)
    
    # Request context
    app.add_middleware(RequestContextMiddleware)
    
    # Exception handlers
    app.add_exception_handler(BaseApplicationException, application_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    # Register routers
    from app.api.routes import api_router
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check endpoints
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for load balancers."""
        return {
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.environment.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """Readiness check for Kubernetes."""
        # Check database connectivity
        from app.services.database import db_manager
        db_healthy = await db_manager.health_check()
        
        services = {
            "database": {"status": "healthy" if db_healthy else "unhealthy"}
        }
        
        all_healthy = all(s["status"] == "healthy" for s in services.values())
        
        return {
            "status": "ready" if all_healthy else "not_ready",
            "services": services,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "AI Enterprise Data Analyst API",
            "docs_url": "/docs" if settings.is_development else None,
            "health_url": "/health"
        }
    
    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.workers,
        log_level=settings.log_level.value.lower()
    )
