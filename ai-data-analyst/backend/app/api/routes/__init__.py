# AI Enterprise Data Analyst - API Routes Package
"""API route exports."""

from fastapi import APIRouter

from app.api.routes import datasets, chat, analyses, auth, ml, analytics, connections, data_quality

# Create main router
api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(analyses.router, prefix="/analyses", tags=["Analyses"])
api_router.include_router(ml.router, prefix="/ml", tags=["Machine Learning"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(connections.router, prefix="/connections", tags=["Database Connections"])
api_router.include_router(data_quality.router, prefix="/quality", tags=["Data Quality"])

__all__ = [
    "api_router",
    "auth",
    "datasets",
    "chat",
    "analyses",
    "ml",
    "analytics",
    "connections",
    "data_quality",
]

