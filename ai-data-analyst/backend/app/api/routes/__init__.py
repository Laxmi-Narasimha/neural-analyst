# AI Enterprise Data Analyst - API Routes Package
"""API route exports."""

from fastapi import APIRouter

from app.api.routes import datasets, chat, analyses, auth, ml, analytics, connections, data_quality, data_speaks, jobs, dashboard, artifacts, metrics, shares, maintenance, billing

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
api_router.include_router(data_speaks.router, prefix="/data-speaks", tags=["Data Speaks"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
api_router.include_router(artifacts.router, prefix="/artifacts", tags=["Artifacts"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
api_router.include_router(shares.router, prefix="/shares", tags=["Shares"])
api_router.include_router(shares.public_router, prefix="/public", tags=["Public"])
api_router.include_router(maintenance.router, prefix="/maintenance", tags=["Maintenance"])
api_router.include_router(billing.router, prefix="/billing", tags=["Billing"])

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
    "data_speaks",
    "jobs",
    "dashboard",
    "artifacts",
    "metrics",
    "shares",
    "maintenance",
    "billing",
]

