# AI Enterprise Data Analyst - Application Package
"""
AI Enterprise Data Analyst - Comprehensive AI-powered data analytics platform.

This package provides:
- Multi-format data ingestion and processing
- Automated exploratory data analysis (EDA)
- Machine learning and deep learning pipelines
- Natural language querying with LLM integration
- Interactive visualizations and dashboards
- Real-time analytics and streaming
"""

__version__ = "3.0.0"
__author__ = "AI Enterprise Team"

# Lazy import to avoid circular dependencies during testing
def get_app():
    from app.main import app
    return app

__all__ = ["get_app", "__version__"]
