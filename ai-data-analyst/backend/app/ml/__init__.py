# AI Enterprise Data Analyst - ML Package
"""Production-grade Machine Learning components.

This package provides lazy imports to avoid circular dependencies during testing.
Import specific modules directly, e.g.:
    from app.ml.data_masking import DataMaskingEngine
"""

__all__ = [
    # Core modules are available via direct import
    "ml_engine",
    "feature_store",
    "data_quality",
    "imputation",
    "causal_inference",
    "ab_testing",
    "anomaly_detection",
    "recommendations",
    "deep_learning",
    "geospatial",
    "streaming",
    "fraud_detection",
    "text_analytics",
    "forecasting",
    "customer_analytics",
    "automl",
    "explainability",
    "model_registry",
    "segmentation",
    "statistical_tests",
    "data_profiling",
    "bi_metrics",
    "association_rules",
    "survival_analysis",
    "monte_carlo",
    "advanced_parsers",
    "export_engine",
    "financial_analytics",
    # New modules
    "data_masking",
    "data_validation",
    "data_quality_report",
    "data_enrichment",
    "data_sampling",
    "outlier_treatment",
    "schema_inference",
    "hypothesis_testing",
    "univariate_analysis",
    "bivariate_analysis",
    "correlation_analysis",
    "distribution_analysis",
    "trend_analysis",
    "time_series_decomposition",
    "clustering_quality",
    "price_analysis",
    "inventory_analysis",
    "profit_analysis",
    "growth_analysis",
    "session_analysis",
    "metric_calculator",
    "rfm_analysis",
    "cohort_analysis",
    "churn_prediction",
    "funnel_analysis",
    "keyword_extraction",
    "text_summarization",
    "sentiment_analysis",
    "data_catalog",
    "data_lineage",
    "experiment_tracker",
    "alert_manager",
    "report_generator",
    "custom_aggregations",
    "benchmarking",
    "eda_automation",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    import importlib
    
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        return module
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
