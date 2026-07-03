from __future__ import annotations

from typing import Any


def eda_p0_plan() -> list[dict[str, Any]]:
    # A safe, fast, always-available default EDA plan for interactive runs.
    return [
        {"operator": "dataset_overview", "params": {}},
        {"operator": "schema_snapshot", "params": {}},
        {"operator": "privacy_risk_scan", "params": {}},
        {"operator": "preview_rows", "params": {"limit": 25}},
        {"operator": "missingness_scan", "params": {}},
        {"operator": "uniqueness_scan", "params": {"max_columns": 200}},
        {"operator": "numeric_summary", "params": {"max_columns": 25}},
        {"operator": "categorical_topk", "params": {"k": 10, "max_columns": 10}},
        {"operator": "text_summary", "params": {"max_columns": 25}},
        {"operator": "segment_summary", "params": {"limit": 50}},
        {"operator": "resample_aggregate", "params": {"freq": "M", "max_points": 200}},
        {"operator": "time_anomaly_scan", "params": {"freq": "M", "max_points": 200}},
        {"operator": "correlation_matrix", "params": {"max_columns": 25}},
        {"operator": "association_scan", "params": {"max_categorical_columns": 20, "max_numeric_columns": 20, "max_pairs": 200}},
        {"operator": "outlier_scan", "params": {"max_columns": 25}},
    ]
