from __future__ import annotations

from typing import Any


def eda_p0_plan() -> list[dict[str, Any]]:
    # A safe, fast, always-available default EDA plan for interactive runs.
    return [
        {"operator": "schema_snapshot", "params": {}},
        {"operator": "preview_rows", "params": {"limit": 25}},
        {"operator": "missingness_scan", "params": {}},
        {"operator": "numeric_summary", "params": {"max_columns": 25}},
        {"operator": "categorical_topk", "params": {"k": 10, "max_columns": 10}},
        {"operator": "correlation_matrix", "params": {"max_columns": 25}},
        {"operator": "outlier_scan", "params": {"max_columns": 25}},
    ]

