import pandas as pd
import pytest
from uuid import uuid4

from app.compute.operators.base import OperatorContext
from app.compute.operators.business import (
    ClusteringScanOperator,
    CohortScanOperator,
    FunnelScanOperator,
    RFMScanOperator,
)
from app.compute.operators.viz import CorrelationHeatmapOperator, DistributionChartOperator, SmartVizOperator


def _ctx(df: pd.DataFrame) -> OperatorContext:
    return OperatorContext(
        dataset_id=uuid4(),
        dataset_version="test",
        df=df,
        profile_report={},
        schema_info={"columns": [{"name": c} for c in df.columns]},
    )


def test_cohort_scan_produces_table_and_chart():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "customer_id": [f"c{i % 10}" for i in range(60)],
        "order_date": dates,
    })
    result = CohortScanOperator().run(_ctx(df), {})
    assert "cohort_matrix" in result.tables
    assert result.summary.get("n_customers", 0) > 0


def test_funnel_scan_steps():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "event": ["view", "cart", "view", "purchase", "view"],
    })
    result = FunnelScanOperator().run(_ctx(df), {})
    assert "funnel" in result.tables
    assert len(result.tables["funnel"]) >= 1


def test_rfm_scan_segments():
    df = pd.DataFrame({
        "customer_id": ["a", "a", "b", "b", "c"],
        "order_date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "amount": [100, 50, 200, 80, 30],
    })
    result = RFMScanOperator().run(_ctx(df), {})
    assert "rfm_segments" in result.tables
    assert result.summary.get("n_customers", 0) >= 3


def test_clustering_scan():
    df = pd.DataFrame({
        "f1": [float(i % 5) for i in range(40)],
        "f2": [float(i % 3) for i in range(40)],
        "f3": [float(i % 7) for i in range(40)],
    })
    result = ClusteringScanOperator().run(_ctx(df), {"n_clusters": 3})
    assert "clusters" in result.tables
    assert result.summary.get("n_clusters") == 3


def test_correlation_heatmap_chart():
    df = pd.DataFrame({
        "a": range(30),
        "b": [x * 2 for x in range(30)],
        "c": [x * 0.5 for x in range(30)],
    })
    result = CorrelationHeatmapOperator().run(_ctx(df), {})
    assert "correlation" in result.tables
    assert "correlation_heatmap" in result.charts


def test_distribution_chart():
    df = pd.DataFrame({"sales": [1, 2, 2, 3, 4, 5, 5, 5, 6]})
    result = DistributionChartOperator().run(_ctx(df), {"column": "sales"})
    assert "distribution" in result.charts


def test_smart_viz_scan():
    df = pd.DataFrame({
        "category": ["A", "B", "A", "C"] * 10,
        "value": [float(i) for i in range(40)],
    })
    result = SmartVizOperator().run(_ctx(df), {"max_charts": 2})
    assert result.summary.get("charts_generated", 0) >= 1
    assert len(result.charts) >= 1