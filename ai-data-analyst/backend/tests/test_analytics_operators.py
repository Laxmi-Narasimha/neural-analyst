import tempfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from app.compute.operators.analytics import (
    ABTestScanOperator,
    DuckDBProfileOperator,
    ForecastScanOperator,
    SentimentScanOperator,
)
from app.compute.operators.base import OperatorContext


def _ctx(
    df: pd.DataFrame,
    *,
    source_file_path: str | None = None,
    source_file_format: str | None = None,
    schema_info: dict | None = None,
) -> OperatorContext:
    return OperatorContext(
        dataset_id=uuid4(),
        dataset_version="test",
        df=df,
        profile_report={"row_count": len(df), "column_count": len(df.columns)},
        schema_info=schema_info or {"columns": [{"name": c} for c in df.columns]},
        source_file_path=source_file_path,
        source_file_format=source_file_format,
    )


def test_duckdb_profile_pandas_fallback_without_file():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = DuckDBProfileOperator().run(_ctx(df), {})
    assert result.summary["engine"] == "pandas_sample"
    assert result.summary["rows"] == 3
    assert result.summary["columns"] == 2


def test_duckdb_profile_csv_pushdown():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.csv"
        pd.DataFrame({"x": range(50), "y": ["a", "b"] * 25}).to_csv(path, index=False)
        result = DuckDBProfileOperator().run(
            _ctx(pd.DataFrame({"x": [1]}), source_file_path=str(path), source_file_format="csv"),
            {},
        )
        assert result.summary["engine"] == "duckdb_pushdown"
        assert int(result.summary.get("row_count", 0)) == 50


def test_sentiment_scan_finds_text_column():
    df = pd.DataFrame(
        {
            "id": range(20),
            "review": [
                "This product is absolutely wonderful and exceeded expectations",
                "Terrible quality, very disappointed with the purchase",
            ]
            * 10,
        }
    )
    result = SentimentScanOperator().run(_ctx(df), {"sample_rows": 20})
    assert "sentiment" in result.tables
    assert result.tables["sentiment"].shape[0] >= 1
    assert result.summary["columns"][0]["column"] == "review"


def test_forecast_scan_infers_columns():
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"month": dates, "revenue": [100 + i * 3 for i in range(24)]})
    result = ForecastScanOperator().run(_ctx(df), {"horizon": 4})
    assert "forecast" in result.tables
    assert result.summary["time_column"] == "month"
    assert result.summary["value_column"] == "revenue"


def test_ab_test_scan_detects_variants():
    df = pd.DataFrame(
        {
            "variant": ["control"] * 50 + ["treatment"] * 50,
            "conversion": [0, 1] * 50,
        }
    )
    result = ABTestScanOperator().run(_ctx(df), {})
    row = result.summary
    assert row["variant_a"] == "control"
    assert row["variant_b"] == "treatment"
    assert "p_value" in row