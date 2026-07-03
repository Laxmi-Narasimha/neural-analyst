import pandas as pd
import numpy as np

from uuid import uuid4

from app.compute.operators.base import OperatorContext
from app.compute.operators.eda import (
    DatasetOverviewOperator,
    SchemaSnapshotOperator,
    PreviewRowsOperator,
    UniquenessOperator,
    TextSummaryOperator,
    SegmentSummaryOperator,
    ResampleAggregateOperator,
    TimeAnomalyScanOperator,
    AssociationScanOperator,
    RelationshipExplainOperator,
    CategoricalTopKOperator,
    PrivacyRiskScanOperator,
    MissingnessPatternsOperator,
    OutlierExplainOperator,
    SegmentDeepDiveOperator,
)


def _ctx(df: pd.DataFrame, *, schema_info: dict | None = None) -> OperatorContext:
    return OperatorContext(
        dataset_id=uuid4(),
        dataset_version="test",
        df=df,
        profile_report={},
        schema_info=schema_info or {},
    )


def test_uniqueness_scan_key_candidates_and_duplicates():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "dup": ["a", "a", "b", "b", "b"],
            "with_nulls": [1, None, 1, None, 1],
        }
    )
    op = UniquenessOperator()
    result = op.run(_ctx(df), {"max_columns": 50})

    out = result.tables["uniqueness"]
    assert set(out.columns) >= {
        "column",
        "non_null_count",
        "null_count",
        "unique_count",
        "unique_ratio",
        "duplicate_count",
        "duplicate_ratio",
        "is_key_candidate",
    }

    id_row = out[out["column"] == "id"].iloc[0].to_dict()
    assert id_row["null_count"] == 0
    assert id_row["unique_count"] == 5
    assert bool(id_row["is_key_candidate"]) is True

    dup_row = out[out["column"] == "dup"].iloc[0].to_dict()
    assert dup_row["unique_count"] == 2
    assert dup_row["duplicate_count"] == 3
    assert bool(dup_row["is_key_candidate"]) is False

    assert "id" in (result.summary.get("key_candidates") or [])


def test_dataset_overview_reports_rows_and_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    op = DatasetOverviewOperator()
    ctx = _ctx(df, schema_info={"columns": [{"name": "a"}, {"name": "b"}]})
    # Profile row_count/column_count are optional; should fall back to sampled shape.
    result = op.run(ctx, {})
    out = result.tables["overview"]
    assert out.shape[0] == 1
    row = out.iloc[0].to_dict()
    assert int(row["rows"]) == 3
    assert int(row["columns"]) == 2
    assert int(row["sampled_rows"]) == 3
    assert int(row["sampled_columns"]) == 2


def test_text_summary_length_stats_no_values_emitted():
    df = pd.DataFrame(
        {
            "text": [" hello ", "", None, "abc", "x" * 10],
            "num": [1, 2, 3, 4, 5],
        }
    )
    op = TextSummaryOperator()
    result = op.run(_ctx(df), {"max_columns": 10})

    out = result.tables["text_summary"]
    assert list(out.columns) == ["column", "non_null_count", "avg_length", "min_length", "max_length", "empty_pct"]
    assert out.shape[0] == 1
    row = out.iloc[0].to_dict()
    assert row["column"] == "text"
    assert row["non_null_count"] == 4
    assert row["min_length"] == 0
    assert row["max_length"] == 10
    # Only structural stats should exist in the output.
    assert "hello" not in str(out.to_dict())


def test_segment_summary_auto_group_and_value_column():
    df = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 10.0, 20.0, 30.0],
        }
    )
    op = SegmentSummaryOperator()
    result = op.run(_ctx(df), {"limit": 10})

    out = result.tables["segments"]
    assert out.shape[0] == 2
    assert list(out.columns) == ["group", "count", "value_mean", "value_sum", "value_median"]

    b = out[out["group"] == "B"].iloc[0].to_dict()
    assert b["count"] == 3
    assert float(b["value_sum"]) == 60.0
    assert float(b["value_mean"]) == 20.0
    assert float(b["value_median"]) == 20.0


def test_resample_aggregate_coarsens_to_max_points_and_formats_period():
    # 400 days of daily points -> with max_points=50 it should coarsen from D to W/M.
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=400, freq="D"),
            "y": np.ones(400, dtype=float),
        }
    )
    schema_info = {"columns": [{"name": "ts", "inferred_type": "datetime"}]}
    op = ResampleAggregateOperator()
    result = op.run(_ctx(df, schema_info=schema_info), {"freq": "D", "max_points": 50, "time_column": "ts", "value_column": "y"})

    out = result.tables["resample"]
    assert out.shape[0] <= 50
    assert list(out.columns) == ["period", "count", "value_mean", "value_sum"]
    assert out["period"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}").all()


def test_association_scan_finds_cat_num_and_cat_cat():
    df = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B", "B", "C", "C", "C"],
            "num": [1.0, 1.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0],
            "cat2": ["X", "X", "Y", "Y", "Y", "Z", "Z", "Z"],
        }
    )
    op = AssociationScanOperator()
    result = op.run(
        _ctx(df),
        {
            "min_group_size": 1,
            "max_pairs": 50,
            "max_levels": 10,
            "max_categorical_columns": 10,
            "max_numeric_columns": 10,
        },
    )

    out = result.tables["associations"]
    assert out.shape[0] >= 2

    cat_num = out[(out["association_type"] == "categorical_numeric_eta2") & (out["column_a"] == "cat") & (out["column_b"] == "num")]
    assert not cat_num.empty
    assert float(cat_num.iloc[0]["score"]) > 0.8

    cat_cat = out[(out["association_type"] == "categorical_categorical_cramers_v")]
    assert not cat_cat.empty
    assert float(cat_cat.iloc[0]["score"]) > 0.9


def test_privacy_risk_scan_surfaces_pii_flags_from_schema_info():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com"], "id": [1, 2]})
    schema_info = {
        "columns": [
            {"name": "email", "inferred_type": "text", "is_potential_pii": True, "is_unique_identifier": False},
            {"name": "id", "inferred_type": "int", "is_potential_pii": False, "is_unique_identifier": True},
        ]
    }
    op = PrivacyRiskScanOperator()
    result = op.run(_ctx(df, schema_info=schema_info), {})
    out = result.tables["risk"]
    assert out.shape[0] == 2
    assert int(result.summary.get("pii_columns") or 0) == 1
    assert int(result.summary.get("identifier_columns") or 0) == 1


def test_missingness_patterns_emits_bounded_tables():
    df = pd.DataFrame(
        {
            "a": [1, None, 3, None, 5, None, 7, 8, None, 10],
            "seg": ["x", "x", "x", "x", "y", "y", "y", "y", "y", "y"],
            "noise": ["p"] * 10,
        }
    )
    op = MissingnessPatternsOperator()
    result = op.run(_ctx(df), {"top_columns": 2, "min_group_count": 2, "k": 5})
    assert "missing_columns" in result.tables
    assert "missingness_by_category" in result.tables
    assert "missingness_numeric_assoc" in result.tables
    assert "missingness_over_time" in result.tables
    cols = result.tables["missing_columns"]
    assert not cols.empty
    assert cols.iloc[0]["null_count"] > 0


def test_outlier_explain_returns_quantiles_and_bounds():
    df = pd.DataFrame({"x": [0, 1, 2, 3, 1000], "y": [1, 1, 1, 1, 1]})
    op = OutlierExplainOperator()
    result = op.run(_ctx(df), {"top_columns": 1})
    out = result.tables["outlier_columns"]
    assert not out.empty
    assert "lower_bound" in out.columns and "upper_bound" in out.columns
    q = result.tables["outlier_quantiles"]
    assert not q.empty
    assert set(["p01", "p99", "min", "max"]).issubset(set(q.columns))


def test_segment_deep_dive_auto_group_and_outputs_diffs():
    df = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B", "B", "C", "C"],
            "value": [1.0, 2.0, 10.0, 20.0, 30.0, 5.0, 6.0],
            "other": [0, 1, 0, 1, 2, 0, 1],
        }
    )
    op = SegmentDeepDiveOperator()
    result = op.run(_ctx(df), {"limit": 5})
    seg = result.tables["segment_summary"]
    assert not seg.empty
    diffs = result.tables["segment_numeric_diff"]
    assert isinstance(diffs, pd.DataFrame)


def test_preview_rows_masks_pii_values_from_schema_info():
    df = pd.DataFrame({"email": ["a@x.com", None], "x": [1, 2]})
    schema_info = {"columns": [{"name": "email", "is_potential_pii": True}, {"name": "x", "is_potential_pii": False}]}
    op = PreviewRowsOperator()
    result = op.run(_ctx(df, schema_info=schema_info), {"limit": 10})
    out = result.tables["preview"]
    assert out.shape[0] == 2
    assert out.loc[0, "email"] == "[REDACTED]"
    assert pd.isna(out.loc[1, "email"])


def test_schema_snapshot_strips_pii_sample_values():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com"], "seg": ["A", "B"]})
    schema_info = {
        "columns": [
            {"name": "email", "is_potential_pii": True, "sample_values": ["a@x.com"]},
            {"name": "seg", "is_potential_pii": False, "sample_values": ["A"]},
        ]
    }
    op = SchemaSnapshotOperator()
    result = op.run(_ctx(df, schema_info=schema_info), {})
    out = result.tables["schema"]
    email = out[out["name"] == "email"].iloc[0].to_dict()
    seg = out[out["name"] == "seg"].iloc[0].to_dict()
    assert email.get("sample_values") == []
    assert seg.get("sample_values") == ["A"]


def test_categorical_topk_excludes_pii_columns():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com", "b@y.com"], "seg": ["A", "A", "B"]})
    schema_info = {"columns": [{"name": "email", "is_potential_pii": True}, {"name": "seg", "is_potential_pii": False}]}
    op = CategoricalTopKOperator()
    result = op.run(_ctx(df, schema_info=schema_info), {"k": 5, "max_columns": 10})
    out = result.tables["categorical_topk"]
    assert "email" not in set(out["column"].astype(str).tolist())
    assert "seg" in set(out["column"].astype(str).tolist())


def test_relationship_explain_numeric_numeric_corr_and_bounded_sample():
    df = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]})
    op = RelationshipExplainOperator()
    result = op.run(_ctx(df), {"column_a": "x", "column_b": "y", "max_points": 3})
    summary = result.tables["relationship_summary"]
    assert summary.shape[0] == 1
    row = summary.iloc[0].to_dict()
    assert row["relationship_type"] == "numeric_numeric"
    assert abs(float(row["metric_value"]) - 1.0) < 1e-9
    sample = result.tables["relationship_sample"]
    assert list(sample.columns) == ["x", "y"]
    assert int(sample.shape[0]) <= 3


def test_relationship_explain_categorical_numeric_eta2():
    df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "num": [1.0, 1.0, 10.0, 10.0]})
    op = RelationshipExplainOperator()
    result = op.run(_ctx(df), {"column_a": "cat", "column_b": "num", "min_group_size": 1})
    summary = result.tables["relationship_summary"]
    assert summary.shape[0] == 1
    row = summary.iloc[0].to_dict()
    assert row["relationship_type"] == "categorical_numeric"
    assert float(row["metric_value"]) > 0.9
    detail = result.tables["relationship_detail"]
    assert not detail.empty


def test_time_anomaly_scan_flags_spike_as_event():
    ts = pd.date_range("2024-01-01", periods=30, freq="D")
    y = np.arange(1, 31, dtype=float)
    y[10] = 1000.0
    df = pd.DataFrame({"ts": ts, "y": y})
    schema_info = {"columns": [{"name": "ts", "inferred_type": "datetime"}]}
    op = TimeAnomalyScanOperator()
    result = op.run(_ctx(df, schema_info=schema_info), {"time_column": "ts", "value_column": "y", "freq": "D", "max_points": 200})
    series = result.tables["time_series"]
    events = result.tables["time_events"]
    assert not series.empty
    assert not events.empty
    assert bool(events["is_anomaly"].any())
