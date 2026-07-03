import pandas as pd

from app.services.dataset_transformations import apply_transform_steps, build_preview_diff


def test_deduplicate_removes_rows_and_reports_metrics():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = apply_transform_steps(df, [{"op": "deduplicate", "params": {"subset": ["a", "b"], "keep": "first"}}])
    assert out.df.shape[0] == 2
    assert out.metrics["steps"][0]["rows_removed"] == 1


def test_fill_missing_constant_fills_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, "x"]})
    out = apply_transform_steps(df, [{"op": "fill_missing", "params": {"columns": ["a"], "strategy": "constant", "value": 0}}])
    assert out.df["a"].isna().sum() == 0


def test_type_convert_coerce_sets_invalid_to_null():
    df = pd.DataFrame({"x": ["1", "bad", "3"]})
    out = apply_transform_steps(df, [{"op": "type_convert", "params": {"column": "x", "to": "int", "errors": "coerce"}}])
    assert str(out.df["x"].dtype) == "Int64"
    assert int(out.df["x"].isna().sum()) == 1


def test_build_preview_diff_tracks_dtype_and_columns():
    before = pd.DataFrame({"x": ["1", "2"], "y": [1.0, 2.0]})
    after = before.copy()
    after["x"] = pd.to_numeric(after["x"], errors="raise").astype("Int64")
    after["z"] = 1
    after = after.drop(columns=["y"])

    diff = build_preview_diff(before=before, after=after, preview_rows=10)
    assert diff["input_columns"] == 2
    assert diff["output_columns"] == 2
    assert diff["added_columns"] == ["z"]
    assert diff["removed_columns"] == ["y"]
    assert any(c["column"] == "x" for c in diff["changed_dtypes"])


def test_filter_rows_and_sort_rows_chain():
    df = pd.DataFrame(
        {
            "city": ["A", "B", "A", "C"],
            "value": [5, 3, 9, 1],
        }
    )
    out = apply_transform_steps(
        df,
        [
            {"op": "filter_rows", "params": {"conditions": [{"column": "city", "op": "eq", "value": "A"}]}},
            {"op": "sort_rows", "params": {"columns": ["value"], "ascending": False}},
        ],
    )
    assert out.df["city"].tolist() == ["A", "A"]
    assert out.df["value"].tolist() == [9, 5]


def test_limit_rows_keeps_last_rows_when_requested():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = apply_transform_steps(df, [{"op": "limit_rows", "params": {"n": 2, "from_end": True}}])
    assert out.df["x"].tolist() == [4, 5]


def test_time_features_adds_expected_columns():
    df = pd.DataFrame({"ts": ["2026-01-01", "2026-01-04"]})
    out = apply_transform_steps(
        df,
        [{"op": "time_features", "params": {"column": "ts", "features": ["year", "day_of_week", "is_weekend"]}}],
    )
    assert "ts_year" in out.df.columns
    assert "ts_day_of_week" in out.df.columns
    assert "ts_is_weekend" in out.df.columns
    assert out.df["ts_year"].tolist() == [2026, 2026]


def test_bin_numeric_creates_codes_column():
    df = pd.DataFrame({"score": [10, 20, 30, 40, 50]})
    out = apply_transform_steps(
        df,
        [{"op": "bin_numeric", "params": {"column": "score", "bins": 3, "strategy": "uniform", "output": "codes"}}],
    )
    assert "score_bin" in out.df.columns
    assert str(out.df["score_bin"].dtype) == "Int64"


def test_clip_outliers_clips_values_with_iqr():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 1000]})
    out = apply_transform_steps(
        df,
        [{"op": "clip_outliers", "params": {"columns": ["x"], "method": "iqr", "action": "clip", "iqr_multiplier": 1.5}}],
    )
    assert out.df["x"].max() < 1000
    assert out.metrics["steps"][0]["per_column"][0]["outliers"] >= 1


def test_clip_outliers_drop_rows_with_any_match():
    df = pd.DataFrame({"x": [1, 2, 3, 1000], "y": [10, 11, 12, 13]})
    out = apply_transform_steps(
        df,
        [{"op": "clip_outliers", "params": {"columns": ["x"], "method": "quantile", "action": "drop", "lower_quantile": 0.0, "upper_quantile": 0.75}}],
    )
    assert out.df.shape[0] == 3
    assert 1000 not in out.df["x"].tolist()


def test_encode_categorical_label_adds_encoded_column():
    df = pd.DataFrame({"city": ["a", "b", "a", None]})
    out = apply_transform_steps(
        df,
        [{"op": "encode_categorical", "params": {"columns": ["city"], "strategy": "label", "drop_original": False}}],
    )
    assert "city_encoded" in out.df.columns
    assert str(out.df["city_encoded"].dtype) == "Int64"


def test_encode_categorical_one_hot_adds_dummy_columns():
    df = pd.DataFrame({"city": ["a", "b", "a"]})
    out = apply_transform_steps(
        df,
        [{"op": "encode_categorical", "params": {"columns": ["city"], "strategy": "one_hot", "drop_original": True}}],
    )
    assert "city" not in out.df.columns
    assert any(c.startswith("city__") for c in out.df.columns)


def test_transforms_handle_spaces_and_punctuation_in_column_names():
    df = pd.DataFrame(
        {
            "Total Sales ($)": [10.0, None, 30.0],
            "Customer Name": [" A ", "b", None],
            "order date": ["2026-01-01", "bad", "2026-01-03"],
            "segment-id": ["x", "y", "x"],
        }
    )

    out = apply_transform_steps(
        df,
        [
            {"op": "fill_missing", "params": {"columns": ["Total Sales ($)"], "strategy": "median"}},
            {"op": "string_normalize", "params": {"columns": ["Customer Name"], "trim": True, "lowercase": True}},
            {"op": "type_convert", "params": {"column": "order date", "to": "datetime", "errors": "coerce"}},
            {"op": "time_features", "params": {"column": "order date", "features": ["year"]}},
            {"op": "encode_categorical", "params": {"columns": ["segment-id"], "strategy": "label", "drop_original": False}},
        ],
    )

    assert out.df["Total Sales ($)"].tolist() == [10.0, 20.0, 30.0]
    assert out.df["Customer Name"].tolist()[:2] == ["a", "b"]
    assert pd.isna(out.df["Customer Name"].iloc[2])
    assert "order date_year" in out.df.columns
    assert "segment-id_encoded" in out.df.columns
