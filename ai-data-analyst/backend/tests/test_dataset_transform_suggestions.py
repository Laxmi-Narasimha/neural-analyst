from __future__ import annotations

from types import SimpleNamespace

from app.services.dataset_transform_suggestions import DatasetTransformSuggestionService


def _col(
    name: str,
    inferred_type: str,
    *,
    null_count: int = 0,
    null_percentage: float = 0.0,
    unique_count: int = 0,
    is_sensitive: bool = False,
    statistics: dict | None = None,
):
    return SimpleNamespace(
        name=name,
        inferred_type=inferred_type,
        null_count=null_count,
        null_percentage=null_percentage,
        unique_count=unique_count,
        is_sensitive=is_sensitive,
        statistics=statistics or {},
    )


def test_transform_suggestion_service_builds_basic_quality_plan():
    dataset = SimpleNamespace(
        row_count=1000,
        quality_report={"warnings": ["Potential duplicate rows detected"]},
        columns=[
            _col("amount", "float", null_count=120, null_percentage=0.12, unique_count=500, statistics={"has_outliers": True}),
            _col("category", "string", null_count=80, null_percentage=0.08, unique_count=8),
            _col(
                "comment",
                "string",
                null_count=0,
                null_percentage=0.0,
                unique_count=500,
                statistics={"sample_values": ["  hello  ", "normal", "many   spaces"]},
            ),
            _col("order_date", "string", null_count=0, null_percentage=0.0, unique_count=900),
            _col(
                "const_col",
                "string",
                null_count=0,
                null_percentage=0.0,
                unique_count=1,
                statistics={"is_constant": True},
            ),
            _col("customer_id", "integer", null_count=0, null_percentage=0.0, unique_count=999),
        ],
    )

    out = DatasetTransformSuggestionService().suggest(dataset, max_steps=12)
    ops = [str(item["step"]["op"]) for item in out.suggestions]

    assert "drop_columns" in ops
    assert "fill_missing" in ops
    assert "string_normalize" in ops
    assert "type_convert" in ops
    assert "deduplicate" in ops
    assert "clip_outliers" in ops
    assert "encode_categorical" in ops
    assert out.summary["suggestion_count"] == len(out.suggestions)


def test_transform_suggestion_service_respects_max_steps():
    dataset = SimpleNamespace(
        row_count=500,
        quality_report={"warnings": ["Potential duplicate rows detected"]},
        columns=[
            _col("x1", "float", null_count=20, null_percentage=0.04, unique_count=300),
            _col("x2", "float", null_count=20, null_percentage=0.04, unique_count=300),
            _col("x3", "float", null_count=20, null_percentage=0.04, unique_count=300),
            _col("event_time", "string", unique_count=480),
            _col("text_col", "string", statistics={"sample_values": ["  a", "b  "]}),
            _col("id", "integer", unique_count=500),
        ],
    )

    out = DatasetTransformSuggestionService().suggest(dataset, max_steps=2)
    assert len(out.suggestions) == 2
    assert out.warnings
