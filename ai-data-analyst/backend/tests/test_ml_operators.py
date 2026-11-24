import pandas as pd
import pytest
from uuid import UUID

from app.compute.operators.base import OperatorContext
from app.compute.operators.ml import MLTaskInferenceOperator, MLTrainBaselineOperator


def _ctx(df: pd.DataFrame) -> OperatorContext:
    return OperatorContext(
        dataset_id=UUID("00000000-0000-0000-0000-000000000001"),
        dataset_version="test-hash",
        df=df,
        profile_report={"row_count": len(df), "column_count": len(df.columns)},
        schema_info={"columns": [{"name": c, "inferred_type": str(df[c].dtype)} for c in df.columns]},
        owner_id=UUID("00000000-0000-0000-0000-00000000beef"),
    )


def test_ml_task_inference_operator():
    df = pd.DataFrame(
        {
            "customer_id": range(100),
            "churn": [0, 1] * 50,
            "revenue": [float(i) for i in range(100)],
            "segment": ["A", "B"] * 50,
        }
    )
    op = MLTaskInferenceOperator()
    result = op.run(_ctx(df), {})
    assert "target_candidates" in result.tables
    assert result.tables["target_candidates"].shape[0] >= 1
    assert result.summary.get("selected_target") is not None


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None,
    reason="sklearn required",
)
def test_ml_train_baseline_operator():
    df = pd.DataFrame(
        {
            "f1": [float(i % 7) for i in range(120)],
            "f2": [float(i % 5) for i in range(120)],
            "target": [i % 2 for i in range(120)],
        }
    )
    op = MLTrainBaselineOperator()
    result = op.run(_ctx(df), {"target_column": "target", "optimize": False, "max_models": 2})
    assert result.summary.get("best_model")
    assert result.summary.get("registered_model")
    assert "model_comparison" in result.tables