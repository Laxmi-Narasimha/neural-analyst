"""Every hostile dataset must either train and score raw rows, or be refused with a clear reason.

A crash (TypeError, KeyError, ...) or a silent misreading fails the test.
Large cases (200k rows, 300 columns) run only with RUN_ML_TESTS=1.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from app.core.exceptions import MLException
from app.ml.model_registry import get_model_registry
from app.ml.training_run import run_training
from tests import ml_robustness_cases as cases

HEAVY = {"200k_rows", "wide_300_cols"}

# Properties that a correct reading of each dataset must have.
EXPECTED = {
    "currency_text_regression_target": {"task": "regression"},
    "messy_yes_no_target": {"task": "binary_classification"},
    "regression_negative_and_zero_target": {"task": "regression"},
    "only_date_features": {"split": "time_ordered"},
    "tz_aware_and_mixed_format_dates": {"split": "time_ordered"},
    "leaky_future_date_column": {"excluded": "delivered_date"},
    "free_text_reviews": {"excluded": "stars"},
    "huge_regression_values": {"excluded": "x"},
    "multiclass_with_singleton_class": {"task": "multiclass_classification"},
}


@pytest.mark.parametrize("name", sorted(cases.cases))
def test_hostile_dataset(name, tmp_path, monkeypatch):
    if name in HEAVY and os.getenv("RUN_ML_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("large case; set RUN_ML_TESTS=1")
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    cases.reseed(0)
    build, expect = cases.cases[name]
    df, target = build()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if expect == "refuse":
            with pytest.raises((MLException, ValueError)):
                run_training(df, target_column=target, registered_name=f"t_{name}", max_models=2)
            return
        run = run_training(df, target_column=target, registered_name=f"t_{name}", max_models=2)
        summary = run.summary()
        bundle = get_model_registry().load_model(run.registered_name, version=run.registered_version)
        raw = df.drop(columns=[target]).head(5)
        preds = bundle.predict(raw)

    assert len(preds) == len(raw)
    assert summary["cv_mean"] is not None
    expected = EXPECTED.get(name, {})
    if "task" in expected:
        assert summary["task"] == expected["task"]
    if "split" in expected:
        assert summary["split"]["strategy"] == expected["split"]
    if "excluded" in expected:
        assert expected["excluded"] in {i["column"] for i in summary["excluded_leaky_columns"]}
    if summary["task"] != "regression":
        assert set(map(str, preds)) <= set(map(str, summary["classes"]))
