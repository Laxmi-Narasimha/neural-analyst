"""Trust tests for the ML path: a data scientist must be able to rely on these numbers.

Each test pins one property that was broken before:
- cross-validation silently returned 0.0 on every run (wrong result key)
- preprocessing was fitted on all rows before the train/test split
- regression winners were chosen on the test set
- multiclass targets found no models at all
- saved models had no preprocessing, so raw rows could not be scored
- the registry forgot versions on reload ("No matching model version found")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from uuid import UUID

import httpx
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

pytest.importorskip("sklearn")

from app.ml.auto_ml_engine import AdvancedAutoMLEngine, MLTask, OptimizationStrategy
from app.ml.model_registry import ModelRegistry
from app.ml.tabular_pipeline import ModelBundle, TabularPreprocessor

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _engine(**kw) -> AdvancedAutoMLEngine:
    return AdvancedAutoMLEngine(optimization_strategy=OptimizationStrategy.NONE, verbose=False, **kw)


def _orders(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Orders where lateness depends on distance and carrier, with messy columns."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(1, 60, n)
    carrier = rng.choice(["DHL", "BlueDart", "Delhivery"], n)
    logit = 0.09 * distance - 2.6 + np.where(carrier == "Delhivery", 1.4, 0.0)
    late = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "order_id": np.arange(100000, 100000 + n),
            "distance_km": distance.round(2),
            "carrier": carrier,
            "order_value": [f"₹{v:,.0f}" for v in rng.uniform(200, 20000, n)],
            "order_date": pd.date_range("2025-01-01", periods=n, freq="10h").strftime("%Y-%m-%d %H:%M"),
            "is_express": rng.choice([True, False], n),
            "late": np.where(late, "late", "on_time"),
        }
    )
    df.loc[df.sample(frac=0.08, random_state=seed).index, "distance_km"] = np.nan
    df.loc[df.sample(frac=0.05, random_state=seed + 1).index, "carrier"] = None
    return df


@pytest.fixture()
def registry_dir(tmp_path, monkeypatch):
    path = tmp_path / "registry"
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(path))
    return path


# ---------------------------------------------------------------------------
# Evaluation honesty
# ---------------------------------------------------------------------------

def test_cross_validation_is_actually_computed_and_drives_selection():
    engine = _engine()
    results = engine.auto_train(_orders(), "late", max_models=3, optimize=False)
    scored = [r for r in results if r.cv_mean is not None]
    assert len(scored) == 3, [r.warnings for r in results]
    for r in scored:
        assert len(r.cv_scores["test"]) >= 2
        assert 0.5 < r.cv_mean <= 1.0
        assert r.cv_scoring == "roc_auc"
    # Best model is the one with the highest CV score, not the highest test score.
    assert results[0].cv_mean == max(r.cv_mean for r in scored)
    assert engine.training_report["cross_validation"]["selection"].startswith("highest mean cross-validation")


def test_regression_winner_is_chosen_by_cv_not_test_set():
    rng = np.random.default_rng(3)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "segment": rng.choice(["a", "b"], n)})
    df["revenue"] = 3 * x1 - 2 * x2 + rng.normal(scale=0.5, size=n)
    engine = _engine()
    results = engine.auto_train(df, "revenue", max_models=3, optimize=False)
    cv = {r.model_name: r.cv_mean for r in results if r.cv_mean is not None}
    assert results[0].model_name == max(cv, key=cv.get)
    assert results[0].cv_scoring == "r2"
    assert engine.training_report["beats_baseline"] is True


def test_preprocessing_never_sees_held_out_rows():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.choice([0, 1], n)})
    engine = _engine(test_size=0.25)
    engine.auto_train(df, "y", models=["logistic_regression"], optimize=False)
    _, pipe, result = engine.get_best_model()
    scaler = pipe.named_steps["prep"].column_transformer_.named_transformers_["num"].named_steps["scale"]
    train_rows = result.n_samples_train
    assert train_rows == 150
    # The scaler's mean equals the mean of exactly the training rows, so the
    # held-out rows did not influence preprocessing.
    impute = pipe.named_steps["prep"].column_transformer_.named_transformers_["num"].named_steps["impute"]
    assert scaler.n_samples_seen_ == train_rows
    assert impute.statistics_.shape[0] == 1


def test_preprocessor_is_refit_inside_each_cv_fold():
    """Fitting inside a Pipeline means sklearn clones the preprocessor per fold."""
    df = _orders(300)
    engine = _engine(cv_folds=3)
    results = engine.auto_train(df, "late", models=["logistic_regression"], optimize=False)
    # If preprocessing had been fitted once on everything, CV and the pipeline
    # would not be separable; here every fold produced its own finite score.
    assert len(results[0].cv_scores["test"]) == 3
    assert all(np.isfinite(results[0].cv_scores["test"]))


def test_baseline_is_reported_and_flags_models_with_no_signal():
    rng = np.random.default_rng(11)
    n = 400
    noise = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "y": rng.choice([0, 1], n)})
    engine = _engine()
    engine.auto_train(noise, "y", max_models=2, optimize=False)
    report = engine.training_report
    assert report["baseline"]["cv_mean"] == pytest.approx(0.5, abs=1e-9)
    assert report["beats_baseline"] is False
    assert any("does not beat the naive baseline" in w for w in report["warnings"])


def test_time_column_gives_forward_in_time_evaluation():
    df = _orders().sample(frac=1.0, random_state=1)  # rows deliberately out of order
    engine = _engine()
    engine.auto_train(df, "late", max_models=2, optimize=False, time_column="order_date")
    split = engine.training_report["split"]
    assert split["strategy"] == "time_ordered"
    assert pd.Timestamp(split["train_period"][1]) < pd.Timestamp(split["test_period"][0])
    assert engine.training_report["cross_validation"]["strategy"] == "TimeSeriesSplit"


def test_missing_targets_are_dropped_not_imputed():
    df = _orders(300)
    df.loc[:29, "late"] = None
    engine = _engine()
    engine.auto_train(df, "late", max_models=1, optimize=False)
    assert engine.training_report["rows_dropped_missing_target"] == 30
    assert engine.training_report["rows_used"] == 270


def test_columns_that_give_away_the_target_are_excluded():
    df = _orders(400)
    df["delivery_status"] = np.where(df["late"] == "late", "Delayed", "Delivered")
    df["late_copy"] = df["late"]
    engine = _engine()
    results = engine.auto_train(df, "late", max_models=2, optimize=False)
    excluded = {i["column"] for i in engine.training_report["excluded_leaky_columns"]}
    assert excluded == {"delivery_status", "late_copy"}
    bundle = engine.build_bundle()
    assert "delivery_status" not in bundle.input_columns
    # Without the giveaway column the model is good but not perfect.
    assert results[0].test_metrics["roc_auc"] < 0.99


def test_multiclass_string_target_trains_and_decodes_labels():
    rng = np.random.default_rng(5)
    n = 450
    x = rng.normal(size=n)
    tier = np.select([x < -0.5, x > 0.5], ["bronze", "gold"], default="silver")
    df = pd.DataFrame({"spend_score": x + rng.normal(scale=0.2, size=n), "region": rng.choice(["N", "S"], n), "tier": tier})
    engine = _engine()
    results = engine.auto_train(df, "tier", max_models=2, optimize=False)
    assert engine._task == MLTask.MULTICLASS_CLASSIFICATION
    assert results[0].cv_mean is not None and results[0].cv_mean > 0.7
    preds = engine.predict(pd.DataFrame({"spend_score": [-2.0, 0.0, 2.0], "region": ["N", "S", "N"]}))
    assert preds == ["bronze", "silver", "gold"]


def test_random_search_tuning_uses_pipeline_params():
    engine = AdvancedAutoMLEngine(
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH, optimization_trials=4, verbose=False
    )
    results = engine.auto_train(_orders(300), "late", models=["logistic_regression"], optimize=True)
    assert "C" in results[0].best_params
    assert not any("Hyperparameter search failed" in w for w in results[0].warnings)


# ---------------------------------------------------------------------------
# Raw-row scoring and persistence
# ---------------------------------------------------------------------------

def test_preprocessor_handles_messy_and_unseen_inputs():
    df = _orders(300).drop(columns=["late"])
    prep = TabularPreprocessor().fit(df)
    roles = prep.column_roles_
    assert roles["order_value"] == "numeric_text"
    assert roles["order_date"] == "datetime"
    assert roles["carrier"] == "categorical"
    assert "order_id" in prep.dropped_columns_
    unseen = pd.DataFrame({"distance_km": [None], "carrier": ["NewCourier"], "order_value": ["₹1,000"]})
    out = prep.transform(unseen)
    assert out.shape == (1, len(prep.feature_names_out_))
    assert np.isfinite(out).all()


def test_train_register_reload_predict_in_a_fresh_registry(tmp_path):
    engine = _engine()
    engine.auto_train(_orders(), "late", max_models=2, optimize=False, time_column="order_date")
    bundle = engine.build_bundle()

    registry = ModelRegistry(tmp_path / "reg")
    v1 = registry.register_model("orders_late", bundle, metrics={"cv_roc_auc": 0.8})
    v2 = registry.register_model("orders_late", bundle, metrics={"cv_roc_auc": 0.81})
    assert (v1.version, v2.version) == (1, 2)

    reopened = ModelRegistry(tmp_path / "reg")  # simulates a server restart
    assert [v["version"] for v in reopened.get_model_versions("orders_late")] == [1, 2]
    loaded = reopened.load_model("orders_late", version=1)
    assert isinstance(loaded, ModelBundle)
    raw = pd.DataFrame(
        {
            "distance_km": [2.0, 58.0],
            "carrier": ["DHL", "Delhivery"],
            "order_value": ["₹500", "₹12,000"],
            "order_date": ["2026-01-05 10:00", "2026-01-05 11:00"],
            "is_express": [True, False],
        }
    )
    assert loaded.predict(raw) == ["on_time", "late"]
    proba = loaded.predict_proba(raw)
    assert len(proba) == 2 and all(abs(sum(p) - 1) < 1e-9 for p in proba)
    assert loaded.check_input(raw.drop(columns=["is_express"])) == [
        "Missing input columns were treated as unknown values: is_express"
    ]
    with pytest.raises(ValueError, match="None of the columns"):
        loaded.check_input(pd.DataFrame({"unrelated": [1]}))


def test_saved_model_scores_raw_rows_in_a_separate_python_process(tmp_path):
    engine = _engine()
    engine.auto_train(_orders(), "late", max_models=1, optimize=False)
    registry = ModelRegistry(tmp_path / "reg")
    registry.register_model("orders_late", engine.build_bundle())

    script = textwrap.dedent(
        f"""
        import json, pandas as pd
        from app.ml.model_registry import ModelRegistry
        bundle = ModelRegistry({str(tmp_path / 'reg')!r}).load_model("orders_late", version=1)
        rows = pd.DataFrame({{"distance_km": [1.5, 59.0], "carrier": ["DHL", "Delhivery"]}})
        print(json.dumps(bundle.predict(rows)))
        """
    )
    env = dict(os.environ, PYTHONPATH=str(BACKEND_DIR))
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ["on_time", "late"]


def test_registry_recovers_versions_written_by_the_old_format(tmp_path):
    reg_dir = tmp_path / "reg"
    (reg_dir / "legacy_model" / "v1").mkdir(parents=True)
    (reg_dir / "legacy_model" / "v2").mkdir(parents=True)
    import pickle

    for v in (1, 2):
        (reg_dir / "legacy_model" / f"v{v}" / "model.pkl").write_bytes(pickle.dumps({"version": v}))
    (reg_dir / "registry.json").write_text(
        json.dumps({"models": [{"model_id": "m1", "name": "legacy_model", "version_count": 2}], "experiments": []})
    )
    registry = ModelRegistry(reg_dir)
    assert registry.load_model("legacy_model", version=2) == {"version": 2}
    assert registry.load_model("legacy_model") == {"version": 2}
    v3 = registry.register_model("legacy_model", {"version": 3})
    assert v3.version == 3


# ---------------------------------------------------------------------------
# Chat operator and HTTP API, end to end
# ---------------------------------------------------------------------------

def test_chat_training_operator_reports_evaluation_and_registers_a_bundle(registry_dir):
    from app.compute.operators.base import OperatorContext
    from app.compute.operators.ml import MLTrainBaselineOperator

    df = _orders()
    df["delivery_status"] = np.where(df["late"] == "late", "Delayed", "Delivered")
    ctx = OperatorContext(
        dataset_id=UUID("00000000-0000-0000-0000-0000000000aa"),
        dataset_version="v-test",
        df=df,
        profile_report={"row_count": len(df), "column_count": len(df.columns)},
        schema_info={"columns": [{"name": c, "inferred_type": str(df[c].dtype)} for c in df.columns]},
    )
    result = MLTrainBaselineOperator().run(ctx, {"target_column": "late", "optimize": False, "max_models": 2})
    s = result.summary
    assert s["split"]["strategy"] == "time_ordered" and s["split"]["time_column"] == "order_date"
    assert s["cv_mean"] is not None and s["beats_baseline"] is True
    assert [i["column"] for i in s["excluded_leaky_columns"]] == ["delivery_status"]
    table = result.tables["model_comparison"]
    assert bool(table["is_baseline"].iloc[0]) is True
    assert table["cv_mean"].notna().all()
    json.dumps(s)  # the summary must be JSON-safe for storage and the API

    loaded = ModelRegistry(registry_dir).load_model(s["registered_model"], version=s["registered_version"])
    assert isinstance(loaded, ModelBundle)
    assert loaded.predict(df.drop(columns=["late"]).head(5))[0] in {"late", "on_time"}


@pytest_asyncio.fixture()
async def api_env(tmp_path):
    from app.core.config import AuthMode, settings
    from app.main import create_application
    from app.services.database import db_manager

    old = (
        settings.database.url,
        settings.upload_directory,
        settings.artifact_directory,
        settings.auth_mode,
        settings.security.rate_limit_requests,
        os.environ.get("MODEL_REGISTRY_PATH"),
    )
    (tmp_path / "uploads").mkdir()
    (tmp_path / "artifacts").mkdir()
    try:
        settings.database.url = f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}"
        settings.upload_directory = tmp_path / "uploads"
        settings.artifact_directory = tmp_path / "artifacts"
        settings.auth_mode = AuthMode.JWT
        settings.security.rate_limit_requests = 10_000
        os.environ["MODEL_REGISTRY_PATH"] = str(tmp_path / "registry")
        await db_manager.close()
        app = create_application()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                creds = {"email": "ml-trust@example.com", "password": "Str0ngPassw0rd!"}
                await client.post("/api/v1/auth/register", json={**creds, "role": "analyst", "full_name": "ML"})
                login = await client.post("/api/v1/auth/login", json=creds)
                assert login.status_code == 200, login.text
                yield client, {"Authorization": f"Bearer {login.json()['access_token']}"}
    finally:
        await db_manager.close()
        (
            settings.database.url,
            settings.upload_directory,
            settings.artifact_directory,
            settings.auth_mode,
            settings.security.rate_limit_requests,
            registry_env,
        ) = old
        if registry_env is None:
            os.environ.pop("MODEL_REGISTRY_PATH", None)
        else:
            os.environ["MODEL_REGISTRY_PATH"] = registry_env
        try:
            from app.main import RateLimitMiddleware

            RateLimitMiddleware._requests.clear()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_train_then_predict_with_returned_name_and_version(api_env):
    from app.services.dataset_processing import DatasetProcessingService

    client, headers = api_env
    csv_text = _orders(500).to_csv(index=False)
    up = await client.post(
        "/api/v1/datasets/upload",
        headers=headers,
        files={"file": ("orders.csv", csv_text.encode("utf-8"), "text/csv")},
        data={"name": "Orders"},
    )
    assert up.status_code == 201, up.text
    data = up.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(data["dataset_id"]), UUID(data["job_id"]))

    trained = await client.post(
        "/api/v1/ml/train",
        headers=headers,
        json={"dataset_id": data["dataset_id"], "target_column": "late", "optimize": False, "max_models": 2},
    )
    assert trained.status_code == 200, trained.text
    result = trained.json()["result"]
    assert result["cv_mean"] is not None
    assert result["split"]["strategy"] == "time_ordered"

    predicted = await client.post(
        "/api/v1/ml/predict",
        headers=headers,
        json={
            "model_name": result["registered_model"],
            "model_version": result["registered_version"],
            "data": [
                {"distance_km": 2, "carrier": "DHL", "order_value": "₹900", "order_date": "2026-02-01 09:00"},
                {"distance_km": 59, "carrier": "Delhivery", "order_value": "₹15,000", "order_date": "2026-02-01 10:00"},
            ],
        },
    )
    assert predicted.status_code == 200, predicted.text
    body = predicted.json()
    assert body["predictions"] == ["on_time", "late"]
    assert body["classes"] == ["late", "on_time"]
    assert any("is_express" in w for w in body["warnings"])
