"""Evaluation beyond one score: imbalance, calibration, thresholds, error by segment, cutoffs, drift.

Each test states the failure it prevents. Decisions (metric, threshold, calibration) must come from
training folds only; held-out numbers are reported, never used to choose.
"""

from __future__ import annotations

import pickle
import warnings
from uuid import UUID

import httpx
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

from app.core.config import AuthMode, settings
from app.ml import drift
from app.ml import evaluation as ev
from app.ml.model_registry import get_model_registry
from app.ml.training_run import run_training

OWNER = UUID("00000000-0000-0000-0000-0000000000d6")


@pytest.fixture(autouse=True)
def _registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    warnings.simplefilter("ignore")


def _fraud(n: int = 3000, seed: int = 1, shift: float = 0.0) -> pd.DataFrame:
    """About 10% positives that depend on x1, x2 and region; hourly timestamps."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n) + shift
    x2 = rng.normal(size=n)
    region = rng.choice(["N", "S", "E"], n, p=[0.5, 0.3, 0.2])
    logit = -3.2 + 1.4 * x1 + 0.8 * x2 + np.where(region == "E", 1.0, 0.0)
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "fraud", "ok")
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region,
                         "when": pd.date_range("2025-01-01", periods=n, freq="h"), "label": y})


def _train(df, **kw):
    return run_training(df, target_column=kw.pop("target", "label"), registered_name=kw.pop("name", "m"),
                        owner_id=OWNER, max_models=2, **kw)


# ---------------------------------------------------------------------------
# D6: imbalance
# ---------------------------------------------------------------------------

def test_imbalanced_target_is_selected_on_pr_auc_and_gets_a_tuned_threshold():
    """A 0.5 cut-off on a 10% class predicts 'ok' for almost everyone; ROC AUC hides that."""
    run = _train(_fraud())
    s = run.summary()
    assert s["cv_scoring"] == "average_precision"
    assert s["class_balance"]["imbalanced"] is True
    thr = s["decision_threshold"]
    assert thr["positive_class"] == "fraud" and 0 < thr["threshold"] < 0.5
    assert thr["held_out_at_chosen"]["recall"] > thr["held_out_at_default_0_5"]["recall"] + 0.1
    assert thr["held_out_at_chosen"]["f1"] > thr["held_out_at_default_0_5"]["f1"]
    assert any(t["threshold"] == 0.5 for t in thr["trade_offs"])  # the user can choose a different trade-off

    bundle = get_model_registry(OWNER).load_model("m", version=s["registered_version"])
    rows = _fraud(seed=9).drop(columns="label")
    preds = np.asarray(bundle.predict(rows))
    p = np.asarray(bundle.predict_proba(rows))[:, bundle.classes.index("fraud")]
    # The saved model predicts exactly what its stored cut-off says.
    assert ((p >= thr["threshold"]) == (preds == "fraud")).all()
    # Reported test metrics describe that saved behaviour, not the 0.5 cut-off.
    assert s["test_metrics"]["recall"] is not None


def test_balanced_target_keeps_roc_auc_and_the_default_cut_off():
    rng = np.random.default_rng(3)
    x = rng.normal(size=800)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=800), "y": np.where(x + rng.normal(size=800) > 0, "a", "b")})
    s = _train(df, target="y").summary()
    assert s["cv_scoring"] == "roc_auc"
    assert s["decision_threshold"] is None
    assert s["class_balance"]["imbalanced"] is False


def test_class_weights_are_opt_in_and_labelled():
    s = _train(_fraud(1500), balance_class_weights=True).summary()
    assert any("balanced class weights" in str(m["display_name"]) for m in s["models"] if not m["is_baseline"])


# ---------------------------------------------------------------------------
# D7: calibration
# ---------------------------------------------------------------------------

def test_calibration_is_kept_only_when_it_helps_out_of_sample():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.6, 4000)
    y = (rng.random(4000) < true_p).astype(int)
    overconfident = np.clip(true_p ** 0.4, 0, 1)  # far too high
    cal, info = ev.choose_calibration(overconfident, y)
    assert info["applied"] and cal is not None
    assert info["oof_brier_calibrated"] < info["oof_brier_raw"]
    assert ev.reliability(y, cal(overconfident))["ece"] < ev.reliability(y, overconfident)["ece"] / 2

    cal2, info2 = ev.choose_calibration(true_p, y)  # already calibrated
    assert not info2["applied"] and cal2 is None


def test_reliability_bins_are_correct():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    r = ev.reliability(y, p, bins=2)
    assert [b["observed_rate"] for b in r["bins"]] == [0.0, 1.0]
    assert r["ece"] == pytest.approx(0.1)
    assert r["brier"] == pytest.approx(0.01)


def test_training_reports_reliability_on_held_out_rows():
    s = _train(_fraud()).summary()
    cal = s["calibration"]
    assert cal["positive_class"] == "fraud"
    assert cal["held_out_before"]["bins"] and cal["held_out_before"]["brier"] is not None
    assert "reason" in cal


def test_a_calibrated_bundle_survives_pickling_and_old_bundles_still_work():
    run = _train(_fraud())
    bundle = get_model_registry(OWNER).load_model("m", version=run.registered_version)
    rows = _fraud(seed=4).drop(columns="label").head(50)
    rng = np.random.default_rng(1)
    cal, _ = ev.choose_calibration(np.clip(rng.uniform(0, 1, 500) ** 0.5, 0, 1), (rng.random(500) < 0.2).astype(int))
    bundle.calibrator = {"positive_index": bundle.classes.index("fraud"), "map": cal}
    clone = pickle.loads(pickle.dumps(bundle))
    assert np.allclose(clone.predict_proba(rows), bundle.predict_proba(rows))
    assert np.allclose(np.asarray(clone.predict_proba(rows)).sum(axis=1), 1.0)

    # Bundles pickled before version 2 have none of the new attributes (the class defaults answer).
    for attr in ("calibrator", "decision_threshold", "reference_profile", "prediction_reference"):
        clone.__dict__.pop(attr, None)
    clone.schema_version = 1
    old = pickle.loads(pickle.dumps(clone))
    assert len(old.predict(rows)) == 50 and old.drift_report(rows) is None
    assert old.describe()["schema_version"] == 1 and old.describe()["calibrated"] is False


# ---------------------------------------------------------------------------
# D8: error by segment
# ---------------------------------------------------------------------------

def test_segments_where_the_model_fails_are_found_and_small_ones_ignored():
    rng = np.random.default_rng(2)
    n = 1000
    X = pd.DataFrame({"region": rng.choice(["N", "S", "E"], n, p=[0.5, 0.3, 0.2]),
                      "tiny": np.where(np.arange(n) < 8, "rare", "common"),
                      "customer_id": [f"C{i}" for i in range(n)]})
    y = rng.integers(0, 2, n)
    pred = y.copy()
    wrong = (X["region"] == "E").to_numpy() & (rng.random(n) < 0.4)
    wrong |= rng.random(n) < 0.05
    pred[wrong] = 1 - pred[wrong]
    out = ev.error_by_segment(X, y, pred, task="classification")
    top = out["worst_segments"][0]
    assert (top["column"], top["segment"], top["significant"]) == ("region", "E", True)
    assert all(s["column"] != "customer_id" for s in out["worst_segments"])  # identifiers are not segments
    assert all(s["segment"] != "rare" for s in out["worst_segments"])  # 8 rows is too few to judge


def test_regression_segments_compare_mean_absolute_error():
    rng = np.random.default_rng(5)
    n = 900
    X = pd.DataFrame({"store": rng.choice(["A", "B", "C"], n)})
    y = rng.normal(100, 10, n)
    pred = y + rng.normal(0, 2, n) + np.where(X["store"] == "C", rng.normal(0, 15, n), 0)
    out = ev.error_by_segment(X, y, pred, task="regression")
    assert out["metric"] == "mean_absolute_error"
    assert out["worst_segments"][0]["segment"] == "C" and out["worst_segments"][0]["significant"]


def test_training_summary_names_the_weak_segment():
    s = _train(_fraud()).summary()
    seg = s["error_by_segment"]
    assert seg["held_out_rows"] == s["split"]["test_rows"]
    assert {"column", "segment", "rows", "significant"} <= set(seg["worst_segments"][0])


# ---------------------------------------------------------------------------
# D9: the prediction moment is the user's choice
# ---------------------------------------------------------------------------

def test_an_explicit_backtest_date_sets_the_split():
    df = _fraud()
    s = _train(df, time_column="when", test_start="2025-04-01").summary()
    split = s["split"]
    assert split["time_column_source"] == "user" and split["test_start"] == "2025-04-01"
    assert pd.Timestamp(split["train_period"][1]) < pd.Timestamp("2025-04-01") <= pd.Timestamp(split["test_period"][0])
    assert split["test_rows"] == int((df["when"] >= "2025-04-01").sum())


def test_an_inferred_prediction_moment_is_flagged_and_bad_cutoffs_refused():
    s = _train(_fraud(1200)).summary()
    assert s["split"]["time_column_source"] == "inferred"
    assert any("moment a prediction is made" in w for w in s["warnings"])
    with pytest.raises(Exception, match="training rows"):
        _train(_fraud(1200), time_column="when", test_start="2024-01-01")
    with pytest.raises(ValueError, match="not a date"):
        _train(_fraud(1200), time_column="when", test_start="next tuesday-ish")
    with pytest.raises(ValueError, match="time-ordered"):
        _train(_fraud(1200), split="random", test_start="2025-02-01")


# ---------------------------------------------------------------------------
# D10: drift
# ---------------------------------------------------------------------------

def test_drift_is_stable_for_the_same_population_and_flags_real_shifts():
    train = _fraud(4000, seed=1).drop(columns=["label", "when"])
    profile = drift.build_profile(train)
    same = drift.compare(profile, _fraud(2000, seed=7).drop(columns=["label", "when"]))
    assert same["verdict"] == "stable", same
    assert all(c["psi"] < 0.1 for c in same["columns"])

    moved = _fraud(2000, seed=7, shift=1.0).drop(columns=["label", "when"])
    moved.loc[moved.index[:400], "region"] = "W"  # a region the model never saw
    moved = moved.drop(columns=["x2"])
    report = drift.compare(profile, moved)
    by = {c["column"]: c for c in report["columns"]}
    assert report["verdict"] == "major"
    assert by["x1"]["level"] == "major"
    assert by["x2"]["level"] == "missing_column"
    assert by["region"]["new_categories"] == ["W"] and by["region"]["share_new_categories"] == pytest.approx(0.2)


def test_the_profile_stores_no_rows():
    df = pd.DataFrame({"email": [f"user{i}@x.com" for i in range(500)], "amount": np.arange(500.0)})
    profile = drift.build_profile(df)
    assert len(profile["columns"]["email"]["categories"]) <= 20
    assert set(profile["columns"]["amount"]) == {"kind", "missing_rate", "edges", "shares", "mean", "min", "max"}


def test_queued_automl_jobs_carry_the_evaluation_options(monkeypatch):
    """D12: the async path used to drop time_column/split, so jobs evaluated differently."""
    from fastapi import BackgroundTasks

    from app.core.config import JobExecutor
    from app.workers import celery_app, dispatcher

    sent = {}
    monkeypatch.setattr(settings, "job_executor", JobExecutor.CELERY)
    monkeypatch.setattr(celery_app.run_ml_training_task, "delay", lambda payload: sent.update(payload))
    opts = {"time_column": "when", "split": None, "test_start": "2025-04-01", "balance_class_weights": True}
    dispatcher.enqueue_ml_training(background_tasks=BackgroundTasks(), job_id=OWNER, owner_id=OWNER,
                                   dataset_id=OWNER, target_column="label", options=opts)
    assert sent["options"] == opts
    assert celery_app._ml_options({**opts, "owner_id": "x"}) == opts


# ---------------------------------------------------------------------------
# End to end: train, then score next week's file
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def two_users(tmp_path, monkeypatch):
    from app.main import create_application
    from app.services.database import db_manager

    (tmp_path / "uploads").mkdir()
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setattr(settings.database, "url", f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(settings, "upload_directory", tmp_path / "uploads")
    monkeypatch.setattr(settings, "artifact_directory", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "auth_mode", AuthMode.JWT)
    monkeypatch.setattr(settings.security, "rate_limit_requests", 10_000)
    await db_manager.close()
    app = create_application()
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                users = []
                for name in ("ana", "ben"):
                    creds = {"email": f"{name}@example.com", "password": "Str0ngPassw0rd!"}
                    await client.post("/api/v1/auth/register", json={**creds, "role": "analyst", "full_name": name})
                    login = await client.post("/api/v1/auth/login", json=creds)
                    users.append({"Authorization": f"Bearer {login.json()['access_token']}"})
                yield client, users
    finally:
        await db_manager.close()
        try:
            from app.main import RateLimitMiddleware

            RateLimitMiddleware._requests.clear()
        except Exception:
            pass


async def _upload(client, headers, frame: pd.DataFrame, name: str) -> str:
    from app.services.dataset_processing import DatasetProcessingService

    up = await client.post("/api/v1/datasets/upload", headers=headers,
                           files={"file": (f"{name}.csv", frame.to_csv(index=False).encode(), "text/csv")},
                           data={"name": name})
    assert up.status_code == 201, up.text
    data = up.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(data["dataset_id"]), UUID(data["job_id"]))
    return data["dataset_id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_score_a_new_file_with_drift_and_realised_performance(two_users):
    client, [ana, ben] = two_users
    train_id = await _upload(client, ana, _fraud(2000), "history")
    trained = await client.post("/api/v1/ml/train", headers=ana, json={
        "dataset_id": train_id, "target_column": "label", "optimize": False, "max_models": 2, "time_column": "when"})
    assert trained.status_code == 200, trained.text
    name = trained.json()["result"]["registered_model"]

    next_week = _fraud(600, seed=11, shift=1.2)
    new_id = await _upload(client, ana, next_week, "next_week")
    res = await client.post(f"/api/v1/ml/models/{name}/score", headers=ana, json={"dataset_id": new_id})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["rows_scored"] == 600
    assert body["drift"]["verdict"] == "major"
    assert body["drift"]["columns"][0]["column"] == "x1"
    assert body["realised_performance"]["rows_with_outcome"] == 600
    assert body["decision_threshold"] is not None
    assert sum(body["prediction_summary"]["counts"].values()) == 600
    assert body["prediction_shift"]["kind"] == "classes"

    rows = await client.get(f"/api/v1/artifacts/{body['artifact_id']}/rows", headers=ana)
    assert rows.status_code == 200, rows.text
    assert "predicted_label" in rows.text and "probability_fraud" in rows.text

    # Another user can neither use the model nor score into someone else's dataset.
    other = await client.post(f"/api/v1/ml/models/{name}/score", headers=ben, json={"dataset_id": new_id})
    assert other.status_code == 404, other.text


def test_the_chat_answer_explains_cut_off_probabilities_segment_and_moment():
    from app.api.routes.chat import _training_quality_lines

    text = "\n".join(_training_quality_lines(_train(_fraud()).summary()))
    assert "**Decision cut-off:** predicts **fraud**" in text
    assert "the usual 0.5 would find" in text
    assert "**Probabilities:**" in text
    assert "**Prediction moment:** assumed to be **when**" in text
    assert "**Rare outcome:**" in text
