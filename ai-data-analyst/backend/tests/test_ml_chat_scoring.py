"""Scoring a file with a saved model from chat ("score this file with my fraud model").

Each test states the failure it prevents: training instead of scoring, a model picked at random,
another user's model, a sampled file, predictions made from mostly missing inputs, and optimistic
accuracy on the model's own training rows.
"""

from __future__ import annotations

import warnings
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest

from app.compute.operators.base import OperatorContext
from app.compute.operators.ml import MLScoreOperator
from app.ml.scoring import score_frame
from app.ml.training_run import run_training
from app.services.ml_model_choice import choose_model
from tests.test_ml_evaluation_depth import _fraud, _upload, two_users  # noqa: F401  (fixture)

OWNER = UUID("00000000-0000-0000-0000-0000000000a1")


@pytest.fixture(autouse=True)
def _registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    warnings.simplefilter("ignore")


def _train(df, *, name, target="label", tags=None, owner=OWNER):
    return run_training(df, target_column=target, registered_name=name, owner_id=owner, max_models=1, tags=tags)


def _prices(n=400, seed=3):
    rng = np.random.default_rng(seed)
    size = rng.uniform(40, 200, n)
    return pd.DataFrame({"size": size, "rooms": rng.integers(1, 6, n), "price": 1000 * size + rng.normal(0, 5000, n)})


# ---------------------------------------------------------------------------
# The shared scoring core
# ---------------------------------------------------------------------------

def test_a_file_missing_most_model_inputs_is_refused_not_guessed():
    run = _train(_fraud(800), name="fraud")
    from app.ml.model_registry import get_model_registry

    bundle = get_model_registry(OWNER).load_model("fraud")
    new = _fraud(200, seed=5)
    with pytest.raises(ValueError, match="Predictions would be mostly guesses"):
        score_frame(bundle, new[["x1", "label"]])
    scored = score_frame(bundle, new.drop(columns=["region"]))
    assert len(scored.table) == 200
    assert any("Missing input columns" in w for w in scored.warnings)
    assert run.registered_name == "fraud"


# ---------------------------------------------------------------------------
# Which model: bound by code against the owner's registry and the file's columns
# ---------------------------------------------------------------------------

def test_model_choice_is_never_a_guess():
    cols = list(_fraud(10).columns)
    none = choose_model(OWNER, columns=cols, message="score this file with my model")
    assert none.choice is None and "no saved models" in none.reason

    _train(_fraud(800), name="fraud_v1")
    only = choose_model(OWNER, columns=cols, message="score this file with my model")
    assert only.choice and only.choice.name == "fraud_v1" and only.choice.target == "label"

    # A model this file cannot feed is not offered.
    _train(_prices(), name="house_prices", target="price")
    still = choose_model(OWNER, columns=cols, message="score this file with my model")
    assert still.choice and still.choice.name == "fraud_v1"
    refused = choose_model(OWNER, columns=cols, requested="house_prices")
    assert refused.choice is None and "cannot score this file" in refused.reason

    # Two fitting models: the target word decides, otherwise the user is asked.
    _train(_fraud(800, seed=2).rename(columns={"label": "flag"}), name="flagger", target="flag")
    by_word = choose_model(OWNER, columns=cols + ["flag"], message="score these rows with my flag model")
    assert by_word.choice and by_word.choice.name == "flagger"
    ask = choose_model(OWNER, columns=cols + ["flag"], message="score this file with my model")
    assert ask.choice is None and {m.name for m in ask.options} == {"fraud_v1", "flagger"}
    assert all("predicts" in m.label for m in ask.options)

    # Another owner sees none of these.
    assert choose_model(uuid4(), columns=cols).choice is None


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------

def _ctx(df, *, dataset_id=None, version="v-new"):
    return OperatorContext(dataset_id=dataset_id or uuid4(), dataset_version=version, df=df,
                           profile_report={}, schema_info={}, owner_id=OWNER)


def test_the_operator_scores_every_row_and_flags_drift():
    _train(_fraud(1500), name="fraud")
    new = _fraud(700, seed=9, shift=1.5)
    res = MLScoreOperator().run(_ctx(new), {"model_name": "fraud"})
    assert len(res.tables["predictions"]) == 700
    assert {"predicted_label", "probability_fraud", "probability_ok"} <= set(res.tables["predictions"].columns)
    assert res.summary["drift"]["verdict"] == "major"
    assert res.tables["drift"].iloc[0]["column"] == "x1"
    assert "drift" in res.charts
    assert res.summary["realised_performance"]["rows_with_outcome"] == 700
    assert res.metrics["model_version"] == 1
    assert not any("optimistic" in w for w in res.summary["warnings"])


def test_scoring_the_training_data_warns_that_accuracy_is_optimistic():
    ds = uuid4()
    df = _fraud(1000)
    _train(df, name="fraud", tags={"dataset_id": str(ds), "dataset_version": "v1"})
    res = MLScoreOperator().run(_ctx(df, dataset_id=ds, version="v1"), {"model_name": "fraud"})
    assert any("same version of the dataset" in w and "optimistic" in w for w in res.summary["warnings"])


def test_another_owners_model_cannot_be_named():
    _train(_fraud(800), name="fraud")
    ctx = OperatorContext(dataset_id=uuid4(), dataset_version="x", df=_fraud(50), profile_report={},
                          schema_info={}, owner_id=uuid4())
    with pytest.raises(ValueError, match="not found"):
        MLScoreOperator().run(ctx, {"model_name": "fraud"})


# ---------------------------------------------------------------------------
# Planning and wording
# ---------------------------------------------------------------------------

def test_scoring_requests_are_not_training_requests():
    from app.api.routes.chat import ChatService

    probe = type("Probe", (), {"_SCORE_RE": ChatService._SCORE_RE})()
    yes = ["score this file with my churn model", "Use my churn model on these rows", "run the fraud model on this data",
           "predict churn with my saved model", "batch predictions please", "rescore the customers"]
    no = ["train a model to predict churn", "average score by region", "what is the credit score distribution",
          "build a model with the price column", "predict churn", "score by region"]
    assert all(ChatService._is_score_intent(probe, m) for m in yes)
    assert not any(ChatService._is_score_intent(probe, m) for m in no)


def test_the_planner_can_name_a_model_and_the_step_is_described_by_code():
    from app.planning.llm_planner import _user_prompt, describe_step
    from app.planning.validator import validate_plan

    checked = validate_plan([{"operator": "ml_score", "params": {"model_name": "fraud"}}], {"columns": []})
    assert not checked.issues
    assert describe_step({"operator": "ml_score", "params": {"model_name": "fraud"}}).endswith("model fraud")
    assert '"MODELS": [{"name": "fraud"' in _user_prompt("q", {"columns": []}, {"models": [{"name": "fraud"}]})


def test_the_answer_states_mix_cut_off_drift_and_realised_accuracy():
    from app.api.routes.chat import _scoring_lines

    _train(_fraud(1500), name="fraud")
    res = MLScoreOperator().run(_ctx(_fraud(600, seed=4, shift=1.5)), {"model_name": "fraud"})
    text = "\n".join(_scoring_lines(res.summary))
    assert "I scored **all 600 rows** with your model **fraud** v1 (predicts **label**)" in text
    assert "**Predictions:**" in text and "**fraud**" in text
    assert "**Drift check:** The new data differs a lot" in text and "x1 PSI" in text
    assert "**Prediction mix vs held-out rows at training:**" in text
    assert "**How it did here:** label is known for 600 rows" in text
    assert "predictions** table artifact" in text


# ---------------------------------------------------------------------------
# End to end through chat, with no language model
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_scores_a_new_file_asks_which_model_and_keeps_users_apart(two_users, monkeypatch):
    from app.api.routes import chat

    monkeypatch.setattr(chat, "get_llm_service", lambda: None)
    client, [ana, ben] = two_users
    history = await _upload(client, ana, _fraud(1500), "history")
    trained = await client.post("/api/v1/ml/train", headers=ana, json={
        "dataset_id": history, "target_column": "label", "optimize": False, "max_models": 1,
        "model_name": "fraud_model"})
    assert trained.status_code == 200, trained.text

    new_id = await _upload(client, ana, _fraud(600, seed=11, shift=1.2), "next_week")
    say = {"dataset_id": new_id, "message": "score this file with my fraud model"}
    res = await client.post("/api/v1/chat", headers=ana, json=say)
    assert res.status_code == 200, res.text
    answer = res.json()["data"]
    assert len(answer["agent_actions"]) == 1, answer["content"]
    action, = answer["agent_actions"]
    assert action["operator"] == "ml_score", answer["content"]
    assert "I scored **all 600 rows**" in answer["content"]
    assert "**Drift check:** The new data differs a lot" in answer["content"]
    table = next(a for a in action["artifacts"] if a["name"] == "ml_score:predictions")
    rows = await client.get(f"/api/v1/artifacts/{table['artifact_id']}/rows", headers=ana)
    assert rows.status_code == 200 and "predicted_label" in rows.text

    # A second fitting model makes "my model" ambiguous: chat asks, then scores with the answer.
    second = await client.post("/api/v1/ml/train", headers=ana, json={
        "dataset_id": history, "target_column": "label", "optimize": False, "max_models": 1,
        "model_name": "fraud_challenger"})
    assert second.status_code == 200, second.text
    ask = await client.post("/api/v1/chat", headers=ana, json={"dataset_id": new_id,
                                                               "message": "run my model on these rows"})
    asked = ask.json()["data"]
    assert asked["agent_actions"] == [], asked["content"]
    assert "Which one should I use?" in asked["content"]
    assert "fraud_challenger (predicts label, v1)" in asked["content"]
    pick = await client.post("/api/v1/chat", headers=ana, json={
        "dataset_id": new_id, "conversation_id": asked["conversation_id"], "message": "fraud_challenger"})
    picked = pick.json()["data"]
    assert picked["agent_actions"][0]["summary"]["model_name"] == "fraud_challenger", picked["content"]

    # Ben has no models; Ana's are invisible to him.
    bens = await _upload(client, ben, _fraud(300, seed=2), "bens")
    other = await client.post("/api/v1/chat", headers=ben, json={"dataset_id": bens, "message": say["message"]})
    assert other.json()["data"]["agent_actions"] == []
    assert "no saved models" in other.json()["data"]["content"]
