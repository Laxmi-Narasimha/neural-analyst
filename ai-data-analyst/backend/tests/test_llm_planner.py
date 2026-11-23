"""LLM planner: the model chooses operators, code validates and computes.

A scripted fake model stands in for OpenAI so every behaviour is deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from app.compute.registry import default_registry
from app.planning.catalog import CATALOG
from app.planning.llm_planner import describe_columns, plan_question
from app.planning.validator import validate_plan
from tests.test_talk_to_data_user_flow import _upload_and_process, talk_client  # noqa: F401  (fixture)

SCHEMA = {
    "columns": [
        {"name": "Order ID", "inferred_type": "integer"},
        {"name": "Branch", "inferred_type": "string", "top_values": {"East": 3, "West": 1}},
        {"name": "Net sales", "inferred_type": "float"},
        {"name": "Invoice date", "inferred_type": "date"},
        {"name": "Customer email", "inferred_type": "email", "is_potential_pii": True, "top_values": {"a@x.com": 1}},
    ]
}


@dataclass
class _Reply:
    content: str


class ScriptedLLM:
    is_available = True

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list] = []

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("planner asked the model more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Reply(reply if isinstance(reply, str) else json.dumps(reply))


@pytest.fixture(autouse=True)
def _planner_on(monkeypatch):
    monkeypatch.setenv("LLM_PLANNER", "auto")


# ---------------------------------------------------------------------------
# Catalog and validator
# ---------------------------------------------------------------------------

def test_every_catalog_operator_is_executable():
    available = set(default_registry().list())
    missing = [name for name in CATALOG if name not in available]
    assert missing == []


def test_validator_binds_columns_and_fills_defaults():
    result = validate_plan(
        [{"operator": "segment_summary", "params": {"group_by": "branch", "value_column": "net_sales", "sort_by": "VALUE_MEAN"}}],
        SCHEMA,
    )
    assert result.ok, result.issues
    assert result.steps == [
        {
            "operator": "segment_summary",
            "params": {"group_by": "Branch", "value_column": "Net sales", "sort_by": "value_mean", "ascending": False, "limit": 50},
        }
    ]


@pytest.mark.parametrize(
    "steps, issue",
    [
        ([{"operator": "segment_summary", "params": {"group_by": "Region"}}], "does not exist"),
        ([{"operator": "measure_aggregate", "params": {"value_column": "Branch", "aggregation": "sum"}}], "needs a numeric"),
        ([{"operator": "resample_aggregate", "params": {"time_column": "Net sales"}}], "needs a datetime"),
        ([{"operator": "segment_summary", "params": {"group_by": "Customer email"}}], "sensitive"),
        ([{"operator": "measure_aggregate", "params": {"value_column": "Net sales", "aggregation": "average"}}], "not one of"),
        ([{"operator": "measure_aggregate", "params": {"value_column": "Net sales"}}], "is required"),
        ([{"operator": "drop_table", "params": {}}], "not available"),
        ([{"operator": "dataset_overview"}] * 5, "at most 4"),
        ([], "no steps"),
        ("not a list", "no steps"),
    ],
)
def test_validator_blocks_bad_plans(steps, issue):
    result = validate_plan(steps, SCHEMA)
    assert not result.ok
    assert any(issue in i for i in result.issues), result.issues


def test_validator_clamps_ranges_and_ignores_unknown_settings():
    result = validate_plan([{"operator": "preview_rows", "params": {"limit": 10_000, "sql": "DROP TABLE x"}}], SCHEMA)
    assert result.ok
    assert result.steps[0]["params"] == {"limit": 200}
    assert any("sql" in n for n in result.notes)


def test_prompt_never_includes_values_of_sensitive_columns():
    described = {c["name"]: c for c in describe_columns(SCHEMA)}
    assert described["Customer email"] == {"name": "Customer email", "type": "categorical", "sensitive": True}
    assert described["Branch"]["examples"] == ["East", "West"]


# ---------------------------------------------------------------------------
# Planner behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_plan_is_returned_bound_to_the_schema():
    llm = ScriptedLLM({"status": "plan", "reason": "average net sales per branch, ranked",
                       "steps": [{"operator": "segment_summary", "params": {"group_by": "BRANCH", "value_column": "Net Sales", "sort_by": "value_mean"}}]})
    out = await plan_question("which branch sells most per order?", SCHEMA, llm=llm)
    assert out.status == "plan"
    assert out.steps[0]["params"]["group_by"] == "Branch"
    assert out.attempts == 1


@pytest.mark.asyncio
async def test_invalid_plan_is_repaired_once_with_the_validation_errors():
    llm = ScriptedLLM(
        {"status": "plan", "steps": [{"operator": "segment_summary", "params": {"group_by": "Region", "value_column": "Revenue"}}]},
        {"status": "plan", "steps": [{"operator": "segment_summary", "params": {"group_by": "Branch", "value_column": "Net sales"}}]},
    )
    out = await plan_question("revenue by region", SCHEMA, llm=llm)
    assert out.status == "plan" and out.attempts == 2
    repair_prompt = llm.calls[1][-1].content
    assert "column 'Region' does not exist" in repair_prompt


@pytest.mark.asyncio
async def test_planner_gives_up_instead_of_running_an_unvalidated_plan():
    bad = {"status": "plan", "steps": [{"operator": "segment_summary", "params": {"group_by": "Region"}}]}
    out = await plan_question("revenue by region", SCHEMA, llm=ScriptedLLM(bad, bad))
    assert out.status == "error" and out.steps == []


@pytest.mark.asyncio
async def test_unexpressed_row_restrictions_become_a_question_not_a_wrong_answer():
    llm = ScriptedLLM({"status": "plan", "unhandled_restrictions": ["for repeat customers only"],
                       "steps": [{"operator": "measure_aggregate", "params": {"value_column": "Net sales", "aggregation": "sum"}}]})
    out = await plan_question("total sales for repeat customers", SCHEMA, llm=llm)
    assert out.status == "clarify"
    assert "repeat customers" in out.question and out.steps == []


@pytest.mark.asyncio
async def test_clarify_and_unsupported_are_passed_through():
    out = await plan_question("trend", SCHEMA, llm=ScriptedLLM(
        {"status": "clarify", "clarification": {"question": "Which date?", "options": ["Invoice date", "Due date"]}}))
    assert (out.status, out.options) == ("clarify", ["Invoice date", "Due date"])
    out = await plan_question("what is the weather", SCHEMA, llm=ScriptedLLM({"status": "unsupported", "reason": "no weather data"}))
    assert (out.status, out.reason) == ("unsupported", "no weather data")


@pytest.mark.asyncio
async def test_non_json_and_provider_errors_fail_closed():
    out = await plan_question("q", SCHEMA, llm=ScriptedLLM("Sure! Here's a plan: segment by branch", "still prose"))
    assert out.status == "error" and out.steps == []
    out = await plan_question("q", SCHEMA, llm=ScriptedLLM(RuntimeError("429 rate limited")))
    assert out.status == "error" and "could not be reached" in out.reason


@pytest.mark.asyncio
async def test_json_wrapped_in_markdown_is_accepted():
    reply = "```json\n" + json.dumps({"status": "plan", "steps": [{"operator": "dataset_overview", "params": {}}]}) + "\n```"
    out = await plan_question("how big is it", SCHEMA, llm=ScriptedLLM(reply))
    assert out.status == "plan"


@pytest.mark.asyncio
async def test_planner_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("LLM_PLANNER", "off")
    llm = ScriptedLLM()
    out = await plan_question("anything", SCHEMA, llm=llm)
    assert out.status == "unavailable" and llm.calls == []


# ---------------------------------------------------------------------------
# End to end through /api/v1/chat
# ---------------------------------------------------------------------------

CSV = (
    "Order ID,Branch,Net sales,Invoice date\n"
    + "".join(f"{i},East,10,2026-01-{(i % 27) + 1:02d}\n" for i in range(1, 31))
    + "".join(f"{i},West,40,2026-0{1 + (i % 4)}-15\n" for i in range(31, 41))
)


@pytest.mark.asyncio
async def test_chat_uses_llm_plan_for_unmatched_questions_and_numbers_come_from_operators(talk_client, monkeypatch):
    from app.api.routes import chat

    llm = ScriptedLLM({
        "status": "plan",
        "reason": "average net sales per order for each branch, highest first",
        "steps": [{"operator": "segment_summary", "params": {"group_by": "branch", "value_column": "net sales", "sort_by": "value_mean"}}],
    })
    monkeypatch.setattr(chat, "get_llm_service", lambda: llm)
    client, headers = talk_client
    dataset_id = await _upload_and_process(client, headers, name="Planner sales", csv_text=CSV)

    res = await client.post("/api/v1/chat", headers=headers, json={
        "dataset_id": str(dataset_id), "message": "which branch has the best average ticket size?",
    })
    assert res.status_code == 200, res.text
    answer = res.json()["data"]
    assert answer["agent_actions"], answer["content"]
    assert len(llm.calls) == 1
    # The model saw the schema but no computed result.
    sent = llm.calls[0][-1].content
    assert "Net sales" in sent and "value_mean" not in sent.split('"QUESTION"')[1]
    action, = answer["agent_actions"]
    assert action["operator"] == "segment_summary"
    table = next(a for a in action["artifacts"] if a["artifact_type"] == "table")
    rows = table["preview"]["preview_rows"]
    assert [(r["group"], r["value_mean"]) for r in rows] == [("West", 40), ("East", 10)]
    assert answer["content"].startswith("Interpreted your question as: average net sales per order")

    # A rule-supported question never reaches the model.
    rule = await client.post("/api/v1/chat", headers=headers, json={
        "dataset_id": str(dataset_id), "message": "show average net sales by branch",
    })
    assert rule.status_code == 200, rule.text
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_chat_turns_model_clarification_and_failure_into_safe_replies(talk_client, monkeypatch):
    from app.api.routes import chat

    client, headers = talk_client
    dataset_id = await _upload_and_process(client, headers, name="Planner sales 2", csv_text=CSV)

    llm = ScriptedLLM({"status": "plan", "unhandled_restrictions": ["big-ticket orders"],
                       "steps": [{"operator": "measure_aggregate", "params": {"value_column": "Net sales", "aggregation": "sum"}}]})
    monkeypatch.setattr(chat, "get_llm_service", lambda: llm)
    res = await client.post("/api/v1/chat", headers=headers, json={
        "dataset_id": str(dataset_id), "message": "how much did big-ticket orders bring in?",
    })
    assert res.status_code == 200, res.text
    answer = res.json()["data"]
    assert answer["agent_actions"] == []
    assert "big-ticket orders" in answer["content"]

    broken = ScriptedLLM(RuntimeError("provider down"))
    monkeypatch.setattr(chat, "get_llm_service", lambda: broken)
    res = await client.post("/api/v1/chat", headers=headers, json={
        "dataset_id": str(dataset_id), "message": "how much did big-ticket orders bring in?",
    })
    assert res.status_code == 200, res.text
    assert res.json()["data"]["agent_actions"] == []
    assert "couldn't match" in res.json()["data"]["content"]
