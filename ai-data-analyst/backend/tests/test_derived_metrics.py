"""Ratios, shares of total and period-over-period change with contribution.

Expected values are computed by hand or with plain pandas; the pandas operators and the
batched DuckDB path must give identical answers on the same rows.
"""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
import pytest

from app.compute.operators.base import OperatorContext
from app.compute.operators.derived import PeriodCompareOperator, RatioOperator, ShareOfTotalOperator, resolve_periods
from app.planning.llm_planner import describe_plan, plan_question
from app.planning.validator import validate_plan
from app.services import chat_disk
from tests.test_llm_planner import ScriptedLLM
from tests.test_talk_to_data_user_flow import _upload_and_process, talk_client  # noqa: F401  (fixture)

SCHEMA = {"columns": [
    {"name": "Region", "inferred_type": "categorical", "unique_count": 2, "top_values": {"North": 5, "South": 3}},
    {"name": "Revenue", "inferred_type": "float"},
    {"name": "Cost", "inferred_type": "float"},
    {"name": "Units", "inferred_type": "integer"},
    {"name": "Day", "inferred_type": "datetime"},
]}


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Region": ["North", "North", "South", "North", "South", "North", "South", "North"],
        "Revenue": [100.0, 200.0, 50.0, 300.0, np.nan, 120.0, 80.0, 60.0],
        "Cost": [60.0, 150.0, 20.0, 200.0, 10.0, 100.0, 40.0, 30.0],
        "Units": [2, 4, 1, 6, 1, 3, 2, 0],
        "Day": ["2026-09-03", "2026-09-20", "2026-09-28", "2026-10-02", "2026-10-05", "2026-10-15", "2026-10-30", "2026-11-04"],
    })


def ctx(df: pd.DataFrame) -> OperatorContext:
    return OperatorContext(dataset_id=uuid4(), dataset_version="v", df=df, profile_report={}, schema_info=SCHEMA)


@pytest.fixture(autouse=True)
def _planner_on(monkeypatch):
    monkeypatch.setenv("LLM_PLANNER", "auto")


PERIODS = {"time_column": "Day", "value_column": "Revenue", "current_start": "2026-10-01", "current_end": "2026-11-01",
           "previous_start": "2026-09-01", "previous_end": "2026-10-01"}


# ---------------------------------------------------------------------------
# Exact values
# ---------------------------------------------------------------------------

def test_margin_ratio_uses_only_complete_rows():
    r = RatioOperator().run(ctx(frame()), {"numerator_column": "Revenue", "subtract_column": "Cost",
                                           "denominator_column": "Revenue", "as_percent": True, "group_by": "Region"})
    by = r.tables["result"].set_index("group")["value"].to_dict()
    north = (780 - 540) / 780 * 100
    south = (130 - 60) / 130 * 100  # the South row with no revenue is left out of both sums
    assert by["North"] == pytest.approx(north) and by["South"] == pytest.approx(south)
    assert r.summary["rows_excluded_missing"] == 1


def test_ratio_with_zero_denominator_is_not_a_number():
    df = pd.DataFrame({"Region": ["North", "South"], "Revenue": [10.0, 5.0], "Cost": [1.0, 1.0], "Units": [2, 0], "Day": ["2026-01-01"] * 2})
    out = RatioOperator().run(ctx(df), {"numerator_column": "Revenue", "denominator_column": "Units", "group_by": "Region"})
    by = out.tables["result"].set_index("group")["value"].to_dict()
    assert by["North"] == 5.0 and pd.isna(by["South"])


def test_share_of_total_adds_to_100():
    out = ShareOfTotalOperator().run(ctx(frame()), {"group_by": "Region", "value_column": "Revenue"}).tables["result"]
    assert out["share_pct"].sum() == pytest.approx(100.0)
    assert out.set_index("group")["value"].to_dict() == {"North": 780.0, "South": 130.0}
    rows = ShareOfTotalOperator().run(ctx(frame()), {"group_by": "Region"}).tables["result"]
    assert rows.set_index("group")["share_pct"].to_dict() == {"North": 62.5, "South": 37.5}


def test_period_compare_totals_change_and_contributions():
    out = PeriodCompareOperator().run(ctx(frame()), {**PERIODS, "group_by": "Region"})
    s = out.summary
    # October: 300 + 120 + 80 (the NaN row is skipped); September: 100 + 200 + 50
    assert (s["overall_current"], s["overall_previous"], s["overall_change"]) == (500.0, 350.0, 150.0)
    assert s["overall_change_pct"] == pytest.approx(150 / 350 * 100)
    by = out.tables["result"].set_index("group")
    assert by.loc["North", "change"] == 120.0 and by.loc["South", "change"] == 30.0
    assert by["contribution_pct"].sum() == pytest.approx(100.0)
    assert by.loc["North", "contribution_pct"] == pytest.approx(80.0)
    assert s["aligned_to_data"] is False


def test_period_compare_mean_and_count():
    mean = PeriodCompareOperator().run(ctx(frame()), {**PERIODS, "aggregation": "mean"}).summary
    assert mean["overall_current"] == pytest.approx(500 / 3) and mean["overall_previous"] == pytest.approx(350 / 3)
    count = PeriodCompareOperator().run(ctx(frame()), {**PERIODS, "aggregation": "count", "value_column": None}).summary
    assert (count["overall_current"], count["overall_previous"]) == (4.0, 3.0)


@pytest.fixture
def today_nov_5(monkeypatch):
    from app.compute.operators import derived
    monkeypatch.setattr(derived, "_today", lambda: date(2026, 11, 5))


def test_unfinished_current_period_is_compared_like_for_like(today_nov_5):
    p = {"current_start": "2026-11-01", "current_end": "2026-12-01", "previous_start": "2026-10-01", "previous_end": "2026-11-01"}
    periods = resolve_periods(p, pd.Timestamp("2026-11-04 13:00"))
    assert periods.current == (date(2026, 11, 1), date(2026, 11, 5))
    assert periods.previous == (date(2026, 10, 1), date(2026, 10, 5)) and periods.aligned
    out = PeriodCompareOperator().run(ctx(frame()), {**PERIODS, **p}).summary
    # 1-4 Nov: 60; 1-4 Oct: 300 (the 5 Oct row has no revenue)
    assert (out["overall_current"], out["overall_previous"]) == (60.0, 300.0)
    assert out["aligned_to_data"] is True and out["current_end"] == "2026-11-05"


def test_a_period_ending_with_today_is_in_progress():
    p = {"current_start": "2026-11-01", "current_end": "2026-11-25", "previous_start": "2026-10-01", "previous_end": "2026-11-01"}
    periods = resolve_periods(p, pd.Timestamp("2026-11-20"), today=date(2026, 11, 24))
    assert periods.aligned and periods.current == (date(2026, 11, 1), date(2026, 11, 21))
    assert periods.previous == (date(2026, 10, 1), date(2026, 10, 21))


def test_finished_period_keeps_the_dates_asked_for_and_says_the_data_ends_early():
    p = {"current_start": "2026-10-01", "current_end": "2026-11-01", "previous_start": "2026-09-01", "previous_end": "2026-10-01"}
    periods = resolve_periods(p, pd.Timestamp("2026-10-26"), today=date(2026, 11, 24))
    assert periods.current == (date(2026, 10, 1), date(2026, 11, 1)) and not periods.aligned
    assert periods.data_ends == date(2026, 10, 26)


# ---------------------------------------------------------------------------
# pandas and DuckDB give the same answer
# ---------------------------------------------------------------------------

def _duckdb_result(df: pd.DataFrame, op: str, params: dict):
    fields = {name: f"c{i}" for i, name in enumerate(df.columns)}
    table = df.rename(columns=fields).copy()
    table[fields["Day"]] = pd.to_datetime(table[fields["Day"]])
    numeric = ["Revenue", "Cost", "Units"]
    with duckdb.connect() as con:
        con.register("incoming", table)
        con.execute("CREATE TABLE matches AS SELECT * FROM incoming")
        return chat_disk._aggregate(con, op, params, fields, numeric, ["Day"])


@pytest.mark.parametrize("op,params,cls", [
    ("ratio", {"numerator_column": "Revenue", "subtract_column": "Cost", "denominator_column": "Revenue",
               "as_percent": True, "group_by": "Region"}, RatioOperator),
    ("ratio", {"numerator_column": "Revenue", "denominator_column": "Units"}, RatioOperator),
    ("share_of_total", {"group_by": "Region", "value_column": "Revenue"}, ShareOfTotalOperator),
    ("share_of_total", {"group_by": "Region"}, ShareOfTotalOperator),
    ("period_compare", {**PERIODS, "group_by": "Region"}, PeriodCompareOperator),
    ("period_compare", {**PERIODS, "aggregation": "mean"}, PeriodCompareOperator),
    ("period_compare", {**PERIODS, "current_end": "2026-12-01", "current_start": "2026-11-01",
                        "previous_start": "2026-10-01", "previous_end": "2026-11-01"}, PeriodCompareOperator),
])
def test_pandas_and_batched_paths_agree(op, params, cls, today_nov_5):
    df = frame()
    pandas_out = cls().run(ctx(df), params)
    sql_out = _duckdb_result(df, op, params)
    a = pandas_out.tables["result"].reset_index(drop=True)
    b = sql_out.tables["result"].reset_index(drop=True)
    if "group" in a:
        a, b = a.assign(group=a["group"].astype(str)).sort_values("group"), b.sort_values("group")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False, atol=1e-9)
    for key, value in pandas_out.summary.items():
        if isinstance(value, float):
            assert sql_out.summary[key] == pytest.approx(value), key
        else:
            assert sql_out.summary[key] == value, key


# ---------------------------------------------------------------------------
# Validation and planning
# ---------------------------------------------------------------------------

def test_period_parameters_are_validated():
    ok = validate_plan([{"operator": "period_compare", "params": {**PERIODS, "aggregation": "count"}}], SCHEMA)
    assert ok.ok and "value_column" not in ok.steps[0]["params"]
    for change, msg in [({"current_start": "Oct 2026"}, "YYYY-MM-DD"), ({"current_end": "2026-02-30"}, "valid calendar date"),
                        ({"previous_end": "2026-10-15"}, "overlap"), ({"current_end": "2026-10-01"}, "start before"),
                        ({"value_column": None}, "give value_column")]:
        bad = validate_plan([{"operator": "period_compare", "params": {**PERIODS, **change}}], SCHEMA)
        assert not bad.ok and any(msg in i for i in bad.issues), (change, bad.issues)
    same = validate_plan([{"operator": "ratio", "params": {"numerator_column": "Revenue", "denominator_column": "Revenue"}}], SCHEMA)
    assert not same.ok


def test_descriptions_name_the_last_included_day():
    text = describe_plan([{"operator": "period_compare", "params": {**PERIODS, "aggregation": "sum", "group_by": "Region"}}])
    assert text == "total Revenue from 2026-10-01 to 2026-10-31 vs 2026-09-01 to 2026-09-30, change by Region"
    assert describe_plan([{"operator": "ratio", "params": {"numerator_column": "Revenue", "subtract_column": "Cost",
                                                           "denominator_column": "Revenue", "as_percent": True}}]) \
        == "total (Revenue - Cost) / total Revenue as a percent"


@pytest.mark.asyncio
async def test_a_date_filter_on_the_compared_column_is_sent_back():
    step = {"operator": "period_compare", "params": PERIODS}
    llm = ScriptedLLM(
        {"status": "plan", "keep_detected_filters": [0], "steps": [step]},
        {"status": "plan", "keep_detected_filters": [], "steps": [step]},
    )
    out = await plan_question("revenue october vs september", SCHEMA, llm=llm, context={
        "detected_filters": [{"op": "date", "start": "2026-10-01", "end": "2026-11-01"}]})
    assert out.status == "plan" and out.attempts == 2 and out.keep_detected == []
    assert "period_compare already selects both periods" in llm.calls[1][-1].content


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

CSV = "Order ID,Region,Revenue,Day\n" + "".join(
    f"{i},{'North' if i % 3 else 'South'},{10 * i},2026-{9 + i % 2:02d}-{1 + i % 27:02d}\n" for i in range(1, 31))


@pytest.mark.asyncio
async def test_comparison_questions_are_refused_without_a_planner(talk_client, monkeypatch):
    """Without the planner, 'X vs Y' must never be answered for one of the two periods."""
    monkeypatch.setenv("LLM_PLANNER", "off")
    client, headers = talk_client
    dataset_id = await _upload_and_process(client, headers, name="Cmp off", csv_text=CSV)
    for q in ["total revenue this month vs last month", "how did revenue change from september 2026 to october 2026",
              "sum revenue for october 2026 versus september 2026", "why did revenue drop in october 2026"]:
        res = await client.post("/api/v1/chat", headers=headers, json={"dataset_id": str(dataset_id), "message": q})
        assert res.json()["data"]["agent_actions"] == [], (q, res.json()["data"]["content"])


@pytest.mark.asyncio
async def test_period_comparison_end_to_end(talk_client, monkeypatch):
    from app.api.routes import chat

    client, headers = talk_client
    dataset_id = await _upload_and_process(client, headers, name="Cmp on", csv_text=CSV)
    rows = pd.read_csv(pd.io.common.StringIO(CSV))
    rows["Day"] = pd.to_datetime(rows["Day"])
    octo = rows[rows["Day"].dt.month == 10]
    sept = rows[rows["Day"].dt.month == 9]
    llm = ScriptedLLM({"status": "plan", "keep_detected_filters": [], "steps": [
        {"operator": "period_compare", "params": {"time_column": "Day", "value_column": "Revenue", "group_by": "Region",
                                                  "current_start": "2026-10-01", "current_end": "2026-11-01",
                                                  "previous_start": "2026-09-01", "previous_end": "2026-10-01"}}]})
    monkeypatch.setattr(chat, "get_llm_service", lambda: llm)
    res = await client.post("/api/v1/chat", headers=headers, json={
        "dataset_id": str(dataset_id), "message": "why did revenue change from september 2026 to october 2026?"})
    body = res.json()["data"]
    assert body["agent_actions"], body["content"]
    text = body["content"]
    from app.api.routes.chat import _fmt_number

    cur, prev = octo["Revenue"].sum(), sept["Revenue"].sum()
    assert f"{_fmt_number(cur)} vs {_fmt_number(prev)}" in text
    assert "2026-10-01 to 2026-10-31 vs 2026-09-01 to 2026-09-30" in text
    assert "The data ends on 2026-10-26" in text, "a finished month keeps its dates and says the data stops early"
    for region in ("North", "South"):
        a = octo.loc[octo["Region"] == region, "Revenue"].sum()
        b = sept.loc[sept["Region"] == region, "Revenue"].sum()
        assert f"{region}: {_fmt_number(b)} → {_fmt_number(a)}" in text
    assert "Change by **Region**" in text and "-0.0%" not in text


@pytest.mark.asyncio
async def test_a_detected_date_restriction_cannot_be_dropped_without_a_comparison():
    """A model dropped 'last month' from 'how much did we sell online last month?'."""
    step = {"operator": "measure_aggregate", "params": {"aggregation": "sum", "value_column": "Revenue"}}
    llm = ScriptedLLM(
        {"status": "plan", "keep_detected_filters": [], "steps": [step]},
        {"status": "plan", "keep_detected_filters": [0], "steps": [step]},
    )
    out = await plan_question("how much revenue", SCHEMA, llm=llm, context={
        "detected_filters": [{"op": "date", "start": "2026-10-01", "end": "2026-11-01"}]})
    assert out.status == "plan" and out.keep_detected == [0] and out.attempts == 2
    assert "is a date restriction from the question" in llm.calls[1][-1].content
