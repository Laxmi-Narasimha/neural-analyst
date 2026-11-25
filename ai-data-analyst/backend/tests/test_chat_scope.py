from datetime import date
from uuid import uuid4

import pandas as pd
import pytest

from app.services.chat_scope import bind_conditions, parse_scope, scoped_dataset
from app.services.dataset_loader import LoadedDataset
from tests.test_talk_to_data_user_flow import _upload_and_process, talk_client


@pytest.mark.asyncio
async def test_trend_scope_edits_and_cached_evidence(talk_client):
    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Orders",
        csv_text="District,Net amount,Invoice day,Channel\nWest,10,2026-01-01,Store\n"
                 "East,500,2026-01-17,Store\nWest,35,2026-03-31,Web\nWest,90,2026-04-01,Store\n")
    conv = None

    async def ask(question):
        nonlocal conv
        r = await client.post("/api/v1/chat", headers=headers, json={
            "dataset_id": str(ds), "conversation_id": conv, "message": question})
        assert r.status_code == 200, r.text
        a = r.json()["data"]
        conv = a["conversation_id"]
        return a

    def trend(a):
        step, = a["agent_actions"]
        t = next(x for x in step["artifacts"] if x["artifact_type"] == "table")
        assert t["operator_params"]["_na_meta"]["row_scope"] == a["metadata"]["analysis_scope"]["filters"]
        return sum(r["value_sum"] for r in t["preview"]["preview_rows"]), t["artifact_id"]

    west = await ask("Show monthly trend for Net amount using Invoice day for West only")
    total, west_id = trend(west)
    assert total == 135
    q1 = await ask("Now only Q1 2026")
    assert trend(q1)[0] == 45
    assert trend(await ask("what about Q2?"))[0] == 90
    assert trend(await ask("what about Q1?")) == trend(q1)
    web = await ask("and Channel = Web")
    assert trend(web)[0] == 35
    no_match = await ask("what about East")
    assert no_match["metadata"]["analysis_scope"]["matched_rows"] == 0
    assert "No rows match" in no_match["content"]
    east = await ask("remove Channel filter")
    assert trend(east)[0] == 500
    assert trend(await ask("switch to West")) == trend(q1)
    assert trend(await ask("all dates")) == (135, west_id)
    assert (await ask("clear all filters"))["metadata"]["analysis_scope"]["filters"] == []
    quarterly = await ask("Show quarterly trend for Net amount using Invoice day for West only")
    proof = next(a for a in quarterly["agent_actions"][0]["artifacts"] if a["artifact_type"] == "table")
    assert [r["value_sum"] for r in proof["preview"]["preview_rows"]] == [45, 90]


@pytest.mark.asyncio
async def test_ambiguous_filters_and_dates_survive_clarification(talk_client):
    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Shipments",
        csv_text="Origin,Destination,Dispatched,Received,Weight\nNorth,South,2025-12-30,2026-01-02,9\n"
                 "South,North,2026-03-01,2026-04-01,7\nNorth,West,2026-03-31,2026-04-02,12\n")
    conv = None

    async def ask(question, dataset=ds):
        nonlocal conv
        r = await client.post("/api/v1/chat", headers=headers, json={
            "dataset_id": str(dataset), "conversation_id": conv, "message": question})
        assert r.status_code == 200, r.text
        a = r.json()["data"]
        conv = a["conversation_id"]
        return a

    a = await ask("How many rows for North only in Q1")
    assert a["clarification"]["param_key"] == "year"
    a = await ask("2026")
    assert {o["value"] for o in a["clarification"]["options"]} == {"Origin", "Destination"}
    a = await ask("Origin")
    assert {o["value"] for o in a["clarification"]["options"]} == {"Dispatched", "Received"}
    a = await ask("Received")
    assert a["metadata"]["analysis_scope"]["matched_rows"] == 1
    assert "1 rows" in a["content"] or "1 row" in a["content"]
    unsupported = await ask("where Weight > 3 or (Origin = South and)")
    assert not unsupported["agent_actions"]
    unsupported = await ask("how many discharged rows")
    assert not unsupported["agent_actions"]
    needs_column = await ask("for North only")
    assert needs_column["clarification"]["options"]
    await ask("cancel clarification")
    assert (await ask("preview rows"))["metadata"]["analysis_scope"]["matched_rows"] == 1
    two_dates = await ask("and Dispatched in Q1 2026")
    assert two_dates["metadata"]["analysis_scope"]["matched_rows"] == 0
    assert {c["column"] for c in two_dates["metadata"]["analysis_scope"]["filters"]} == {"Origin", "Received", "Dispatched"}
    assert (await ask("remove Dispatched filter"))["metadata"]["analysis_scope"]["matched_rows"] == 1
    other = await _upload_and_process(client, headers, name="Labs", csv_text="Reading,Assay\n3,X\n9,Y\n")
    a = await ask("how many rows", other)
    assert not a["metadata"]["analysis_scope"]["filters"] and "2 rows" in a["content"]


@pytest.mark.parametrize(("question", "expected"), [
    ("how many rows where μg/L >= 10 and Passed = true", [2]),
    ('how many rows where Site in ("R and D","Ward B") and μg/L between 2 and 9', [1]),
    ('how many rows where Site != "R and D"', [1, 2]),
    ("how many rows where Reading time between 2024-02-01 and 2024-02-29", [1, 2]),
    ("how many rows from 2024-02-29 until 2024-03-01", [2]),
    ("how many rows during February 2024", [1, 2]),
    ("how many rows where Site is missing", [3]),
    ("how many rows last month", [0]),
    ("how many rows where Reading time in Q1 2024 and Site = Intake", [2]),
    ("how many rows where Site = Intake or Ward B", [1, 2]),
    ("how many rows from February 1, 2024 through 29 February 2024", [1, 2]),
    ("how many rows where Reading time != 2024-02-29", [0, 1, 3]),
    ("how many rows where Reading time in (2024-02-01, 2024-02-29)", [1, 2]),
])
def test_typed_row_conditions(question, expected):
    df = pd.DataFrame({"μg/L": [1, 8, 11, 2], "Passed": [False, False, True, True],
                       "Site": ["R and D", "Ward B", "Intake", None],
                       "Reading time": ["2024-01-18", "2024-02-01", "2024-02-29 23:59:59", "2024-03-01"]})
    schema = {"columns": [dict(name=c, inferred_type=t) for c, t in zip(df, ["integer", "boolean", "string", "datetime"])]}
    loaded = LoadedDataset(uuid4(), "readings", "v1", "full.csv", "csv", df, {}, schema)
    edit = parse_scope(question, schema, today=date(2024, 2, 10), bindings={})
    conditions = bind_conditions(edit, [], loaded, [], {})
    narrowed = scoped_dataset(loaded, conditions)
    assert narrowed.df.index.tolist() == expected
    assert not narrowed.file_path
    assert narrowed.profile_report["row_count"] == len(expected)


def test_calendar_words_can_also_be_column_names():
    frame = pd.DataFrame({"Month": [1, 2], "Date": ["2024-01-20", "2024-02-02"]})
    schema = {"columns": [{"name": "Month", "inferred_type": "integer"}, {"name": "Date", "inferred_type": "datetime"}]}
    loaded = LoadedDataset(uuid4(), "observations", "v1", "rows.csv", "csv", frame, {}, schema)
    edit = parse_scope("how many rows last month and Month = 1", schema, today=date(2024, 2, 10), bindings={})
    conditions = bind_conditions(edit, [], loaded, [], {})
    assert scoped_dataset(loaded, conditions).df.index.tolist() == [0]

@pytest.mark.parametrize("file_format", ["csv", "tsv", "json", "parquet", "xlsx"])
@pytest.mark.asyncio
async def test_measure_and_date_scope_across_upload_formats(talk_client, file_format):
    from io import BytesIO
    from uuid import UUID
    from app.services.dataset_processing import DatasetProcessingService

    client, headers = talk_client
    frame = pd.DataFrame({
        "Test site": ["North", "South", "North", "North"],
        "Reading": [2, 83, 7, 11],
        "Observed": ["2024-01-01", "2024-01-02", "2024-03-31", "2024-04-01"],
    })
    buf = BytesIO()
    if file_format == "parquet":
        frame.to_parquet(buf, index=False)
    elif file_format == "xlsx":
        frame.to_excel(buf, index=False)
    elif file_format == "json":
        buf.write(frame.to_json(orient="records").encode())
    else:
        buf.write(frame.to_csv(index=False, sep="\t" if file_format == "tsv" else ",").encode())
    r = await client.post("/api/v1/datasets/upload", headers=headers,
                          files={"file": (f"assays.{file_format}", buf.getvalue())}, data={"name": "Assays"})
    assert r.status_code == 201, r.text
    upload = r.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(upload["dataset_id"]), UUID(upload["job_id"]))
    r = await client.post("/api/v1/chat", headers=headers, json={
        "dataset_id": upload["dataset_id"], "message": "sum of Reading where Test site = North in Q1 2024",
    })
    assert r.status_code == 200, r.text
    a = r.json()["data"]
    assert a["metadata"]["analysis_scope"]["matched_rows"] == 2, a
    assert a["agent_actions"][0]["artifacts"][0]["preview"]["preview_rows"] == [{"value": 9}]
    r = await client.post("/api/v1/chat", headers=headers, json={
        "conversation_id": a["conversation_id"], "message": "what about South?",
    })
    a = r.json()["data"]
    assert a["metadata"]["analysis_scope"]["matched_rows"] == 1
    assert a["agent_actions"][0]["artifacts"][0]["preview"]["preview_rows"] == [{"value": 83}]
    r = await client.post("/api/v1/chat", headers=headers, json={
        "conversation_id": a["conversation_id"], "message": "now median Reading",
    })
    changed = r.json()["data"]
    assert changed["metadata"]["analysis_scope"]["filters"] == a["metadata"]["analysis_scope"]["filters"]
    assert changed["metadata"]["analysis_scope"]["steps"][0]["params"]["aggregation"] == "median"


@pytest.mark.asyncio
async def test_dataset_version_change_needs_a_fresh_scope(talk_client):
    from app.models import Dataset
    from app.services.database import db_manager

    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Versioned", csv_text="Place,Qty\nA,5\nB,6\n")
    url = "/api/v1/chat"
    first = await client.post(url, headers=headers, json={"dataset_id": str(ds), "message": "sum Qty where Place = A"})
    conversation_id = first.json()["data"]["conversation_id"]
    async with db_manager.session() as session:
        record = await session.get(Dataset, ds)
        record.profile_report = {**record.profile_report, "file_hash": "second-version"}
        await session.commit()
    stale = await client.post(url, headers=headers, json={"conversation_id": conversation_id, "message": "what about B"})
    assert not stale.json()["data"]["agent_actions"]
    assert "version changed" in stale.json()["data"]["content"]
    fresh = await client.post(url, headers=headers, json={"conversation_id": conversation_id, "message": "start over"})
    scope = fresh.json()["data"]["metadata"]["analysis_scope"]
    assert not scope["filters"] and scope["dataset_version"] == "second-version"


@pytest.mark.asyncio
async def test_followup_in_older_conversation_requests_original_calculation(talk_client):
    from uuid import UUID
    from app.models import Conversation
    from app.services.database import db_manager

    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Legacy readings", csv_text="Place,Qty\nA,5\nB,6\n")
    url = "/api/v1/chat"
    first = await client.post(url, headers=headers, json={"dataset_id": str(ds), "message": "sum Qty where Place = A"})
    conversation_id = first.json()["data"]["conversation_id"]
    async with db_manager.session() as session:
        conv = await session.get(Conversation, UUID(conversation_id))
        conv.context = {"turn_history": conv.context["turn_history"]}
        await session.commit()
    follow = await client.post(url, headers=headers, json={"conversation_id": conversation_id, "message": "what about B"})
    assert not follow.json()["data"]["agent_actions"]
    assert "Restate the calculation" in follow.json()["data"]["content"]
    resumed = await client.post(url, headers=headers, json={"conversation_id": conversation_id, "message": "sum Qty where Place = B"})
    assert resumed.json()["data"]["metadata"]["analysis_scope"]["matched_rows"] == 1


@pytest.mark.asyncio
async def test_overlapping_followups_cannot_overwrite_each_others_scope(talk_client, monkeypatch):
    import asyncio
    from app.compute.executor import ComputeExecutor

    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Readings", csv_text="Site,Qty\nA,5\nB,6\nC,7\n")
    first = await client.post("/api/v1/chat", headers=headers, json={"dataset_id": str(ds), "message": "sum Qty where Site = A"})
    conversation = first.json()["data"]["conversation_id"]
    paused, resume = asyncio.Event(), asyncio.Event()
    original = ComputeExecutor.run_plan

    async def pause_one_turn(self, **kwargs):
        if any(c.get("value") == "B" for c in kwargs.get("row_scope", [])):
            paused.set()
            await asyncio.wait_for(resume.wait(), 10)
        return await original(self, **kwargs)

    monkeypatch.setattr(ComputeExecutor, "run_plan", pause_one_turn)

    async def ask(question):
        return await client.post("/api/v1/chat", headers=headers, json={"conversation_id": conversation, "message": question})

    delayed = asyncio.create_task(ask("what about B"))
    try:
        await asyncio.wait_for(paused.wait(), 10)
        latest = await ask("what about C")
        assert latest.status_code == 200
    finally:
        resume.set()
    assert (await delayed).status_code == 409
    after = (await ask("count records")).json()["data"]["metadata"]["analysis_scope"]
    assert after["filters"] == [{"column": "Site", "op": "eq", "value": "C"}]


@pytest.mark.asyncio
async def test_exact_scope_beyond_interactive_sample(talk_client):
    client, headers = talk_client
    data = "station,units\n" + "main,1\n" * 200_000 + "remote,987\n"
    ds = await _upload_and_process(client, headers, name="Meter readings", csv_text=data)
    response = await client.post("/api/v1/chat", headers=headers, json={
        "dataset_id": str(ds), "message": "total units where station = remote"})
    a = response.json()["data"]
    assert a["metadata"]["sample_rows"] is None
    assert a["metadata"]["analysis_scope"]["matched_rows"] == 1
    assert a["agent_actions"][0]["artifacts"][0]["preview"]["preview_rows"] == [{"value": 987}]
