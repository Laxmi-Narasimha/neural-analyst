from datetime import date
from io import BytesIO
from uuid import UUID

import pandas as pd
import pytest

from app.services import dataset_batches
from app.services.chat_scope import fiscal_period
from app.services.dataset_loader import DatasetLoaderService
from app.services.dataset_processing import DatasetProcessingService
from tests.test_talk_to_data_user_flow import talk_client, _upload_and_process


def fiscal(text, start="April", label="Ending year"):
    return fiscal_period(text, date(2024, 2, 29), {"fiscal_start": start, "fiscal_label": label})


@pytest.mark.parametrize(("text", "month", "label", "lo", "hi"), [
    ("FY2024 Q4", "April", "Ending year", "2024-01-01", "2024-04-01"),
    ("fiscal Q1 2024", "October", "Starting year", "2024-10-01", "2025-01-01"),
    ("FY 2023/24", "July", "Ending year", "2023-07-01", "2024-07-01"),
    ("last fiscal quarter", "January", "Ending year", "2023-10-01", "2024-01-01"),
    ("this fiscal year", "August", "Starting year", "2023-08-01", "2024-08-01"),
    ("previous financial year", "April", "Ending year", "2022-04-01", "2023-04-01"),
    ("FYTD", "April", "Ending year", "2023-04-01", "2024-03-01"),
    ("from FY2023 Q4 through FY2024 Q1", "April", "Ending year", "2023-01-01", "2023-07-01"),
])
def test_fiscal_boundaries(text, month, label, lo, hi):
    period = fiscal(text, month, label)
    assert (period["start"], period["end"]) == (lo, hi)


def conversation(client, headers, dataset):
    conversation_id = None
    async def ask(message):
        nonlocal conversation_id
        r = await client.post("/api/v1/chat", headers=headers, json={
            "dataset_id": str(dataset), "conversation_id": conversation_id, "message": message})
        assert r.status_code == 200, r.text
        answer = r.json()["data"]
        conversation_id = answer["conversation_id"]
        return answer
    return ask


def table(answer):
    assert answer["agent_actions"], answer["content"]
    artifact = next(a for a in answer["agent_actions"][0]["artifacts"] if a["artifact_type"] == "table")
    return artifact["preview"]["preview_rows"]


@pytest.mark.asyncio
async def test_fiscal_boolean_followups_and_dataset_isolation(talk_client):
    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Freight",
        csv_text="Port,Late,Boxes,Arrival\nWest,1,10,2025-04-01\nWest,0,20,2025-06-30\n"
                 "East,1,30,2025-05-02\nEast,0,80,2025-05-03\nEast,1,42,2025-07-01\nWest,1,90,2025-03-31\n")
    ask = conversation(client, headers, ds)
    start = await ask("sum Boxes in Q1 FY2026 where Port = West or Late = 1")
    assert start["clarification"]["param_key"] == "fiscal_start"
    convention = await ask("April")
    assert convention["clarification"]["param_key"] == "fiscal_label"
    first = await ask("Ending year")
    assert table(first) == [{"value": 60}] and first["metadata"]["analysis_scope"]["matched_rows"] == 3
    assert table(await ask("what about Q2")) == [{"value": 42}]
    assert table(await ask("what about Q1")) == [{"value": 60}]
    assert table(await ask("switch Port to East")) == [{"value": 120}]
    assert table(await ask("remove Port filter")) == [{"value": 40}]
    assert table(await ask("or Port = West")) == [{"value": 60}]
    assert table(await ask("remove Port filter")) == [{"value": 40}]
    assert table(await ask("all dates")) == [{"value": 172}]
    fresh = conversation(client, headers, ds)
    assert (await fresh("how many rows in fiscal Q1"))["clarification"]["param_key"] == "fiscal_start"
    assert (await fresh("October"))["clarification"]["param_key"] == "fiscal_year"
    full = await fresh("2024-25")
    assert full["metadata"]["analysis_scope"]["matched_rows"] == 0
    span = await fresh("sum Boxes between Q3 FY2024-25 and Q4 FY2024-25")
    assert table(span) == [{"value": 182}]


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["csv", "tsv", "parquet", "json", "xlsx"])
async def test_batched_exact_operators_across_file_formats(talk_client, monkeypatch, fmt):
    client, headers = talk_client
    monkeypatch.setattr(dataset_batches, "LARGE_BYTES", 1)
    monkeypatch.setattr(dataset_batches, "BATCH_ROWS", 4)
    frame = pd.DataFrame({
        "Instrument": ["Control", "Laser", "Laser", "Probe"] * 12,
        "Batch code": ["A", "B", "A", "B"] * 12,
        "Reading": list(range(48)),
        "Reference": [2 * i for i in range(48)],
        "Recorded": pd.date_range("2024-09-20", periods=48).strftime("%Y-%m-%d"),
    })
    buf = BytesIO()
    if fmt == "parquet":
        frame.to_parquet(buf, index=False)
    elif fmt == "json":
        buf.write(frame.to_json(orient="records").encode())
    elif fmt == "xlsx":
        frame.to_excel(buf, index=False)
    else:
        buf.write(frame.to_csv(index=False, sep="\t" if fmt == "tsv" else ",").encode())
    upload = await client.post("/api/v1/datasets/upload", headers=headers,
        files={"file": (f"readings.{fmt}", buf.getvalue())}, data={"name": "Instrument readings"})
    assert upload.status_code == 201, upload.text
    ids = upload.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(ids["dataset_id"]), UUID(ids["job_id"]))

    async def reject_full_load(*args, **kwargs):
        pytest.fail("Large chat invoked the eager dataset loader")
    monkeypatch.setattr(DatasetLoaderService, "load_dataset", reject_full_load)
    ask = conversation(client, headers, ids["dataset_id"])
    requested = await ask("median Reading where (Instrument = Laser or Batch code = A) and Recorded in FY2024-25 Q1")
    assert requested["clarification"]["param_key"] == "fiscal_start", requested["content"]
    answer = await ask("October")
    selected = frame[(frame.Recorded >= "2024-10-01") & ((frame.Instrument == "Laser") | (frame["Batch code"] == "A"))]
    assert table(answer) == [{"value": selected.Reading.median()}]
    scope = answer["metadata"]["analysis_scope"]
    assert scope["execution"] == "batched" and not scope["sampled_operators"]
    assert scope["matched_rows"] == scope["analyzed_rows"] == len(selected)
    assert sorted((r["group"], r["value"]) for r in table(await ask("sum Reading by Instrument"))) == sorted(selected.groupby("Instrument").Reading.sum().items())
    trend = table(await ask("Show monthly trend for Reading using Recorded"))
    assert [r["value_sum"] for r in trend] == list(selected.groupby(selected.Recorded.str[:7]).Reading.sum())
    preview = table(await ask("preview rows"))
    assert preview[0]["Reading"] == selected.iloc[0].Reading
    assert table(await ask("sum Reading")) == [{"value": selected.Reading.sum()}]
    stats = await ask("describe numeric columns")
    summary = next(a for a in stats["agent_actions"] if a["operator"] == "numeric_summary")
    row = summary["artifacts"][0]["preview"].get("preview_rows")
    if not row:
        row = next(a["preview"]["preview_rows"] for a in summary["artifacts"] if a["artifact_type"] == "table")
    stats_reading = next(r for r in row if r["column"] == "Reading")
    assert stats_reading["50%"] == selected.Reading.median()
    corr = await ask("show correlations")
    assert corr["metadata"]["sample_rows"] is None
    assert table(corr)[0]["corr"] == pytest.approx(1.0)
    sample = await ask("find outliers")
    assert sample["metadata"]["analysis_scope"]["matched_rows"] == len(selected)
    assert sample["metadata"]["analysis_scope"]["sampled_operators"] == ["outlier_scan"]
    assert sample["metadata"]["sample_rows"] == 4 and "Sample-based" in sample["content"]
    none = await ask("and Reading > 2000")
    assert none["metadata"]["analysis_scope"]["matched_rows"] == 0


@pytest.mark.asyncio
async def test_and_or_precedence_quotes_and_overlap(talk_client):
    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Spares",
        csv_text="Location,Part,Units\nA,R or D,4\nA,Fit,12\nB,R or D,3\nB,Fit,15\n")
    async def total(expr):
        return table(await conversation(client, headers, ds)("sum Units where " + expr))[0]["value"]
    assert await total('Location = A or Part = "R or D" and Units < 10') == 19
    assert await total('(Location = A or Part = "R or D") and Units < 10') == 7
    assert await total('Location = B and (Units between 1 and 5 or Part in ("Fit", "R or D"))') == 18


@pytest.mark.asyncio
async def test_compact_parquet_late_match_and_bounded_batches(talk_client, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    client, headers = talk_client
    buf = BytesIO()
    with pq.ParquetWriter(buf, pa.schema([("Station", pa.string()), ("Units", pa.int64())])) as writer:
        batch = pa.table({"Station": ["main"] * 5000, "Units": [5] * 5000})
        for _ in range(40):
            writer.write_table(batch)
        writer.write_table(pa.table({"Station": ["remote"], "Units": [7801]}))
    assert len(buf.getvalue()) < dataset_batches.LARGE_BYTES
    observed = []
    original = dataset_batches.frames
    def watched(*args, **kwargs):
        for chunk in original(*args, **kwargs):
            observed.append(len(chunk))
            yield chunk
    monkeypatch.setattr(dataset_batches, "frames", watched)
    upload = await client.post("/api/v1/datasets/upload", headers=headers,
        files={"file": ("meters.parquet", buf.getvalue())}, data={"name": "Energy meters"})
    ids = upload.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(ids["dataset_id"]), UUID(ids["job_id"]))
    ask = conversation(client, headers, ids["dataset_id"])
    observed.clear()
    result = await ask("sum Units for remote only")
    assert table(result) == [{"value": 7801}]
    assert max(observed) <= dataset_batches.BATCH_ROWS
    assert sum(observed) >= 200001
    all_rows = await ask("or Units > 0")
    assert table(all_rows) == [{"value": 1007801}]
    assert all_rows["metadata"]["analysis_scope"]["matched_rows"] == 200001


def test_json_lines_profiles_fields_first_seen_in_later_batches(tmp_path, monkeypatch):
    path = tmp_path / "observations.json"
    path.write_text('{"a":1}\n{"a":2}\n{"a":3,"late":"new"}\n', encoding="utf8")
    monkeypatch.setattr(dataset_batches, "BATCH_ROWS", 2)
    sample, count = dataset_batches.profile_sample(path, "json")
    assert count == 3 and len(sample) == 2 and "late" in sample
    schema = {"columns": [{"name": "a"}, {"name": "late"}]}
    frames = list(dataset_batches.frames(path, "json", schema))
    assert frames[0].late.isna().all()
    assert frames[1].late.tolist() == ["new"]
