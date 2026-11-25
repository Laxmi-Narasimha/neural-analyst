from datetime import date, datetime
from uuid import UUID

import pandas as pd
import pytest
import xlwt

from app.services import dataset_batches
from app.services.chat_scope import ScopeQuestion, fiscal_period
from app.services.dataset_loader import DatasetLoaderService
from app.services.dataset_processing import DatasetProcessingService
from app.services.retail_calendar import CUSTOM, JANUARY, RetailCalendar
from tests.test_chat_fiscal_disk import conversation, table
from tests.test_talk_to_data_user_flow import _upload_and_process, talk_client


@pytest.mark.parametrize("pattern,unit,start,stop", [
    ("4-4-5", "P2", "2023-02-26", "2023-03-26"),
    ("4-5-4", "P2", "2023-02-26", "2023-04-02"),
    ("5-4-4", "P2", "2023-03-05", "2023-04-02"),
    ("4-4-5", "Q4", "2023-10-29", "2024-02-04"),
    ("4-5-4", "W53", "2024-01-28", "2024-02-04"),
    ("4-5-4", "FYTD", "2023-01-29", "2024-02-01"),
    ("4-4-5", "last retail period", "2023-11-26", "2023-12-24"),
])
def test_published_january_anchor_and_patterns(pattern, unit, start, stop):
    cfg = {"retail_pattern": pattern, "retail_preset": JANUARY, "active_retail_anchor": 2024}
    interval = fiscal_period(unit, date(2024, 1, 31), cfg)
    assert (interval["start"], interval["end"]) == (start, stop)


def test_custom_anchor_52_and_53_week_invariants():
    cal = RetailCalendar("5-4-4", 8, 4, False, False, 1)
    for year in range(1995, 2051):
        bounds = cal.periods(year)
        assert (bounds[-1] - bounds[0]).days in (364, 371)
        assert cal.periods(year + 1)[0] == bounds[-1]
        assert all(b.weekday() == 5 for b in bounds)
        assert (bounds[-1] - pd.Timedelta(days=1)).month == 8
        lengths = [(b-a).days // 7 for a,b in zip(bounds,bounds[1:])]
        assert lengths[1:] == ([5,4,4]*4)[1:]
        if (bounds[-1] - bounds[0]).days == 371:
            assert lengths[0] == 6
            assert cal.interval(year, "W", 53)["end"] == bounds[-1].isoformat()
        else:
            assert lengths[0] == 5
            with pytest.raises(ScopeQuestion, match="no W53"):
                cal.interval(year, "W", 53)


def make_xls(path, header, rows, *, mac=False):
    book = xlwt.Workbook()
    book.dates_1904 = mac
    sheet = book.add_sheet("Observations")
    date_style = xlwt.easyxf(num_format_str="YYYY-MM-DD HH:MM")
    for j, name in enumerate(header):
        sheet.write(0, j, name)
    for i, values in enumerate(rows, 1):
        for j, val in enumerate(values):
            if val is not None:
                if isinstance(val, date):
                    sheet.write(i, j, val, date_style)
                else:
                    sheet.write(i, j, val)
    book.add_sheet("Notes").write(0, 0, "Separate worksheet")
    book.save(str(path))


@pytest.mark.parametrize("mac", [False, True])
def test_binary_excel_types_sparse_rows_and_string_continuations(tmp_path, mac):
    path = tmp_path / "instrument.xls"
    text = "ab測試😀" * 1700
    make_xls(path, ["Location", "Time", "Accepted", "Reading", None], [
        (text, datetime(2024, 2, 29, 16, 45), True, 1.5, None),
        (None, None, None, None, None),
        ("東京", datetime(2025, 1, 1), False, -301, "tail"),
    ], mac=mac)
    from app.services.legacy_excel import xls_frames
    actual = pd.concat(xls_frames(path, 2), ignore_index=True)
    reference = pd.read_excel(path, engine="xlrd")
    pd.testing.assert_frame_equal(actual, reference, check_dtype=False)


@pytest.mark.asyncio
async def test_large_xls_retail_followups_and_last_record(talk_client, tmp_path, monkeypatch):
    client, headers = talk_client
    path = tmp_path / "historic-sales.xls"
    header = ["Depot", "Channel", "Amount", "Posted", "Memo"]
    def rows():
        for i in range(10_000):
            yield "East", "Desk", 1, date(2023, 11, 1), str(i) + "z" * 1780
        yield "West", "Desk", 10, date(2023, 12, 24), "first day of P12"
        yield "West", "Web", 20, date(2024, 1, 28), "both branches"
        yield "East", "Web", 7, date(2024, 2, 3), "last day of W53"
        yield "East", "Web", 999, date(2024, 2, 4), "next fiscal year"
        yield "West", "Desk", 101, date(2024, 2, 2), "last source record"
    make_xls(path, header, rows())
    assert path.stat().st_size > dataset_batches.LARGE_BYTES
    observed = []
    original = dataset_batches.frames
    def watched(*args, **kwargs):
        for batch in original(*args, **kwargs):
            observed.append(len(batch))
            yield batch
    monkeypatch.setattr(dataset_batches, "frames", watched)
    def reject_eager(*args, **kwargs):
        pytest.fail("Binary Excel upload or chat took an eager path")
    monkeypatch.setattr(pd, "read_excel", reject_eager)
    monkeypatch.setattr(DatasetLoaderService, "load_dataset", reject_eager)
    with path.open("rb") as file:
        response = await client.post("/api/v1/datasets/upload", headers=headers,
            files={"file": (path.name, file, "application/vnd.ms-excel")}, data={"name": "Historical sales"})
    assert response.status_code == 201, response.text
    ids = response.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(ids["dataset_id"]), UUID(ids["job_id"]))
    ask = conversation(client, headers, ids["dataset_id"])
    first = await ask("sum Amount in 4-4-5 P12 FY2023 where Depot = West or Channel = Web")
    assert first["clarification"]["param_key"] == "retail_preset"
    p12 = await ask(JANUARY)
    assert table(p12) == [{"value": 138}]
    assert p12["metadata"]["analysis_scope"]["execution"] == "batched"
    assert p12["metadata"]["analysis_scope"]["matched_rows"] == 4
    w53 = await ask("what about week 53")
    assert table(w53) == [{"value": 128}]
    assert w53["metadata"]["analysis_scope"]["matched_rows"] == 3
    invalid = await ask("what about W53 FY2024")
    assert not invalid["agent_actions"] and "no W53" in invalid["content"]
    # Rejecting W53 in a 52-week year leaves the previous successful scope active.
    assert table(await ask("and Amount > 100")) == [{"value": 101}]
    assert table(await ask("what about P1 FY2024")) == [{"value": 999}]
    await ask("clear all filters")
    grouped = await ask("sum Amount by retail period using Posted in FY2023 where Depot = West or Channel = Web")
    assert table(grouped) == [{"group": "FY2023 P12", "value": 138}]
    weekly = await ask("sum Amount by retail week using Posted")
    assert table(weekly) == [{"group": "FY2023 W48", "value": 10}, {"group": "FY2023 W53", "value": 128}]
    assert max(observed) <= dataset_batches.BATCH_ROWS
    assert len(observed) >= 2


@pytest.mark.asyncio
async def test_custom_calendar_and_filter_followups_on_shipments(talk_client):
    client, headers = talk_client
    ds = await _upload_and_process(client, headers, name="Shipments",
        csv_text="Hub,Delivered,Boxes\nNorth,2024-08-30,100\nNorth,2024-08-31,4\n"
                 "South,2024-09-01,8\nNorth,2024-10-04,15\nNorth,2024-10-05,30\n")
    ask = conversation(client, headers, ds)
    first = await ask("sum Boxes for North only in retail period 1 FY2025")
    assert first["clarification"]["param_key"] == "retail_pattern"
    for answer, key in [("5-4-4","retail_preset"), (CUSTOM,"retail_month"), ("August","retail_weekday"),
                        ("Friday","retail_rule"), ("Last","retail_label"), ("Anchor year","retail_extra")]:
        assert (await ask(answer))["clarification"]["param_key"] == key
    assert table(await ask("P1")) == [{"value": 19}]
    assert table(await ask("what about P2")) == [{"value": 30}]
    assert table(await ask("what about Q1")) == [{"value": 49}]
    assert table(await ask("switch Hub to South")) == [{"value": 8}]
    grouped = await ask("sum Boxes by retail period using Delivered")
    assert table(grouped) == [{"group": "FY2025 P01", "value": 8}]
    fresh = conversation(client, headers, ds)
    assert (await fresh("sum Boxes in fiscal Q1 FY2025"))["clarification"]["param_key"] == "fiscal_start"



@pytest.mark.asyncio
async def test_tall_lab_xls_nrf_and_two_date_columns(talk_client, tmp_path):
    client, headers = talk_client
    path = tmp_path / "lab-history.xls"
    early = date(2023, 1, 1)
    def rows():
        for i in range(65531):  # Four late observations and one header fill a BIFF8 sheet.
            yield "bench", early, early, i % 17
        yield "remote", date(2024, 3, 2), date(2024, 3, 5), 25
        yield "remote", date(2024, 3, 3), date(2024, 3, 7), 71
        yield "remote", date(2024, 4, 6), date(2024, 4, 12), 81
        yield "remote", date(2024, 4, 7), date(2024, 4, 15), 133
    make_xls(path, ["Station", "Taken", "Logged", "Reading"], rows(), mac=True)
    with path.open("rb") as stream:
        r = await client.post("/api/v1/datasets/upload", headers=headers, files={"file": (path.name, stream)},
                              data={"name": "Laboratory instruments"})
    assert r.status_code == 201, r.text
    ids = r.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(ids["dataset_id"]), UUID(ids["job_id"]))
    ask = conversation(client, headers, ids["dataset_id"])
    first = await ask("median Reading in NRF P1 FY2024 where Station = remote")
    assert {c["value"] for c in first["clarification"]["options"]} == {"Taken", "Logged"}
    assert table(await ask("Taken")) == [{"value": 25}]
    p2 = await ask("what about P2")
    assert p2["metadata"]["analysis_scope"]["matched_rows"] == 2
    assert table(p2) == [{"value": 76}]
    assert table(await ask("what about P3")) == [{"value": 133}]
    assert table(await ask("what about P2")) == table(p2)
    grouped = conversation(client, headers, ids["dataset_id"])
    need_axis = await grouped("sum Reading by NRF period")
    assert need_axis["clarification"]["param_key"] == "retail_axis"
    by_period = await grouped("Logged")
    assert table(by_period)
    by_week = await grouped("now by retail week")
    assert by_week["metadata"]["analysis_scope"]["steps"][0]["params"]["time_column"] == "Logged"
    assert all(" W" in row["group"] for row in table(by_week))
