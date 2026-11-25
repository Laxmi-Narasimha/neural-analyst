from datetime import datetime
from uuid import UUID

import openpyxl
import pytest
import xlwt

from app.models import Conversation, Dataset
from app.services.database import db_manager
from app.services import dataset_batches
from app.services.dataset_processing import DatasetProcessingService
from tests.test_chat_fiscal_disk import table
from tests.test_talk_to_data_user_flow import talk_client


def write_book(path, sheets):
    legacy = path.suffix == ".xls"
    book = xlwt.Workbook() if legacy else openpyxl.Workbook()
    date_style = xlwt.easyxf(num_format_str="YYYY-MM-DD") if legacy else None
    if not legacy:
        book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.add_sheet(name) if legacy else book.create_sheet(name)
        for i, row in enumerate(rows):
            if legacy:
                for j, cell in enumerate(row):
                    if isinstance(cell, datetime):
                        sheet.write(i, j, cell, date_style)
                    else:
                        sheet.write(i, j, cell)
            else:
                sheet.append(row)
    book.save(str(path))


async def open_chat(client, headers, path):
    with path.open("rb") as file:
        upload = await client.post("/api/v1/datasets/upload", headers=headers, files={"file": (path.name, file)}, data={"name": "Stores"})
    assert upload.status_code == 201, upload.text
    ids = upload.json()["data"]
    dataset = ids["dataset_id"]
    await DatasetProcessingService().process_dataset(UUID(dataset), UUID(ids["job_id"]))
    detail = (await client.get("/api/v1/datasets/" + dataset, headers=headers)).json()["data"]
    assert detail["status"] == "ready", detail.get("error_message")
    cid = None
    async def ask(message, sheets=None):
        nonlocal cid
        response = await client.post("/api/v1/chat", headers=headers, json={"dataset_id": dataset,
            "conversation_id": cid, "message": message, "context": {} if sheets is None else {"worksheets": sheets}})
        assert response.status_code == 200, response.text
        result = response.json()["data"]
        cid = result["conversation_id"]
        return result

    return detail, ask


@pytest.mark.parametrize("fmt", ["xls", "xlsx"])
@pytest.mark.asyncio
async def test_selected_sheets_filters_and_evidence(talk_client, tmp_path, monkeypatch, fmt):
    client, headers = talk_client
    path = tmp_path / ("stores." + fmt)
    # Cross a batch boundary with matching and nonmatching records in both worksheets.
    monkeypatch.setattr(dataset_batches, "BATCH_ROWS", 4)
    may, june = datetime(2024, 5, 8), datetime(2024, 6, 1)
    a, b = "Store A", "東京 and Central"
    write_book(path, {
        a: [["Amount", "Region", "Posted", "Channel"], [11, "West", may, "Web"],
            [42, "North", june, "Desk"], [5, "East", may, "Web"]],
        b: [["Channel", "Region", "Posted", "Extra", "Amount"],
            *[["Desk", "East", may, 1, 19] for _ in range(8)],
            ["Desk", "West", may, 2, 7], ["Web", "West", june, 3, 55]],
        "Inventory": [["SKU", "Units"], ["Blue", 24]],
        "Blank": [],
    })
    detail, ask = await open_chat(client, headers, path)
    assert [(s["name"], s["rows"]) for s in detail["schema_info"]["worksheet_catalog"]] == [(a, 3), (b, 10), ("Inventory", 1), ("Blank", 0)]
    default = await ask("sum Amount")
    assert table(default) == [{"value": 58}]
    # Emulate a saved conversation from the first-worksheet-only release.
    async with db_manager.session() as session:
        conv = await session.get(Conversation, UUID(default["conversation_id"]))
        scope = dict(conv.context["analysis_scope"])
        for key in ("worksheets", "column_types", "worksheet_column"):
            scope.pop(key)
        conv.context = {**conv.context, "analysis_scope": scope}
        await session.commit()
    assert table(await ask("sum Amount")) == [{"value": 58}]
    other = await ask("sum Amount", [b])
    assert table(other) == [{"value": 214}]
    first = await ask("sum Amount in May 2024 where Region = West or Channel = Web", [b, a, b])
    assert table(first) == [{"value": 23}]
    scope = first["metadata"]["analysis_scope"]
    assert scope["worksheets"] == [a, b] and scope["matched_rows"] == 3
    evidence = next(item for item in first["agent_actions"][0]["artifacts"] if item["artifact_type"] == "table")
    assert evidence["operator_params"]["_na_meta"]["worksheets"] == [a, b]
    assert evidence["operator_params"]["_na_meta"]["row_scope"] == scope["filters"]
    # Extra exists only on the second tab. Its sole matching value is 2.
    partial = await ask("sum Extra")
    assert table(partial) == [{"value": 2}]
    assert "treated as blank" in partial["content"]
    response = await ask("sum Amount by Worksheet")
    assert table(response) == [{"group": a, "value": 16}, {"group": b, "value": 7}]
    assert response["metadata"]["analysis_scope"]["filters"] == scope["filters"]
    narrowed = await ask('sum Amount on sheet "' + b + '"')
    assert table(narrowed) == [{"value": 7}]
    restored = await client.get("/api/v1/chat/conversations/" + narrowed["conversation_id"], headers=headers)
    assert restored.json()["data"]["worksheets"] == [b]
    for selection in ([], ["Unknown"], ["Inventory"]):
        rejected = await ask("sum Amount", selection)
        assert not rejected["agent_actions"]
    assert table(await ask("sum Amount")) == [{"value": 7}]
    assert table(await ask("clear filters")) == [{"value": 214}]
    reset = await ask("start over", ["Inventory"])
    assert reset["metadata"]["analysis_scope"]["worksheets"] == ["Inventory"]
    assert table(await ask("sum Units")) == [{"value": 24}]


@pytest.mark.parametrize("fmt", ["xls", "xlsx"])
@pytest.mark.asyncio
async def test_lab_workbook_clarification_and_large_later_sheet(talk_client, tmp_path, fmt):
    client, headers = talk_client
    path = tmp_path / ("lab." + fmt)
    day, later = datetime(2024, 2, 29), datetime(2024, 3, 1)
    columns = ["Reading", "Taken", "Reported", "Worksheet"]
    tail = dataset_batches.BATCH_ROWS + 17
    a, b = "Lab, East", "Lab West"
    write_book(path, {"Cover": [], a: [columns, [12, day, later, "Sensor A"]],
        b: [columns, *[[1, later, later, "Sensor B"] for _ in range(tail)], [25, day, later, "Sensor B"]],
        "Text": [["Reading"], ["unavailable"]]})
    detail, ask = await open_chat(client, headers, path)
    # Previously processed workbooks may have only the original first-sheet metadata.
    async with db_manager.session() as session:
        dataset = await session.get(Dataset, UUID(detail["id"]))
        dataset.schema_info = {k: v for k, v in dataset.schema_info.items() if k != "worksheet_catalog"}
        await session.commit()
    detail = (await client.get("/api/v1/datasets/" + detail["id"], headers=headers)).json()["data"]
    assert detail["schema_info"]["worksheet_catalog"][2]["rows"] == tail + 1
    pending = await ask('sum Reading in February 2024 across sheets "Lab, East" and "Lab West"')
    assert {c["value"] for c in pending["clarification"]["options"]} == {"Taken", "Reported"}
    assert pending["metadata"]["worksheets"] == [a, b]
    wrong = await ask("Taken", [a])
    assert not wrong["agent_actions"] and wrong["metadata"]["worksheets"] == [a, b]
    result = await ask("Taken")
    assert table(result) == [{"value": 37}]
    assert result["metadata"]["analysis_scope"]["worksheet_column"] == "Source Worksheet"
    assert table(await ask("sum Reading by Source Worksheet")) == [{"group": b, "value": 25}, {"group": a, "value": 12}]
    single = await ask('switch to sheet "Lab West"')
    assert table(single) == [{"group": b, "value": 25}]
    incompatible = await ask("sum Reading", [b, "Text"])
    assert "different types" in incompatible["content"]
    assert table(await ask("sum Reading")) == [{"value": 25}]
    assert table(await ask("clear filters")) == [{"value": tail + 25}]
