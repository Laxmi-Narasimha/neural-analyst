from __future__ import annotations

from pathlib import Path
from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from app.core.config import settings
from app.main import create_application
from app.services.database import db_manager
from app.services.dataset_processing import DatasetProcessingService


@pytest_asyncio.fixture(scope="module")
async def client_and_headers(tmp_path_factory):
    base = tmp_path_factory.mktemp("dataset_sql")
    db_file = base / "test.db"
    uploads_dir = base / "uploads"
    artifacts_dir = base / "artifacts"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    old_db_url = settings.database.url
    old_upload_dir = settings.upload_directory
    old_artifact_dir = settings.artifact_directory

    try:
        settings.database.url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
        settings.upload_directory = uploads_dir
        settings.artifact_directory = artifacts_dir

        await db_manager.close()

        app = create_application()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                email = "sql@example.com"
                password = "Str0ngPassw0rd!"

                r = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": password, "role": "analyst", "full_name": "SQL Tester"},
                )
                assert r.status_code in {200, 201, 400}

                login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
                assert login.status_code == 200, login.text
                token = login.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}

                yield client, headers
    finally:
        await db_manager.close()
        settings.database.url = old_db_url
        settings.upload_directory = old_upload_dir
        settings.artifact_directory = old_artifact_dir


async def _upload_and_process(client: httpx.AsyncClient, headers: dict[str, str], csv_text: str | None = None) -> UUID:
    if not csv_text:
        csv_text = "\n".join(
            [
                "id,value,category,email",
                "1,10,A,a@x.com",
                "2,20,B,b@y.com",
                "3,30,A,c@z.com",
            ]
        )
    files = {"file": ("data.csv", csv_text.encode("utf-8"), "text/csv")}
    data = {"name": "SQL Dataset"}
    res = await client.post("/api/v1/datasets/upload", headers=headers, files=files, data=data)
    assert res.status_code == 201, res.text
    payload = res.json()
    dataset_id = UUID(payload["data"]["dataset_id"])
    job_id = UUID(payload["data"]["job_id"])
    await DatasetProcessingService().process_dataset(dataset_id, job_id)
    return dataset_id


@pytest.mark.asyncio
async def test_dataset_query_sql_endpoint_returns_artifact(client_and_headers):
    client, headers = client_and_headers
    dataset_id = await _upload_and_process(client, headers)

    q = {"query": "select count(*) as n from dataset"}
    res = await client.post(f"/api/v1/datasets/{dataset_id}/query", headers=headers, json=q)
    assert res.status_code == 200, res.text
    data = res.json()["data"]
    assert data["row_count"] == 1
    assert data["artifact"]["artifact_type"] == "table"
    assert data["artifact"]["dataset_id"] == str(dataset_id)


@pytest.mark.asyncio
async def test_dataset_query_sql_endpoint_blocks_file_table_functions(client_and_headers):
    client, headers = client_and_headers
    dataset_id = await _upload_and_process(client, headers)

    q = {"query": "select * from read_csv_auto('C:/secrets.csv')"}
    res = await client.post(f"/api/v1/datasets/{dataset_id}/query", headers=headers, json=q)
    assert res.status_code == 400, res.text


@pytest.mark.asyncio
async def test_artifact_rows_endpoint_pages_table_artifacts(client_and_headers):
    client, headers = client_and_headers
    dataset_id = await _upload_and_process(client, headers)

    q = {"query": "select id, value, category from dataset order by id"}
    res = await client.post(f"/api/v1/datasets/{dataset_id}/query", headers=headers, json=q)
    assert res.status_code == 200, res.text
    artifact_id = res.json()["data"]["artifact"]["id"]

    rows = await client.get(f"/api/v1/artifacts/{artifact_id}/rows?offset=0&limit=2", headers=headers)
    assert rows.status_code == 200, rows.text
    payload = rows.json()["data"]
    assert payload["artifact_id"] == artifact_id
    assert payload["offset"] == 0
    assert payload["limit"] == 2
    assert isinstance(payload["columns"], list)
    assert isinstance(payload["rows"], list)
    assert len(payload["rows"]) == 2


@pytest.mark.asyncio
async def test_dataset_query_sql_redacts_pii_in_preview_and_rows(client_and_headers):
    client, headers = client_and_headers
    dataset_id = await _upload_and_process(client, headers)

    q = {"query": "select email from dataset order by id"}
    res = await client.post(f"/api/v1/datasets/{dataset_id}/query", headers=headers, json=q)
    assert res.status_code == 200, res.text
    data = res.json()["data"]

    preview = data["artifact"].get("preview") or {}
    assert isinstance(preview, dict)
    preview_rows = preview.get("preview_rows") or []
    assert isinstance(preview_rows, list)
    assert preview_rows and isinstance(preview_rows[0], dict)
    assert preview_rows[0].get("email") == "[REDACTED]"

    artifact_id = data["artifact"]["id"]
    rows = await client.get(f"/api/v1/artifacts/{artifact_id}/rows?offset=0&limit=3", headers=headers)
    assert rows.status_code == 200, rows.text
    payload = rows.json()["data"]
    assert payload["rows"][0].get("email") == "[REDACTED]"


@pytest.mark.asyncio
async def test_dataset_query_sql_falls_back_when_duckdb_missing(client_and_headers, monkeypatch):
    from app.api.routes import datasets as datasets_route

    def _missing_duckdb(*, file_path: str, file_format: str, sql: str):
        raise ModuleNotFoundError("No module named 'duckdb'")

    monkeypatch.setattr(datasets_route, "_run_duckdb_query_sync", _missing_duckdb)

    client, headers = client_and_headers
    dataset_id = await _upload_and_process(client, headers)

    q = {"query": "select category, count(*) as n from dataset group by category order by category"}
    res = await client.post(f"/api/v1/datasets/{dataset_id}/query", headers=headers, json=q)
    assert res.status_code == 200, res.text
    rows = (res.json()["data"]["artifact"]["preview"] or {}).get("preview_rows") or []
    assert rows and rows[0]["category"] == "A"


@pytest.mark.asyncio
async def test_artifact_rows_falls_back_when_duckdb_missing(client_and_headers, monkeypatch):
    from app.api.routes import artifacts as artifacts_route

    def _missing_duckdb(*, data_path: str, data_format: str, offset: int, limit: int):
        raise ModuleNotFoundError("No module named 'duckdb'")

    monkeypatch.setattr(artifacts_route, "_run_duckdb_slice_sync", _missing_duckdb)

    client, headers = client_and_headers
    dataset_id = await _upload_and_process(client, headers)

    q = {"query": "select id, value from dataset order by id"}
    res = await client.post(f"/api/v1/datasets/{dataset_id}/query", headers=headers, json=q)
    assert res.status_code == 200, res.text
    artifact_id = res.json()["data"]["artifact"]["id"]

    rows = await client.get(f"/api/v1/artifacts/{artifact_id}/rows?offset=1&limit=2", headers=headers)
    assert rows.status_code == 200, rows.text
    payload = rows.json()["data"]
    assert len(payload["rows"]) == 2


@pytest.mark.asyncio
async def test_dataset_transform_suggest_endpoint_returns_plan(client_and_headers):
    client, headers = client_and_headers
    csv_text = "\n".join(
        [
            "id,amount,category,order_date,comment,const_col",
            "1,10,A,2026-01-01,  hello  ,x",
            "2,,A,2026-01-02,world,x",
            "3,30,,2026-01-03,many   spaces,x",
            "3,30,,2026-01-03,many   spaces,x",
        ]
    )
    dataset_id = await _upload_and_process(client, headers, csv_text=csv_text)

    res = await client.post(
        f"/api/v1/datasets/{dataset_id}/transform/suggest",
        headers=headers,
        json={"max_steps": 10},
    )
    assert res.status_code == 200, res.text
    data = res.json()["data"]
    suggestions = data.get("suggestions") or []
    assert suggestions and isinstance(suggestions, list)

    ops = [str((s.get("step") or {}).get("op")) for s in suggestions if isinstance(s, dict)]
    assert "fill_missing" in ops
    assert "drop_columns" in ops
    assert "string_normalize" in ops
