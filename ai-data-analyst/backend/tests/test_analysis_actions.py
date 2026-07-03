from __future__ import annotations

from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from app.core.config import settings
from app.main import create_application
from app.services.analysis_execution import AnalysisExecutionService
from app.services.database import db_manager
from app.services.dataset_processing import DatasetProcessingService


@pytest_asyncio.fixture(scope="module")
async def client_and_headers(tmp_path_factory):
    base = tmp_path_factory.mktemp("analysis_actions")
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
                email = "actions@example.com"
                password = "Str0ngPassw0rd!"

                r = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": password, "role": "analyst", "full_name": "Actions Tester"},
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


async def _upload_csv(client: httpx.AsyncClient, headers: dict[str, str], *, name: str, csv_text: str) -> tuple[UUID, UUID]:
    files = {"file": ("data.csv", csv_text.encode("utf-8"), "text/csv")}
    data = {"name": name}
    res = await client.post("/api/v1/datasets/upload", headers=headers, files=files, data=data)
    assert res.status_code == 201, res.text
    payload = res.json()
    assert payload.get("status") == "success"
    ds_id = UUID(payload["data"]["dataset_id"])
    job_id = UUID(payload["data"]["job_id"])
    return ds_id, job_id


@pytest.mark.asyncio
async def test_action_run_creates_child_analysis_and_updates_parent_feed(client_and_headers):
    client, headers = client_and_headers

    csv_text = "\n".join(
        [
            "id,value,segment,date",
            "1,10,A,2020-01-01",
            "2,20,A,2020-01-02",
            "3,,B,2020-01-03",
            "4,9999,B,2020-01-04",
            "5,50,C,2020-01-05",
        ]
    )

    dataset_id, job_id = await _upload_csv(client, headers, name="Actions Dataset", csv_text=csv_text)
    await DatasetProcessingService().process_dataset(dataset_id, job_id)

    created = await client.post(
        "/api/v1/analyses",
        headers=headers,
        json={"name": "Parent EDA", "dataset_id": str(dataset_id), "analysis_type": "eda", "config": {"sample_rows": 5000}},
    )
    assert created.status_code == 201, created.text
    parent_id = UUID(created.json()["data"]["id"])

    detail = await client.get(f"/api/v1/analyses/{parent_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    parent_job_id = UUID(detail.json()["data"]["config"]["job_id"])
    await AnalysisExecutionService().run_analysis(parent_id, parent_job_id)

    run = await client.post(
        f"/api/v1/analyses/{parent_id}/actions/run",
        headers=headers,
        json={"action_id": "missingness_patterns", "params": {}},
    )
    assert run.status_code == 200, run.text
    item = run.json()["data"]
    child_id = UUID(item["analysis_id"])

    child_detail = await client.get(f"/api/v1/analyses/{child_id}", headers=headers)
    assert child_detail.status_code == 200, child_detail.text
    child_job_id = UUID(child_detail.json()["data"]["config"]["job_id"])
    await AnalysisExecutionService().run_analysis(child_id, child_job_id)

    feed = await client.get(f"/api/v1/analyses/{parent_id}/actions", headers=headers)
    assert feed.status_code == 200, feed.text
    items = feed.json()["data"]
    assert isinstance(items, list)
    match = [it for it in items if str(it.get("analysis_id")) == str(child_id)]
    assert match, items
    it = match[0]
    assert str(it.get("status")).lower() == "completed"
    assert isinstance(it.get("artifacts"), list)
    assert len(it["artifacts"]) >= 1

