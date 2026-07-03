from __future__ import annotations

from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from app.core.config import settings
from app.main import create_application
from app.services.database import db_manager
from app.services.dataset_processing import DatasetProcessingService


@pytest_asyncio.fixture(scope="module")
async def cache_client(tmp_path_factory):
    base = tmp_path_factory.mktemp("cache_integration")
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
                email = "cache@example.com"
                password = "Str0ngPassw0rd!"

                r = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": password, "role": "analyst", "full_name": "Cache Tester"},
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
    assert payload.get("status") == "success", payload
    ds_id = UUID(payload["data"]["dataset_id"])
    job_id = UUID(payload["data"]["job_id"])
    return ds_id, job_id


def _artifact_ids_from_agent_actions(actions: list[dict]) -> list[str]:
    ids: list[str] = []
    for act in actions or []:
        if not isinstance(act, dict):
            continue
        arts = act.get("artifacts")
        if not isinstance(arts, list):
            continue
        for a in arts:
            if not isinstance(a, dict):
                continue
            aid = a.get("artifact_id")
            if aid:
                ids.append(str(aid))
    ids.sort()
    return ids


@pytest.mark.asyncio
async def test_chat_reuses_cached_artifacts(cache_client):
    client, headers = cache_client

    csv_text = "\n".join(
        [
            "id,value,category,date,text",
            "1,10,A,2020-01-01,hello",
            "2,20,B,2020-01-02,world",
            "3,,A,2020-01-03,missing value",
            "4,40,C,2020-01-04,more text",
        ]
    )

    dataset_id, job_id = await _upload_csv(client, headers, name="Cache Dataset", csv_text=csv_text)
    await DatasetProcessingService().process_dataset(dataset_id, job_id)

    q = {"message": "How many rows and columns are there?", "dataset_id": str(dataset_id)}
    first = await client.post("/api/v1/chat", headers=headers, json=q)
    assert first.status_code == 200, first.text
    a1 = first.json()["data"]["agent_actions"]
    ids1 = _artifact_ids_from_agent_actions(a1)
    assert ids1, "Expected artifacts from first grounded chat call"

    second = await client.post("/api/v1/chat", headers=headers, json=q)
    assert second.status_code == 200, second.text
    a2 = second.json()["data"]["agent_actions"]
    ids2 = _artifact_ids_from_agent_actions(a2)
    assert ids2, "Expected artifacts from second grounded chat call"

    # Cache should return the same evidence artifacts (no recomputation, no new artifact IDs).
    assert ids2 == ids1

    # At least one operator summary should reflect cache hit.
    cache_flags = []
    for act in a2:
        if not isinstance(act, dict):
            continue
        summary = act.get("summary") if isinstance(act.get("summary"), dict) else {}
        if summary.get("_cache_hit") is True:
            cache_flags.append(True)
    assert cache_flags, "Expected at least one cached operator in second call"
