from __future__ import annotations

from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from app.core.config import settings
from app.main import create_application
from app.services.database import db_manager
from app.compute.artifacts import ArtifactStore
from app.services.artifact_index import ArtifactIndexService


@pytest_asyncio.fixture(scope="module")
async def share_client(tmp_path_factory):
    base = tmp_path_factory.mktemp("report_shares")
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
                email = "shares@example.com"
                password = "Str0ngPassw0rd!"

                r = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": password, "role": "analyst", "full_name": "Share Tester"},
                )
                assert r.status_code in {200, 201, 400}

                login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
                assert login.status_code == 200, login.text
                token = login.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}

                me = await client.get("/api/v1/auth/me", headers=headers)
                assert me.status_code == 200, me.text
                user_id = UUID(me.json()["user_id"])

                yield client, headers, user_id
    finally:
        await db_manager.close()
        settings.database.url = old_db_url
        settings.upload_directory = old_upload_dir
        settings.artifact_directory = old_artifact_dir


@pytest.mark.asyncio
async def test_report_share_create_and_public_fetch(share_client):
    client, headers, user_id = share_client

    store = ArtifactStore()
    ref = store.write_report(
        name="test_report_share",
        content="# Shared Report\n\nHello world.",
        report_format="markdown",
        dataset_id=None,
        dataset_version=None,
        operator_name="test",
        operator_params={"k": "v"},
    )

    async with db_manager.session() as session:
        await ArtifactIndexService(session).index_many(owner_id=user_id, refs=[ref])

    created = await client.post(f"/api/v1/shares/reports/{ref.artifact_id}", headers=headers, json={})
    assert created.status_code == 200, created.text
    data = created.json()["data"]
    token = str(data["share_token"])
    assert token

    public = await client.get(f"/api/v1/public/reports/{token}")
    assert public.status_code == 200, public.text
    payload = public.json()["data"]
    assert str(payload["artifact_id"]) == str(ref.artifact_id)
    assert "# Shared Report" in str(payload["content"])


@pytest.mark.asyncio
async def test_report_share_invalid_token_404(share_client):
    client, _headers, _user_id = share_client
    res = await client.get("/api/v1/public/reports/not-a-real-token")
    assert res.status_code == 404

