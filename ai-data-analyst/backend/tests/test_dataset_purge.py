from __future__ import annotations

from pathlib import Path
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select

from app.core.config import settings
from app.main import create_application
from app.models import Artifact, Dataset, DatasetVersion, Job
from app.services.analysis_execution import AnalysisExecutionService
from app.services.database import db_manager
from app.services.dataset_processing import DatasetProcessingService
from app.services.dataset_purge import DatasetPurgeService
from app.services.object_store import get_object_store


@pytest_asyncio.fixture(scope="module")
async def purge_client(tmp_path_factory):
    base = tmp_path_factory.mktemp("purge_integration")
    db_file = base / "test.db"
    uploads_dir = base / "uploads"
    artifacts_dir = base / "artifacts"
    models_dir = base / "models"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    old_db_url = settings.database.url
    old_upload_dir = settings.upload_directory
    old_artifact_dir = settings.artifact_directory
    old_models_dir = settings.ml.model_storage_path

    try:
        settings.database.url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
        settings.upload_directory = uploads_dir
        settings.artifact_directory = artifacts_dir
        settings.ml.model_storage_path = models_dir

        await db_manager.close()

        app = create_application()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                email = "purge@example.com"
                password = "Str0ngPassw0rd!"

                r = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": password, "role": "analyst", "full_name": "Purge Tester"},
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
        settings.ml.model_storage_path = old_models_dir


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


@pytest.mark.asyncio
async def test_dataset_purge_deletes_blobs_and_metadata(purge_client):
    client, headers = purge_client

    csv_text = "\n".join(
        [
            "id,value,category,date",
            "1,10,A,2020-01-01",
            "2,20,B,2020-01-02",
            "3,,A,2020-01-03",
            "4,40,C,2020-01-04",
        ]
    )

    dataset_id, ingest_job_id = await _upload_csv(client, headers, name="Purge Dataset", csv_text=csv_text)
    await DatasetProcessingService().process_dataset(dataset_id, ingest_job_id)

    # Run a small EDA analysis to generate artifacts for this dataset.
    created = await client.post(
        "/api/v1/analyses",
        headers=headers,
        json={"name": "Purge EDA", "dataset_id": str(dataset_id), "analysis_type": "eda", "config": {"sample_rows": 5000}},
    )
    assert created.status_code == 201, created.text
    analysis_id = UUID(created.json()["data"]["id"])

    detail = await client.get(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    job_id = UUID(detail.json()["data"]["config"]["job_id"])
    await AnalysisExecutionService().run_analysis(analysis_id, job_id)

    # Capture storage paths before purge.
    async with db_manager.session() as session:
        ds = (await session.execute(select(Dataset).where(Dataset.id == dataset_id))).scalars().first()
        assert ds is not None
        ds_path = str(ds.file_path)

        ver_paths = (await session.execute(
            select(DatasetVersion.file_path).where(DatasetVersion.dataset_id == dataset_id)
        )).scalars().all()

        art_paths = (await session.execute(
            select(Artifact.manifest_path, Artifact.data_path).where(Artifact.dataset_id == dataset_id)
        )).all()

    # Trigger purge via API (creates the job record and soft-deletes immediately).
    pur = await client.post(f"/api/v1/datasets/{dataset_id}/purge", headers=headers)
    assert pur.status_code == 200, pur.text
    purge_job_id = UUID(pur.json()["data"]["job_id"])

    # Execute purge deterministically (avoid relying on background tasks in tests).
    await DatasetPurgeService().purge_dataset(dataset_id, purge_job_id)

    # Dataset is gone.
    ds_get = await client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert ds_get.status_code == 404

    # Job is completed (and retained even though dataset metadata is deleted).
    job_get = await client.get(f"/api/v1/jobs/{purge_job_id}", headers=headers)
    assert job_get.status_code == 200, job_get.text
    assert str(job_get.json()["data"]["status"]).lower() == "completed"

    # Dataset-scoped artifacts are removed from the index.
    arts = await client.get(f"/api/v1/artifacts?dataset_id={dataset_id}", headers=headers)
    assert arts.status_code == 200, arts.text
    assert arts.json()["data"] == []

    # Storage paths are deleted (local backend).
    obj = get_object_store()
    assert not obj.exists(ds_path)
    for p in ver_paths:
        assert not obj.exists(str(p))
    for mp, dp in art_paths:
        assert not obj.exists(str(mp))
        if dp:
            assert not obj.exists(str(dp))

    # Metadata rows are hard-deleted.
    async with db_manager.session() as session:
        assert (await session.execute(select(Dataset).where(Dataset.id == dataset_id))).scalars().first() is None
        assert (await session.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id))).scalars().first() is None
        assert (await session.execute(select(Artifact).where(Artifact.dataset_id == dataset_id))).scalars().first() is None
        # Purge job should remain but no longer be FK-linked to the dataset.
        job = (await session.execute(select(Job).where(Job.id == purge_job_id))).scalars().first()
        assert job is not None
        assert job.dataset_id is None

