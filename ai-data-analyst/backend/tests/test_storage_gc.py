from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from app.core.config import settings
from app.models import Artifact, ArtifactType, Dataset, DatasetStatus, DatasetVersion, User
from app.services.database import db_manager
from app.services.storage_gc import StorageGCService


@pytest_asyncio.fixture(scope="module")
async def gc_context(tmp_path_factory):
    base = tmp_path_factory.mktemp("storage_gc")
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
        await db_manager.create_tables()

        # Seed DB with user + dataset + version + artifact.
        user_id = uuid4()
        ds_id = uuid4()
        version_id = uuid4()
        artifact_id = uuid4()

        # Live files
        dataset_path = uploads_dir / "dataset.csv"
        dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
        version_path = uploads_dir / "dataset_v1.parquet"
        version_path.write_text("parquet", encoding="utf-8")
        artifact_manifest = artifacts_dir / "manifest.json"
        artifact_data = artifacts_dir / "report.md"
        artifact_manifest.write_text('{"data_path": "%s"}' % artifact_data.as_posix(), encoding="utf-8")
        artifact_data.write_text("# Report", encoding="utf-8")

        async with db_manager.session() as session:
            session.add(
                User(
                    id=user_id,
                    email="gc@example.com",
                    hashed_password="x",
                    full_name="GC Tester",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                )
            )
            session.add(
                Dataset(
                    id=ds_id,
                    name="GC Dataset",
                    description=None,
                    original_filename="dataset.csv",
                    file_path=str(dataset_path),
                    file_size_bytes=dataset_path.stat().st_size,
                    file_format="csv",
                    status=DatasetStatus.READY,
                    owner_id=user_id,
                )
            )
            session.add(
                DatasetVersion(
                    id=version_id,
                    dataset_id=ds_id,
                    owner_id=user_id,
                    version_hash="v1",
                    file_path=str(version_path),
                    file_format="parquet",
                    file_size_bytes=version_path.stat().st_size,
                )
            )
            session.add(
                Artifact(
                    id=artifact_id,
                    owner_id=user_id,
                    dataset_id=ds_id,
                    artifact_type=ArtifactType.REPORT,
                    name="GC Report",
                    manifest_path=str(artifact_manifest),
                    data_path=str(artifact_data),
                    preview={},
                )
            )
            await session.commit()

        # Orphan files
        orphan_upload = uploads_dir / "orphan.csv"
        orphan_upload.write_text("x,y\n3,4\n", encoding="utf-8")
        orphan_artifact = artifacts_dir / "orphan.json"
        orphan_artifact.write_text("{}", encoding="utf-8")

        yield {
            "user_id": user_id,
            "dataset_path": dataset_path,
            "version_path": version_path,
            "artifact_manifest": artifact_manifest,
            "artifact_data": artifact_data,
            "orphan_upload": orphan_upload,
            "orphan_artifact": orphan_artifact,
        }
    finally:
        await db_manager.close()
        settings.database.url = old_db_url
        settings.upload_directory = old_upload_dir
        settings.artifact_directory = old_artifact_dir


@pytest.mark.asyncio
async def test_storage_gc_deletes_orphans(gc_context):
    async with db_manager.session() as session:
        service = StorageGCService(session)
        result = await service.run(dry_run=False, min_age_days=0)

    assert result["deleted_files"] >= 2

    assert gc_context["dataset_path"].exists()
    assert gc_context["version_path"].exists()
    assert gc_context["artifact_manifest"].exists()
    assert gc_context["artifact_data"].exists()

    assert not gc_context["orphan_upload"].exists()
    assert not gc_context["orphan_artifact"].exists()
