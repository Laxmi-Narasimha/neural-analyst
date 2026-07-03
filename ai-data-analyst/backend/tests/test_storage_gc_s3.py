from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from app.core.config import ObjectStoreBackend, settings
from app.models import Artifact, ArtifactType, Dataset, DatasetStatus, DatasetVersion, User
from app.services.database import db_manager
from app.services.object_store import S3ObjectInfo
from app.services.storage_gc import StorageGCService


class _FakeS3ObjectStore:
    def __init__(self, objects: list[S3ObjectInfo], manifest_payloads: dict[str, bytes]) -> None:
        self.backend = ObjectStoreBackend.S3
        self._objects = list(objects)
        self._manifest_payloads = dict(manifest_payloads)
        self.deleted: list[str] = []

    def list_s3_objects(self, *, prefixes=None, max_keys: int = 200_000):
        return list(self._objects)[: int(max_keys)]

    def delete(self, storage_path: str) -> bool:
        self.deleted.append(str(storage_path))
        return True

    def read_bytes(self, storage_path: str) -> bytes:
        return self._manifest_payloads.get(str(storage_path), b"{}")


@pytest_asyncio.fixture(scope="module")
async def gc_s3_context(tmp_path_factory):
    base = tmp_path_factory.mktemp("storage_gc_s3")
    db_file = base / "test.db"

    old_db_url = settings.database.url
    old_upload_dir = settings.upload_directory
    old_artifact_dir = settings.artifact_directory

    try:
        settings.database.url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
        settings.upload_directory = base / "uploads"
        settings.artifact_directory = base / "artifacts"
        Path(settings.upload_directory).mkdir(parents=True, exist_ok=True)
        Path(settings.artifact_directory).mkdir(parents=True, exist_ok=True)

        await db_manager.close()
        await db_manager.create_tables()

        user_id = uuid4()
        ds_id = uuid4()
        version_id = uuid4()
        artifact_id = uuid4()

        referenced_upload = "s3://bucket/neural-analyst/uploads/user/referenced.csv"
        referenced_version = "s3://bucket/neural-analyst/uploads/user/referenced_v1.parquet"
        referenced_manifest = "s3://bucket/neural-analyst/artifacts/user/manifest.json"
        referenced_data = "s3://bucket/neural-analyst/artifacts/user/report.md"

        async with db_manager.session() as session:
            session.add(
                User(
                    id=user_id,
                    email="gc-s3@example.com",
                    hashed_password="x",
                    full_name="GC S3 Tester",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                )
            )
            session.add(
                Dataset(
                    id=ds_id,
                    name="GC Dataset S3",
                    description=None,
                    original_filename="dataset.csv",
                    file_path=referenced_upload,
                    file_size_bytes=123,
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
                    file_path=referenced_version,
                    file_format="parquet",
                    file_size_bytes=456,
                )
            )
            session.add(
                Artifact(
                    id=artifact_id,
                    owner_id=user_id,
                    dataset_id=ds_id,
                    artifact_type=ArtifactType.REPORT,
                    name="GC S3 Report",
                    manifest_path=referenced_manifest,
                    data_path=None,
                    preview={},
                )
            )
            await session.commit()

        now = time.time()
        orphan_old = S3ObjectInfo(
            uri="s3://bucket/neural-analyst/artifacts/user/orphan_old.json",
            key="neural-analyst/artifacts/user/orphan_old.json",
            size_bytes=10,
            last_modified_epoch=now - (10 * 86400),
        )
        orphan_recent = S3ObjectInfo(
            uri="s3://bucket/neural-analyst/artifacts/user/orphan_recent.json",
            key="neural-analyst/artifacts/user/orphan_recent.json",
            size_bytes=10,
            last_modified_epoch=now - (1 * 86400),
        )
        fake_store = _FakeS3ObjectStore(
            objects=[
                S3ObjectInfo(
                    uri=referenced_upload,
                    key="neural-analyst/uploads/user/referenced.csv",
                    size_bytes=1,
                    last_modified_epoch=now - (20 * 86400),
                ),
                S3ObjectInfo(
                    uri=referenced_version,
                    key="neural-analyst/uploads/user/referenced_v1.parquet",
                    size_bytes=1,
                    last_modified_epoch=now - (20 * 86400),
                ),
                S3ObjectInfo(
                    uri=referenced_manifest,
                    key="neural-analyst/artifacts/user/manifest.json",
                    size_bytes=1,
                    last_modified_epoch=now - (20 * 86400),
                ),
                S3ObjectInfo(
                    uri=referenced_data,
                    key="neural-analyst/artifacts/user/report.md",
                    size_bytes=1,
                    last_modified_epoch=now - (20 * 86400),
                ),
                orphan_old,
                orphan_recent,
            ],
            manifest_payloads={
                referenced_manifest: ('{"data_path": "%s"}' % referenced_data).encode("utf-8"),
            },
        )

        yield {"fake_store": fake_store, "orphan_old_uri": orphan_old.uri, "orphan_recent_uri": orphan_recent.uri}
    finally:
        await db_manager.close()
        settings.database.url = old_db_url
        settings.upload_directory = old_upload_dir
        settings.artifact_directory = old_artifact_dir


@pytest.mark.asyncio
async def test_storage_gc_s3_deletes_only_orphans(gc_s3_context, monkeypatch):
    fake_store = gc_s3_context["fake_store"]
    monkeypatch.setattr("app.services.storage_gc.get_object_store", lambda: fake_store)

    async with db_manager.session() as session:
        service = StorageGCService(session)
        result = await service.run(dry_run=False, min_age_days=7, include_s3=True, s3_max_scan=1000)

    assert result["scanned_s3_objects"] >= 6
    assert result["deleted_s3_objects"] == 1
    assert gc_s3_context["orphan_old_uri"] in fake_store.deleted
    assert gc_s3_context["orphan_recent_uri"] not in fake_store.deleted
