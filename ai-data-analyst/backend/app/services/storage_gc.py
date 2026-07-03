from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models import Artifact as ArtifactModel
from app.models import Dataset as DatasetModel
from app.models import DatasetVersion as DatasetVersionModel
from app.services.object_store import ObjectStoreBackend, get_object_store, is_s3_uri

logger = get_logger(__name__)


class StorageGCService:
    """
    Best-effort storage garbage collector for managed uploads/artifacts.

    Safety rules:
    - Local: only deletes files under managed roots (uploads/artifacts).
    - S3: only scans/deletes objects under managed prefixes (uploads/artifacts).
    - Only deletes files older than min_age_days (default 7).
    - Skips any path referenced by a non-deleted DB record.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._obj = get_object_store()

    async def run(
        self,
        *,
        dry_run: bool = True,
        min_age_days: int = 7,
        include_cache: bool = False,
        include_s3: bool = True,
        s3_max_scan: int = 200_000,
        max_examples: int = 25,
    ) -> dict[str, Any]:
        min_age_days = int(min_age_days or 0)
        dry_run = bool(dry_run)
        include_cache = bool(include_cache)
        include_s3 = bool(include_s3)
        s3_max_scan = int(max(1, s3_max_scan))

        local_referenced, s3_referenced = await self._collect_referenced_paths()
        roots = self._roots(include_cache=include_cache)

        now = time.time()
        min_age_seconds = min_age_days * 86400

        scanned_local = 0
        scanned_s3 = 0
        deleted_local = 0
        deleted_s3 = 0
        skipped_active_local = 0
        skipped_active_s3 = 0
        skipped_recent_local = 0
        skipped_recent_s3 = 0
        failed_local = 0
        failed_s3 = 0
        deleted_examples: list[str] = []
        skipped_examples: list[str] = []
        skipped_recent_examples: list[str] = []

        for root in roots:
            for p in self._iter_files(root):
                scanned_local += 1
                norm = self._norm_local(p)
                if norm in local_referenced:
                    skipped_active_local += 1
                    if len(skipped_examples) < max_examples:
                        skipped_examples.append(str(p))
                    continue

                if min_age_seconds > 0:
                    try:
                        mtime = p.stat().st_mtime
                        if (now - mtime) < min_age_seconds:
                            skipped_recent_local += 1
                            if len(skipped_recent_examples) < max_examples:
                                skipped_recent_examples.append(str(p))
                            continue
                    except Exception:
                        # If we cannot stat, err on the side of safety.
                        skipped_recent_local += 1
                        continue

                if dry_run:
                    deleted_local += 1
                    if len(deleted_examples) < max_examples:
                        deleted_examples.append(str(p))
                    continue

                try:
                    ok = self._obj.delete(str(p))
                    if ok:
                        deleted_local += 1
                        if len(deleted_examples) < max_examples:
                            deleted_examples.append(str(p))
                    else:
                        failed_local += 1
                except Exception:
                    failed_local += 1

        if include_s3 and self._obj.backend == ObjectStoreBackend.S3:
            try:
                objects = self._obj.list_s3_objects(prefixes=["uploads", "artifacts"], max_keys=s3_max_scan)
            except Exception as e:
                logger.warning("S3 GC list failed", error=str(e))
                objects = []
                failed_s3 += 1

            for obj in objects:
                uri = str(obj.uri or "").strip()
                if not uri:
                    continue
                scanned_s3 += 1

                if uri in s3_referenced:
                    skipped_active_s3 += 1
                    if len(skipped_examples) < max_examples:
                        skipped_examples.append(uri)
                    continue

                if min_age_seconds > 0 and obj.last_modified_epoch is not None:
                    if (now - float(obj.last_modified_epoch)) < min_age_seconds:
                        skipped_recent_s3 += 1
                        if len(skipped_recent_examples) < max_examples:
                            skipped_recent_examples.append(uri)
                        continue

                if dry_run:
                    deleted_s3 += 1
                    if len(deleted_examples) < max_examples:
                        deleted_examples.append(uri)
                    continue

                try:
                    ok = self._obj.delete(uri)
                    if ok:
                        deleted_s3 += 1
                        if len(deleted_examples) < max_examples:
                            deleted_examples.append(uri)
                    else:
                        failed_s3 += 1
                except Exception:
                    failed_s3 += 1

        scanned = scanned_local + scanned_s3
        deleted = deleted_local + deleted_s3
        skipped_active = skipped_active_local + skipped_active_s3
        skipped_recent = skipped_recent_local + skipped_recent_s3
        failed = failed_local + failed_s3

        return {
            "dry_run": dry_run,
            "min_age_days": min_age_days,
            "include_s3": include_s3,
            "roots": [str(r) for r in roots],
            "scanned_files": int(scanned),
            "deleted_files": int(deleted),
            "skipped_active": int(skipped_active),
            "skipped_recent": int(skipped_recent),
            "failed": int(failed),
            "scanned_local_files": int(scanned_local),
            "scanned_s3_objects": int(scanned_s3),
            "deleted_local_files": int(deleted_local),
            "deleted_s3_objects": int(deleted_s3),
            "skipped_active_local": int(skipped_active_local),
            "skipped_active_s3": int(skipped_active_s3),
            "skipped_recent_local": int(skipped_recent_local),
            "skipped_recent_s3": int(skipped_recent_s3),
            "failed_local": int(failed_local),
            "failed_s3": int(failed_s3),
            "deleted_examples": deleted_examples,
            "skipped_examples": skipped_examples,
            "skipped_recent_examples": skipped_recent_examples,
        }

    async def _collect_referenced_paths(self) -> tuple[set[str], set[str]]:
        local_paths: set[str] = set()
        s3_paths: set[str] = set()

        def _add(sp: str | None) -> None:
            if not sp:
                return
            if is_s3_uri(sp):
                s3_paths.add(str(sp).strip())
                return
            local_norm = self._norm_local(Path(sp))
            if local_norm:
                local_paths.add(local_norm)

        ds_paths = (
            await self._session.execute(
                select(DatasetModel.file_path).where(DatasetModel.is_deleted == False)  # noqa: E712
            )
        ).scalars().all()
        for p in ds_paths:
            _add(p)

        version_paths = (
            await self._session.execute(
                select(DatasetVersionModel.file_path).where(DatasetVersionModel.is_deleted == False)  # noqa: E712
            )
        ).scalars().all()
        for p in version_paths:
            _add(p)

        artifacts = (
            await self._session.execute(
                select(ArtifactModel.manifest_path, ArtifactModel.data_path).where(ArtifactModel.is_deleted == False)  # noqa: E712
            )
        ).all()
        for manifest_path, data_path in artifacts:
            _add(manifest_path)
            _add(data_path)
            # If data_path is missing in DB, attempt to read manifest for it.
            if data_path is None and manifest_path:
                for dp in self._data_paths_from_manifest(str(manifest_path)):
                    _add(dp)

        return local_paths, s3_paths

    def _data_paths_from_manifest(self, manifest_path: str) -> list[str]:
        try:
            raw = self._obj.read_bytes(manifest_path)
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        dp = payload.get("data_path")
        if dp:
            return [str(dp)]
        return []

    def _roots(self, *, include_cache: bool) -> list[Path]:
        roots: list[Path] = []
        try:
            roots.append(Path(settings.upload_directory))
        except Exception:
            pass
        try:
            roots.append(Path(settings.artifact_directory))
        except Exception:
            pass
        if include_cache:
            try:
                roots.append(Path(settings.object_store.cache_dir))
            except Exception:
                pass
        out: list[Path] = []
        for r in roots:
            try:
                out.append(r.resolve(strict=False))
            except Exception:
                out.append(r)
        return out

    def _iter_files(self, root: Path) -> Iterable[Path]:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            return []
        for p in root.rglob("*"):
            try:
                if not p.is_file():
                    continue
                if p.name.endswith(".part"):
                    continue
                yield p
            except Exception:
                continue

    def _norm_local(self, path: Path) -> str:
        try:
            p = path.resolve(strict=False)
        except Exception:
            p = path
        try:
            return os.path.normcase(os.path.normpath(str(p)))
        except Exception:
            return str(p)
