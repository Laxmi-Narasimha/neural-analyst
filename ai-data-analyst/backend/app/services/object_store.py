from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional
from uuid import UUID

from app.core.config import ObjectStoreBackend, settings
from app.core.logging import LogContext, get_logger

logger = get_logger(__name__)


_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def is_s3_uri(value: str) -> bool:
    v = str(value or "").strip()
    return bool(_S3_URI_RE.match(v))


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str


@dataclass(frozen=True)
class S3ObjectInfo:
    uri: str
    key: str
    size_bytes: int
    last_modified_epoch: float | None


def parse_s3_uri(uri: str) -> S3Location:
    m = _S3_URI_RE.match(str(uri or "").strip())
    if not m:
        raise ValueError("not an s3 uri")
    return S3Location(bucket=str(m.group(1)), key=str(m.group(2)))


def build_s3_uri(*, bucket: str, key: str) -> str:
    b = str(bucket or "").strip()
    k = str(key or "").lstrip("/")
    if not b:
        raise ValueError("bucket is required")
    if not k:
        raise ValueError("key is required")
    return f"s3://{b}/{k}"


def _safe_basename(value: str) -> str:
    base = str(value or "").strip()
    base = base.replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return base or "upload"


def _guess_ext_from_name(name: str) -> str:
    try:
        ext = Path(str(name or "")).suffix
        if ext and len(ext) <= 10:
            return ext
    except Exception:
        pass
    return ""


class ObjectStoreError(RuntimeError):
    pass


class ObjectStore:
    """
    Storage abstraction for uploads and artifacts.

    Design notes:
    - The app stores dataset and artifact file locations as strings.
      - Local backend stores absolute filesystem paths.
      - S3 backend stores `s3://bucket/key` URIs.
    - Compute/EDA parsers typically require seekable files. In S3 mode we download
      to a local cache directory and return a local path for the caller.
    """

    def __init__(self) -> None:
        self._backend = settings.object_store.backend
        self._cache_dir = Path(settings.object_store.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def backend(self) -> ObjectStoreBackend:
        return self._backend

    def _s3_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except Exception as e:
            raise ObjectStoreError("boto3 is required for S3 storage backend") from e

        cfg = None
        if bool(settings.object_store.s3_force_path_style):
            cfg = Config(s3={"addressing_style": "path"})

        kwargs = {
            "region_name": str(settings.object_store.s3_region or "us-east-1"),
            "endpoint_url": settings.object_store.s3_endpoint_url,
            "config": cfg,
        }
        # Allow default AWS credential chain when not explicitly configured.
        ak = settings.object_store.s3_access_key_id.get_secret_value() if settings.object_store.s3_access_key_id else None
        sk = (
            settings.object_store.s3_secret_access_key.get_secret_value()
            if settings.object_store.s3_secret_access_key
            else None
        )
        st = settings.object_store.s3_session_token.get_secret_value() if settings.object_store.s3_session_token else None
        if ak and sk:
            kwargs["aws_access_key_id"] = ak
            kwargs["aws_secret_access_key"] = sk
        if st:
            kwargs["aws_session_token"] = st

        self._client = boto3.client("s3", **{k: v for k, v in kwargs.items() if v is not None})
        return self._client

    def _require_s3_bucket(self) -> str:
        bucket = str(settings.object_store.s3_bucket or "").strip()
        if not bucket:
            raise ObjectStoreError("OBJECT_STORE_S3_BUCKET (or S3_BUCKET) is required when backend=s3")
        return bucket

    def _prefix(self) -> str:
        p = str(settings.object_store.s3_prefix or "").strip().strip("/")
        return p

    def _key(self, *parts: str) -> str:
        pieces = [str(p or "").strip().strip("/") for p in parts if str(p or "").strip().strip("/")]
        key = "/".join(pieces)
        if not key:
            raise ObjectStoreError("invalid empty object key")
        return key

    def s3_uri_for(self, *, key: str) -> str:
        bucket = self._require_s3_bucket()
        full_key = self._key(self._prefix(), key) if self._prefix() else self._key(key)
        return build_s3_uri(bucket=bucket, key=full_key)

    def _s3_prefixed_key(self, key: str) -> str:
        k = self._key(key)
        p = self._prefix()
        if p:
            return self._key(p, k)
        return k

    def _s3_uploads_key(self, *, owner_id: UUID, filename: str) -> str:
        safe_name = _safe_basename(filename)
        stamp = hashlib.sha256(os.urandom(32)).hexdigest()[:12]
        return self._key("uploads", f"{owner_id}", f"{stamp}_{safe_name}")

    def _s3_artifacts_key(self, *, rel_key: str) -> str:
        return self._key("artifacts", rel_key)

    def list_s3_objects(
        self,
        *,
        prefixes: Optional[list[str]] = None,
        max_keys: int = 200_000,
    ) -> list[S3ObjectInfo]:
        """
        List managed S3 objects for orphan cleanup and maintenance.

        prefixes are relative to the app namespace (e.g., uploads/, artifacts/).
        """
        if self._backend != ObjectStoreBackend.S3:
            return []

        bucket = self._require_s3_bucket()
        client = self._s3_client()
        max_keys = int(max(1, max_keys))

        pref = prefixes or ["uploads", "artifacts"]
        base_prefixes: list[str] = []
        for p in pref:
            raw = str(p or "").strip().strip("/")
            if not raw:
                continue
            base_prefixes.append(self._s3_prefixed_key(raw))

        out: list[S3ObjectInfo] = []
        for key_prefix in base_prefixes:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
                for item in page.get("Contents", []) or []:
                    k = str(item.get("Key") or "").strip()
                    if not k:
                        continue
                    lm = item.get("LastModified")
                    lm_ts = None
                    try:
                        lm_ts = float(lm.timestamp()) if lm is not None else None
                    except Exception:
                        lm_ts = None
                    out.append(
                        S3ObjectInfo(
                            uri=build_s3_uri(bucket=bucket, key=k),
                            key=k,
                            size_bytes=int(item.get("Size") or 0),
                            last_modified_epoch=lm_ts,
                        )
                    )
                    if len(out) >= max_keys:
                        return out

        return out

    def list_s3_uris(
        self,
        *,
        prefixes: Optional[list[str]] = None,
        max_keys: int = 200_000,
    ) -> list[str]:
        return [obj.uri for obj in self.list_s3_objects(prefixes=prefixes, max_keys=max_keys)]

    def storage_health(self) -> dict[str, object]:
        """
        Best-effort health check for configured storage backend.
        """
        if self._backend == ObjectStoreBackend.LOCAL:
            checks: dict[str, str] = {}
            ok = True
            for name, root in (
                ("uploads", settings.upload_directory),
                ("artifacts", settings.artifact_directory),
                ("cache", settings.object_store.cache_dir),
            ):
                try:
                    p = Path(root)
                    p.mkdir(parents=True, exist_ok=True)
                    checks[name] = "healthy"
                except Exception:
                    checks[name] = "unhealthy"
                    ok = False
            return {"backend": "local", "status": "healthy" if ok else "unhealthy", "checks": checks}

        bucket = self._require_s3_bucket()
        try:
            self._s3_client().head_bucket(Bucket=bucket)
            return {"backend": "s3", "status": "healthy", "bucket": bucket}
        except Exception as e:
            return {"backend": "s3", "status": "unhealthy", "bucket": bucket, "detail": str(e)}

    def _cache_path_for_uri(self, uri: str, *, filename_hint: Optional[str] = None) -> Path:
        # Stable cache key based on uri, with an optional extension for nicer tooling.
        h = hashlib.sha256(str(uri).encode("utf-8")).hexdigest()
        ext = _guess_ext_from_name(filename_hint or uri)
        sub = self._cache_dir / h[:2] / h[2:4]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{h}{ext}"

    def _managed_roots(self) -> list[Path]:
        roots: list[Path] = []
        try:
            roots.append(Path(settings.upload_directory))
        except Exception:
            pass
        try:
            roots.append(Path(settings.artifact_directory))
        except Exception:
            pass
        try:
            roots.append(Path(settings.ml.model_storage_path))
        except Exception:
            pass
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

    def _within_root(self, path: Path, root: Path) -> bool:
        try:
            p = path.resolve(strict=False)
            r = root.resolve(strict=False)
            return p == r or r in p.parents
        except Exception:
            # Fallback to case-insensitive string prefix check (Windows-friendly).
            try:
                p = Path(os.path.normcase(os.path.normpath(str(path))))
                r = Path(os.path.normcase(os.path.normpath(str(root))))
                return str(p).startswith(str(r))
            except Exception:
                return False

    def exists(self, storage_path: str) -> bool:
        p = str(storage_path or "").strip()
        if not p:
            return False
        if is_s3_uri(p):
            loc = parse_s3_uri(p)
            try:
                self._s3_client().head_object(Bucket=loc.bucket, Key=loc.key)
                return True
            except Exception:
                return False
        return Path(p).exists()

    def delete(self, storage_path: str) -> bool:
        """
        Best-effort deletion of an upload/artifact path.

        Safety:
        - Local paths are only deleted if they live under managed roots
          (uploads/, artifacts/, storage_cache/). This avoids accidental deletion
          of arbitrary filesystem paths.
        """
        sp = str(storage_path or "").strip()
        if not sp:
            return False

        if is_s3_uri(sp):
            loc = parse_s3_uri(sp)
            try:
                self._s3_client().delete_object(Bucket=loc.bucket, Key=loc.key)
            except Exception as e:
                raise ObjectStoreError(f"failed to delete {sp}: {e}") from e
            # Best-effort: also remove any cached local copy.
            try:
                cached = self._cache_path_for_uri(sp)
                cached.unlink(missing_ok=True)
            except Exception:
                pass
            return True

        p = Path(sp)
        roots = self._managed_roots()
        if not any(self._within_root(p, r) for r in roots):
            logger.warning("Refusing to delete path outside managed roots", path=str(p))
            return False

        try:
            if p.is_dir():
                # Only allow deleting directories that are not the root itself.
                if any(self._within_root(p, r) and p.resolve(strict=False) == r.resolve(strict=False) for r in roots):
                    logger.warning("Refusing to delete managed root directory", path=str(p))
                    return False
                shutil.rmtree(p, ignore_errors=True)
                return True
            p.unlink(missing_ok=True)
            return True
        except Exception as e:
            raise ObjectStoreError(f"failed to delete local path {p}: {e}") from e

    def delete_many(self, paths: list[str]) -> dict[str, int]:
        deleted = 0
        skipped = 0
        failed = 0
        for sp in paths:
            try:
                ok = self.delete(sp)
                if ok:
                    deleted += 1
                else:
                    skipped += 1
            except Exception:
                failed += 1
        return {"deleted": int(deleted), "skipped": int(skipped), "failed": int(failed)}

    def prune_cache(self) -> dict[str, int]:
        """
        Best-effort pruning of the local cache dir used for remote objects.

        Rules:
        - TTL pruning first (cache_ttl_days)
        - then size-based pruning (cache_max_bytes), deleting oldest files
        """
        root = Path(settings.object_store.cache_dir)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            return {"deleted_files": 0, "deleted_bytes": 0}

        ttl_days = int(getattr(settings.object_store, "cache_ttl_days", 0) or 0)
        max_bytes = int(getattr(settings.object_store, "cache_max_bytes", 0) or 0)
        ttl_seconds = ttl_days * 86400

        now = time.time()
        files: list[tuple[float, int, Path]] = []
        total_bytes = 0
        for p in root.rglob("*"):
            try:
                if not p.is_file():
                    continue
                if p.name.endswith(".part"):
                    continue
                st = p.stat()
                mtime = float(st.st_mtime)
                size = int(st.st_size)
                files.append((mtime, size, p))
                total_bytes += size
            except Exception:
                continue

        deleted_files = 0
        deleted_bytes = 0

        # Age-based pruning
        if ttl_seconds > 0:
            for mtime, size, p in sorted(files, key=lambda x: x[0]):
                try:
                    if (now - mtime) <= ttl_seconds:
                        continue
                    p.unlink(missing_ok=True)
                    deleted_files += 1
                    deleted_bytes += size
                    total_bytes -= size
                except Exception:
                    continue

        # Size-based pruning
        if max_bytes > 0 and total_bytes > max_bytes:
            # Rebuild list of remaining files.
            remaining: list[tuple[float, int, Path]] = []
            for p in root.rglob("*"):
                try:
                    if p.is_file() and not p.name.endswith(".part"):
                        st = p.stat()
                        remaining.append((float(st.st_mtime), int(st.st_size), p))
                except Exception:
                    continue
            for mtime, size, p in sorted(remaining, key=lambda x: x[0]):
                if total_bytes <= max_bytes:
                    break
                try:
                    p.unlink(missing_ok=True)
                    deleted_files += 1
                    deleted_bytes += size
                    total_bytes -= size
                except Exception:
                    continue

        return {"deleted_files": int(deleted_files), "deleted_bytes": int(deleted_bytes)}

    def ensure_local_path(
        self,
        storage_path: str,
        *,
        expected_size_bytes: Optional[int] = None,
        filename_hint: Optional[str] = None,
    ) -> Path:
        """
        Return a local filesystem path for a storage_path.

        - local backend: returns Path(storage_path)
        - s3 uri: downloads into cache_dir (best-effort reuse) and returns cached path
        """
        sp = str(storage_path or "").strip()
        if not sp:
            raise ObjectStoreError("missing storage_path")

        if not is_s3_uri(sp):
            return Path(sp)

        loc = parse_s3_uri(sp)
        dest = self._cache_path_for_uri(sp, filename_hint=filename_hint)

        try:
            if dest.exists():
                if expected_size_bytes is None:
                    return dest
                try:
                    if int(dest.stat().st_size) == int(expected_size_bytes):
                        return dest
                except Exception:
                    pass
        except Exception:
            pass

        tmp = Path(str(dest) + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)

        client = self._s3_client()
        context = LogContext(component="ObjectStore", operation="ensure_local_path")
        try:
            # Ensure any previous partial file doesn't get used accidentally.
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

            client.download_file(loc.bucket, loc.key, str(tmp))
            os.replace(str(tmp), str(dest))
            return dest
        except Exception as e:
            logger.error(
                "Failed to download S3 object to local cache",
                context=context,
                bucket=loc.bucket,
                key=loc.key,
                error=str(e),
            )
            raise ObjectStoreError(f"failed to download {sp}") from e
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def put_upload(
        self,
        *,
        owner_id: UUID,
        original_filename: str,
        src: BinaryIO,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Persist an uploaded file and return its storage_path string.

        Callers must position src at the beginning of the file.
        """
        context = LogContext(component="ObjectStore", operation="put_upload")
        if self._backend == ObjectStoreBackend.LOCAL:
            upload_dir = Path(settings.upload_directory)
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_basename(original_filename)
            # Preserve existing on-disk behavior: a unique filename under uploads/.
            nonce = hashlib.sha256(os.urandom(32)).hexdigest()[:12]
            path = upload_dir / f"{owner_id}_{nonce}_{safe_name}"
            with path.open("wb") as f:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return str(path)

        # S3 backend
        bucket = self._require_s3_bucket()
        key = self._s3_uploads_key(owner_id=owner_id, filename=original_filename)
        full_key = self._key(self._prefix(), key) if self._prefix() else key

        extra: dict[str, str] = {}
        if content_type:
            extra["ContentType"] = str(content_type)
        try:
            self._s3_client().upload_fileobj(src, bucket, full_key, ExtraArgs=extra or None)
        except TypeError:
            # boto3 requires ExtraArgs omitted when empty.
            self._s3_client().upload_fileobj(src, bucket, full_key)
        except Exception as e:
            logger.error("Upload to S3 failed", context=context, bucket=bucket, key=full_key, error=str(e))
            raise ObjectStoreError("upload failed") from e

        return build_s3_uri(bucket=bucket, key=full_key)

    def put_upload_file(
        self,
        *,
        owner_id: UUID,
        original_filename: str,
        local_path: Path,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Persist a local dataset file into the configured backend and return its storage_path string.

        - local backend: returns the given local_path unchanged
        - s3 backend: uploads to S3 under the uploads/ prefix and returns an s3:// uri
        """
        lp = Path(local_path)
        if not lp.exists():
            raise ObjectStoreError(f"local file missing: {lp}")

        if self._backend == ObjectStoreBackend.LOCAL:
            return str(lp)

        bucket = self._require_s3_bucket()
        key = self._s3_uploads_key(owner_id=owner_id, filename=original_filename)
        full_key = self._key(self._prefix(), key) if self._prefix() else key

        extra: dict[str, str] = {}
        if content_type:
            extra["ContentType"] = str(content_type)
        try:
            if extra:
                self._s3_client().upload_file(str(lp), bucket, full_key, ExtraArgs=extra)
            else:
                self._s3_client().upload_file(str(lp), bucket, full_key)
        except Exception as e:
            raise ObjectStoreError(f"failed to upload file: {e}") from e

        return build_s3_uri(bucket=bucket, key=full_key)

    def put_artifact_bytes(self, *, rel_key: str, data: bytes, content_type: Optional[str] = None) -> str:
        """
        Persist a small artifact blob (json/markdown) and return its storage_path string.

        rel_key is relative to the artifact root (no leading slash).
        """
        if self._backend == ObjectStoreBackend.LOCAL:
            p = Path(settings.artifact_directory) / self._key(rel_key)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return str(p)

        bucket = self._require_s3_bucket()
        rk = self._s3_artifacts_key(rel_key=rel_key)
        full_key = self._key(self._prefix(), rk) if self._prefix() else rk
        args: dict[str, str] = {}
        if content_type:
            args["ContentType"] = str(content_type)
        try:
            if args:
                self._s3_client().put_object(Bucket=bucket, Key=full_key, Body=data, **args)
            else:
                self._s3_client().put_object(Bucket=bucket, Key=full_key, Body=data)
        except Exception as e:
            raise ObjectStoreError(f"failed to write artifact bytes: {e}") from e
        return build_s3_uri(bucket=bucket, key=full_key)

    def put_artifact_file(
        self,
        *,
        rel_key: str,
        local_path: Path,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Persist a file-backed artifact (parquet/csv/html) and return its storage_path string.
        """
        lp = Path(local_path)
        if not lp.exists():
            raise ObjectStoreError(f"local artifact file missing: {lp}")

        if self._backend == ObjectStoreBackend.LOCAL:
            dst = Path(settings.artifact_directory) / self._key(rel_key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace to avoid partial reads.
            tmp = Path(str(dst) + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            # Stream copy to avoid loading large artifacts into memory.
            with lp.open("rb") as src, tmp.open("wb") as dst_f:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst_f.write(chunk)
            os.replace(str(tmp), str(dst))
            return str(dst)

        bucket = self._require_s3_bucket()
        rk = self._s3_artifacts_key(rel_key=rel_key)
        full_key = self._key(self._prefix(), rk) if self._prefix() else rk
        extra: dict[str, str] = {}
        if content_type:
            extra["ContentType"] = str(content_type)
        try:
            if extra:
                self._s3_client().upload_file(str(lp), bucket, full_key, ExtraArgs=extra)
            else:
                self._s3_client().upload_file(str(lp), bucket, full_key)
        except Exception as e:
            raise ObjectStoreError(f"failed to upload artifact file: {e}") from e
        return build_s3_uri(bucket=bucket, key=full_key)

    def read_bytes(self, storage_path: str) -> bytes:
        sp = str(storage_path or "").strip()
        if not sp:
            raise ObjectStoreError("missing storage_path")
        if is_s3_uri(sp):
            loc = parse_s3_uri(sp)
            try:
                resp = self._s3_client().get_object(Bucket=loc.bucket, Key=loc.key)
                body = resp.get("Body")
                return body.read() if body is not None else b""
            except Exception as e:
                raise ObjectStoreError(f"failed to read {sp}: {e}") from e
        return Path(sp).read_bytes()

    def presign_download_url(
        self,
        *,
        storage_path: str,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        expires_seconds: Optional[int] = None,
    ) -> Optional[str]:
        sp = str(storage_path or "").strip()
        if not sp or not is_s3_uri(sp):
            return None

        loc = parse_s3_uri(sp)
        exp = int(expires_seconds or settings.object_store.presign_expires_seconds)
        exp = max(60, min(exp, 7 * 24 * 3600))

        params: dict[str, str] = {"Bucket": loc.bucket, "Key": loc.key}
        if filename:
            params["ResponseContentDisposition"] = f'attachment; filename="{_safe_basename(filename)}"'
        if content_type:
            params["ResponseContentType"] = str(content_type)

        try:
            return self._s3_client().generate_presigned_url("get_object", Params=params, ExpiresIn=exp)
        except Exception as e:
            raise ObjectStoreError(f"failed to presign url: {e}") from e


_singleton: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    global _singleton
    if _singleton is None:
        _singleton = ObjectStore()
    return _singleton
