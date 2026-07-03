from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import pandas as pd

from app.core.config import settings
from app.core.exceptions import DataProcessingException
from app.core.logging import LogContext, get_logger
from app.core.serialization import to_jsonable
from app.services.object_store import ObjectStoreBackend, get_object_store, is_s3_uri

logger = get_logger(__name__)


class ArtifactType(str, Enum):
    METRIC = "metric"
    TABLE = "table"
    CHART = "chart"
    REPORT = "report"


class TableStorageFormat(str, Enum):
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: UUID
    artifact_type: ArtifactType
    name: str
    created_at: str
    storage_path: str
    preview: dict[str, Any]
    dataset_id: Optional[UUID] = None
    dataset_version: Optional[str] = None
    operator_name: Optional[str] = None
    operator_params: Optional[dict[str, Any]] = None


def _safe_json_dumps(obj: Any) -> str:
    try:
        import orjson

        return orjson.dumps(obj).decode("utf-8")
    except Exception:
        return json.dumps(obj, ensure_ascii=True, default=str)


class ArtifactStore:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or settings.artifact_directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._obj = get_object_store()

    def _rel_key_for(self, artifact_id: UUID, suffix: str) -> str:
        # Keep the existing deterministic 2-level shard layout, but represent it as a
        # forward-slash key so it also works for object storage.
        return f"{artifact_id.hex[:2]}/{artifact_id.hex[2:4]}/{artifact_id.hex}{suffix}"

    def _local_path_for_rel_key(self, rel_key: str) -> Path:
        return self.base_dir / str(rel_key).replace("/", os.sep)

    def _s3_uri_for_rel_key(self, rel_key: str) -> str:
        return self._obj.s3_uri_for(key=f"artifacts/{rel_key}")

    def _write_bytes(self, *, rel_key: str, data: bytes, content_type: Optional[str] = None) -> str:
        if self._obj.backend == ObjectStoreBackend.LOCAL:
            p = self._local_path_for_rel_key(rel_key)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return str(p)
        return self._obj.put_artifact_bytes(rel_key=rel_key, data=data, content_type=content_type)

    def _write_file(self, *, rel_key: str, local_path: Path, content_type: Optional[str] = None) -> str:
        lp = Path(local_path)
        if not lp.exists():
            raise DataProcessingException(f"Artifact temp file missing: {lp}")

        if self._obj.backend == ObjectStoreBackend.LOCAL:
            dst = self._local_path_for_rel_key(rel_key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(dst) + ".part")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                with lp.open("rb") as src, tmp.open("wb") as out:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                tmp.replace(dst)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
            return str(dst)

        return self._obj.put_artifact_file(rel_key=rel_key, local_path=lp, content_type=content_type)

    def write_metric(
        self,
        *,
        name: str,
        value: float | int | str | bool,
        unit: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        dataset_id: Optional[UUID] = None,
        dataset_version: Optional[str] = None,
        operator_name: Optional[str] = None,
        operator_params: Optional[dict[str, Any]] = None,
        operator_summary: Optional[dict[str, Any]] = None,
    ) -> ArtifactRef:
        artifact_id = uuid4()
        safe_value = to_jsonable(value)
        safe_details = to_jsonable(details or {})
        safe_operator_params = to_jsonable(operator_params or {})
        safe_operator_summary = to_jsonable(operator_summary or {}) if operator_summary is not None else None
        payload = {
            "type": ArtifactType.METRIC.value,
            "name": name,
            "value": safe_value,
            "unit": unit,
            "details": safe_details,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "dataset_version": dataset_version,
            "operator_name": operator_name,
            "operator_params": safe_operator_params,
            **({"operator_summary": safe_operator_summary} if safe_operator_summary is not None else {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        rel_key = self._rel_key_for(artifact_id, ".json")
        storage_path = self._write_bytes(
            rel_key=rel_key,
            data=_safe_json_dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        preview = {"value": safe_value, "unit": unit, "details": safe_details}
        if safe_operator_summary is not None:
            preview["operator_summary"] = safe_operator_summary
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.METRIC,
            name=name,
            created_at=payload["created_at"],
            storage_path=str(storage_path),
            preview=preview,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            operator_name=operator_name,
            operator_params=safe_operator_params,
        )

    def write_table(
        self,
        *,
        name: str,
        df: pd.DataFrame,
        preview_rows: int = 25,
        storage_format: TableStorageFormat = TableStorageFormat.PARQUET,
        dataset_id: Optional[UUID] = None,
        dataset_version: Optional[str] = None,
        operator_name: Optional[str] = None,
        operator_params: Optional[dict[str, Any]] = None,
        operator_summary: Optional[dict[str, Any]] = None,
    ) -> ArtifactRef:
        artifact_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        preview_df = df.head(preview_rows).copy()
        preview = to_jsonable(
            {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": [str(c) for c in df.columns.tolist()],
                "preview_rows": preview_df.to_dict(orient="records"),
            }
        )
        safe_operator_summary = to_jsonable(operator_summary or {}) if operator_summary is not None else None
        if safe_operator_summary is not None and isinstance(preview, dict):
            preview["operator_summary"] = safe_operator_summary

        context = LogContext(component="ArtifactStore", operation="write_table")
        effective_format = storage_format
        tmp_dir = Path(settings.object_store.cache_dir) / "artifact_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_data = tmp_dir / f"{artifact_id.hex}.data"

        try:
            data_rel_key: str
            data_storage_path: str

            if storage_format == TableStorageFormat.PARQUET:
                try:
                    tmp_data = tmp_data.with_suffix(".parquet")
                    df.to_parquet(tmp_data, index=False)
                    data_rel_key = self._rel_key_for(artifact_id, ".parquet")
                    data_storage_path = self._write_file(rel_key=data_rel_key, local_path=tmp_data)
                except Exception as e:
                    logger.warning(
                        "Parquet write failed; falling back to CSV",
                        context=context,
                        artifact_id=str(artifact_id),
                        error=str(e),
                    )
                    effective_format = TableStorageFormat.CSV
                    tmp_data = tmp_data.with_suffix(".csv")
                    df.to_csv(tmp_data, index=False)
                    data_rel_key = self._rel_key_for(artifact_id, ".csv")
                    data_storage_path = self._write_file(rel_key=data_rel_key, local_path=tmp_data, content_type="text/csv")
            elif storage_format == TableStorageFormat.CSV:
                tmp_data = tmp_data.with_suffix(".csv")
                df.to_csv(tmp_data, index=False)
                data_rel_key = self._rel_key_for(artifact_id, ".csv")
                data_storage_path = self._write_file(rel_key=data_rel_key, local_path=tmp_data, content_type="text/csv")
            else:
                payload = {
                    "type": ArtifactType.TABLE.value,
                    "name": name,
                    "created_at": created_at,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "data": to_jsonable(df.to_dict(orient="records")),
                }
                tmp_data = tmp_data.with_suffix(".json")
                tmp_data.write_text(_safe_json_dumps(payload), encoding="utf-8")
                data_rel_key = self._rel_key_for(artifact_id, ".data.json")
                data_storage_path = self._write_file(rel_key=data_rel_key, local_path=tmp_data, content_type="application/json")

        except Exception as e:
            logger.error(
                "Failed to write artifact table",
                context=context,
                artifact_id=str(artifact_id),
                format=storage_format.value,
                error=str(e),
            )
            raise DataProcessingException(f"Failed to write table artifact: {e}") from e
        finally:
            try:
                tmp_data.unlink(missing_ok=True)
            except Exception:
                pass

        manifest = {
            "type": ArtifactType.TABLE.value,
            "name": name,
            "created_at": created_at,
            "data_format": effective_format.value,
            "data_path": str(data_storage_path),
            "preview": preview,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "dataset_version": dataset_version,
            "operator_name": operator_name,
            "operator_params": to_jsonable(operator_params or {}),
            **({"operator_summary": safe_operator_summary} if safe_operator_summary is not None else {}),
        }
        manifest_rel_key = self._rel_key_for(artifact_id, ".json")
        manifest_path = self._write_bytes(
            rel_key=manifest_rel_key,
            data=_safe_json_dumps(manifest).encode("utf-8"),
            content_type="application/json",
        )

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.TABLE,
            name=name,
            created_at=created_at,
            storage_path=str(manifest_path),
            preview=preview,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            operator_name=operator_name,
            operator_params=to_jsonable(operator_params or {}),
        )

    def write_chart(
        self,
        *,
        name: str,
        spec: dict[str, Any],
        dataset_id: Optional[UUID] = None,
        dataset_version: Optional[str] = None,
        operator_name: Optional[str] = None,
        operator_params: Optional[dict[str, Any]] = None,
        operator_summary: Optional[dict[str, Any]] = None,
    ) -> ArtifactRef:
        artifact_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        safe_spec = to_jsonable(spec)
        safe_operator_params = to_jsonable(operator_params or {})
        safe_operator_summary = to_jsonable(operator_summary or {}) if operator_summary is not None else None
        payload = {
            "type": ArtifactType.CHART.value,
            "name": name,
            "created_at": created_at,
            "spec": safe_spec,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "dataset_version": dataset_version,
            "operator_name": operator_name,
            "operator_params": safe_operator_params,
            **({"operator_summary": safe_operator_summary} if safe_operator_summary is not None else {}),
        }
        rel_key = self._rel_key_for(artifact_id, ".json")
        storage_path = self._write_bytes(
            rel_key=rel_key,
            data=_safe_json_dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        preview = {"spec": safe_spec}
        if safe_operator_summary is not None:
            preview["operator_summary"] = safe_operator_summary
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.CHART,
            name=name,
            created_at=created_at,
            storage_path=str(storage_path),
            preview=preview,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            operator_name=operator_name,
            operator_params=safe_operator_params,
        )

    def write_report(
        self,
        *,
        name: str,
        content: str,
        report_format: str = "markdown",
        dataset_id: Optional[UUID] = None,
        dataset_version: Optional[str] = None,
        operator_name: Optional[str] = None,
        operator_params: Optional[dict[str, Any]] = None,
        operator_summary: Optional[dict[str, Any]] = None,
    ) -> ArtifactRef:
        artifact_id = uuid4()
        created_at = datetime.utcnow().isoformat()

        fmt = str(report_format or "markdown").lower().strip()
        if fmt not in {"markdown", "md", "html", "json", "text"}:
            fmt = "markdown"
        suffix = ".md" if fmt in {"markdown", "md"} else ".html" if fmt == "html" else ".json" if fmt == "json" else ".txt"

        safe_operator_params = to_jsonable(operator_params or {})
        safe_operator_summary = to_jsonable(operator_summary or {}) if operator_summary is not None else None
        tmp_dir = Path(settings.object_store.cache_dir) / "artifact_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_data = (tmp_dir / f"{artifact_id.hex}").with_suffix(suffix)
        try:
            tmp_data.write_text(str(content or ""), encoding="utf-8")
        except Exception as e:
            raise DataProcessingException(f"Failed to write report artifact: {e}") from e

        preview_text = str(content or "")
        preview = to_jsonable(
            {
                "format": fmt,
                "bytes": int(len(preview_text.encode("utf-8"))),
                "preview_text": preview_text[:2000],
            }
        )
        if safe_operator_summary is not None and isinstance(preview, dict):
            preview["operator_summary"] = safe_operator_summary

        ct = "text/plain"
        if suffix == ".html":
            ct = "text/html"
        elif suffix == ".json":
            ct = "application/json"

        data_rel_key = self._rel_key_for(artifact_id, suffix)
        data_path = self._write_file(rel_key=data_rel_key, local_path=tmp_data, content_type=ct)
        try:
            tmp_data.unlink(missing_ok=True)
        except Exception:
            pass

        manifest = {
            "type": ArtifactType.REPORT.value,
            "name": name,
            "created_at": created_at,
            "data_format": fmt,
            "data_path": str(data_path),
            "preview": preview,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "dataset_version": dataset_version,
            "operator_name": operator_name,
            "operator_params": safe_operator_params,
            **({"operator_summary": safe_operator_summary} if safe_operator_summary is not None else {}),
        }
        manifest_rel_key = self._rel_key_for(artifact_id, ".json")
        manifest_path = self._write_bytes(
            rel_key=manifest_rel_key,
            data=_safe_json_dumps(manifest).encode("utf-8"),
            content_type="application/json",
        )

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.REPORT,
            name=name,
            created_at=created_at,
            storage_path=str(manifest_path),
            preview=preview,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            operator_name=operator_name,
            operator_params=safe_operator_params,
        )

    def read_manifest(self, artifact_id: UUID) -> dict[str, Any]:
        rel_key = self._rel_key_for(artifact_id, ".json")
        local_path = self._local_path_for_rel_key(rel_key)

        candidates: list[str] = []
        if local_path.exists():
            candidates.append(str(local_path))
        try:
            candidates.append(self._s3_uri_for_rel_key(rel_key))
        except Exception:
            pass

        last_err: Exception | None = None
        for sp in candidates:
            try:
                raw = self._obj.read_bytes(sp)
                return json.loads(raw.decode("utf-8"))
            except Exception as e:
                last_err = e
                continue

        raise DataProcessingException(f"Artifact manifest missing: {artifact_id}") from last_err

    def read_json_path(self, artifact_path: str) -> dict[str, Any]:
        sp = str(artifact_path or "").strip()
        if not sp:
            raise DataProcessingException("Artifact file missing: empty path")
        try:
            raw = self._obj.read_bytes(sp) if is_s3_uri(sp) else Path(sp).read_bytes()
            return json.loads(raw.decode("utf-8"))
        except FileNotFoundError as e:
            raise DataProcessingException(f"Artifact file missing: {sp}") from e
        except Exception as e:
            raise DataProcessingException(f"Failed to read artifact json: {e}") from e
