from __future__ import annotations

import json
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

    def _path_for(self, artifact_id: UUID, suffix: str) -> Path:
        subdir = self.base_dir / artifact_id.hex[:2] / artifact_id.hex[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{artifact_id.hex}{suffix}"

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
    ) -> ArtifactRef:
        artifact_id = uuid4()
        safe_value = to_jsonable(value)
        safe_details = to_jsonable(details or {})
        safe_operator_params = to_jsonable(operator_params or {})
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
            "created_at": datetime.utcnow().isoformat(),
        }
        path = self._path_for(artifact_id, ".json")
        path.write_text(_safe_json_dumps(payload), encoding="utf-8")
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.METRIC,
            name=name,
            created_at=payload["created_at"],
            storage_path=str(path),
            preview={"value": safe_value, "unit": unit, "details": safe_details},
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

        context = LogContext(component="ArtifactStore", operation="write_table")
        try:
            data_path: Path
            if storage_format == TableStorageFormat.PARQUET:
                data_path = self._path_for(artifact_id, ".parquet")
                df.to_parquet(data_path, index=False)
            elif storage_format == TableStorageFormat.CSV:
                data_path = self._path_for(artifact_id, ".csv")
                df.to_csv(data_path, index=False)
            else:
                payload = {
                    "type": ArtifactType.TABLE.value,
                    "name": name,
                    "created_at": created_at,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "data": to_jsonable(df.to_dict(orient="records")),
                }
                data_path = self._path_for(artifact_id, ".data.json")
                data_path.write_text(_safe_json_dumps(payload), encoding="utf-8")
        except Exception as e:
            logger.error(
                "Failed to write artifact table",
                context=context,
                artifact_id=str(artifact_id),
                format=storage_format.value,
                error=str(e),
            )
            raise DataProcessingException(f"Failed to write table artifact: {e}") from e

        manifest = {
            "type": ArtifactType.TABLE.value,
            "name": name,
            "created_at": created_at,
            "data_format": storage_format.value,
            "data_path": str(data_path),
            "preview": preview,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "dataset_version": dataset_version,
            "operator_name": operator_name,
            "operator_params": to_jsonable(operator_params or {}),
        }
        manifest_path = self._path_for(artifact_id, ".json")
        manifest_path.write_text(_safe_json_dumps(manifest), encoding="utf-8")

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
    ) -> ArtifactRef:
        artifact_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        safe_spec = to_jsonable(spec)
        safe_operator_params = to_jsonable(operator_params or {})
        payload = {
            "type": ArtifactType.CHART.value,
            "name": name,
            "created_at": created_at,
            "spec": safe_spec,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "dataset_version": dataset_version,
            "operator_name": operator_name,
            "operator_params": safe_operator_params,
        }
        path = self._path_for(artifact_id, ".json")
        path.write_text(_safe_json_dumps(payload), encoding="utf-8")
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.CHART,
            name=name,
            created_at=created_at,
            storage_path=str(path),
            preview={"spec": safe_spec},
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            operator_name=operator_name,
            operator_params=safe_operator_params,
        )

    def read_manifest(self, artifact_id: UUID) -> dict[str, Any]:
        manifest_path = self._path_for(artifact_id, ".json")
        if not manifest_path.exists():
            raise DataProcessingException(f"Artifact manifest missing: {artifact_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def read_json_path(self, artifact_path: str) -> dict[str, Any]:
        path = Path(artifact_path)
        if not path.exists():
            raise DataProcessingException(f"Artifact file missing: {artifact_path}")
        return json.loads(path.read_text(encoding="utf-8"))
