# AI Enterprise Data Analyst - Model Registry
# MLflow-inspired model versioning, tracking, and deployment

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
from uuid import uuid4
import json
import os
import pickle
import hashlib

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)

_REGISTRY_FORMAT = 2


def _round_metric(value: Any) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return round(f, 4) if np.isfinite(f) else None


def _jsonable(value: Any) -> Any:
    """Make params/metrics safe for registry.json (numpy scalars, NaN, nested)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


# ============================================================================
# Registry Types
# ============================================================================

class ModelStage(str, Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(str, Enum):
    """Model status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


@dataclass
class ModelVersion:
    """Single model version."""
    
    version: int
    model_id: str
    stage: ModelStage
    status: ModelStatus
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    description: str = ""
    
    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    
    # Artifacts
    model_path: Optional[str] = None
    artifact_paths: list[str] = field(default_factory=list)
    
    # Lineage
    parent_version: Optional[int] = None
    training_data_hash: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "stage": self.stage.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "metrics": {k: _round_metric(v) for k, v in self.metrics.items()},
            "params": self.params
        }

    def to_storage_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data["metrics"] = {k: _jsonable(v) for k, v in self.metrics.items()}
        data["params"] = _jsonable(self.params)
        data.update(
            {
                "model_path": self.model_path,
                "artifact_paths": list(self.artifact_paths),
                "parent_version": self.parent_version,
                "training_data_hash": self.training_data_hash,
            }
        )
        return data

    @classmethod
    def from_storage_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        return cls(
            version=int(data["version"]),
            model_id=str(data.get("model_id", "")),
            stage=ModelStage(data.get("stage", ModelStage.DEVELOPMENT.value)),
            status=ModelStatus(data.get("status", ModelStatus.ACTIVE.value)),
            created_at=_parse_dt(data.get("created_at")),
            created_by=str(data.get("created_by", "system")),
            description=str(data.get("description", "")),
            metrics=dict(data.get("metrics") or {}),
            params=dict(data.get("params") or {}),
            model_path=data.get("model_path"),
            artifact_paths=list(data.get("artifact_paths") or []),
            parent_version=data.get("parent_version"),
            training_data_hash=data.get("training_data_hash"),
        )


@dataclass
class RegisteredModel:
    """Registered model with all versions."""
    
    model_id: str
    name: str
    description: str = ""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    tags: dict[str, str] = field(default_factory=dict)
    versions: list[ModelVersion] = field(default_factory=list)
    
    # Current production version
    production_version: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "version_count": len(self.versions),
            "production_version": self.production_version,
            "latest_version": self.versions[-1].version if self.versions else None
        }

    def to_storage_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data["versions"] = [v.to_storage_dict() for v in self.versions]
        return data

    @classmethod
    def from_storage_dict(cls, data: dict[str, Any]) -> "RegisteredModel":
        versions = [ModelVersion.from_storage_dict(v) for v in data.get("versions") or []]
        versions.sort(key=lambda v: v.version)
        return cls(
            model_id=str(data["model_id"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
            tags=dict(data.get("tags") or {}),
            versions=versions,
            production_version=data.get("production_version"),
        )


@dataclass
class Experiment:
    """ML experiment tracking."""
    
    experiment_id: str
    name: str
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    runs: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "run_count": len(self.runs)
        }


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    MLflow-inspired model registry.
    
    Features:
    - Model versioning
    - Stage transitions
    - Experiment tracking
    - Model serving
    """
    
    def __init__(self, storage_path: str | Path = "./model_registry"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._models: dict[str, RegisteredModel] = {}
        self._experiments: dict[str, Experiment] = {}

        self._load_registry()

    @property
    def storage_path(self) -> Path:
        return self._storage_path

    def _resolve_artifact(self, model_path: Optional[str]) -> Optional[Path]:
        if not model_path:
            return None
        p = Path(model_path)
        if p.is_absolute():
            return p
        # Stored relative to the registry folder; older entries were relative to the cwd.
        candidate = self._storage_path / p
        return candidate if candidate.exists() or not p.exists() else p

    def _recover_versions_from_disk(self, model: RegisteredModel) -> None:
        """Rebuild versions for entries written before versions were persisted."""
        model_dir = self._storage_path / model.name
        if not model_dir.is_dir():
            return
        for vdir in sorted(model_dir.glob("v*"), key=lambda d: d.name):
            artifact = vdir / "model.pkl"
            suffix = vdir.name[1:]
            if not artifact.exists() or not suffix.isdigit():
                continue
            model.versions.append(
                ModelVersion(
                    version=int(suffix),
                    model_id=model.model_id,
                    stage=ModelStage.DEVELOPMENT,
                    status=ModelStatus.ACTIVE,
                    created_at=datetime.utcfromtimestamp(artifact.stat().st_mtime),
                    description="recovered from registry folder",
                    model_path=str(artifact.relative_to(self._storage_path)),
                )
            )
        model.versions.sort(key=lambda v: v.version)

    def _load_registry(self) -> None:
        """Load registry from disk, including every saved version."""
        registry_file = self._storage_path / "registry.json"
        if not registry_file.exists():
            return
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read model registry {registry_file}: {e}")
            return
        for model_data in data.get("models", []):
            try:
                model = RegisteredModel.from_storage_dict(model_data)
            except Exception as e:
                logger.warning(f"Skipping unreadable registry entry {model_data.get('name')!r}: {e}")
                continue
            if not model.versions:
                self._recover_versions_from_disk(model)
            self._models[model.name] = model
        for exp_data in data.get("experiments", []):
            try:
                self._experiments[exp_data["name"]] = Experiment(
                    experiment_id=str(exp_data["experiment_id"]),
                    name=str(exp_data["name"]),
                    created_at=_parse_dt(exp_data.get("created_at")),
                    runs=list(exp_data.get("runs") or []),
                )
            except Exception:
                continue

    def _save_registry(self) -> None:
        """Write registry.json atomically so a crash never leaves it half-written."""
        registry_file = self._storage_path / "registry.json"
        data = {
            "format": _REGISTRY_FORMAT,
            "models": [m.to_storage_dict() for m in self._models.values()],
            "experiments": [
                {**e.to_dict(), "runs": _jsonable(e.runs)} for e in self._experiments.values()
            ],
        }
        tmp = registry_file.with_suffix(f".json.{uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, registry_file)
    
    def register_model(
        self,
        name: str,
        model: Any,
        metrics: dict[str, float] = None,
        params: dict[str, Any] = None,
        description: str = "",
        tags: dict[str, str] = None
    ) -> ModelVersion:
        """Register a new model or version."""
        # Get or create registered model
        if name not in self._models:
            self._models[name] = RegisteredModel(
                model_id=str(uuid4()),
                name=name,
                description=description,
                tags=tags or {}
            )
        
        registered = self._models[name]
        version_num = max((v.version for v in registered.versions), default=0) + 1

        # Save model artifact
        model_path = self._storage_path / name / f"v{version_num}"
        model_path.mkdir(parents=True, exist_ok=True)
        model_file = model_path / "model.pkl"

        payload = pickle.dumps(model)
        with open(model_file, 'wb') as f:
            f.write(payload)

        # Create version
        version = ModelVersion(
            version=version_num,
            model_id=registered.model_id,
            stage=ModelStage.DEVELOPMENT,
            status=ModelStatus.ACTIVE,
            metrics=metrics or {},
            params=params or {},
            model_path=str(model_file.relative_to(self._storage_path)),
            description=description,
            training_data_hash=(tags or {}).get("dataset_version") or None,
        )
        
        registered.versions.append(version)
        registered.updated_at = datetime.utcnow()
        
        self._save_registry()
        
        logger.info(f"Registered model {name} version {version_num}")
        return version
    
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage
    ) -> ModelVersion:
        """Transition model version to new stage."""
        if name not in self._models:
            raise ValidationException(f"Model {name} not found")
        
        registered = self._models[name]
        version_obj = None
        
        for v in registered.versions:
            if v.version == version:
                version_obj = v
                break
        
        if version_obj is None:
            raise ValidationException(f"Version {version} not found")
        
        # If transitioning to production, archive current production
        if stage == ModelStage.PRODUCTION:
            for v in registered.versions:
                if v.stage == ModelStage.PRODUCTION:
                    v.stage = ModelStage.ARCHIVED
            registered.production_version = version
        
        version_obj.stage = stage
        self._save_registry()
        
        logger.info(f"Model {name} v{version} transitioned to {stage.value}")
        return version_obj
    
    def load_model(
        self,
        name: str,
        version: int = None,
        stage: ModelStage = None
    ) -> Any:
        """Load a model from registry."""
        if name not in self._models:
            raise ValidationException(f"Model {name} not found")
        
        registered = self._models[name]
        
        # Find version
        version_obj = None
        
        if version is not None:
            for v in registered.versions:
                if v.version == int(version):
                    version_obj = v
                    break
        elif stage:
            for v in registered.versions:
                if v.stage == stage:
                    version_obj = v
                    break
        else:
            # Latest
            version_obj = registered.versions[-1] if registered.versions else None
        
        if version_obj is None:
            available = ", ".join(str(v.version) for v in registered.versions) or "none"
            wanted = f"version {version}" if version is not None else (f"stage {stage.value}" if stage else "latest")
            raise ValidationException(
                f"No {wanted} found for model {name}; available versions: {available}"
            )

        artifact = self._resolve_artifact(version_obj.model_path)
        if artifact is None or not artifact.exists():
            raise ValidationException(f"Model artifact for {name} v{version_obj.version} is missing")

        # Artifacts are written only by this service into its own folder.
        with open(artifact, 'rb') as f:
            return pickle.load(f)
    
    def get_production_model(self, name: str) -> Any:
        """Get current production model."""
        return self.load_model(name, stage=ModelStage.PRODUCTION)
    
    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        return [m.to_dict() for m in self._models.values()]
    
    def get_model_versions(self, name: str) -> list[dict[str, Any]]:
        """Get all versions of a model."""
        if name not in self._models:
            return []
        return [v.to_dict() for v in self._models[name].versions]
    
    def compare_versions(
        self,
        name: str,
        version1: int,
        version2: int
    ) -> dict[str, Any]:
        """Compare two model versions."""
        if name not in self._models:
            raise ValidationException(f"Model {name} not found")
        
        registered = self._models[name]
        v1 = v2 = None
        
        for v in registered.versions:
            if v.version == version1:
                v1 = v
            if v.version == version2:
                v2 = v
        
        if v1 is None or v2 is None:
            raise ValidationException("Version not found")
        
        # Compare metrics
        metric_comparison = {}
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        
        for metric in all_metrics:
            m1 = v1.metrics.get(metric, 0)
            m2 = v2.metrics.get(metric, 0)
            metric_comparison[metric] = {
                "v1": m1,
                "v2": m2,
                "change": m2 - m1,
                "change_pct": ((m2 - m1) / m1 * 100) if m1 != 0 else 0
            }
        
        return {
            "version1": version1,
            "version2": version2,
            "metrics": metric_comparison,
            "recommendation": "v2" if sum(v2.metrics.values()) > sum(v1.metrics.values()) else "v1"
        }
    
    # Experiment tracking
    def create_experiment(self, name: str) -> Experiment:
        """Create new experiment."""
        exp = Experiment(
            experiment_id=str(uuid4()),
            name=name
        )
        self._experiments[name] = exp
        return exp
    
    def log_run(
        self,
        experiment_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        model: Any = None,
        model_name: str = None
    ) -> dict[str, Any]:
        """Log experiment run."""
        if experiment_name not in self._experiments:
            self.create_experiment(experiment_name)
        
        exp = self._experiments[experiment_name]
        
        run = {
            "run_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "params": params,
            "metrics": metrics
        }
        
        exp.runs.append(run)
        
        # Optionally register model
        if model and model_name:
            version = self.register_model(
                model_name, model, metrics, params
            )
            run["model_version"] = version.version
        
        self._save_registry()
        return run
    
    def get_experiment_runs(self, name: str) -> list[dict[str, Any]]:
        """Get all runs for an experiment."""
        if name not in self._experiments:
            return []
        return self._experiments[name].runs


# Factory function
def get_model_registry(path: Optional[str] = None) -> ModelRegistry:
    """Get model registry instance (MODEL_REGISTRY_PATH overrides the default folder)."""
    return ModelRegistry(path or os.getenv("MODEL_REGISTRY_PATH") or "./model_registry")
