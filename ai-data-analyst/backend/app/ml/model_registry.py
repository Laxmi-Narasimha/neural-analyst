# AI Enterprise Data Analyst - Model Registry
# MLflow-inspired model versioning, tracking, and deployment

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
from uuid import UUID, uuid4
import json
import os
import pickle
import hashlib
import re
import shutil
import time

from filelock import FileLock

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)

_REGISTRY_FORMAT = 2
_OWNERS_DIR = "owners"
_UNSAFE_DIR_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_LOCK_TIMEOUT_SECONDS = 60


def model_dir_name(name: str) -> str:
    """Folder for a model name. Names come from column headers in uploaded files, so they may hold
    path separators, '..', reserved device names or non-ASCII text; the folder is always a plain
    ASCII name, with a hash suffix whenever the name had to change, so two names never share one."""
    raw = str(name)
    safe = _UNSAFE_DIR_CHARS.sub("_", raw).strip("._")[:60] or "model"
    if safe != raw or safe.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}:
        safe = f"{safe}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"
    return safe


def validate_model_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text or len(text) > 200 or any(ord(ch) < 32 for ch in text):
        raise ValidationException("Model names must be 1-200 printable characters")
    return text


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
        # Every change is read-modify-write of registry.json under this lock, so two trainings
        # finishing together (threads, workers or processes) never lose a version or reuse a folder.
        self._lock = FileLock(str(self._storage_path / "registry.lock"), timeout=_LOCK_TIMEOUT_SECONDS)

        self._load_registry()

    def _reload(self) -> None:
        self._models = {}
        self._experiments = {}
        self._load_registry()

    @property
    def storage_path(self) -> Path:
        return self._storage_path

    def _resolve_artifact(self, model_path: Optional[str]) -> Optional[Path]:
        """Artifacts are unpickled, so only files inside this registry's own folder are ever opened."""
        if not model_path:
            return None
        p = Path(model_path)
        # Stored relative to the registry folder; older entries were relative to the cwd.
        candidate = p if p.is_absolute() else self._storage_path / p
        if not p.is_absolute() and not candidate.exists() and p.exists():
            candidate = p
        root = self._storage_path.resolve()
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValidationException("Model artifact is outside this registry")
        return resolved

    def _recover_versions_from_disk(self, model: RegisteredModel) -> None:
        """Rebuild versions for entries written before versions were persisted."""
        model_dir = self._storage_path / model_dir_name(model.name)
        if not model_dir.is_dir():
            model_dir = self._storage_path / model.name
        try:
            inside = model_dir.resolve().parent == self._storage_path.resolve()
        except OSError:
            inside = False
        if not inside or not model_dir.is_dir():
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
        # Windows refuses to replace a file another process is reading for a moment; retry briefly.
        for attempt in range(20):
            try:
                os.replace(tmp, registry_file)
                return
            except PermissionError:
                if attempt == 19:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(0.05)
    
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
        name = validate_model_name(name)
        payload = pickle.dumps(model)  # outside the lock: it can take a while for big models
        with self._lock:
            self._reload()
            return self._register_locked(name, payload, metrics, params, description, tags)

    def _register_locked(self, name, payload, metrics, params, description, tags) -> ModelVersion:
        if name not in self._models:
            self._models[name] = RegisteredModel(
                model_id=str(uuid4()),
                name=name,
                description=description,
                tags=tags or {}
            )
        
        registered = self._models[name]
        version_num = max((v.version for v in registered.versions), default=0) + 1

        model_path = self._storage_path / model_dir_name(name) / f"v{version_num}"
        model_path.mkdir(parents=True, exist_ok=True)
        model_file = model_path / "model.pkl"
        tmp_file = model_path / f"model.{uuid4().hex}.tmp"
        with open(tmp_file, 'wb') as f:
            f.write(payload)
        os.replace(tmp_file, model_file)

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
        with self._lock:
            self._reload()
            return self._transition_locked(name, version, stage)

    def _transition_locked(self, name: str, version: int, stage: ModelStage) -> ModelVersion:
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
        
        def num(value: Any) -> Optional[float]:
            try:
                f = float(value)
            except (TypeError, ValueError):
                return None
            return f if np.isfinite(f) else None

        for metric in sorted(all_metrics):
            m1, m2 = num(v1.metrics.get(metric)), num(v2.metrics.get(metric))
            change = None if m1 is None or m2 is None else m2 - m1
            metric_comparison[metric] = {
                "v1": m1,
                "v2": m2,
                "change": change,
                "change_pct": (change / abs(m1) * 100) if change is not None and m1 else None,
            }
        
        # Cross-validated scores follow scikit-learn's convention (higher is better, errors are
        # negated), so they are the one fair basis for a pick. Raw test metrics mix directions.
        cv = [m for m in sorted(all_metrics) if m.startswith("cv_")
              and metric_comparison[m]["change"] is not None]
        recommendation = None
        basis = None
        if cv:
            basis = cv[0]
            recommendation = "v2" if metric_comparison[basis]["change"] > 0 else "v1"
        return {
            "version1": version1,
            "version2": version2,
            "metrics": metric_comparison,
            "recommendation": recommendation,
            "recommendation_basis": basis,
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
        run = {
            "run_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "params": params,
            "metrics": metrics
        }
        if model and model_name:
            version = self.register_model(model_name, model, metrics, params)
            run["model_version"] = version.version
        with self._lock:
            self._reload()
            if experiment_name not in self._experiments:
                self.create_experiment(experiment_name)
            self._experiments[experiment_name].runs.append(run)
            self._save_registry()
        return run
    
    def delete_models_for_dataset(self, dataset_id: Any) -> list[str]:
        """Remove every model trained on one dataset, with its files. A trained model can memorise
        rows of its training data, so deleting a dataset must delete its models too."""
        wanted = str(dataset_id)
        removed: list[str] = []
        with self._lock:
            self._reload()
            for name in [n for n, m in self._models.items() if str((m.tags or {}).get("dataset_id")) == wanted]:
                for folder in {self._storage_path / model_dir_name(name), self._storage_path / name}:
                    try:
                        if folder.is_dir() and folder.resolve().parent == self._storage_path.resolve():
                            shutil.rmtree(folder)
                    except OSError as e:
                        logger.warning(f"Could not delete model folder {folder.name}: {e}")
                del self._models[name]
                removed.append(name)
            if removed:
                self._save_registry()
        return removed

    def get_experiment_runs(self, name: str) -> list[dict[str, Any]]:
        """Get all runs for an experiment."""
        if name not in self._experiments:
            return []
        return self._experiments[name].runs


def registry_root() -> Path:
    """Top folder for every owner's registry (MODEL_REGISTRY_PATH overrides the default)."""
    return Path(os.getenv("MODEL_REGISTRY_PATH") or "./model_registry")


def _owner_key(owner_id: Any) -> str:
    try:
        return str(UUID(str(owner_id)))
    except (TypeError, ValueError, AttributeError):
        raise ValidationException("A model registry belongs to one signed-in owner") from None


def get_model_registry(owner_id: Any) -> ModelRegistry:
    """The registry of one owner. Models are never shared between owners: each owner has a
    separate folder, so a name or version from someone else simply does not exist here."""
    if owner_id is None:
        raise ValidationException("A model registry belongs to one signed-in owner")
    return ModelRegistry(registry_root() / _OWNERS_DIR / _owner_key(owner_id))


def migrate_shared_registry(owner_of_dataset: Callable[[str], Optional[Any]]) -> dict[str, Any]:
    """Move models from the old single shared registry into their owners' registries.

    The owner is found from each model's dataset_id tag. Models whose owner cannot be found stay in
    the old registry, which no API reads any more, so they are never served to anyone. Safe to run
    on every start: it is idempotent and holds both registries' locks while moving a model.
    """
    root = registry_root()
    report: dict[str, Any] = {"moved": [], "unclaimed": [], "conflicts": []}
    if not (root / "registry.json").exists():
        return report
    legacy = ModelRegistry(root)
    with legacy._lock:
        legacy._reload()
        for name in list(legacy._models):
            model = legacy._models[name]
            dataset_id = str((model.tags or {}).get("dataset_id") or "")
            owner = owner_of_dataset(dataset_id) if dataset_id else None
            if owner is None:
                report["unclaimed"].append(name)
                continue
            target = get_model_registry(owner)
            with target._lock:
                target._reload()
                if name in target._models:
                    report["conflicts"].append(name)
                    continue
                dest_dir = target.storage_path / model_dir_name(name)
                for v in model.versions:
                    try:
                        src = legacy._resolve_artifact(v.model_path)
                    except ValidationException:
                        src = None
                    if src is None or not src.exists():
                        continue
                    dest = dest_dir / f"v{v.version}" / "model.pkl"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    v.model_path = str(dest.relative_to(target.storage_path))
                model.versions = [v for v in model.versions
                                  if v.model_path and (target.storage_path / v.model_path).exists()]
                target._models[name] = model
                target._save_registry()
            del legacy._models[name]
            legacy._save_registry()
            for folder in {root / model_dir_name(name), root / name}:
                try:
                    if folder.is_dir() and folder.resolve().parent == root.resolve():
                        shutil.rmtree(folder, ignore_errors=True)
                except OSError:
                    pass
            report["moved"].append(name)
    if report["moved"] or report["unclaimed"] or report["conflicts"]:
        logger.info(
            "Shared model registry migration",
            moved=len(report["moved"]), unclaimed=len(report["unclaimed"]),
            conflicts=len(report["conflicts"]),
        )
    return report
