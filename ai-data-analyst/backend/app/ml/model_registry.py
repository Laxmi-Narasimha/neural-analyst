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
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "params": self.params
        }


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
    
    def __init__(self, storage_path: str = "./model_registry"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        self._models: dict[str, RegisteredModel] = {}
        self._experiments: dict[str, Experiment] = {}
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self._storage_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct models
                    for model_data in data.get("models", []):
                        model = RegisteredModel(
                            model_id=model_data["model_id"],
                            name=model_data["name"],
                            description=model_data.get("description", ""),
                            tags=model_data.get("tags", {})
                        )
                        self._models[model.name] = model
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self._storage_path / "registry.json"
        data = {
            "models": [m.to_dict() for m in self._models.values()],
            "experiments": [e.to_dict() for e in self._experiments.values()]
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
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
        version_num = len(registered.versions) + 1
        
        # Save model artifact
        model_path = self._storage_path / name / f"v{version_num}"
        model_path.mkdir(parents=True, exist_ok=True)
        model_file = model_path / "model.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Create version
        version = ModelVersion(
            version=version_num,
            model_id=registered.model_id,
            stage=ModelStage.DEVELOPMENT,
            status=ModelStatus.ACTIVE,
            metrics=metrics or {},
            params=params or {},
            model_path=str(model_file),
            description=description
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
        
        if version:
            for v in registered.versions:
                if v.version == version:
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
            raise ValidationException("No matching model version found")
        
        if version_obj.model_path is None:
            raise ValidationException("Model artifact not found")
        
        with open(version_obj.model_path, 'rb') as f:
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
def get_model_registry(path: str = "./model_registry") -> ModelRegistry:
    """Get model registry instance."""
    return ModelRegistry(path)
