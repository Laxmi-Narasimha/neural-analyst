from __future__ import annotations

import os
import platform
from typing import Any

from app.core.config import get_settings
from app.services.llm_service import get_llm_service


def _env_present(name: str) -> bool:
    value = os.environ.get(name, "").strip()
    return bool(value)


def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except Exception:
        return False


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def _detect_llm_provider(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("ollama/"):
        return "ollama"
    if m.startswith("anthropic/") or "claude" in m:
        return "anthropic"
    if m.startswith("gemini/") or "gemini" in m:
        return "google"
    if m.startswith("mistral/"):
        return "mistral"
    return "openai"


def _llm_ready(model: str, keys: dict[str, bool]) -> bool:
    provider = _detect_llm_provider(model)
    if provider == "ollama":
        return True
    if provider == "anthropic":
        return keys.get("anthropic", False)
    if provider == "google":
        return keys.get("gemini", False)
    if provider == "mistral":
        return keys.get("mistral", False)
    return keys.get("openai", False)


def build_system_info() -> dict[str, Any]:
    settings = get_settings()
    llm = get_llm_service()
    model = settings.llm_model

    keys = {
        "openai": _env_present("OPENAI_API_KEY") or bool(settings.openai.api_key),
        "anthropic": _env_present("ANTHROPIC_API_KEY"),
        "gemini": _env_present("GEMINI_API_KEY"),
        "mistral": _env_present("MISTRAL_API_KEY"),
    }

    provider = _detect_llm_provider(model)
    llm_ready = _llm_ready(model, keys)

    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment.value,
        "deployment_mode": settings.deployment_mode.value,
        "auth_mode": settings.auth_mode.value,
        "job_executor": settings.job_executor.value,
        "object_store_backend": settings.object_store.backend.value,
        "vector_store": settings.vector_store.value,
        "llm_model": model,
        "llm_provider": provider,
        "llm_configured": llm_ready,
        "llm_available": llm.is_available,
        "narrator_mode": settings.narrator.mode.value,
        "api_keys_present": keys,
        "compute": {
            "runs_on": "your_machine",
            "description": "All pandas/sklearn/AutoML runs in your container or local RAM — not on Neural Analyst cloud.",
            "duckdb_available": _duckdb_available(),
            "sklearn_available": _sklearn_available(),
            "max_upload_size_mb": settings.max_upload_size_mb,
            "ml_training_sample_rows": settings.ml.automl_training_sample_rows,
        },
        "capabilities": {
            "talk_to_data_without_llm": True,
            "chat_without_llm": False,
            "distributed_jobs": settings.job_executor.value == "celery",
            "s3_storage": settings.object_store.backend.value == "s3",
        },
        "self_host": {
            "docker_compose": "docker compose up --build",
            "docker_compose_celery": "docker compose -f docker-compose.yml -f docker-compose.celery.yml up --build",
            "windows_script": "scripts/self_host.ps1",
            "unix_script": "scripts/self_host.sh",
            "env_file": "ai-data-analyst/backend/.env",
            "docs": "docs/API_KEYS.md",
        },
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
        },
    }