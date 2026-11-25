"""A self-contained demo workspace: its own database, uploads, artifacts and model registry.

The demo never touches your development data. Everything lives under ``demo_workspace/`` next to
the backend (git-ignored) and can be wiped with ``python -m demo reset``.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.getenv("DEMO_WORKSPACE") or BACKEND / "demo_workspace")
DATA = Path(__file__).resolve().parent / "data"


def demo_env() -> dict[str, str]:
    """Environment that points the backend at the demo workspace (local mode: no login)."""
    ws = WORKSPACE.resolve()
    return {
        "DATABASE_URL": f"sqlite+aiosqlite:///{(ws / 'demo.db').as_posix()}",
        "DB_AUTO_CREATE_TABLES": "true",
        "UPLOAD_DIRECTORY": str(ws / "uploads"),
        "ARTIFACT_DIRECTORY": str(ws / "artifacts"),
        "MODEL_REGISTRY_PATH": str(ws / "models"),
        "STORAGE_CACHE_DIR": str(ws / "storage_cache"),
        "OBJECT_STORE_BACKEND": "local",
        "AUTH_MODE": "local",
        "ENVIRONMENT": "development",
        "JOB_EXECUTOR": "local",
        "VECTOR_STORE": "none",
    }


def apply_demo_env() -> None:
    """Point this process at the workspace. Call before importing anything from ``app``."""
    for sub in ("uploads", "artifacts", "models", "storage_cache"):
        (WORKSPACE / sub).mkdir(parents=True, exist_ok=True)
    os.environ.update(demo_env())
