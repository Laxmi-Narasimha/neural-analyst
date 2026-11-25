"""Move models out of the old single shared registry into each owner's own registry.

Before per-owner registries, every trained model sat in one folder that every signed-in user
could list, load and promote. On start-up each such model is moved to the owner of the dataset it
was trained on (from its dataset_id tag). Models whose dataset no longer exists stay in the old
folder, which nothing serves any more.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.core.logging import get_logger
from app.ml.model_registry import migrate_shared_registry, registry_root

logger = get_logger(__name__)


def _legacy_dataset_ids() -> set[UUID]:
    path = registry_root() / "registry.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    ids: set[UUID] = set()
    for model in data.get("models") or []:
        raw = ((model or {}).get("tags") or {}).get("dataset_id")
        try:
            ids.add(UUID(str(raw)))
        except (TypeError, ValueError):
            continue
    return ids


async def migrate_shared_model_registry() -> dict[str, Any]:
    from app.models import Dataset
    from app.services.database import db_manager

    ids = _legacy_dataset_ids()
    if not (registry_root() / "registry.json").exists():
        return {"moved": [], "unclaimed": [], "conflicts": []}
    owners: dict[str, UUID] = {}
    if ids:
        async with db_manager.session() as session:
            rows = await session.execute(select(Dataset.id, Dataset.owner_id).where(Dataset.id.in_(ids)))
            owners = {str(i): o for i, o in rows.all() if o is not None}
    return await asyncio.to_thread(migrate_shared_registry, lambda dataset_id: owners.get(str(dataset_id)))
