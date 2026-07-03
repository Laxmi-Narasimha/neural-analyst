"""backfill artifact operator_params_hash

Revision ID: 4c1e9e67e3d7
Revises: 1b2f9df9b0b4
Create Date: 2026-02-10 00:00:00.000000
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = "4c1e9e67e3d7"
down_revision = "1b2f9df9b0b4"
branch_labels = None
depends_on = None


def _canonical_dumps(obj: Any) -> str:
    # Keep identical behavior to app.utils.hashing.canonical_dumps:
    # - sort_keys=True for determinism
    # - compact separators to avoid whitespace differences
    # - ensure_ascii=True so DB content is stable across locales
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha256_hex(text_value: str) -> str:
    return hashlib.sha256(str(text_value).encode("utf-8")).hexdigest()


def upgrade() -> None:
    bind = op.get_bind()
    # Best-effort backfill for deployments that already have artifacts in the DB.
    # This improves cache hit rate immediately after upgrading.
    rows = bind.execute(text("SELECT id, operator_params FROM artifacts WHERE operator_params_hash IS NULL")).mappings()
    for r in rows:
        aid = r.get("id")
        params = r.get("operator_params")
        if params is None:
            params = {}
        # sqlite may return JSON as a string; postgres returns dict-like.
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        try:
            digest = _sha256_hex(_canonical_dumps(params))
        except Exception:
            digest = _sha256_hex(_canonical_dumps({}))
        bind.execute(
            text("UPDATE artifacts SET operator_params_hash = :h WHERE id = :id"),
            {"h": digest, "id": aid},
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(text("UPDATE artifacts SET operator_params_hash = NULL"))

