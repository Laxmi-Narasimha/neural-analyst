from __future__ import annotations

from typing import Any


REDACTION_TOKEN = "[REDACTED]"


def mask_pii_value(value: Any) -> Any:
    # Preserve nulls so "missingness" stays meaningful in UI.
    if value is None:
        return None
    return REDACTION_TOKEN


def mask_pii_rows(rows: list[dict[str, Any]], pii_columns: set[str]) -> list[dict[str, Any]]:
    if not rows or not pii_columns:
        return rows
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        for c in pii_columns:
            if c in rr:
                rr[c] = mask_pii_value(rr.get(c))
        out.append(rr)
    return out


def mask_preview(preview: dict[str, Any], pii_columns: set[str]) -> dict[str, Any]:
    if not preview or not pii_columns:
        return preview
    p = dict(preview)
    rows = p.get("preview_rows")
    if isinstance(rows, list):
        # Keep structure stable; do not mutate original.
        p["preview_rows"] = mask_pii_rows([r for r in rows if isinstance(r, dict)], pii_columns)
    return p

