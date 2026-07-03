from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.serialization import to_jsonable


def canonical_dumps(obj: Any) -> str:
    """
    Canonical JSON encoding for stable hashing.

    - sort_keys=True ensures deterministic output across runs
    - separators minimize size and avoid whitespace differences
    """
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def sha256_hexdigest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def hash_json(obj: Any) -> str:
    return sha256_hexdigest(canonical_dumps(obj))

