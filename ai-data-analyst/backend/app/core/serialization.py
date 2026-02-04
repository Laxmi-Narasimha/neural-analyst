from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable primitives.

    This is used before persisting to SQLAlchemy JSON columns (SQLite's JSON
    serialization is strict and will error on numpy scalar types).
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (UUID,)):
        return str(value)

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = to_jsonable(v)
        return out

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    # Fallback: preserve the value as a string rather than failing a DB commit.
    return str(value)

