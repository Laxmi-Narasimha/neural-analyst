"""Has the data changed since the model was trained?

At training time a compact profile of every input the model uses is stored with it: decile edges
and shares for numbers, the most common categories for text, and missing rates. When a new file is
scored, the same summary is computed for it and compared with the population stability index (PSI)
per column:

    PSI < 0.10       stable
    0.10 – 0.25      moderate shift: check the column
    > 0.25           major shift: predictions on these rows are less trustworthy

The profile holds no rows, only bin edges, category names (at most 20 per column) and shares.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

_MAX_CATEGORIES = 20
_EPS = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _level(psi: Optional[float]) -> str:
    if psi is None:
        return "unknown"
    return "stable" if psi < 0.10 else ("moderate" if psi <= 0.25 else "major")


def _numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cleaned = series.astype("string").str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _is_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    non_null = series.dropna()
    return len(non_null) > 0 and _numeric(non_null).notna().mean() >= 0.95


def build_profile(X: pd.DataFrame, columns: Optional[list[str]] = None) -> dict[str, Any]:
    """Reference profile of the training rows (raw values, before preprocessing)."""
    cols = [c for c in (columns or list(X.columns)) if c in X.columns]
    profile: dict[str, Any] = {"rows": int(len(X)), "columns": {}}
    for col in cols:
        s = X[col]
        missing = float(s.isna().mean()) if len(s) else 0.0
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if _is_numeric(s):
            v = _numeric(s).dropna()
            if v.nunique() < 2:
                continue
            edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
            if len(edges) < 3:
                continue
            inner = edges[1:-1]
            counts = np.bincount(np.searchsorted(inner, v.to_numpy(), side="right"), minlength=len(inner) + 1)
            profile["columns"][str(col)] = {
                "kind": "numeric", "missing_rate": missing, "edges": [float(e) for e in inner],
                "shares": (counts / counts.sum()).tolist(),
                "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            }
        else:
            text = s.astype("string").str.strip().dropna()
            if text.empty:
                continue
            shares = text.value_counts(normalize=True)
            top = shares.head(_MAX_CATEGORIES)
            profile["columns"][str(col)] = {
                "kind": "categorical", "missing_rate": missing,
                "categories": [str(k) for k in top.index], "shares": [float(x) for x in top.values],
                "other_share": float(max(0.0, 1.0 - top.sum())),
            }
    return profile


def compare(profile: dict[str, Any], new: pd.DataFrame) -> dict[str, Any]:
    """Per-column drift of ``new`` against a training profile, worst first."""
    rows = []
    for col, ref in (profile.get("columns") or {}).items():
        if col not in new.columns:
            rows.append({"column": col, "kind": ref["kind"], "psi": None, "level": "missing_column",
                         "note": "column is absent from the new data; the model treats it as unknown"})
            continue
        s = new[col]
        missing_now = float(s.isna().mean()) if len(s) else 0.0
        item: dict[str, Any] = {"column": col, "kind": ref["kind"],
                                "missing_rate_training": round(ref["missing_rate"], 4),
                                "missing_rate_new": round(missing_now, 4)}
        if ref["kind"] == "numeric":
            v = _numeric(s).dropna()
            if v.empty:
                item.update(psi=None, level="unknown", note="no numeric values in the new data")
                rows.append(item)
                continue
            counts = np.bincount(np.searchsorted(np.asarray(ref["edges"]), v.to_numpy(), side="right"),
                                 minlength=len(ref["edges"]) + 1)
            psi = _psi(ref["shares"], counts / counts.sum())
            outside = float(((v < ref["min"]) | (v > ref["max"])).mean())
            item.update(mean_training=round(ref["mean"], 4), mean_new=round(float(v.mean()), 4),
                        share_outside_training_range=round(outside, 4))
        else:
            text = s.astype("string").str.strip().dropna()
            if text.empty:
                item.update(psi=None, level="unknown", note="no values in the new data")
                rows.append(item)
                continue
            shares = text.value_counts(normalize=True)
            cats = ref["categories"]
            actual = [float(shares.get(c, 0.0)) for c in cats]
            actual_other = float(max(0.0, 1.0 - sum(actual)))
            psi = _psi(list(ref["shares"]) + [ref["other_share"]], actual + [actual_other])
            unseen = [str(k) for k in shares.index if str(k) not in set(cats)]
            item.update(new_categories=unseen[:10], share_new_categories=round(actual_other, 4))
        item.update(psi=round(psi, 4), level=_level(psi))
        if abs(missing_now - ref["missing_rate"]) >= 0.10 and item["level"] == "stable":
            item["level"] = "moderate"
            item["note"] = "missing values changed by 10+ percentage points"
        rows.append(item)
    order = {"missing_column": 0, "major": 1, "moderate": 2, "unknown": 3, "stable": 4}
    rows.sort(key=lambda r: (order.get(r["level"], 5), -(r.get("psi") or 0)))
    worst = [r for r in rows if r["level"] in {"major", "missing_column"}]
    moderate = [r for r in rows if r["level"] == "moderate"]
    if worst:
        verdict = "major"
        message = ("The new data differs a lot from the training data in "
                   + ", ".join(r["column"] for r in worst[:5]) + "; treat these predictions with caution "
                   "and consider retraining on recent data.")
    elif moderate:
        verdict = "moderate"
        message = "Some inputs shifted moderately (" + ", ".join(r["column"] for r in moderate[:5]) + ")."
    else:
        verdict = "stable"
        message = "The new data looks like the training data on every input the model uses."
    return {"verdict": verdict, "message": message, "rows_scored": int(len(new)),
            "training_rows": profile.get("rows"), "columns": rows,
            "method": "Population stability index per input against the training split (bands 0.10 / 0.25)."}


def prediction_shift(reference: Optional[dict[str, Any]], predictions: list[Any], probabilities: Optional[list[list[float]]],
                     classes: Optional[list[Any]]) -> Optional[dict[str, Any]]:
    """How the predicted outcome mix compares with the held-out predictions made at training time."""
    if not reference:
        return None
    if classes is not None and reference.get("class_shares"):
        now = pd.Series([str(p) for p in predictions]).value_counts(normalize=True)
        ref = reference["class_shares"]
        keys = list(ref.keys())
        psi = _psi([ref[k] for k in keys], [float(now.get(k, 0.0)) for k in keys])
        return {"kind": "classes", "training": ref, "new": {k: round(float(now.get(k, 0.0)), 4) for k in keys},
                "psi": round(psi, 4), "level": _level(psi)}
    if reference.get("mean") is not None and predictions:
        vals = np.asarray(predictions, dtype=float)
        return {"kind": "values", "training_mean": reference["mean"], "new_mean": round(float(vals.mean()), 4),
                "training_std": reference.get("std"), "new_std": round(float(vals.std()), 4)}
    return None
