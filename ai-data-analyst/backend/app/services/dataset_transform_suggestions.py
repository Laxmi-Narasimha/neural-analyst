from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_NUMERIC_TYPES = {"integer", "float"}
_STRING_TYPES = {"string", "text", "categorical", "email", "url", "phone", "uuid", "unknown"}
_TIME_NAME_HINT_RE = re.compile(r"(date|time|timestamp|created|updated|_at$|^dt_|_dt$)", flags=re.IGNORECASE)
_ID_NAME_HINT_RE = re.compile(r"(^id$|_id$|^id_|identifier|uuid|key$)", flags=re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s{2,}")


@dataclass(frozen=True)
class TransformPlanSuggestion:
    suggestions: list[dict[str, Any]]
    warnings: list[str]
    summary: dict[str, Any]


def _col_name(col: Any) -> str:
    return str(getattr(col, "name", "") or "").strip()


def _col_type(col: Any) -> str:
    return str(getattr(col, "inferred_type", "") or "").strip().lower()


def _col_null_pct(col: Any) -> float:
    try:
        return float(getattr(col, "null_percentage", 0.0) or 0.0)
    except Exception:
        return 0.0


def _col_null_count(col: Any) -> int:
    try:
        return int(getattr(col, "null_count", 0) or 0)
    except Exception:
        return 0


def _col_unique_count(col: Any) -> int:
    try:
        return int(getattr(col, "unique_count", 0) or 0)
    except Exception:
        return 0


def _col_sensitive(col: Any) -> bool:
    direct = bool(getattr(col, "is_sensitive", False))
    if direct:
        return True
    stats = getattr(col, "statistics", None)
    if isinstance(stats, dict):
        if bool(stats.get("is_potential_pii")):
            return True
        if bool(stats.get("is_sensitive")):
            return True
    return False


def _col_is_constant(col: Any) -> bool:
    stats = getattr(col, "statistics", None)
    if isinstance(stats, dict):
        return bool(stats.get("is_constant", False))
    return False


def _col_has_outliers(col: Any) -> bool:
    direct = bool(getattr(col, "has_outliers", False))
    if direct:
        return True
    stats = getattr(col, "statistics", None)
    if isinstance(stats, dict):
        return bool(stats.get("has_outliers", False))
    return False


def _string_needs_normalization(col: Any) -> bool:
    stats = getattr(col, "statistics", None)
    if not isinstance(stats, dict):
        return False
    samples = stats.get("sample_values")
    if not isinstance(samples, list):
        return False
    for raw in samples[:10]:
        if raw is None:
            continue
        s = str(raw)
        if s != s.strip():
            return True
        if _MULTISPACE_RE.search(s):
            return True
    return False


class DatasetTransformSuggestionService:
    """
    Deterministic, no-LLM transformation planner for the "basic analyst" stage.

    The planner generates a bounded, reproducible transform pipeline from dataset
    metadata only, with conservative defaults to minimize destructive changes.
    """

    def suggest(
        self,
        dataset: Any,
        *,
        max_steps: int = 8,
        include_drop_columns: bool = True,
        include_string_normalization: bool = True,
    ) -> TransformPlanSuggestion:
        max_steps_i = max(1, min(int(max_steps or 8), 25))
        columns = list(getattr(dataset, "columns", None) or [])
        row_count = int(getattr(dataset, "row_count", 0) or 0)
        quality_report = getattr(dataset, "quality_report", None)
        quality_warnings = (
            list(quality_report.get("warnings") or [])
            if isinstance(quality_report, dict)
            else []
        )

        suggestions: list[dict[str, Any]] = []
        warnings: list[str] = []

        skipped_steps = 0

        def push(step: dict[str, Any], reason: str, impact: str | None = None) -> None:
            nonlocal skipped_steps
            if len(suggestions) >= max_steps_i:
                skipped_steps += 1
                return
            payload = {"step": step, "reason": str(reason)}
            if impact:
                payload["impact"] = str(impact)
            suggestions.append(payload)

        # 1) Drop effectively empty/constant columns first (safe and high-value).
        if include_drop_columns:
            drop_cols: list[str] = []
            for c in columns:
                name = _col_name(c)
                if not name:
                    continue
                null_pct = _col_null_pct(c)
                if _col_is_constant(c):
                    drop_cols.append(name)
                    continue
                if null_pct >= 0.98:
                    drop_cols.append(name)
            if drop_cols:
                push(
                    {"op": "drop_columns", "params": {"columns": sorted(set(drop_cols))}},
                    "Drop columns that are constant or almost entirely missing.",
                    f"Reduce noise in {len(set(drop_cols))} low-information column(s).",
                )

        # 2) Missing-value treatment by inferred type.
        numeric_missing: list[str] = []
        categorical_missing: list[str] = []
        for c in columns:
            name = _col_name(c)
            if not name:
                continue
            nulls = _col_null_count(c)
            null_pct = _col_null_pct(c)
            if nulls <= 0 or null_pct >= 0.98:
                continue
            t = _col_type(c)
            if t in _NUMERIC_TYPES:
                numeric_missing.append(name)
            else:
                categorical_missing.append(name)

        if numeric_missing:
            push(
                {"op": "fill_missing", "params": {"columns": sorted(set(numeric_missing)), "strategy": "median"}},
                "Fill missing numeric values with median to preserve robustness against outliers.",
                f"Impute missing values in {len(set(numeric_missing))} numeric column(s).",
            )

        if categorical_missing:
            push(
                {"op": "fill_missing", "params": {"columns": sorted(set(categorical_missing)), "strategy": "mode"}},
                "Fill missing categorical/text values with mode for stable grouping behavior.",
                f"Impute missing values in {len(set(categorical_missing))} non-numeric column(s).",
            )

        # 2b) Outlier clipping for numeric columns flagged by profiling.
        outlier_numeric: list[str] = []
        for c in columns:
            name = _col_name(c)
            if not name:
                continue
            if _col_type(c) not in _NUMERIC_TYPES:
                continue
            if _col_has_outliers(c):
                outlier_numeric.append(name)
        if outlier_numeric:
            push(
                {
                    "op": "clip_outliers",
                    "params": {
                        "columns": sorted(set(outlier_numeric))[:6],
                        "method": "iqr",
                        "action": "clip",
                        "iqr_multiplier": 1.5,
                    },
                },
                "Clip extreme numeric outliers (IQR rule) to stabilize summary statistics and models.",
                f"Bound high-leverage values in {len(set(outlier_numeric))} numeric column(s).",
            )

        # 3) String normalization for whitespace inconsistencies.
        if include_string_normalization:
            normalize_cols: list[str] = []
            for c in columns:
                name = _col_name(c)
                if not name:
                    continue
                t = _col_type(c)
                if t not in _STRING_TYPES:
                    continue
                if _string_needs_normalization(c):
                    normalize_cols.append(name)
            if normalize_cols:
                push(
                    {
                        "op": "string_normalize",
                        "params": {
                            "columns": sorted(set(normalize_cols)),
                            "trim": True,
                            "lowercase": False,
                            "uppercase": False,
                            "regex_replace": [{"pattern": r"\s{2,}", "repl": " "}],
                        },
                    },
                    "Normalize string whitespace to reduce category fragmentation.",
                    f"Standardize text formatting in {len(set(normalize_cols))} column(s).",
                )

        # 4) Datetime coercion for likely time-like columns still typed as string.
        time_like_cols: list[str] = []
        for c in columns:
            name = _col_name(c)
            if not name:
                continue
            t = _col_type(c)
            if t == "datetime":
                continue
            if t not in _STRING_TYPES:
                continue
            if _TIME_NAME_HINT_RE.search(name):
                time_like_cols.append(name)
        for col in sorted(set(time_like_cols)):
            push(
                {"op": "type_convert", "params": {"column": col, "to": "datetime", "errors": "coerce"}},
                f"Convert '{col}' to datetime for time-based analysis.",
                "Enable time trends, period grouping, and anomaly scans.",
            )

        # 5) Dedup recommendation when key-like columns are present.
        key_candidates: list[str] = []
        for c in columns:
            name = _col_name(c)
            if not name:
                continue
            if _col_sensitive(c):
                continue
            if _ID_NAME_HINT_RE.search(name):
                key_candidates.append(name)
                continue
            if row_count > 0 and _col_unique_count(c) >= int(row_count * 0.98) and _col_null_pct(c) <= 0.02:
                key_candidates.append(name)
        duplicate_warning_present = any("duplicate" in str(w).lower() for w in quality_warnings)
        if key_candidates or duplicate_warning_present:
            params: dict[str, Any] = {"keep": "first"}
            subset = sorted(set(key_candidates))[:3]
            if subset:
                params["subset"] = subset
            push(
                {"op": "deduplicate", "params": params},
                "Remove duplicate rows using stable key candidates where available.",
                "Prevent over-counting and downstream metric inflation.",
            )

        # 6) Categorical encoding suggestion for low-cardinality non-sensitive columns.
        enc_candidates: list[str] = []
        for c in columns:
            name = _col_name(c)
            if not name:
                continue
            if _col_sensitive(c):
                continue
            if _ID_NAME_HINT_RE.search(name):
                continue
            t = _col_type(c)
            if t not in _STRING_TYPES:
                continue
            uniq = _col_unique_count(c)
            if uniq < 2:
                continue
            if uniq <= 40:
                enc_candidates.append(name)
        if enc_candidates:
            push(
                {
                    "op": "encode_categorical",
                    "params": {
                        "columns": sorted(set(enc_candidates))[:6],
                        "strategy": "label",
                        "drop_original": False,
                        "max_categories": 100,
                    },
                },
                "Encode low-cardinality categorical columns for downstream modeling and driver analysis.",
                f"Prepare {len(set(enc_candidates))} categorical column(s) for ML-ready workflows.",
            )

        if skipped_steps > 0:
            warnings.append(
                f"Skipped {skipped_steps} additional suggestion(s) because max_steps={max_steps_i}."
            )

        summary = {
            "columns_scanned": len(columns),
            "row_count": row_count,
            "suggestion_count": len(suggestions),
            "max_steps": max_steps_i,
            "include_drop_columns": bool(include_drop_columns),
            "include_string_normalization": bool(include_string_normalization),
        }
        return TransformPlanSuggestion(suggestions=suggestions, warnings=warnings, summary=summary)
