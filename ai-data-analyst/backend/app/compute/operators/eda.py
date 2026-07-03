from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.compute.operators.base import OperatorContext, OperatorResult


def _safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return numeric
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric


_REDACTION_TOKEN = "[REDACTED]"


def _mask_pii_preview(df: pd.DataFrame, pii_cols: set[str]) -> pd.DataFrame:
    if df is None or df.empty or not pii_cols:
        return df
    for c in pii_cols:
        if c not in df.columns:
            continue
        try:
            s = df[c]
            df[c] = s.where(s.isna(), _REDACTION_TOKEN)
        except Exception:
            # Best-effort: never fail preview due to masking.
            df[c] = _REDACTION_TOKEN
    return df


def _pandas_resample_freq(freq: str) -> str:
    code = str(freq or "").upper().strip()
    # Pandas deprecated "M" (month-end) in favor of "ME". Keep API accepting "M".
    if code == "M":
        return "ME"
    return code


@dataclass(frozen=True)
class SchemaSnapshotOperator:
    name: str = "schema_snapshot"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        columns = []
        schema_columns = (ctx.schema_info or {}).get("columns")
        if isinstance(schema_columns, list):
            for col in schema_columns:
                if not isinstance(col, dict):
                    continue
                safe_col = dict(col)
                stats = safe_col.get("statistics") if isinstance(safe_col.get("statistics"), dict) else {}
                is_pii = bool(safe_col.get("is_potential_pii") or stats.get("is_potential_pii"))
                if is_pii:
                    # PII-safe defaults: avoid surfacing raw example values in schema outputs.
                    if "sample_values" in safe_col:
                        safe_col["sample_values"] = []
                    if "value_distribution" in safe_col:
                        safe_col["value_distribution"] = {}
                    if isinstance(safe_col.get("statistics"), dict):
                        st = dict(safe_col["statistics"])
                        st.pop("sample_values", None)
                        st.pop("value_distribution", None)
                        safe_col["statistics"] = st
                columns.append(safe_col)

        if not columns:
            columns = [{"name": str(c)} for c in ctx.df.columns.tolist()]

        out = pd.DataFrame(columns)
        return OperatorResult(
            tables={"schema": out},
            metrics={},
            charts={},
            summary={"columns": int(out.shape[0])},
        )


@dataclass(frozen=True)
class DatasetOverviewOperator:
    name: str = "dataset_overview"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        profile = ctx.profile_report or {}

        sampled_rows = int(df.shape[0]) if df is not None else 0
        sampled_cols = int(df.shape[1]) if df is not None else 0

        rows = profile.get("row_count")
        cols = profile.get("column_count")
        try:
            rows = int(rows) if rows is not None else sampled_rows
        except Exception:
            rows = sampled_rows
        try:
            cols = int(cols) if cols is not None else sampled_cols
        except Exception:
            cols = sampled_cols

        mem_bytes = 0
        if df is not None:
            try:
                mem_bytes = int(df.memory_usage(deep=True).sum())
            except Exception:
                mem_bytes = 0

        out = pd.DataFrame(
            [
                {
                    "rows": int(rows),
                    "columns": int(cols),
                    "sampled_rows": int(sampled_rows),
                    "sampled_columns": int(sampled_cols),
                    "sample_memory_bytes": int(mem_bytes),
                }
            ]
        )

        return OperatorResult(
            tables={"overview": out},
            metrics={"rows": int(rows), "columns": int(cols), "sampled_rows": int(sampled_rows)},
            charts={},
            summary={"rows": int(rows), "columns": int(cols), "sampled_rows": int(sampled_rows), "sample_memory_bytes": int(mem_bytes)},
        )


@dataclass(frozen=True)
class PreviewRowsOperator:
    name: str = "preview_rows"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        limit = int(params.get("limit", 25))
        limit = max(1, min(limit, 200))
        out = ctx.df.head(limit).copy()
        out = _mask_pii_preview(out, _pii_columns(ctx))
        return OperatorResult(
            tables={"preview": out},
            metrics={"preview_rows": int(out.shape[0])},
            charts={},
            summary={"preview_rows": int(out.shape[0])},
        )


@dataclass(frozen=True)
class MissingnessOperator:
    name: str = "missingness_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        total_rows = int(df.shape[0]) if df is not None else 0
        if total_rows == 0:
            out = pd.DataFrame(columns=["column", "null_count", "null_pct"])
            return OperatorResult(
                tables={"missingness": out},
                metrics={"missing_pct": 0.0},
                charts={},
                summary={"rows": 0, "columns": int(df.shape[1]) if df is not None else 0},
            )

        null_counts = df.isna().sum().astype(int)
        out = (
            pd.DataFrame({"column": null_counts.index, "null_count": null_counts.values})
            .assign(null_pct=lambda x: x["null_count"] / float(total_rows))
            .sort_values("null_pct", ascending=False)
            .reset_index(drop=True)
        )

        missing_pct = float(null_counts.sum() / float(total_rows * max(int(df.shape[1]), 1)))

        return OperatorResult(
            tables={"missingness": out},
            metrics={"missing_pct": missing_pct},
            charts={},
            summary={"rows": total_rows, "columns": int(df.shape[1]), "missing_pct": missing_pct},
        )


@dataclass(frozen=True)
class UniquenessOperator:
    name: str = "uniqueness_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        total_rows = int(df.shape[0]) if df is not None else 0
        if total_rows == 0:
            out = pd.DataFrame(
                columns=[
                    "column",
                    "non_null_count",
                    "null_count",
                    "unique_count",
                    "unique_ratio",
                    "duplicate_count",
                    "duplicate_ratio",
                    "is_key_candidate",
                ]
            )
            return OperatorResult(
                tables={"uniqueness": out},
                metrics={"columns_scanned": 0, "key_candidates": 0},
                charts={},
                summary={"rows": 0, "columns": int(df.shape[1]) if df is not None else 0},
            )

        max_columns = int(params.get("max_columns", 200))
        max_columns = max(1, min(max_columns, 1000))
        cols = df.columns.tolist()[:max_columns]

        rows: list[dict[str, Any]] = []
        key_candidates: list[str] = []

        for col in cols:
            s = df[col]
            null_count = int(s.isna().sum())
            non_null_count = int(s.shape[0] - null_count)
            if non_null_count <= 0:
                unique_count = 0
            else:
                try:
                    unique_count = int(s.nunique(dropna=True))
                except Exception:
                    # Fallback: cast to string for nunique on mixed types.
                    unique_count = int(s.astype(str).nunique(dropna=True))

            unique_ratio = float(unique_count) / float(max(non_null_count, 1))
            duplicate_count = int(max(non_null_count - unique_count, 0))
            duplicate_ratio = float(duplicate_count) / float(max(non_null_count, 1))

            is_key_candidate = bool(unique_ratio >= 0.995 and null_count == 0)
            if is_key_candidate:
                key_candidates.append(str(col))

            rows.append(
                {
                    "column": str(col),
                    "non_null_count": non_null_count,
                    "null_count": null_count,
                    "unique_count": unique_count,
                    "unique_ratio": unique_ratio,
                    "duplicate_count": duplicate_count,
                    "duplicate_ratio": duplicate_ratio,
                    "is_key_candidate": is_key_candidate,
                }
            )

        out = pd.DataFrame(rows).sort_values("unique_ratio", ascending=False).reset_index(drop=True)

        # Keep the summary small and deterministic.
        key_candidates = key_candidates[:10]

        return OperatorResult(
            tables={"uniqueness": out},
            metrics={
                "columns_scanned": int(len(cols)),
                "key_candidates": int(len(key_candidates)),
            },
            charts={},
            summary={"key_candidates": key_candidates},
        )


@dataclass(frozen=True)
class NumericSummaryOperator:
    name: str = "numeric_summary"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        numeric = _safe_numeric_df(ctx.df)
        if numeric.empty:
            out = pd.DataFrame(columns=["column"])
            return OperatorResult(
                tables={"numeric_summary": out},
                metrics={"numeric_columns": 0},
                charts={},
                summary={"numeric_columns": 0},
            )

        max_cols = int(params.get("max_columns", 25))
        max_cols = max(1, min(max_cols, 200))
        cols = numeric.columns.tolist()[:max_cols]
        sub = numeric[cols]

        desc = sub.describe(percentiles=[0.25, 0.5, 0.75]).transpose().reset_index()
        desc = desc.rename(columns={"index": "column"})

        return OperatorResult(
            tables={"numeric_summary": desc},
            metrics={"numeric_columns": int(len(cols))},
            charts={},
            summary={"numeric_columns": int(len(cols))},
        )


@dataclass(frozen=True)
class CategoricalTopKOperator:
    name: str = "categorical_topk"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        pii = _pii_columns(ctx)
        k = int(params.get("k", 10))
        k = max(3, min(k, 50))
        max_columns = int(params.get("max_columns", 10))
        max_columns = max(1, min(max_columns, 50))

        # Heuristic: choose low-cardinality non-numeric columns.
        non_numeric = df.select_dtypes(exclude=[np.number])
        candidates: list[str] = []
        for col in non_numeric.columns.tolist():
            if str(col) in pii:
                continue
            try:
                nunique = int(non_numeric[col].nunique(dropna=True))
            except Exception:
                continue
            if 1 <= nunique <= 1000:
                candidates.append(col)

        selected = candidates[:max_columns]
        rows: list[dict[str, Any]] = []
        for col in selected:
            series = df[col]
            vc = series.value_counts(dropna=False).head(k)
            total = float(series.shape[0]) if series is not None else 0.0
            for value, count in vc.items():
                rows.append(
                    {
                        "column": col,
                        "value": None if (isinstance(value, float) and np.isnan(value)) else str(value),
                        "count": int(count),
                        "pct": float(count) / total if total > 0 else 0.0,
                    }
                )

        out = pd.DataFrame(rows)
        return OperatorResult(
            tables={"categorical_topk": out},
            metrics={"categorical_columns": int(len(selected))},
            charts={},
            summary={"categorical_columns": int(len(selected)), "k": k},
        )


@dataclass(frozen=True)
class TextSummaryOperator:
    name: str = "text_summary"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            out = pd.DataFrame(columns=["column", "non_null_count", "avg_length", "min_length", "max_length", "empty_pct"])
            return OperatorResult(
                tables={"text_summary": out},
                metrics={"text_columns": 0},
                charts={},
                summary={"text_columns": 0},
            )

        max_columns = int(params.get("max_columns", 25))
        max_columns = max(1, min(max_columns, 200))

        # Prefer string-like columns. Keep it deterministic: scan in column order.
        candidates = df.select_dtypes(include=["object", "string"]).columns.tolist()
        selected = candidates[:max_columns]

        rows: list[dict[str, Any]] = []
        for col in selected:
            s = df[col].dropna()
            if s.empty:
                rows.append(
                    {
                        "column": str(col),
                        "non_null_count": 0,
                        "avg_length": 0.0,
                        "min_length": 0,
                        "max_length": 0,
                        "empty_pct": 0.0,
                    }
                )
                continue

            # Convert to string but do not emit values.
            ss = s.astype(str)
            lengths = ss.str.len()
            empty = (ss.str.strip() == "").sum()
            n = int(ss.shape[0])

            rows.append(
                {
                    "column": str(col),
                    "non_null_count": n,
                    "avg_length": float(lengths.mean()) if n else 0.0,
                    "min_length": int(lengths.min()) if n else 0,
                    "max_length": int(lengths.max()) if n else 0,
                    "empty_pct": float(empty) / float(n) if n else 0.0,
                }
            )

        out = pd.DataFrame(rows).sort_values("avg_length", ascending=False).reset_index(drop=True)
        return OperatorResult(
            tables={"text_summary": out},
            metrics={"text_columns": int(len(selected))},
            charts={},
            summary={"text_columns": int(len(selected))},
        )


def _infer_time_column(ctx: OperatorContext) -> str | None:
    cols = (ctx.schema_info or {}).get("columns")
    if isinstance(cols, list):
        for c in cols:
            if not isinstance(c, dict):
                continue
            inferred = str(c.get("inferred_type") or "").lower()
            name = c.get("name")
            if inferred in {"datetime", "date"} and name in ctx.df.columns:
                return str(name)
    # Fallback to actual pandas dtype detection.
    dt_cols = ctx.df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    return str(dt_cols[0]) if dt_cols else None


def _pii_columns(ctx: OperatorContext) -> set[str]:
    cols = (ctx.schema_info or {}).get("columns")
    out: set[str] = set()
    if not isinstance(cols, list):
        return out
    for c in cols:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name:
            continue
        is_pii = bool(c.get("is_potential_pii"))
        if not is_pii and isinstance(c.get("statistics"), dict):
            is_pii = bool(c["statistics"].get("is_potential_pii"))
        if is_pii:
            out.add(str(name))
    return out


@dataclass(frozen=True)
class CorrelationOperator:
    name: str = "correlation_matrix"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        numeric = _safe_numeric_df(ctx.df)
        pii = _pii_columns(ctx)
        if pii and not numeric.empty:
            numeric = numeric[[c for c in numeric.columns.tolist() if str(c) not in pii]]
        if numeric.empty:
            out = pd.DataFrame(columns=["column_a", "column_b", "corr"])
            return OperatorResult(
                tables={"correlations": out},
                metrics={"numeric_columns": 0},
                charts={},
                summary={"numeric_columns": 0},
            )

        max_cols = int(params.get("max_columns", 25))
        max_cols = max(2, min(max_cols, 100))
        cols = numeric.columns.tolist()[:max_cols]
        sub = numeric[cols]
        corr = sub.corr(numeric_only=True)

        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                val = corr.loc[a, b]
                if pd.isna(val):
                    continue
                pairs.append({"column_a": a, "column_b": b, "corr": float(val)})
        out = pd.DataFrame(pairs).sort_values("corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

        return OperatorResult(
            tables={"correlations": out.head(200)},
            metrics={"numeric_columns": int(len(cols))},
            charts={},
            summary={"numeric_columns": int(len(cols)), "pairs": int(out.shape[0])},
        )


def _normalize_categorical(series: pd.Series, *, max_levels: int) -> tuple[pd.Series, int]:
    max_levels = max(2, int(max_levels))
    s = series.where(~series.isna(), "__MISSING__").astype(str)
    vc = s.value_counts(dropna=False)
    if int(vc.shape[0]) <= max_levels:
        return s, int(vc.shape[0])

    top = set(vc.head(max_levels - 1).index.tolist())
    s2 = s.where(s.isin(top), "__OTHER__")
    return s2, int(s2.nunique(dropna=False))


@dataclass(frozen=True)
class AssociationScanOperator:
    name: str = "association_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            out = pd.DataFrame(columns=["column_a", "column_b", "association_type", "score", "n", "levels_a", "levels_b"])
            return OperatorResult(
                tables={"associations": out},
                metrics={"pairs": 0},
                charts={},
                summary={"reason": "empty_dataset"},
            )

        max_cats = int(params.get("max_categorical_columns", 20))
        max_cats = max(1, min(max_cats, 200))
        max_nums = int(params.get("max_numeric_columns", 20))
        max_nums = max(1, min(max_nums, 200))
        max_pairs = int(params.get("max_pairs", 200))
        max_pairs = max(10, min(max_pairs, 2000))
        max_levels = int(params.get("max_levels", 50))
        max_levels = max(5, min(max_levels, 200))
        min_group = int(params.get("min_group_size", 10))
        min_group = max(1, min(min_group, 10_000))

        pii = _pii_columns(ctx)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if str(c) not in pii][:max_nums]

        cat_candidates = [c for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist() if str(c) not in pii]
        cat_cols: list[str] = []
        normalized: dict[str, pd.Series] = {}
        levels: dict[str, int] = {}

        for col in cat_candidates:
            if len(cat_cols) >= max_cats:
                break
            try:
                s_norm, lvl = _normalize_categorical(df[col], max_levels=max_levels)
            except Exception:
                continue
            if int(lvl) < 2:
                continue
            cat_cols.append(str(col))
            normalized[str(col)] = s_norm
            levels[str(col)] = int(lvl)

        rows: list[dict[str, Any]] = []

        # categorical-numeric: correlation ratio / eta-squared (bounded and deterministic)
        for c in cat_cols:
            s_cat = normalized[c]
            for ncol in numeric_cols:
                if len(rows) >= max_pairs:
                    break

                sub = pd.DataFrame({"cat": s_cat, "num": df[ncol]}).dropna(subset=["num"])
                if sub.empty:
                    continue

                counts = sub["cat"].value_counts(dropna=False)
                keep = counts[counts >= min_group].index
                sub = sub[sub["cat"].isin(keep)]
                if sub.empty:
                    continue
                if int(sub["cat"].nunique(dropna=False)) < 2:
                    continue

                vals = sub["num"].astype(float)
                overall = float(vals.mean())
                ss_total = float(((vals - overall) ** 2).sum())
                if ss_total <= 0.0:
                    continue

                g = sub.groupby("cat", dropna=False)["num"].agg(["count", "mean"])
                ss_between = float((g["count"] * ((g["mean"] - overall) ** 2)).sum())
                score = ss_between / ss_total

                rows.append(
                    {
                        "column_a": str(c),
                        "column_b": str(ncol),
                        "association_type": "categorical_numeric_eta2",
                        "score": float(score),
                        "n": int(vals.shape[0]),
                        "levels_a": int(sub["cat"].nunique(dropna=False)),
                        "levels_b": None,
                    }
                )

            if len(rows) >= max_pairs:
                break

        # categorical-categorical: Cramer's V (bounded by max_levels)
        for i, a in enumerate(cat_cols):
            if len(rows) >= max_pairs:
                break
            for b in cat_cols[i + 1 :]:
                if len(rows) >= max_pairs:
                    break

                s1 = normalized[a]
                s2 = normalized[b]
                ct = pd.crosstab(s1, s2, dropna=False)
                if ct.empty:
                    continue
                obs = ct.to_numpy(dtype=float, copy=False)
                n = float(obs.sum())
                if n <= 0:
                    continue

                row_sums = obs.sum(axis=1, keepdims=True)
                col_sums = obs.sum(axis=0, keepdims=True)
                expected = (row_sums @ col_sums) / n
                with np.errstate(divide="ignore", invalid="ignore"):
                    chi2 = np.nansum(np.where(expected > 0, ((obs - expected) ** 2) / expected, 0.0))
                phi2 = float(chi2) / n

                r, k = obs.shape
                denom = float(min(k - 1, r - 1))
                if denom <= 0:
                    continue

                v = float(np.sqrt(phi2 / denom))
                if not np.isfinite(v):
                    continue

                rows.append(
                    {
                        "column_a": str(a),
                        "column_b": str(b),
                        "association_type": "categorical_categorical_cramers_v",
                        "score": float(v),
                        "n": int(n),
                        "levels_a": int(levels.get(a) or ct.shape[0]),
                        "levels_b": int(levels.get(b) or ct.shape[1]),
                    }
                )

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("score", ascending=False).head(max_pairs).reset_index(drop=True)
        else:
            out = pd.DataFrame(columns=["column_a", "column_b", "association_type", "score", "n", "levels_a", "levels_b"])

        return OperatorResult(
            tables={"associations": out},
            metrics={
                "pairs": int(out.shape[0]),
                "categorical_columns": int(len(cat_cols)),
                "numeric_columns": int(len(numeric_cols)),
            },
            charts={},
            summary={
                "categorical_columns": int(len(cat_cols)),
                "numeric_columns": int(len(numeric_cols)),
                "pairs": int(out.shape[0]),
                "max_pairs": int(max_pairs),
                "max_levels": int(max_levels),
                "min_group_size": int(min_group),
            },
        )


@dataclass(frozen=True)
class RelationshipExplainOperator:
    name: str = "relationship_explain"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
            return OperatorResult(
                tables={
                    "relationship_summary": empty,
                    "relationship_detail": pd.DataFrame(),
                    "relationship_sample": pd.DataFrame(),
                },
                metrics={},
                charts={},
                summary={"reason": "empty_dataset"},
            )

        col_a = params.get("column_a") or params.get("a")
        col_b = params.get("column_b") or params.get("b")
        col_a = str(col_a or "").strip()
        col_b = str(col_b or "").strip()
        if not col_a or not col_b or col_a not in df.columns or col_b not in df.columns:
            empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
            return OperatorResult(
                tables={
                    "relationship_summary": empty,
                    "relationship_detail": pd.DataFrame(),
                    "relationship_sample": pd.DataFrame(),
                },
                metrics={},
                charts={},
                summary={"reason": "missing_columns", "column_a": col_a or None, "column_b": col_b or None},
            )

        pii = _pii_columns(ctx)
        if col_a in pii or col_b in pii:
            empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
            return OperatorResult(
                tables={
                    "relationship_summary": empty,
                    "relationship_detail": pd.DataFrame(),
                    "relationship_sample": pd.DataFrame(),
                },
                metrics={},
                charts={},
                summary={"reason": "pii_columns_blocked", "column_a": col_a, "column_b": col_b},
            )

        max_levels = int(params.get("max_levels", 30))
        max_levels = max(5, min(max_levels, 200))
        max_points = int(params.get("max_points", 200))
        max_points = max(2, min(max_points, 2000))
        min_group = int(params.get("min_group_size", 10))
        min_group = max(1, min(min_group, 100_000))

        s_a = df[col_a]
        s_b = df[col_b]
        a_num = bool(pd.api.types.is_numeric_dtype(s_a))
        b_num = bool(pd.api.types.is_numeric_dtype(s_b))
        a_dt = bool(pd.api.types.is_datetime64_any_dtype(s_a))
        b_dt = bool(pd.api.types.is_datetime64_any_dtype(s_b))

        if a_dt or b_dt:
            empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
            return OperatorResult(
                tables={
                    "relationship_summary": empty,
                    "relationship_detail": pd.DataFrame(),
                    "relationship_sample": pd.DataFrame(),
                },
                metrics={},
                charts={},
                summary={"reason": "datetime_not_supported", "column_a": col_a, "column_b": col_b},
            )

        if a_num and b_num:
            a_vals = pd.to_numeric(s_a, errors="coerce")
            b_vals = pd.to_numeric(s_b, errors="coerce")
            sub = pd.DataFrame({"a": a_vals, "b": b_vals}).dropna()
            if sub.empty:
                empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
                return OperatorResult(
                    tables={
                        "relationship_summary": empty,
                        "relationship_detail": pd.DataFrame(),
                        "relationship_sample": pd.DataFrame(),
                    },
                    metrics={},
                    charts={},
                    summary={"reason": "no_overlap_after_coerce", "column_a": col_a, "column_b": col_b},
                )

            n = int(sub.shape[0])
            corr = float(sub["a"].corr(sub["b"])) if n >= 2 else float("nan")

            # Binned evidence: quantify how b changes across quantiles of a (bounded).
            detail = pd.DataFrame(columns=["bin", "a_min", "a_max", "count", "b_mean", "b_median"])
            try:
                bins = int(params.get("bins", 10))
                bins = max(4, min(bins, 20))
                bins = min(bins, int(sub["a"].nunique(dropna=True)) or bins)
                if bins >= 2:
                    qbins = pd.qcut(sub["a"], q=bins, duplicates="drop")
                    g = sub.groupby(qbins, observed=False, dropna=False).agg(a_min=("a", "min"), a_max=("a", "max"), count=("a", "size"), b_mean=("b", "mean"), b_median=("b", "median"))
                    detail = g.reset_index().rename(columns={"a": "bin"})
                    detail["bin"] = detail["bin"].astype(str)
                    detail = detail.sort_values("a_min", ascending=True).reset_index(drop=True)
            except Exception:
                detail = pd.DataFrame(columns=["bin", "a_min", "a_max", "count", "b_mean", "b_median"])

            # Deterministic sample: evenly-spaced points over sorted a.
            sample = sub.sort_values("a", ascending=True).reset_index(drop=True)
            if int(sample.shape[0]) > max_points:
                idx = np.linspace(0, int(sample.shape[0]) - 1, num=max_points, dtype=int)
                sample = sample.iloc[idx]
            sample_tbl = sample.rename(columns={"a": col_a, "b": col_b})[[col_a, col_b]].reset_index(drop=True)

            summary = pd.DataFrame(
                [
                    {
                        "relationship_type": "numeric_numeric",
                        "column_a": col_a,
                        "column_b": col_b,
                        "metric": "pearson_r",
                        "metric_value": corr,
                        "n": n,
                    }
                ]
            )

            return OperatorResult(
                tables={
                    "relationship_summary": summary,
                    "relationship_detail": detail,
                    "relationship_sample": sample_tbl,
                },
                metrics={"pearson_r": corr, "n": n},
                charts={},
                summary={"relationship_type": "numeric_numeric", "column_a": col_a, "column_b": col_b, "n": n},
            )

        # categorical-numeric
        if a_num != b_num:
            cat_col = col_a if not a_num else col_b
            num_col = col_b if not a_num else col_a

            s_cat, levels = _normalize_categorical(df[cat_col], max_levels=max_levels)
            s_num = pd.to_numeric(df[num_col], errors="coerce")
            sub = pd.DataFrame({"cat": s_cat, "num": s_num}).dropna(subset=["num"])
            if sub.empty:
                empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
                return OperatorResult(
                    tables={
                        "relationship_summary": empty,
                        "relationship_detail": pd.DataFrame(),
                        "relationship_sample": pd.DataFrame(),
                    },
                    metrics={},
                    charts={},
                    summary={"reason": "no_numeric_values", "column_a": col_a, "column_b": col_b},
                )

            counts = sub["cat"].value_counts(dropna=False)
            keep = counts[counts >= min_group].index
            sub = sub[sub["cat"].isin(keep)]
            if sub.empty or int(sub["cat"].nunique(dropna=False)) < 2:
                empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
                return OperatorResult(
                    tables={
                        "relationship_summary": empty,
                        "relationship_detail": pd.DataFrame(),
                        "relationship_sample": pd.DataFrame(),
                    },
                    metrics={},
                    charts={},
                    summary={"reason": "insufficient_groups", "column_a": col_a, "column_b": col_b, "min_group_size": int(min_group)},
                )

            vals = sub["num"].astype(float)
            n = int(vals.shape[0])
            overall = float(vals.mean())
            ss_total = float(((vals - overall) ** 2).sum())
            eta2 = float("nan")
            if ss_total > 0.0:
                g = sub.groupby("cat", dropna=False)["num"].agg(["count", "mean"])
                ss_between = float((g["count"] * ((g["mean"] - overall) ** 2)).sum())
                eta2 = float(ss_between / ss_total)

            # Detail: per-category stats (bounded).
            detail = sub.groupby("cat", dropna=False)["num"].agg(count="size", mean="mean", median="median", p25=lambda s: float(s.quantile(0.25)), p75=lambda s: float(s.quantile(0.75)))
            detail = detail.reset_index().rename(columns={"cat": "category"})
            detail["category"] = detail["category"].astype(str)
            detail["delta_vs_overall_mean"] = detail["mean"].astype(float) - float(overall)
            detail["_abs_delta"] = detail["delta_vs_overall_mean"].abs()
            detail = detail.sort_values(["_abs_delta", "count"], ascending=[False, False]).drop(columns=["_abs_delta"])
            detail = detail.head(max_levels).reset_index(drop=True)

            summary = pd.DataFrame(
                [
                    {
                        "relationship_type": "categorical_numeric",
                        "column_a": col_a,
                        "column_b": col_b,
                        "metric": "eta2",
                        "metric_value": eta2,
                        "n": n,
                    }
                ]
            )

            return OperatorResult(
                tables={
                    "relationship_summary": summary,
                    "relationship_detail": detail,
                    "relationship_sample": pd.DataFrame(),
                },
                metrics={"eta2": eta2, "n": n, "levels": int(levels)},
                charts={},
                summary={
                    "relationship_type": "categorical_numeric",
                    "column_a": col_a,
                    "column_b": col_b,
                    "n": n,
                    "min_group_size": int(min_group),
                },
            )

        # categorical-categorical
        s1, levels_a = _normalize_categorical(df[col_a], max_levels=max_levels)
        s2, levels_b = _normalize_categorical(df[col_b], max_levels=max_levels)
        ct = pd.crosstab(s1, s2, dropna=False)
        if ct.empty:
            empty = pd.DataFrame(columns=["relationship_type", "column_a", "column_b", "metric", "metric_value", "n"])
            return OperatorResult(
                tables={
                    "relationship_summary": empty,
                    "relationship_detail": pd.DataFrame(),
                    "relationship_sample": pd.DataFrame(),
                },
                metrics={},
                charts={},
                summary={"reason": "empty_crosstab", "column_a": col_a, "column_b": col_b},
            )

        obs = ct.to_numpy(dtype=float, copy=False)
        n = float(obs.sum())
        v = float("nan")
        if n > 0:
            row_sums = obs.sum(axis=1, keepdims=True)
            col_sums = obs.sum(axis=0, keepdims=True)
            expected = (row_sums @ col_sums) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = np.nansum(np.where(expected > 0, ((obs - expected) ** 2) / expected, 0.0))
            phi2 = float(chi2) / float(n)
            r, k = obs.shape
            denom = float(min(k - 1, r - 1))
            if denom > 0:
                v = float(np.sqrt(phi2 / denom))

        # Detail: top co-occurring pairs (bounded).
        long = ct.stack().reset_index()
        long.columns = ["value_a", "value_b", "count"]
        long["value_a"] = long["value_a"].astype(str)
        long["value_b"] = long["value_b"].astype(str)
        long["count"] = long["count"].astype(int)
        long = long.sort_values("count", ascending=False).reset_index(drop=True)
        if n > 0:
            long["pct"] = long["count"].astype(float) / float(n)
        else:
            long["pct"] = np.nan
        long = long.head(250).reset_index(drop=True)

        summary = pd.DataFrame(
            [
                {
                    "relationship_type": "categorical_categorical",
                    "column_a": col_a,
                    "column_b": col_b,
                    "metric": "cramers_v",
                    "metric_value": v,
                    "n": int(n),
                }
            ]
        )

        return OperatorResult(
            tables={
                "relationship_summary": summary,
                "relationship_detail": long,
                "relationship_sample": pd.DataFrame(),
            },
            metrics={"cramers_v": v, "n": int(n), "levels_a": int(levels_a), "levels_b": int(levels_b)},
            charts={},
            summary={"relationship_type": "categorical_categorical", "column_a": col_a, "column_b": col_b, "n": int(n)},
        )


@dataclass(frozen=True)
class ResampleAggregateOperator:
    name: str = "resample_aggregate"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            out = pd.DataFrame(columns=["period", "count", "value_mean", "value_sum"])
            return OperatorResult(
                tables={"resample": out},
                metrics={},
                charts={},
                summary={"reason": "empty_dataset"},
            )

        freq = str(params.get("freq", "M")).upper().strip()
        max_points = int(params.get("max_points", 200))
        max_points = max(10, min(max_points, 5000))

        time_col = params.get("time_column")
        if time_col is None:
            time_col = _infer_time_column(ctx)
        if time_col is None or str(time_col) not in df.columns:
            out = pd.DataFrame(columns=["period", "count", "value_mean", "value_sum"])
            return OperatorResult(
                tables={"resample": out},
                metrics={},
                charts={},
                summary={"reason": "no_time_column_detected"},
            )

        value_col = params.get("value_column")
        if value_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = numeric_cols[0] if numeric_cols else None

        allowed_freqs = ["D", "W", "M"]
        if freq not in allowed_freqs:
            freq = "M"

        sub = df[[str(time_col)] + ([str(value_col)] if value_col and str(value_col) in df.columns else [])].copy()
        sub[str(time_col)] = pd.to_datetime(sub[str(time_col)], errors="coerce", utc=False)
        sub = sub.dropna(subset=[str(time_col)])
        if sub.empty:
            out = pd.DataFrame(columns=["period", "count", "value_mean", "value_sum"])
            return OperatorResult(
                tables={"resample": out},
                metrics={},
                charts={},
                summary={"reason": "no_valid_timestamps"},
            )

        # Choose a frequency that respects max_points (deterministic coarsening only).
        chosen_freq = freq
        for f in allowed_freqs[allowed_freqs.index(freq) :]:
            n_points = int(sub.set_index(str(time_col)).resample(_pandas_resample_freq(f)).size().shape[0])
            chosen_freq = f
            if n_points <= max_points:
                break

        g = sub.set_index(str(time_col)).sort_index().resample(_pandas_resample_freq(chosen_freq))
        out = pd.DataFrame({"period": g.size().index})
        out["count"] = g.size().astype(int).values

        if value_col and str(value_col) in sub.columns and pd.api.types.is_numeric_dtype(sub[str(value_col)]):
            out["value_mean"] = g[str(value_col)].mean().astype(float).values
            out["value_sum"] = g[str(value_col)].sum().astype(float).values
        else:
            out["value_mean"] = np.nan
            out["value_sum"] = np.nan

        periods = pd.to_datetime(out["period"], errors="coerce")
        if getattr(periods.dt, "tz", None) is not None:
            periods = periods.dt.tz_convert(None)
        out["period"] = periods.dt.strftime("%Y-%m-%d %H:%M:%S")

        return OperatorResult(
            tables={"resample": out},
            metrics={
                "points": int(out.shape[0]),
            },
            charts={},
            summary={
                "time_column": str(time_col),
                "value_column": str(value_col) if value_col else None,
                "freq": chosen_freq,
                "points": int(out.shape[0]),
            },
        )


@dataclass(frozen=True)
class TimeAnomalyScanOperator:
    name: str = "time_anomaly_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            return OperatorResult(
                tables={
                    "time_series": pd.DataFrame(columns=["period", "count", "metric", "metric_kind"]),
                    "time_events": pd.DataFrame(columns=["period", "event_type", "metric", "delta", "pct_change"]),
                },
                metrics={},
                charts={},
                summary={"reason": "empty_dataset"},
            )

        time_col = params.get("time_column") or _infer_time_column(ctx)
        if time_col is None or str(time_col) not in df.columns:
            return OperatorResult(
                tables={
                    "time_series": pd.DataFrame(columns=["period", "count", "metric", "metric_kind"]),
                    "time_events": pd.DataFrame(columns=["period", "event_type", "metric", "delta", "pct_change"]),
                },
                metrics={},
                charts={},
                summary={"reason": "no_time_column_detected"},
            )

        freq = str(params.get("freq", "M")).upper().strip()
        max_points = int(params.get("max_points", 200))
        max_points = max(10, min(max_points, 5000))
        allowed_freqs = ["D", "W", "M"]
        if freq not in allowed_freqs:
            freq = "M"

        pii = _pii_columns(ctx)
        value_col = params.get("value_column")
        value_kind = "count"
        blocked_value_col = False
        if value_col is None:
            numeric_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns.tolist() if str(c) not in pii]
            value_col = numeric_cols[0] if numeric_cols else None
        else:
            value_col = str(value_col)
            if value_col in pii:
                blocked_value_col = True
                value_col = None

        cols = [str(time_col)]
        if value_col and str(value_col) in df.columns:
            cols.append(str(value_col))
            value_kind = "sum"

        sub = df[cols].copy()
        sub[str(time_col)] = pd.to_datetime(sub[str(time_col)], errors="coerce", utc=False)
        sub = sub.dropna(subset=[str(time_col)])
        if sub.empty:
            return OperatorResult(
                tables={
                    "time_series": pd.DataFrame(columns=["period", "count", "metric", "metric_kind"]),
                    "time_events": pd.DataFrame(columns=["period", "event_type", "metric", "delta", "pct_change"]),
                },
                metrics={},
                charts={},
                summary={"reason": "no_valid_timestamps"},
            )

        indexed = sub.set_index(str(time_col)).sort_index()
        chosen = freq
        for f in allowed_freqs[allowed_freqs.index(freq) :]:
            n_points = int(indexed.resample(_pandas_resample_freq(f)).size().shape[0])
            chosen = f
            if n_points <= max_points:
                break

        g = indexed.resample(_pandas_resample_freq(chosen))
        counts = g.size().astype(int)

        metric = counts.astype(float)
        if value_col and str(value_col) in indexed.columns and pd.api.types.is_numeric_dtype(indexed[str(value_col)]):
            metric = g[str(value_col)].sum(min_count=1).astype(float)
            value_kind = "sum"
        else:
            value_col = None
            value_kind = "count"

        out = pd.DataFrame({"period": counts.index, "count": counts.values})
        out["metric"] = metric.values
        out["metric_kind"] = value_kind

        # Anomalies: IQR on the metric (bounded, robust).
        m = out["metric"].astype(float)
        m_nonnull = m.dropna()
        iqr_k = float(params.get("iqr_k", 3.0))
        if not np.isfinite(iqr_k):
            iqr_k = 3.0
        iqr_k = float(max(1.0, min(iqr_k, 10.0)))

        lo = hi = None
        anomaly_mask = pd.Series(False, index=out.index)
        if int(m_nonnull.shape[0]) >= 6:
            q1 = float(m_nonnull.quantile(0.25))
            q3 = float(m_nonnull.quantile(0.75))
            iqr = float(q3 - q1)
            if np.isfinite(iqr) and iqr > 0:
                lo = float(q1 - iqr_k * iqr)
                hi = float(q3 + iqr_k * iqr)
                anomaly_mask = (m < lo) | (m > hi)

        # Change points: biggest absolute deltas between adjacent periods.
        d = m.diff()
        abs_d = d.abs()
        max_cps = int(params.get("max_change_points", 10))
        max_cps = max(3, min(max_cps, 50))
        cp_idx = abs_d.dropna().sort_values(ascending=False).head(max_cps).index
        cp_mask = out.index.isin(cp_idx)

        max_events = int(params.get("max_events", 50))
        max_events = max(10, min(max_events, 200))

        events_rows: list[dict[str, Any]] = []
        for i, row in out.iterrows():
            is_anom = bool(anomaly_mask.loc[i]) if i in anomaly_mask.index else False
            is_cp = bool(cp_mask[i]) if i < len(cp_mask) else False
            if not is_anom and not is_cp:
                continue

            metric_val = row.get("metric")
            delta = float(d.loc[i]) if i in d.index and pd.notna(d.loc[i]) else None
            prev = float(m.loc[i - 1]) if i > 0 and pd.notna(m.loc[i - 1]) else None
            pct = None
            if delta is not None and prev is not None and prev != 0:
                pct = float(delta) / float(prev)

            event_type = "change_point" if is_cp and not is_anom else ("anomaly" if is_anom and not is_cp else "anomaly_change_point")
            events_rows.append(
                {
                    "period": row.get("period"),
                    "event_type": event_type,
                    "metric": float(metric_val) if metric_val is not None and pd.notna(metric_val) else None,
                    "delta": delta,
                    "pct_change": pct,
                    "is_anomaly": bool(is_anom),
                    "is_change_point": bool(is_cp),
                }
            )

        events = pd.DataFrame(events_rows)
        if not events.empty:
            events = events.drop_duplicates(subset=["period", "event_type"]).head(max_events).reset_index(drop=True)
        else:
            events = pd.DataFrame(columns=["period", "event_type", "metric", "delta", "pct_change", "is_anomaly", "is_change_point"])

        periods = pd.to_datetime(out["period"], errors="coerce")
        if getattr(periods.dt, "tz", None) is not None:
            periods = periods.dt.tz_convert(None)
        out["period"] = periods.dt.strftime("%Y-%m-%d %H:%M:%S")

        if not events.empty:
            ep = pd.to_datetime(events["period"], errors="coerce")
            if getattr(ep.dt, "tz", None) is not None:
                ep = ep.dt.tz_convert(None)
            events["period"] = ep.dt.strftime("%Y-%m-%d %H:%M:%S")

        anomalies = int(anomaly_mask.sum()) if anomaly_mask is not None else 0
        change_points = int(len(cp_idx)) if cp_idx is not None else 0

        return OperatorResult(
            tables={"time_series": out, "time_events": events},
            metrics={"points": int(out.shape[0]), "anomalies": int(anomalies), "change_points": int(change_points)},
            charts={},
            summary={
                "time_column": str(time_col),
                "value_column": str(value_col) if value_col else None,
                "value_column_blocked": bool(blocked_value_col),
                "freq": str(chosen),
                "points": int(out.shape[0]),
                "anomalies": int(anomalies),
                "change_points": int(change_points),
                "iqr_k": float(iqr_k),
            },
        )


@dataclass(frozen=True)
class SegmentSummaryOperator:
    name: str = "segment_summary"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            out = pd.DataFrame(columns=["group", "count", "value_mean", "value_sum", "value_median"])
            return OperatorResult(
                tables={"segments": out},
                metrics={},
                charts={},
                summary={"reason": "empty_dataset"},
            )

        limit = int(params.get("limit", 50))
        limit = max(5, min(limit, 200))

        group_by = params.get("group_by")
        pii = _pii_columns(ctx)
        if group_by is not None and str(group_by) in pii:
            out = pd.DataFrame(columns=["group", "count", "value_mean", "value_sum", "value_median"])
            return OperatorResult(
                tables={"segments": out},
                metrics={},
                charts={},
                summary={"reason": "group_by_is_pii"},
            )
        if group_by is None:
            non_numeric = df.select_dtypes(exclude=[np.number]).copy()
            candidates: list[tuple[str, int]] = []
            for col in non_numeric.columns.tolist():
                if str(col) in pii:
                    continue
                try:
                    nunique = int(non_numeric[col].nunique(dropna=True))
                except Exception:
                    continue
                if 2 <= nunique <= 50:
                    candidates.append((str(col), nunique))
            # Deterministic: pick the highest-cardinality categorical within bounds.
            candidates.sort(key=lambda x: x[1], reverse=True)
            group_by = candidates[0][0] if candidates else None

        if group_by is None or str(group_by) not in df.columns:
            out = pd.DataFrame(columns=["group", "count", "value_mean", "value_sum", "value_median"])
            return OperatorResult(
                tables={"segments": out},
                metrics={},
                charts={},
                summary={"reason": "no_categorical_group_column_detected"},
            )

        value_column = params.get("value_column")
        if value_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_column = numeric_cols[0] if numeric_cols else None

        # Always compute count; compute mean/sum when a numeric column exists.
        g = df.groupby(str(group_by), dropna=False)
        counts = g.size().rename("count")

        out = counts.reset_index().rename(columns={str(group_by): "group"})
        out["group"] = out["group"].astype(str)

        if value_column and str(value_column) in df.columns and pd.api.types.is_numeric_dtype(df[str(value_column)]):
            means = g[str(value_column)].mean().rename("value_mean")
            sums = g[str(value_column)].sum().rename("value_sum")
            medians = g[str(value_column)].median().rename("value_median")
            means_df = means.reset_index().rename(columns={str(group_by): "group"})
            means_df["group"] = means_df["group"].astype(str)
            sums_df = sums.reset_index().rename(columns={str(group_by): "group"})
            sums_df["group"] = sums_df["group"].astype(str)
            medians_df = medians.reset_index().rename(columns={str(group_by): "group"})
            medians_df["group"] = medians_df["group"].astype(str)
            out = out.merge(means_df, on="group", how="left")
            out = out.merge(sums_df, on="group", how="left")
            out = out.merge(medians_df, on="group", how="left")
        else:
            out["value_mean"] = np.nan
            out["value_sum"] = np.nan
            out["value_median"] = np.nan

        sort_by = str(params.get("sort_by") or "count").strip().lower()
        ascending = bool(params.get("ascending", False))
        if sort_by not in {"count", "value_sum", "value_mean", "value_median"}:
            sort_by = "count"
        if sort_by != "count" and sort_by not in out.columns:
            sort_by = "count"
        if sort_by != "count" and out[sort_by].isna().all():
            sort_by = "count"

        sort_cols = [sort_by]
        sort_flags = [ascending]
        if sort_by != "count":
            sort_cols.append("count")
            sort_flags.append(False)

        out = out.sort_values(sort_cols, ascending=sort_flags).head(limit).reset_index(drop=True)

        return OperatorResult(
            tables={"segments": out},
            metrics={"segments": int(out.shape[0])},
            charts={},
            summary={
                "group_by": str(group_by),
                "value_column": str(value_column) if value_column else None,
                "limit": int(limit),
                "sort_by": str(sort_by),
                "ascending": bool(ascending),
            },
        )


@dataclass(frozen=True)
class OutlierScanOperator:
    name: str = "outlier_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        numeric = _safe_numeric_df(ctx.df)
        if numeric.empty:
            out = pd.DataFrame(columns=["column", "outlier_count", "outlier_pct", "method"])
            return OperatorResult(
                tables={"outliers": out},
                metrics={"numeric_columns": 0},
                charts={},
                summary={"numeric_columns": 0},
            )

        max_cols = int(params.get("max_columns", 25))
        max_cols = max(1, min(max_cols, 100))
        cols = numeric.columns.tolist()[:max_cols]
        sub = numeric[cols]

        rows: list[dict[str, Any]] = []
        total = float(sub.shape[0])
        for col in cols:
            series = sub[col].dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outliers = ((series < lo) | (series > hi)).sum()
            rows.append(
                {
                    "column": col,
                    "outlier_count": int(outliers),
                    "outlier_pct": float(outliers) / total if total > 0 else 0.0,
                    "method": "iqr",
                    "lower_bound": lo,
                    "upper_bound": hi,
                }
            )

        out = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)
        return OperatorResult(
            tables={"outliers": out},
            metrics={"numeric_columns": int(len(cols))},
            charts={},
            summary={"numeric_columns": int(len(cols)), "scanned": int(len(rows))},
        )


@dataclass(frozen=True)
class PrivacyRiskScanOperator:
    name: str = "privacy_risk_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        rows: list[dict[str, Any]] = []
        cols = (ctx.schema_info or {}).get("columns")
        if isinstance(cols, list) and cols:
            for c in cols:
                if not isinstance(c, dict):
                    continue
                name = c.get("name")
                if not name:
                    continue

                stats = c.get("statistics") if isinstance(c.get("statistics"), dict) else {}
                is_pii = bool(c.get("is_potential_pii") or stats.get("is_potential_pii"))
                is_uid = bool(c.get("is_unique_identifier") or stats.get("is_unique_identifier"))

                rows.append(
                    {
                        "column": str(name),
                        "inferred_type": str(c.get("inferred_type") or ""),
                        "is_potential_pii": bool(is_pii),
                        "is_unique_identifier": bool(is_uid),
                        "null_percentage": c.get("null_percentage"),
                        "unique_percentage": c.get("unique_percentage"),
                        "is_constant": bool(c.get("is_constant") or stats.get("is_constant")),
                        "has_outliers": bool(c.get("has_outliers") or stats.get("has_outliers")),
                    }
                )
        else:
            for col in ctx.df.columns.tolist():
                rows.append(
                    {
                        "column": str(col),
                        "inferred_type": str(ctx.df[str(col)].dtype),
                        "is_potential_pii": False,
                        "is_unique_identifier": False,
                        "null_percentage": None,
                        "unique_percentage": None,
                        "is_constant": False,
                        "has_outliers": False,
                    }
                )

        out = pd.DataFrame(rows)
        pii_cols = int(out["is_potential_pii"].sum()) if not out.empty and "is_potential_pii" in out.columns else 0
        uid_cols = int(out["is_unique_identifier"].sum()) if not out.empty and "is_unique_identifier" in out.columns else 0

        return OperatorResult(
            tables={"risk": out},
            metrics={"pii_columns": int(pii_cols), "identifier_columns": int(uid_cols), "columns": int(out.shape[0])},
            charts={},
            summary={"pii_columns": int(pii_cols), "identifier_columns": int(uid_cols), "columns": int(out.shape[0])},
        )


@dataclass(frozen=True)
class MissingnessPatternsOperator:
    name: str = "missingness_patterns"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            return OperatorResult(
                tables={
                    "missing_columns": pd.DataFrame(columns=["column", "null_count", "null_pct"]),
                    "missingness_by_category": pd.DataFrame(
                        columns=["target_column", "group_column", "group_value", "group_count", "missing_pct", "delta_vs_overall"]
                    ),
                    "missingness_numeric_assoc": pd.DataFrame(
                        columns=["target_column", "related_numeric", "mean_missing", "mean_present", "delta", "missing_count"]
                    ),
                    "missingness_over_time": pd.DataFrame(columns=["period", "target_column", "missing_pct", "count"]),
                },
                metrics={},
                charts={},
                summary={"reason": "empty_dataset"},
            )

        pii = _pii_columns(ctx)
        top_n = int(params.get("top_columns", 3))
        top_n = max(1, min(top_n, 8))

        cols_param = params.get("columns") or params.get("column")
        if isinstance(cols_param, str):
            cols_param = [c.strip() for c in cols_param.split(",") if c.strip()]
        selected: list[str] = []
        if isinstance(cols_param, list):
            selected = [str(c) for c in cols_param if str(c) and str(c) in df.columns]

        total_rows = int(df.shape[0])
        null_counts = df.isna().sum().astype(int)
        missing_tbl = (
            pd.DataFrame({"column": null_counts.index, "null_count": null_counts.values})
            .assign(null_pct=lambda x: x["null_count"] / float(max(total_rows, 1)))
            .sort_values("null_pct", ascending=False)
            .reset_index(drop=True)
        )

        if not selected:
            selected = (
                missing_tbl[missing_tbl["null_count"] > 0]["column"].astype(str).head(top_n).tolist()
            )
        selected = selected[:top_n]

        missing_cols = missing_tbl[missing_tbl["column"].astype(str).isin(selected)].copy().reset_index(drop=True)

        # Choose bounded categorical columns for grouping (exclude PII/time/high-cardinality).
        time_col = params.get("time_column") or _infer_time_column(ctx)
        max_cat = int(params.get("max_categorical_columns", 10))
        max_cat = max(1, min(max_cat, 25))
        max_unique = int(params.get("max_categorical_unique", 50))
        max_unique = max(2, min(max_unique, 200))
        min_group_count = int(params.get("min_group_count", 25))
        min_group_count = max(1, min(min_group_count, 1000))
        k = int(params.get("k", 8))
        k = max(3, min(k, 25))

        cat_candidates: list[str] = []
        non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in non_num:
            c = str(col)
            if c in pii:
                continue
            if time_col and c == str(time_col):
                continue
            try:
                nunique = int(df[c].nunique(dropna=True))
            except Exception:
                continue
            if 2 <= nunique <= max_unique:
                cat_candidates.append(c)
        cat_candidates = cat_candidates[:max_cat]

        by_cat_rows: list[dict[str, Any]] = []
        num_assoc_rows: list[dict[str, Any]] = []

        # Numeric association candidates.
        numeric_cols = _safe_numeric_df(df).columns.tolist()
        numeric_cols = [str(c) for c in numeric_cols if str(c) not in pii]
        max_numeric_assoc = int(params.get("max_numeric_assoc", 20))
        max_numeric_assoc = max(5, min(max_numeric_assoc, 100))
        numeric_cols = numeric_cols[:max_numeric_assoc]

        for target in selected:
            if target not in df.columns:
                continue
            miss = df[target].isna()
            miss_n = int(miss.sum())
            if miss_n <= 0:
                continue
            overall = float(miss_n) / float(max(total_rows, 1))

            # Missingness by category.
            for gcol in cat_candidates:
                s = df[gcol].astype("string").fillna("NULL")
                tmp = pd.DataFrame({"g": s, "m": miss.astype(int)})
                try:
                    gsize = tmp.groupby("g", dropna=False).size().rename("group_count")
                    gmiss = tmp.groupby("g", dropna=False)["m"].sum().rename("missing_count")
                except Exception:
                    continue
                out = pd.concat([gsize, gmiss], axis=1).reset_index().rename(columns={"g": "group_value"})
                out["missing_pct"] = out["missing_count"].astype(float) / out["group_count"].astype(float).clip(lower=1.0)
                out = out[out["group_count"] >= int(min_group_count)]
                if out.empty:
                    continue
                out["delta_vs_overall"] = out["missing_pct"].astype(float) - float(overall)
                out = out.sort_values(["missing_pct", "group_count"], ascending=[False, False]).head(k)
                for _, r in out.iterrows():
                    by_cat_rows.append(
                        {
                            "target_column": str(target),
                            "group_column": str(gcol),
                            "group_value": str(r.get("group_value") or ""),
                            "group_count": int(r.get("group_count") or 0),
                            "missing_pct": float(r.get("missing_pct") or 0.0),
                            "delta_vs_overall": float(r.get("delta_vs_overall") or 0.0),
                        }
                    )

            # Missingness association with numeric features (mean shift).
            present = ~miss
            present_n = int(present.sum())
            if present_n <= 0:
                continue

            for ncol in numeric_cols:
                if ncol == str(target):
                    continue
                try:
                    s = pd.to_numeric(df[ncol], errors="coerce")
                except Exception:
                    continue
                m0 = float(s[present].mean()) if present_n > 0 else float("nan")
                m1 = float(s[miss].mean()) if miss_n > 0 else float("nan")
                if not np.isfinite(m0) or not np.isfinite(m1):
                    continue
                delta = float(m1 - m0)
                num_assoc_rows.append(
                    {
                        "target_column": str(target),
                        "related_numeric": str(ncol),
                        "mean_missing": float(m1),
                        "mean_present": float(m0),
                        "delta": delta,
                        "missing_count": int(miss_n),
                    }
                )

        by_cat = pd.DataFrame(by_cat_rows)
        num_assoc = pd.DataFrame(num_assoc_rows)
        if not num_assoc.empty and "delta" in num_assoc.columns:
            num_assoc["_abs_delta"] = num_assoc["delta"].abs()
            num_assoc = num_assoc.sort_values("_abs_delta", ascending=False).drop(columns=["_abs_delta"]).head(200).reset_index(drop=True)

        # Missingness over time (optional).
        over_time = pd.DataFrame(columns=["period", "target_column", "missing_pct", "count"])
        chosen_freq = None
        if time_col and str(time_col) in df.columns:
            freq = str(params.get("freq", "M")).upper().strip()
            max_points = int(params.get("max_points", 200))
            max_points = max(10, min(max_points, 5000))
            allowed_freqs = ["D", "W", "M"]
            if freq not in allowed_freqs:
                freq = "M"

            sub = df[[str(time_col)] + [c for c in selected if c in df.columns]].copy()
            sub[str(time_col)] = pd.to_datetime(sub[str(time_col)], errors="coerce", utc=False)
            sub = sub.dropna(subset=[str(time_col)])
            if not sub.empty:
                flags = pd.DataFrame({c: sub[c].isna().astype(int) for c in selected if c in sub.columns})
                tmp = pd.concat([sub[[str(time_col)]].reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
                tmp = tmp.set_index(str(time_col)).sort_index()

                chosen = freq
                for f in allowed_freqs[allowed_freqs.index(freq) :]:
                    n_points = int(tmp.resample(_pandas_resample_freq(f)).size().shape[0])
                    chosen = f
                    if n_points <= max_points:
                        break
                chosen_freq = chosen

                g = tmp.resample(_pandas_resample_freq(chosen))
                counts = g.size().astype(int)
                sums = g.sum(numeric_only=True)
                rows: list[dict[str, Any]] = []
                for col in sums.columns.tolist():
                    miss = sums[col].astype(float)
                    denom = counts.astype(float).clip(lower=1.0)
                    pct = (miss / denom).astype(float)
                    for idx, val in pct.items():
                        rows.append(
                            {
                                "period": idx,
                                "target_column": str(col),
                                "missing_pct": float(val),
                                "count": int(counts.loc[idx]),
                            }
                        )
                over_time = pd.DataFrame(rows)
                if not over_time.empty:
                    periods = pd.to_datetime(over_time["period"], errors="coerce")
                    if getattr(periods.dt, "tz", None) is not None:
                        periods = periods.dt.tz_convert(None)
                    over_time["period"] = periods.dt.strftime("%Y-%m-%d %H:%M:%S")

        return OperatorResult(
            tables={
                "missing_columns": missing_cols,
                "missingness_by_category": by_cat,
                "missingness_numeric_assoc": num_assoc,
                "missingness_over_time": over_time,
            },
            metrics={"targets": int(len(selected))},
            charts={},
            summary={
                "targets": [str(c) for c in selected],
                "time_column": str(time_col) if time_col and str(time_col) in df.columns else None,
                "freq": chosen_freq,
            },
        )


@dataclass(frozen=True)
class OutlierExplainOperator:
    name: str = "outlier_explain"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        numeric = _safe_numeric_df(ctx.df)
        pii = _pii_columns(ctx)
        if not pii:
            pii = set()
        if not numeric.empty:
            numeric = numeric[[c for c in numeric.columns.tolist() if str(c) not in pii]]

        if numeric.empty:
            return OperatorResult(
                tables={
                    "outlier_columns": pd.DataFrame(columns=["column", "outlier_count", "outlier_pct", "lower_bound", "upper_bound"]),
                    "outlier_quantiles": pd.DataFrame(columns=["column", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "min", "max"]),
                    "outliers_over_time": pd.DataFrame(columns=["period", "outlier_rows", "outlier_row_pct", "count"]),
                },
                metrics={"numeric_columns": 0},
                charts={},
                summary={"numeric_columns": 0},
            )

        top_n = int(params.get("top_columns", 3))
        top_n = max(1, min(top_n, 8))

        cols_param = params.get("columns") or params.get("column")
        if isinstance(cols_param, str):
            cols_param = [c.strip() for c in cols_param.split(",") if c.strip()]
        selected: list[str] = []
        if isinstance(cols_param, list):
            selected = [str(c) for c in cols_param if str(c) and str(c) in numeric.columns]

        # Auto-pick columns by outlier_pct when not specified.
        total = float(numeric.shape[0])
        bounds: dict[str, tuple[float, float, int, float]] = {}
        rows: list[dict[str, Any]] = []
        for col in numeric.columns.tolist():
            series = numeric[col].dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outliers = int(((series < lo) | (series > hi)).sum())
            pct = float(outliers) / float(total) if total > 0 else 0.0
            bounds[str(col)] = (lo, hi, outliers, pct)
            rows.append({"column": str(col), "outlier_count": outliers, "outlier_pct": pct, "lower_bound": lo, "upper_bound": hi})

        scan = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)
        if not selected:
            selected = scan["column"].astype(str).head(top_n).tolist()
        selected = selected[:top_n]

        explain_cols = scan[scan["column"].astype(str).isin(selected)].copy().reset_index(drop=True)

        q_rows: list[dict[str, Any]] = []
        for col in selected:
            s = numeric[str(col)].dropna()
            if s.empty:
                continue
            q = s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
            q_rows.append(
                {
                    "column": str(col),
                    "p01": float(q.get(0.01)),
                    "p05": float(q.get(0.05)),
                    "p25": float(q.get(0.25)),
                    "p50": float(q.get(0.5)),
                    "p75": float(q.get(0.75)),
                    "p95": float(q.get(0.95)),
                    "p99": float(q.get(0.99)),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
            )
        quantiles = pd.DataFrame(q_rows)

        over_time = pd.DataFrame(columns=["period", "outlier_rows", "outlier_row_pct", "count"])
        time_col = params.get("time_column") or _infer_time_column(ctx)
        chosen_freq = None
        if time_col and str(time_col) in ctx.df.columns and selected:
            freq = str(params.get("freq", "M")).upper().strip()
            max_points = int(params.get("max_points", 200))
            max_points = max(10, min(max_points, 5000))
            allowed_freqs = ["D", "W", "M"]
            if freq not in allowed_freqs:
                freq = "M"

            sub = ctx.df[[str(time_col)]].copy()
            sub[str(time_col)] = pd.to_datetime(sub[str(time_col)], errors="coerce", utc=False)
            sub = sub.dropna(subset=[str(time_col)])
            if not sub.empty:
                # Combined outlier mask across selected columns (bounded output).
                mask_any = None
                for col in selected:
                    b = bounds.get(str(col))
                    if not b:
                        continue
                    lo, hi, _, _ = b
                    s = pd.to_numeric(ctx.df[str(col)], errors="coerce")
                    m = (s < lo) | (s > hi)
                    mask_any = m if mask_any is None else (mask_any | m)
                if mask_any is not None:
                    tmp = pd.DataFrame({"t": sub[str(time_col)].values, "out": mask_any.loc[sub.index].astype(int).values})
                    tmp["t"] = pd.to_datetime(tmp["t"], errors="coerce", utc=False)
                    tmp = tmp.dropna(subset=["t"])
                    if not tmp.empty:
                        tmp = tmp.set_index("t").sort_index()
                        chosen = freq
                        for f in allowed_freqs[allowed_freqs.index(freq) :]:
                            n_points = int(tmp.resample(_pandas_resample_freq(f)).size().shape[0])
                            chosen = f
                            if n_points <= max_points:
                                break
                        chosen_freq = chosen
                        g = tmp.resample(_pandas_resample_freq(chosen))
                        counts = g.size().astype(int)
                        outs = g["out"].sum().astype(int)
                        outdf = pd.DataFrame({"period": counts.index, "count": counts.values, "outlier_rows": outs.values})
                        outdf["outlier_row_pct"] = outdf["outlier_rows"].astype(float) / outdf["count"].astype(float).clip(lower=1.0)
                        periods = pd.to_datetime(outdf["period"], errors="coerce")
                        if getattr(periods.dt, "tz", None) is not None:
                            periods = periods.dt.tz_convert(None)
                        outdf["period"] = periods.dt.strftime("%Y-%m-%d %H:%M:%S")
                        over_time = outdf[["period", "outlier_rows", "outlier_row_pct", "count"]]

        return OperatorResult(
            tables={"outlier_columns": explain_cols, "outlier_quantiles": quantiles, "outliers_over_time": over_time},
            metrics={"numeric_columns": int(numeric.shape[1]), "targets": int(len(selected))},
            charts={},
            summary={
                "targets": [str(c) for c in selected],
                "time_column": str(time_col) if time_col and str(time_col) in ctx.df.columns else None,
                "freq": chosen_freq,
            },
        )


@dataclass(frozen=True)
class SegmentDeepDiveOperator:
    name: str = "segment_deep_dive"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or int(df.shape[0]) == 0:
            return OperatorResult(
                tables={
                    "segment_summary": pd.DataFrame(columns=["group", "count", "value_mean", "value_sum", "value_median"]),
                    "segment_numeric_diff": pd.DataFrame(columns=["group", "column", "group_mean", "overall_mean", "delta", "count"]),
                },
                metrics={},
                charts={},
                summary={"reason": "empty_dataset"},
            )

        pii = _pii_columns(ctx)
        limit = int(params.get("limit", 10))
        limit = max(3, min(limit, 25))

        group_by = params.get("group_by")
        if group_by is None:
            non_numeric = df.select_dtypes(exclude=[np.number]).copy()
            candidates: list[tuple[str, int]] = []
            for col in non_numeric.columns.tolist():
                c = str(col)
                if c in pii:
                    continue
                try:
                    nunique = int(non_numeric[c].nunique(dropna=True))
                except Exception:
                    continue
                if 2 <= nunique <= 50:
                    candidates.append((c, nunique))
            candidates.sort(key=lambda x: x[1], reverse=True)
            group_by = candidates[0][0] if candidates else None

        if group_by is None or str(group_by) not in df.columns:
            return OperatorResult(
                tables={
                    "segment_summary": pd.DataFrame(columns=["group", "count", "value_mean", "value_sum", "value_median"]),
                    "segment_numeric_diff": pd.DataFrame(columns=["group", "column", "group_mean", "overall_mean", "delta", "count"]),
                },
                metrics={},
                charts={},
                summary={"reason": "no_segment_column_detected"},
            )

        value_col = params.get("value_column")
        if value_col is None:
            numeric_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns.tolist() if str(c) not in pii]
            value_col = numeric_cols[0] if numeric_cols else None

        g = df.groupby(str(group_by), dropna=False)
        counts = g.size().rename("count")
        out = counts.reset_index().rename(columns={str(group_by): "group"})
        out["group"] = out["group"].astype(str)

        if value_col and str(value_col) in df.columns and pd.api.types.is_numeric_dtype(df[str(value_col)]):
            out["value_mean"] = g[str(value_col)].mean().astype(float).values
            out["value_sum"] = g[str(value_col)].sum().astype(float).values
            out["value_median"] = g[str(value_col)].median().astype(float).values
        else:
            out["value_mean"] = np.nan
            out["value_sum"] = np.nan
            out["value_median"] = np.nan

        out = out.sort_values(["count"], ascending=False).head(limit).reset_index(drop=True)

        # Numeric diffs vs overall (bounded).
        max_numeric = int(params.get("max_numeric_columns", 8))
        max_numeric = max(1, min(max_numeric, 25))
        numeric_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns.tolist() if str(c) not in pii]
        numeric_cols = numeric_cols[:max_numeric]
        diff_rows: list[dict[str, Any]] = []
        if numeric_cols:
            overall = df[numeric_cols].mean(numeric_only=True)
            top_groups = out["group"].astype(str).tolist()
            sub = df[df[str(group_by)].astype(str).isin(top_groups)][[str(group_by)] + numeric_cols].copy()
            sub[str(group_by)] = sub[str(group_by)].astype(str)
            means = sub.groupby(str(group_by), dropna=False)[numeric_cols].mean(numeric_only=True)
            sizes = sub.groupby(str(group_by), dropna=False).size()
            for grp in means.index.tolist():
                for col in numeric_cols:
                    try:
                        gm = float(means.loc[grp, col])
                        om = float(overall[col])
                    except Exception:
                        continue
                    if not np.isfinite(gm) or not np.isfinite(om):
                        continue
                    diff_rows.append(
                        {
                            "group": str(grp),
                            "column": str(col),
                            "group_mean": gm,
                            "overall_mean": om,
                            "delta": float(gm - om),
                            "count": int(sizes.loc[grp]) if grp in sizes.index else None,
                        }
                    )
        diffs = pd.DataFrame(diff_rows)
        if not diffs.empty and "delta" in diffs.columns:
            diffs["_abs"] = diffs["delta"].abs()
            diffs = diffs.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(250).reset_index(drop=True)

        return OperatorResult(
            tables={"segment_summary": out, "segment_numeric_diff": diffs},
            metrics={"group_by": str(group_by), "segments": int(out.shape[0])},
            charts={},
            summary={"group_by": str(group_by), "value_column": str(value_col) if value_col else None, "segments": int(out.shape[0])},
        )


P0_EDA_OPERATORS = [
    SchemaSnapshotOperator(),
    DatasetOverviewOperator(),
    PreviewRowsOperator(),
    MissingnessOperator(),
    UniquenessOperator(),
    NumericSummaryOperator(),
    CategoricalTopKOperator(),
    TextSummaryOperator(),
    CorrelationOperator(),
    AssociationScanOperator(),
    RelationshipExplainOperator(),
    ResampleAggregateOperator(),
    TimeAnomalyScanOperator(),
    SegmentSummaryOperator(),
    OutlierScanOperator(),
    PrivacyRiskScanOperator(),
    MissingnessPatternsOperator(),
    OutlierExplainOperator(),
    SegmentDeepDiveOperator(),
]
