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


@dataclass(frozen=True)
class SchemaSnapshotOperator:
    name: str = "schema_snapshot"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        columns = []
        schema_columns = (ctx.schema_info or {}).get("columns")
        if isinstance(schema_columns, list):
            for col in schema_columns:
                if isinstance(col, dict):
                    columns.append(col)

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
class PreviewRowsOperator:
    name: str = "preview_rows"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        limit = int(params.get("limit", 25))
        limit = max(1, min(limit, 200))
        out = ctx.df.head(limit).copy()
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
        k = int(params.get("k", 10))
        k = max(3, min(k, 50))
        max_columns = int(params.get("max_columns", 10))
        max_columns = max(1, min(max_columns, 50))

        # Heuristic: choose low-cardinality non-numeric columns.
        non_numeric = df.select_dtypes(exclude=[np.number])
        candidates: list[str] = []
        for col in non_numeric.columns.tolist():
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
class CorrelationOperator:
    name: str = "correlation_matrix"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        numeric = _safe_numeric_df(ctx.df)
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


P0_EDA_OPERATORS = [
    SchemaSnapshotOperator(),
    PreviewRowsOperator(),
    MissingnessOperator(),
    NumericSummaryOperator(),
    CategoricalTopKOperator(),
    CorrelationOperator(),
    OutlierScanOperator(),
]

