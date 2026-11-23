from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.compute.chart_helpers import build_chart, chart_to_spec
from app.compute.operators.base import OperatorContext, OperatorResult
from app.ml.cohort_analysis import CohortAnalysisEngine, CohortConfig, CohortPeriod, CohortType
from app.ml.funnel_analysis import FunnelAnalysisEngine, FunnelConfig
from app.ml.rfm_analysis import RFMAnalysisEngine, RFMConfig
from app.ml.visualization import ChartType, VisualizationEngine


def _pick_col(df: pd.DataFrame, hints: list[str], *, numeric: bool = False) -> str | None:
    for c in df.columns:
        cn = str(c).lower()
        if any(h in cn for h in hints):
            if not numeric or pd.api.types.is_numeric_dtype(df[c]):
                return str(c)
    if numeric:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return str(nums[0]) if nums else None
    for c in df.columns:
        if df[c].dtype == object:
            return str(c)
    return str(df.columns[0]) if len(df.columns) else None


@dataclass(frozen=True)
class CohortScanOperator:
    name: str = "cohort_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        customer_col = params.get("customer_id_column") or _pick_col(df, ["customer", "user", "account", "client", "id"])
        date_col = params.get("date_column") or _pick_col(df, ["date", "time", "created", "signup", "order"])
        if not customer_col or not date_col:
            raise ValueError("Need customer and date columns for cohort analysis")

        period = str(params.get("period") or "M").upper()
        config = CohortConfig(
            cohort_type=CohortType.RETENTION,
            period=CohortPeriod(period) if period in {"D", "W", "M", "Q", "Y"} else CohortPeriod.MONTHLY,
            max_periods=int(params.get("max_periods") or 8),
        )
        result = CohortAnalysisEngine(config=config, verbose=False).analyze(df, customer_id_col=customer_col, date_col=date_col)

        matrix = result.cohort_matrix
        if matrix is None or matrix.empty:
            raise ValueError("Cohort matrix could not be computed")

        table = matrix.reset_index().rename(columns={"index": "cohort"})
        charts = {}
        try:
            engine = VisualizationEngine()
            chart = engine.create_chart(
                matrix.reset_index(),
                chart_type=ChartType.HEATMAP,
                title="Cohort retention heatmap",
            )
            charts["cohort_heatmap"] = chart_to_spec(chart)
        except Exception:
            pass

        summary = {
            "customer_column": customer_col,
            "date_column": date_col,
            "n_cohorts": int(result.n_cohorts),
            "n_customers": int(result.n_customers),
            "avg_retention_period_1": float((result.avg_retention or {}).get(1, 0)),
        }
        return OperatorResult(
            tables={"cohort_matrix": table},
            metrics=summary,
            charts=charts,
            summary=summary,
        )


@dataclass(frozen=True)
class FunnelScanOperator:
    name: str = "funnel_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        user_col = params.get("user_id_column") or _pick_col(df, ["user", "customer", "session", "visitor"])
        event_col = params.get("event_column") or _pick_col(df, ["event", "step", "stage", "action", "status"])
        steps = params.get("steps")
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.split(",") if s.strip()]

        config = FunnelConfig(user_id_col=user_col, event_col=event_col, steps=steps)
        result = FunnelAnalysisEngine(config=config, verbose=False).analyze(df, user_id_col=user_col, event_col=event_col, steps=steps)

        rows = []
        for step in result.steps:
            rows.append(
                {
                    "step": step.name,
                    "count": int(step.count),
                    "conversion_rate": round(float(step.conversion_rate) * 100, 2),
                    "overall_rate": round(float(step.overall_rate) * 100, 2),
                    "drop_off_rate": round(float(step.drop_off_rate) * 100, 2),
                }
            )
        table = pd.DataFrame(rows)

        charts = {}
        if not table.empty:
            charts["funnel_chart"] = build_chart(
                table,
                chart_type=ChartType.FUNNEL,
                x="step",
                y="count",
                title="Conversion funnel",
            )

        summary = {
            "user_column": user_col,
            "event_column": event_col,
            "overall_conversion_pct": round(float(result.overall_conversion) * 100, 2),
            "biggest_drop_step": result.biggest_drop_step,
            "biggest_drop_rate_pct": round(float(result.biggest_drop_rate) * 100, 2),
        }
        return OperatorResult(
            tables={"funnel": table},
            metrics=summary,
            charts=charts,
            summary=summary,
        )


@dataclass(frozen=True)
class RFMScanOperator:
    name: str = "rfm_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        customer_col = params.get("customer_id_column")
        date_col = params.get("date_column")
        amount_col = params.get("amount_column") or _pick_col(df, ["amount", "revenue", "sales", "price", "total"], numeric=True)

        result = RFMAnalysisEngine(config=RFMConfig(), verbose=False).analyze(
            df,
            customer_id_col=customer_col,
            date_col=date_col,
            amount_col=amount_col,
        )

        seg = result.segment_summary or {}
        seg_rows = []
        for name, stats in seg.items():
            if not isinstance(stats, dict):
                continue
            seg_rows.append({"segment": str(name), **stats})
        seg_table = pd.DataFrame(seg_rows) if seg_rows else pd.DataFrame([{"note": "no segments"}])

        rfm_sample = result.customer_rfm.head(50) if isinstance(result.customer_rfm, pd.DataFrame) else pd.DataFrame()

        charts = {}
        if seg_rows:
            counts = pd.DataFrame([{"Segment": r["segment"], "customers": r.get("count", 0)} for r in seg_rows])
            charts["rfm_segments"] = build_chart(
                counts,
                chart_type=ChartType.BAR,
                x="Segment",
                y="customers",
                title="RFM customer segments",
            )

        top_segment = seg_rows[0]["segment"] if seg_rows else ""
        summary = {
            "n_customers": int(result.n_customers),
            "top_segment": top_segment,
            "segments": len(seg_rows),
        }
        return OperatorResult(
            tables={"rfm_segments": seg_table, "rfm_customers_sample": rfm_sample},
            metrics={"n_customers": int(result.n_customers)},
            charts=charts,
            summary=summary,
        )


@dataclass(frozen=True)
class ClusteringScanOperator:
    name: str = "clustering_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        max_cols = int(params.get("max_columns") or 8)
        k = int(params.get("n_clusters") or 4)
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:max_cols]
        if len(numeric) < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")

        sample = df[numeric].dropna().head(int(params.get("sample_rows") or 5000))
        if len(sample) < k:
            raise ValueError("Not enough rows for clustering")

        scaled = StandardScaler().fit_transform(sample)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        out = sample.copy()
        out["cluster"] = labels

        summary_rows = []
        for cl in sorted(out["cluster"].unique()):
            sub = out[out["cluster"] == cl]
            summary_rows.append({"cluster": int(cl), "size": int(len(sub)), **{c: round(float(sub[c].mean()), 4) for c in numeric[:4]}})
        table = pd.DataFrame(summary_rows)

        charts = {}
        if len(numeric) >= 2:
            scatter_df = out[[numeric[0], numeric[1], "cluster"]].head(500)
            charts["cluster_scatter"] = build_chart(
                scatter_df,
                chart_type=ChartType.SCATTER,
                x=numeric[0],
                y=numeric[1],
                color="cluster",
                title=f"K-Means clusters (k={k})",
            )

        return OperatorResult(
            tables={"clusters": table},
            metrics={"n_clusters": k, "rows_clustered": int(len(sample))},
            charts=charts,
            summary={"n_clusters": k, "features": numeric},
        )


P0_BUSINESS_OPERATORS = [
    CohortScanOperator(),
    FunnelScanOperator(),
    RFMScanOperator(),
    ClusteringScanOperator(),
]