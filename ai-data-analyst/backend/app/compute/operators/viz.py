from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.compute.chart_helpers import build_chart, chart_to_spec
from app.compute.operators.base import OperatorContext, OperatorResult
from app.ml.visualization import ChartType, VisualizationEngine


@dataclass(frozen=True)
class SmartVizOperator:
    """Auto-generate charts for key columns in the dataset."""

    name: str = "smart_viz_scan"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        max_charts = int(params.get("max_charts") or 3)
        engine = VisualizationEngine()
        charts_list = engine.quick_visualization(df.head(int(params.get("sample_rows") or 5000)))

        charts: dict[str, Any] = {}
        for i, chart in enumerate(charts_list[:max_charts]):
            charts[f"auto_chart_{i + 1}"] = chart_to_spec(chart)

        summary = {"charts_generated": len(charts), "chart_titles": [c.config.title for c in charts_list[:max_charts]]}
        return OperatorResult(tables={}, metrics=summary, charts=charts, summary=summary)


@dataclass(frozen=True)
class CorrelationHeatmapOperator:
    name: str = "correlation_heatmap"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        max_cols = int(params.get("max_columns") or 12)
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:max_cols]
        if len(numeric) < 2:
            raise ValueError("Need at least 2 numeric columns")

        corr = df[numeric].corr().round(4)
        engine = VisualizationEngine()
        chart = engine.create_chart(corr.reset_index(), chart_type=ChartType.HEATMAP, title="Correlation heatmap")
        from app.compute.chart_helpers import chart_to_spec

        chart_spec = chart_to_spec(chart)

        table = corr.reset_index().rename(columns={"index": "column"})
        return OperatorResult(
            tables={"correlation": table},
            metrics={"columns": len(numeric)},
            charts={"correlation_heatmap": chart_spec},
            summary={"columns": numeric},
        )


@dataclass(frozen=True)
class DistributionChartOperator:
    name: str = "distribution_chart"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        df = ctx.df
        if df is None or df.empty:
            raise ValueError("Empty dataset")

        column = params.get("column")
        if not column:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            column = numeric[0] if numeric else None
        if not column or column not in df.columns:
            raise ValueError("column required for distribution chart")

        sample = df[[column]].dropna().head(int(params.get("sample_rows") or 5000))
        chart = build_chart(sample, chart_type=ChartType.HISTOGRAM, x=str(column), title=f"Distribution of {column}")
        return OperatorResult(
            tables={},
            metrics={"column": str(column), "samples": int(len(sample))},
            charts={"distribution": chart},
            summary={"column": str(column)},
        )


P0_VIZ_OPERATORS = [
    SmartVizOperator(),
    CorrelationHeatmapOperator(),
    DistributionChartOperator(),
]