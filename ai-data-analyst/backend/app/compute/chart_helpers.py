from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.visualization import Chart, ChartType, VisualizationEngine


def chart_to_spec(chart: Chart) -> dict[str, Any]:
    spec = chart.json_spec if isinstance(chart.json_spec, dict) and chart.json_spec else {}
    if not spec and chart.figure is not None:
        try:
            spec = chart.figure.to_dict()
        except Exception:
            spec = {}
    return {
        "chart_type": chart.config.chart_type.value,
        "title": chart.config.title,
        "spec": spec,
    }


def build_chart(
    df: pd.DataFrame,
    *,
    chart_type: ChartType | None = None,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str = "",
) -> dict[str, Any]:
    engine = VisualizationEngine()
    chart = engine.create_chart(df, chart_type=chart_type, x=x, y=y, color=color, title=title)
    return chart_to_spec(chart)


def importance_bar_chart(importance: dict[str, float], *, title: str = "Feature importance", top_n: int = 15) -> dict[str, Any]:
    rows = sorted(importance.items(), key=lambda x: -abs(float(x[1])))[:top_n]
    df = pd.DataFrame([{"feature": k, "importance": float(v)} for k, v in rows])
    return build_chart(df, chart_type=ChartType.BAR, x="feature", y="importance", title=title)