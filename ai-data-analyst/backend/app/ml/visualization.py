# AI Enterprise Data Analyst - Visualization Engine
# Plotly-based visualization with smart chart selection

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Chart Types
# ============================================================================

class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    AREA = "area"
    FUNNEL = "funnel"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    WATERFALL = "waterfall"
    CANDLESTICK = "candlestick"
    PARALLEL = "parallel"
    SANKEY = "sankey"
    MAP = "map"


class ColorTheme(str, Enum):
    """Color themes."""
    DEFAULT = "plotly"
    DARK = "plotly_dark"
    MINIMAL = "simple_white"
    PRESENTATION = "presentation"


@dataclass
class ChartConfig:
    """Chart configuration."""
    
    chart_type: ChartType
    title: str = ""
    
    x_column: str = ""
    y_column: str = ""
    color_column: str = ""
    size_column: str = ""
    
    # Styling
    theme: ColorTheme = ColorTheme.DEFAULT
    width: int = 800
    height: int = 500
    
    # Options
    show_legend: bool = True
    interactive: bool = True
    animation: bool = False
    
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chart:
    """Generated chart."""
    
    config: ChartConfig
    figure: Any  # Plotly figure
    html: str = ""
    json_spec: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.config.chart_type.value,
            "title": self.config.title,
            "spec": self.json_spec
        }


# ============================================================================
# Chart Builders
# ============================================================================

class PlotlyChartBuilder:
    """Build Plotly charts."""
    
    def __init__(self):
        self._plotly_available = self._check_plotly()
    
    def _check_plotly(self) -> bool:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            return True
        except ImportError:
            logger.warning("Plotly not installed")
            return False
    
    def build(self, df: pd.DataFrame, config: ChartConfig) -> Chart:
        """Build chart based on config."""
        if not self._plotly_available:
            return self._fallback_chart(config)
        
        import plotly.express as px
        import plotly.graph_objects as go
        
        fig = None
        
        if config.chart_type == ChartType.BAR:
            fig = self._build_bar(df, config, px)
        elif config.chart_type == ChartType.LINE:
            fig = self._build_line(df, config, px)
        elif config.chart_type == ChartType.SCATTER:
            fig = self._build_scatter(df, config, px)
        elif config.chart_type == ChartType.PIE:
            fig = self._build_pie(df, config, px)
        elif config.chart_type == ChartType.HISTOGRAM:
            fig = self._build_histogram(df, config, px)
        elif config.chart_type == ChartType.BOX:
            fig = self._build_box(df, config, px)
        elif config.chart_type == ChartType.HEATMAP:
            fig = self._build_heatmap(df, config, px, go)
        elif config.chart_type == ChartType.AREA:
            fig = self._build_area(df, config, px)
        elif config.chart_type == ChartType.FUNNEL:
            fig = self._build_funnel(df, config, px)
        elif config.chart_type == ChartType.TREEMAP:
            fig = self._build_treemap(df, config, px)
        else:
            # Default to bar chart
            fig = self._build_bar(df, config, px)
        
        # Apply theme and layout
        self._apply_layout(fig, config)
        
        return Chart(
            config=config,
            figure=fig,
            html=fig.to_html(include_plotlyjs='cdn'),
            json_spec=fig.to_dict()
        )
    
    def _build_bar(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build bar chart."""
        kwargs = {
            "x": config.x_column,
            "y": config.y_column,
            "title": config.title
        }
        
        if config.color_column:
            kwargs["color"] = config.color_column
        
        return px.bar(df, **kwargs)
    
    def _build_line(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build line chart."""
        kwargs = {
            "x": config.x_column,
            "y": config.y_column,
            "title": config.title
        }
        
        if config.color_column:
            kwargs["color"] = config.color_column
        
        return px.line(df, **kwargs)
    
    def _build_scatter(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build scatter plot."""
        kwargs = {
            "x": config.x_column,
            "y": config.y_column,
            "title": config.title
        }
        
        if config.color_column:
            kwargs["color"] = config.color_column
        if config.size_column:
            kwargs["size"] = config.size_column
        
        return px.scatter(df, **kwargs)
    
    def _build_pie(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build pie chart."""
        return px.pie(
            df,
            names=config.x_column,
            values=config.y_column,
            title=config.title
        )
    
    def _build_histogram(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build histogram."""
        kwargs = {"x": config.x_column, "title": config.title}
        
        if config.color_column:
            kwargs["color"] = config.color_column
        
        return px.histogram(df, **kwargs)
    
    def _build_box(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build box plot."""
        kwargs = {"y": config.y_column, "title": config.title}
        
        if config.x_column:
            kwargs["x"] = config.x_column
        if config.color_column:
            kwargs["color"] = config.color_column
        
        return px.box(df, **kwargs)
    
    def _build_heatmap(self, df: pd.DataFrame, config: ChartConfig, px, go) -> Any:
        """Build heatmap."""
        # If numeric DataFrame, use correlation
        numeric = df.select_dtypes(include=[np.number])
        
        if len(numeric.columns) > 1:
            corr = numeric.corr()
            return px.imshow(
                corr,
                title=config.title or "Correlation Heatmap",
                color_continuous_scale="RdBu_r"
            )
        
        return px.density_heatmap(
            df,
            x=config.x_column,
            y=config.y_column,
            title=config.title
        )
    
    def _build_area(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build area chart."""
        kwargs = {
            "x": config.x_column,
            "y": config.y_column,
            "title": config.title
        }
        
        if config.color_column:
            kwargs["color"] = config.color_column
        
        return px.area(df, **kwargs)
    
    def _build_funnel(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build funnel chart."""
        return px.funnel(
            df,
            x=config.y_column,
            y=config.x_column,
            title=config.title
        )
    
    def _build_treemap(self, df: pd.DataFrame, config: ChartConfig, px) -> Any:
        """Build treemap."""
        path = [config.x_column]
        if config.color_column:
            path.append(config.color_column)
        
        return px.treemap(
            df,
            path=path,
            values=config.y_column,
            title=config.title
        )
    
    def _apply_layout(self, fig, config: ChartConfig) -> None:
        """Apply layout and styling."""
        fig.update_layout(
            template=config.theme.value,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            title_x=0.5
        )
    
    def _fallback_chart(self, config: ChartConfig) -> Chart:
        """Fallback when Plotly not available."""
        return Chart(
            config=config,
            figure=None,
            html="<p>Plotly not installed</p>",
            json_spec={"error": "Plotly not available"}
        )


# ============================================================================
# Smart Chart Selector
# ============================================================================

class SmartChartSelector:
    """Intelligently select best chart type for data."""
    
    def select(
        self,
        df: pd.DataFrame,
        x_column: str = None,
        y_column: str = None,
        purpose: str = None
    ) -> ChartType:
        """Select best chart type based on data characteristics."""
        # Get column types
        if x_column and x_column in df.columns:
            x_type = self._get_column_type(df[x_column])
        else:
            x_type = "unknown"
        
        if y_column and y_column in df.columns:
            y_type = self._get_column_type(df[y_column])
        else:
            y_type = "unknown"
        
        # Purpose-based selection
        if purpose:
            purpose = purpose.lower()
            if "trend" in purpose or "time" in purpose:
                return ChartType.LINE
            if "distribution" in purpose:
                return ChartType.HISTOGRAM
            if "comparison" in purpose:
                return ChartType.BAR
            if "relationship" in purpose or "correlation" in purpose:
                return ChartType.SCATTER
            if "composition" in purpose or "proportion" in purpose:
                return ChartType.PIE
        
        # Data-driven selection
        if x_type == "datetime":
            return ChartType.LINE
        
        if x_type == "categorical" and y_type == "numeric":
            n_categories = df[x_column].nunique() if x_column else 0
            if n_categories <= 5:
                return ChartType.PIE
            return ChartType.BAR
        
        if x_type == "numeric" and y_type == "numeric":
            return ChartType.SCATTER
        
        if y_type == "numeric" and not x_column:
            return ChartType.HISTOGRAM
        
        # Default
        return ChartType.BAR
    
    def _get_column_type(self, series: pd.Series) -> str:
        """Get semantic column type."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
            return "categorical"
        return "unknown"


# ============================================================================
# Visualization Engine
# ============================================================================

class VisualizationEngine:
    """
    Unified visualization engine.
    
    Features:
    - Smart chart selection
    - Multiple chart types
    - Theme support
    - Interactive charts
    - Export to HTML/JSON
    """
    
    def __init__(self):
        self.builder = PlotlyChartBuilder()
        self.selector = SmartChartSelector()
    
    def create_chart(
        self,
        df: pd.DataFrame,
        chart_type: ChartType = None,
        x: str = None,
        y: str = None,
        color: str = None,
        title: str = "",
        **kwargs
    ) -> Chart:
        """Create a chart."""
        # Auto-select chart type if not specified
        if chart_type is None:
            chart_type = self.selector.select(df, x, y, kwargs.get("purpose"))
        
        config = ChartConfig(
            chart_type=chart_type,
            title=title,
            x_column=x or "",
            y_column=y or "",
            color_column=color or "",
            **kwargs
        )
        
        return self.builder.build(df, config)
    
    def quick_visualization(
        self,
        df: pd.DataFrame,
        columns: list[str] = None
    ) -> list[Chart]:
        """Generate quick visualizations for a DataFrame."""
        charts = []
        
        if columns is None:
            numeric = df.select_dtypes(include=[np.number]).columns[:3].tolist()
            categorical = df.select_dtypes(include=['object', 'category']).columns[:2].tolist()
        else:
            numeric = [c for c in columns if df[c].dtype in ['int64', 'float64']]
            categorical = [c for c in columns if df[c].dtype == 'object']
        
        # Histograms for numeric
        for col in numeric:
            charts.append(self.create_chart(
                df, ChartType.HISTOGRAM, x=col, title=f"Distribution of {col}"
            ))
        
        # Bar charts for categorical
        for col in categorical:
            value_counts = df[col].value_counts().head(10).reset_index()
            value_counts.columns = [col, 'count']
            charts.append(self.create_chart(
                value_counts, ChartType.BAR, x=col, y='count',
                title=f"Top values in {col}"
            ))
        
        # Correlation heatmap
        if len(numeric) > 1:
            charts.append(self.create_chart(
                df[numeric], ChartType.HEATMAP, title="Correlation Matrix"
            ))
        
        return charts
    
    def create_dashboard(
        self,
        charts: list[Chart],
        title: str = "Dashboard",
        cols: int = 2
    ) -> str:
        """Create HTML dashboard from multiple charts."""
        html_parts = [
            f"""
            <html>
            <head>
                <title>{title}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .dashboard {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 20px; }}
                    .chart {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; }}
                    h1 {{ text-align: center; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="dashboard">
            """
        ]
        
        for i, chart in enumerate(charts):
            div_id = f"chart_{i}"
            html_parts.append(f"""
                <div class="chart">
                    <div id="{div_id}"></div>
                    <script>
                        Plotly.newPlot('{div_id}', {chart.json_spec.get('data', [])}, {chart.json_spec.get('layout', {{}})});
                    </script>
                </div>
            """)
        
        html_parts.append("</div></body></html>")
        
        return "".join(html_parts)


# Factory function
def get_visualization_engine() -> VisualizationEngine:
    """Get visualization engine instance."""
    return VisualizationEngine()
