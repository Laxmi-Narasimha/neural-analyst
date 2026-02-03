# AI Enterprise Data Analyst - Visualization Agent
# Production-grade data visualization with Plotly

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
)
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


class ChartType(str, Enum):
    """Available chart types."""
    # Basic charts
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    PIE = "pie"
    
    # Advanced charts
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    FUNNEL = "funnel"
    WATERFALL = "waterfall"
    
    # Statistical charts
    CORRELATION_MATRIX = "correlation_matrix"
    DISTRIBUTION = "distribution"
    REGRESSION = "regression"
    PAIR_PLOT = "pair_plot"
    
    # Time series
    TIME_SERIES = "time_series"
    CANDLESTICK = "candlestick"


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    
    chart_type: ChartType
    title: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    facet_column: Optional[str] = None
    
    # Style
    theme: str = "plotly_white"
    width: int = 800
    height: int = 500
    show_legend: bool = True
    
    # Options
    aggregation: Optional[str] = None  # sum, mean, count, etc.
    orientation: str = "v"  # v or h
    nbins: int = 30  # for histograms
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "color_column": self.color_column
        }


class VisualizationEngine:
    """
    Visualization engine using Plotly for interactive charts.
    
    Features:
    - Auto chart type recommendation
    - Interactive visualizations
    - Statistical visualizations
    - Responsive design
    """
    
    def __init__(self) -> None:
        if not HAS_PLOTLY:
            logger.warning("Plotly not installed. Visualizations will be limited.")
    
    def create_chart(
        self,
        df: pd.DataFrame,
        config: ChartConfig
    ) -> dict[str, Any]:
        """
        Create a chart based on configuration.
        
        Returns Plotly figure as JSON for frontend rendering.
        """
        if not HAS_PLOTLY:
            return {"error": "Plotly not installed"}
        
        chart_type = config.chart_type
        
        # Basic charts
        if chart_type == ChartType.BAR:
            fig = self._create_bar(df, config)
        elif chart_type == ChartType.LINE:
            fig = self._create_line(df, config)
        elif chart_type == ChartType.SCATTER:
            fig = self._create_scatter(df, config)
        elif chart_type == ChartType.HISTOGRAM:
            fig = self._create_histogram(df, config)
        elif chart_type == ChartType.BOX:
            fig = self._create_box(df, config)
        elif chart_type == ChartType.VIOLIN:
            fig = self._create_violin(df, config)
        elif chart_type == ChartType.PIE:
            fig = self._create_pie(df, config)
        elif chart_type == ChartType.HEATMAP:
            fig = self._create_heatmap(df, config)
        elif chart_type == ChartType.CORRELATION_MATRIX:
            fig = self._create_correlation_matrix(df, config)
        elif chart_type == ChartType.DISTRIBUTION:
            fig = self._create_distribution(df, config)
        elif chart_type == ChartType.PAIR_PLOT:
            fig = self._create_pair_plot(df, config)
        elif chart_type == ChartType.TIME_SERIES:
            fig = self._create_time_series(df, config)
        else:
            fig = self._create_scatter(df, config)  # Default fallback
        
        # Apply theme and layout
        fig.update_layout(
            template=config.theme,
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        # Return as JSON for frontend
        return {
            "chart_type": chart_type.value,
            "config": config.to_dict(),
            "plotly_json": fig.to_json()
        }
    
    def _create_bar(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create bar chart."""
        return px.bar(
            df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            orientation=config.orientation,
            title=config.title
        )
    
    def _create_line(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create line chart."""
        return px.line(
            df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            title=config.title
        )
    
    def _create_scatter(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create scatter plot."""
        return px.scatter(
            df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            size=config.size_column,
            title=config.title,
            trendline="ols" if config.x_column and config.y_column else None
        )
    
    def _create_histogram(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create histogram."""
        return px.histogram(
            df,
            x=config.x_column,
            color=config.color_column,
            nbins=config.nbins,
            title=config.title
        )
    
    def _create_box(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create box plot."""
        return px.box(
            df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            title=config.title
        )
    
    def _create_violin(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create violin plot."""
        return px.violin(
            df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            box=True,
            points="outliers",
            title=config.title
        )
    
    def _create_pie(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create pie chart."""
        if config.y_column:
            values = df.groupby(config.x_column)[config.y_column].sum()
        else:
            values = df[config.x_column].value_counts()
        
        return px.pie(
            names=values.index,
            values=values.values,
            title=config.title
        )
    
    def _create_heatmap(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create heatmap."""
        # Create pivot table if x and y specified
        if config.x_column and config.y_column:
            pivot = df.pivot_table(
                index=config.y_column,
                columns=config.x_column,
                aggfunc='count'
            )
            return px.imshow(pivot, title=config.title)
        else:
            # Correlation heatmap of numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            return px.imshow(
                numeric_df.corr(),
                title=config.title or "Correlation Heatmap"
            )
    
    def _create_correlation_matrix(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create correlation matrix heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(title=config.title or "Correlation Matrix")
        return fig
    
    def _create_distribution(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create distribution plot with histogram and KDE."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Box Plot"])
        
        col = config.x_column or df.select_dtypes(include=[np.number]).columns[0]
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df[col], name="Histogram", nbinsx=config.nbins),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df[col], name="Box Plot"),
            row=1, col=2
        )
        
        fig.update_layout(title=config.title or f"Distribution of {col}")
        return fig
    
    def _create_pair_plot(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create pair plot (scatter matrix)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5
        
        return px.scatter_matrix(
            df,
            dimensions=numeric_cols,
            color=config.color_column,
            title=config.title or "Pair Plot"
        )
    
    def _create_time_series(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Create time series plot."""
        fig = go.Figure()
        
        y_cols = [config.y_column] if config.y_column else df.select_dtypes(include=[np.number]).columns[:3]
        
        for col in y_cols:
            fig.add_trace(go.Scatter(
                x=df[config.x_column],
                y=df[col],
                mode='lines',
                name=col
            ))
        
        fig.update_layout(
            title=config.title or "Time Series",
            xaxis_title=config.x_column,
            yaxis_title="Value"
        )
        return fig
    
    def recommend_chart(
        self,
        df: pd.DataFrame,
        question: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Recommend appropriate chart types based on data.
        
        Returns list of recommended charts with configurations.
        """
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Distribution charts for numeric columns
        for col in numeric_cols[:3]:
            recommendations.append({
                "chart_type": ChartType.HISTOGRAM.value,
                "reason": f"Show distribution of {col}",
                "config": {"x_column": col}
            })
        
        # Correlation if multiple numeric columns
        if len(numeric_cols) >= 2:
            recommendations.append({
                "chart_type": ChartType.CORRELATION_MATRIX.value,
                "reason": "Show correlations between numeric variables",
                "config": {}
            })
            recommendations.append({
                "chart_type": ChartType.SCATTER.value,
                "reason": f"Show relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                "config": {"x_column": numeric_cols[0], "y_column": numeric_cols[1]}
            })
        
        # Bar charts for categorical
        for col in categorical_cols[:2]:
            if df[col].nunique() <= 20:
                recommendations.append({
                    "chart_type": ChartType.BAR.value,
                    "reason": f"Show counts by {col}",
                    "config": {"x_column": col}
                })
        
        # Time series if datetime present
        if datetime_cols and numeric_cols:
            recommendations.append({
                "chart_type": ChartType.TIME_SERIES.value,
                "reason": f"Show {numeric_cols[0]} over time",
                "config": {"x_column": datetime_cols[0], "y_column": numeric_cols[0]}
            })
        
        return recommendations[:5]


class VisualizationAgent(BaseAgent[dict[str, Any]]):
    """
    Visualization Agent for creating data visualizations.
    
    Capabilities:
    - Auto chart recommendation
    - Interactive Plotly charts
    - Statistical visualizations
    - Dashboard creation
    """
    
    name: str = "VisualizationAgent"
    description: str = "Create interactive data visualizations"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.engine = VisualizationEngine()
    
    def _register_tools(self) -> None:
        """Register visualization tools."""
        
        self.register_tool(AgentTool(
            name="create_chart",
            description="Create a specific type of chart",
            function=self._create_chart,
            parameters={
                "data": {"type": "object"},
                "chart_type": {"type": "string"},
                "x_column": {"type": "string"},
                "y_column": {"type": "string"},
                "color_column": {"type": "string"},
                "title": {"type": "string"}
            },
            required_params=["data", "chart_type"]
        ))
        
        self.register_tool(AgentTool(
            name="recommend_visualizations",
            description="Recommend visualizations for the data",
            function=self._recommend_visualizations,
            parameters={
                "data": {"type": "object"},
                "question": {"type": "string"}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="create_dashboard",
            description="Create multiple related visualizations",
            function=self._create_dashboard,
            parameters={
                "data": {"type": "object"},
                "charts": {"type": "array"}
            },
            required_params=["data"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute visualization task."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a data visualization expert. Recommend the best visualizations."
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "chart_types": list(ChartType)
        }
    
    async def _create_chart(
        self,
        data: dict,
        chart_type: str,
        x_column: str = None,
        y_column: str = None,
        color_column: str = None,
        title: str = None
    ) -> dict[str, Any]:
        """Create a chart."""
        df = pd.DataFrame(data)
        
        config = ChartConfig(
            chart_type=ChartType(chart_type),
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            title=title
        )
        
        return self.engine.create_chart(df, config)
    
    async def _recommend_visualizations(
        self,
        data: dict,
        question: str = None
    ) -> dict[str, Any]:
        """Recommend visualizations for the data."""
        df = pd.DataFrame(data)
        recommendations = self.engine.recommend_chart(df, question)
        
        return {
            "recommendations": recommendations,
            "data_summary": {
                "rows": len(df),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
            }
        }
    
    async def _create_dashboard(
        self,
        data: dict,
        charts: list[dict] = None
    ) -> dict[str, Any]:
        """Create a dashboard with multiple charts."""
        df = pd.DataFrame(data)
        
        if charts is None:
            # Auto-generate dashboard
            recommendations = self.engine.recommend_chart(df)
            charts = [r["config"] | {"chart_type": r["chart_type"]} for r in recommendations[:4]]
        
        dashboard_charts = []
        for chart_config in charts:
            config = ChartConfig(
                chart_type=ChartType(chart_config.get("chart_type", "scatter")),
                x_column=chart_config.get("x_column"),
                y_column=chart_config.get("y_column"),
                color_column=chart_config.get("color_column"),
                title=chart_config.get("title")
            )
            result = self.engine.create_chart(df, config)
            dashboard_charts.append(result)
        
        return {
            "status": "success",
            "charts": dashboard_charts,
            "chart_count": len(dashboard_charts)
        }


# Factory function
def get_visualization_agent() -> VisualizationAgent:
    """Get visualization agent instance."""
    return VisualizationAgent()
