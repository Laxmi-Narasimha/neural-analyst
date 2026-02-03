# AI Enterprise Data Analyst - Automated Report Generator
# Generate comprehensive analysis reports in multiple formats

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Report Types
# ============================================================================

class ReportFormat(str, Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    EXCEL = "excel"


class ReportType(str, Enum):
    """Types of analysis reports."""
    EDA = "exploratory_data_analysis"
    STATISTICAL = "statistical_analysis"
    ML_MODEL = "ml_model_report"
    AB_TEST = "ab_test_results"
    FORECAST = "forecast_report"
    EXECUTIVE = "executive_summary"
    CUSTOM = "custom"


@dataclass
class ReportSection:
    """Single report section."""
    
    title: str
    content: str
    section_type: str = "text"  # text, table, chart, code
    
    data: Any = None
    order: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "type": self.section_type,
            "order": self.order
        }


@dataclass
class Report:
    """Complete analysis report."""
    
    title: str
    report_type: ReportType
    
    sections: list[ReportSection] = field(default_factory=list)
    
    author: str = "AI Data Analyst"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_section(
        self,
        title: str,
        content: str,
        section_type: str = "text",
        data: Any = None
    ) -> None:
        """Add a section to the report."""
        self.sections.append(ReportSection(
            title=title,
            content=content,
            section_type=section_type,
            data=data,
            order=len(self.sections)
        ))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "type": self.report_type.value,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "sections": [s.to_dict() for s in sorted(self.sections, key=lambda x: x.order)],
            "metadata": self.metadata
        }


# ============================================================================
# Report Formatters
# ============================================================================

class HTMLFormatter:
    """Format report as HTML."""
    
    def format(self, report: Report) -> str:
        """Generate HTML report."""
        sections_html = []
        
        for section in sorted(report.sections, key=lambda x: x.order):
            sections_html.append(self._format_section(section))
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 40px;
            background: #f5f5f5;
            color: #333;
        }}
        .report-container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .chart {{ background: #f8f9fa; padding: 20px; border-radius: 4px; margin: 20px 0; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .section {{ margin-bottom: 40px; }}
        .metric {{ display: inline-block; background: #3498db; color: white; padding: 10px 20px; margin: 5px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="report-container">
        <h1>{report.title}</h1>
        <div class="meta">
            Generated by {report.author} on {report.created_at.strftime('%Y-%m-%d %H:%M')}
        </div>
        {''.join(sections_html)}
    </div>
</body>
</html>
"""
    
    def _format_section(self, section: ReportSection) -> str:
        """Format a single section."""
        content = section.content
        
        if section.section_type == "table" and section.data is not None:
            if isinstance(section.data, pd.DataFrame):
                content += section.data.to_html(classes='table', index=False)
            elif isinstance(section.data, dict):
                content += self._dict_to_table(section.data)
        
        elif section.section_type == "code":
            content = f"<pre><code>{content}</code></pre>"
        
        elif section.section_type == "chart":
            content = f'<div class="chart">{content}</div>'
        
        return f"""
        <div class="section">
            <h2>{section.title}</h2>
            <div>{content}</div>
        </div>
        """
    
    def _dict_to_table(self, data: dict) -> str:
        """Convert dict to HTML table."""
        rows = "".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in data.items())
        return f"<table>{rows}</table>"


class MarkdownFormatter:
    """Format report as Markdown."""
    
    def format(self, report: Report) -> str:
        """Generate Markdown report."""
        lines = [
            f"# {report.title}",
            "",
            f"*Generated by {report.author} on {report.created_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            ""
        ]
        
        for section in sorted(report.sections, key=lambda x: x.order):
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            
            if section.section_type == "table" and section.data is not None:
                if isinstance(section.data, pd.DataFrame):
                    lines.append(section.data.to_markdown(index=False))
                elif isinstance(section.data, dict):
                    for k, v in section.data.items():
                        lines.append(f"- **{k}**: {v}")
                lines.append("")
            
            elif section.section_type == "code":
                lines.append("```")
                lines.append(section.content)
                lines.append("```")
                lines.append("")
        
        return "\n".join(lines)


class JSONFormatter:
    """Format report as JSON."""
    
    def format(self, report: Report) -> str:
        """Generate JSON report."""
        return json.dumps(report.to_dict(), indent=2, default=str)


# ============================================================================
# Report Generators
# ============================================================================

class EDAReportGenerator:
    """Generate EDA reports."""
    
    def generate(
        self,
        df: pd.DataFrame,
        title: str = "Exploratory Data Analysis Report"
    ) -> Report:
        """Generate comprehensive EDA report."""
        report = Report(title=title, report_type=ReportType.EDA)
        
        # Overview
        report.add_section(
            "Dataset Overview",
            f"This dataset contains **{len(df):,}** rows and **{len(df.columns)}** columns.",
            "text"
        )
        
        # Data types
        dtype_counts = df.dtypes.value_counts().to_dict()
        report.add_section(
            "Data Types",
            "Distribution of column data types:",
            "table",
            {str(k): v for k, v in dtype_counts.items()}
        )
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Percentage': missing_pct.values
        }).query('Missing > 0')
        
        if len(missing_df) > 0:
            report.add_section(
                "Missing Values",
                f"Found missing values in {len(missing_df)} columns:",
                "table",
                missing_df
            )
        else:
            report.add_section("Missing Values", "No missing values found.", "text")
        
        # Numeric summary
        numeric = df.select_dtypes(include=['number'])
        if len(numeric.columns) > 0:
            report.add_section(
                "Numeric Summary",
                "Statistical summary of numeric columns:",
                "table",
                numeric.describe().T.round(2)
            )
        
        # Categorical summary
        categorical = df.select_dtypes(include=['object', 'category'])
        if len(categorical.columns) > 0:
            cat_summary = []
            for col in categorical.columns[:10]:
                cat_summary.append({
                    'Column': col,
                    'Unique': categorical[col].nunique(),
                    'Top': categorical[col].mode().iloc[0] if len(categorical[col].mode()) > 0 else 'N/A',
                    'Freq': categorical[col].value_counts().iloc[0] if len(categorical[col]) > 0 else 0
                })
            report.add_section(
                "Categorical Summary",
                "Overview of categorical columns:",
                "table",
                pd.DataFrame(cat_summary)
            )
        
        return report


class MLModelReportGenerator:
    """Generate ML model reports."""
    
    def generate(
        self,
        model_name: str,
        metrics: dict[str, float],
        feature_importance: dict[str, float] = None,
        predictions: pd.DataFrame = None
    ) -> Report:
        """Generate ML model report."""
        report = Report(
            title=f"ML Model Report: {model_name}",
            report_type=ReportType.ML_MODEL
        )
        
        # Model overview
        report.add_section(
            "Model Overview",
            f"This report summarizes the performance of **{model_name}**.",
            "text"
        )
        
        # Metrics
        report.add_section(
            "Performance Metrics",
            "Key performance metrics:",
            "table",
            {k: f"{v:.4f}" for k, v in metrics.items()}
        )
        
        # Feature importance
        if feature_importance:
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])
            
            report.add_section(
                "Feature Importance",
                "Top features by importance:",
                "table",
                pd.DataFrame([
                    {"Feature": k, "Importance": f"{v:.4f}"}
                    for k, v in sorted_importance.items()
                ])
            )
        
        return report


# ============================================================================
# Report Engine
# ============================================================================

class ReportEngine:
    """
    Automated report generation engine.
    
    Features:
    - Multiple report types (EDA, ML, A/B test)
    - Multiple output formats (HTML, Markdown, JSON)
    - Template system
    - Chart embedding
    - Export functionality
    """
    
    def __init__(self):
        self.formatters = {
            ReportFormat.HTML: HTMLFormatter(),
            ReportFormat.MARKDOWN: MarkdownFormatter(),
            ReportFormat.JSON: JSONFormatter()
        }
        
        self.generators = {
            ReportType.EDA: EDAReportGenerator(),
            ReportType.ML_MODEL: MLModelReportGenerator()
        }
    
    def generate_eda_report(
        self,
        df: pd.DataFrame,
        title: str = "EDA Report"
    ) -> Report:
        """Generate EDA report."""
        return self.generators[ReportType.EDA].generate(df, title)
    
    def generate_model_report(
        self,
        model_name: str,
        metrics: dict,
        **kwargs
    ) -> Report:
        """Generate ML model report."""
        return self.generators[ReportType.ML_MODEL].generate(
            model_name, metrics, **kwargs
        )
    
    def export(
        self,
        report: Report,
        format: ReportFormat = ReportFormat.HTML,
        path: str = None
    ) -> str:
        """Export report to specified format."""
        formatter = self.formatters.get(format)
        
        if not formatter:
            raise ValueError(f"Unknown format: {format}")
        
        content = formatter.format(report)
        
        if path:
            Path(path).write_text(content, encoding='utf-8')
            logger.info(f"Report saved to {path}")
        
        return content
    
    def create_custom_report(
        self,
        title: str,
        sections: list[dict]
    ) -> Report:
        """Create custom report from sections."""
        report = Report(title=title, report_type=ReportType.CUSTOM)
        
        for section in sections:
            report.add_section(
                title=section.get("title", ""),
                content=section.get("content", ""),
                section_type=section.get("type", "text"),
                data=section.get("data")
            )
        
        return report


# Factory function
def get_report_engine() -> ReportEngine:
    """Get report engine instance."""
    return ReportEngine()
