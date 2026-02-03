# AI Enterprise Data Analyst - Report Generator Engine
# Production-grade automated report generation
# Handles: any analysis results, multiple formats

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class ReportFormat(str, Enum):
    """Report output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


class ReportSection(str, Enum):
    """Report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    DATA_OVERVIEW = "data_overview"
    KEY_FINDINGS = "key_findings"
    DETAILED_ANALYSIS = "detailed_analysis"
    RECOMMENDATIONS = "recommendations"
    APPENDIX = "appendix"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReportContent:
    """Content for a report section."""
    section: ReportSection
    title: str
    content: str
    tables: List[pd.DataFrame] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneratedReport:
    """Complete generated report."""
    title: str
    subtitle: str
    generated_at: datetime
    format: ReportFormat
    
    sections: List[ReportContent] = field(default_factory=list)
    
    # Full content
    full_content: str = ""
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "generated_at": self.generated_at.isoformat(),
            "format": self.format.value,
            "sections": [s.section.value for s in self.sections],
            "content_length": len(self.full_content)
        }


# ============================================================================
# Report Generator Engine
# ============================================================================

class ReportGeneratorEngine:
    """
    Report Generator engine.
    
    Features:
    - Multiple output formats
    - Auto-generated sections
    - Executive summary generation
    - Recommendations extraction
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def generate(
        self,
        data: pd.DataFrame = None,
        analysis_results: Dict[str, Any] = None,
        title: str = "Data Analysis Report",
        format: ReportFormat = ReportFormat.MARKDOWN
    ) -> GeneratedReport:
        """Generate comprehensive report."""
        start_time = datetime.now()
        
        if self.verbose:
            logger.info(f"Generating {format.value} report")
        
        sections = []
        
        # Executive Summary
        sections.append(self._generate_executive_summary(data, analysis_results))
        
        # Data Overview
        if data is not None:
            sections.append(self._generate_data_overview(data))
        
        # Key Findings
        sections.append(self._generate_key_findings(analysis_results))
        
        # Recommendations
        sections.append(self._generate_recommendations(analysis_results))
        
        # Generate full content
        if format == ReportFormat.MARKDOWN:
            full_content = self._to_markdown(title, sections)
        elif format == ReportFormat.HTML:
            full_content = self._to_html(title, sections)
        elif format == ReportFormat.JSON:
            full_content = self._to_json(title, sections)
        else:
            full_content = self._to_text(title, sections)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GeneratedReport(
            title=title,
            subtitle=f"Generated on {datetime.now().strftime('%Y-%m-%d')}",
            generated_at=datetime.now(),
            format=format,
            sections=sections,
            full_content=full_content,
            processing_time_sec=processing_time
        )
    
    def _generate_executive_summary(
        self,
        data: pd.DataFrame,
        results: Dict[str, Any]
    ) -> ReportContent:
        """Generate executive summary."""
        summary_points = []
        
        if data is not None:
            summary_points.append(f"Dataset contains {len(data):,} records across {len(data.columns)} variables.")
        
        if results:
            for key, value in results.items():
                if isinstance(value, dict) and 'summary' in value:
                    summary_points.append(f"{key}: {value['summary']}")
        
        content = "\n".join(f"- {p}" for p in summary_points[:5])
        
        return ReportContent(
            section=ReportSection.EXECUTIVE_SUMMARY,
            title="Executive Summary",
            content=content or "Analysis completed successfully."
        )
    
    def _generate_data_overview(self, data: pd.DataFrame) -> ReportContent:
        """Generate data overview section."""
        lines = [
            f"- **Rows:** {len(data):,}",
            f"- **Columns:** {len(data.columns)}",
            f"- **Missing Values:** {data.isna().sum().sum():,} ({data.isna().sum().sum() / data.size * 100:.1f}%)",
            f"- **Numeric Columns:** {len(data.select_dtypes(include=[np.number]).columns)}",
            f"- **Categorical Columns:** {len(data.select_dtypes(include=['object']).columns)}"
        ]
        
        return ReportContent(
            section=ReportSection.DATA_OVERVIEW,
            title="Data Overview",
            content="\n".join(lines)
        )
    
    def _generate_key_findings(self, results: Dict[str, Any]) -> ReportContent:
        """Generate key findings section."""
        findings = []
        
        if results:
            for key, value in results.items():
                if isinstance(value, dict):
                    if 'key_insights' in value:
                        findings.extend(value['key_insights'][:3])
                    if 'top_findings' in value:
                        findings.extend(value['top_findings'][:3])
        
        if not findings:
            findings = ["Analysis completed - review detailed results for specific insights."]
        
        content = "\n".join(f"{i+1}. {f}" for i, f in enumerate(findings[:10]))
        
        return ReportContent(
            section=ReportSection.KEY_FINDINGS,
            title="Key Findings",
            content=content
        )
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> ReportContent:
        """Generate recommendations section."""
        recommendations = []
        
        if results:
            for key, value in results.items():
                if isinstance(value, dict):
                    if 'recommendations' in value:
                        recommendations.extend(value['recommendations'][:3])
        
        if not recommendations:
            recommendations = [
                "Review the detailed analysis for actionable insights.",
                "Consider collecting additional data for deeper analysis.",
                "Monitor key metrics on an ongoing basis."
            ]
        
        content = "\n".join(f"- {r}" for r in recommendations[:10])
        
        return ReportContent(
            section=ReportSection.RECOMMENDATIONS,
            title="Recommendations",
            content=content
        )
    
    def _to_markdown(self, title: str, sections: List[ReportContent]) -> str:
        """Convert to Markdown format."""
        lines = [f"# {title}", "", f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]
        
        for section in sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        
        return "\n".join(lines)
    
    def _to_html(self, title: str, sections: List[ReportContent]) -> str:
        """Convert to HTML format."""
        html = [f"<html><head><title>{title}</title></head><body>"]
        html.append(f"<h1>{title}</h1>")
        
        for section in sections:
            html.append(f"<h2>{section.title}</h2>")
            html.append(f"<p>{section.content.replace(chr(10), '<br/>')}</p>")
        
        html.append("</body></html>")
        return "\n".join(html)
    
    def _to_json(self, title: str, sections: List[ReportContent]) -> str:
        """Convert to JSON format."""
        import json
        data = {
            "title": title,
            "sections": {s.section.value: s.content for s in sections}
        }
        return json.dumps(data, indent=2)
    
    def _to_text(self, title: str, sections: List[ReportContent]) -> str:
        """Convert to plain text format."""
        lines = [title.upper(), "=" * len(title), ""]
        
        for section in sections:
            lines.append(section.title.upper())
            lines.append("-" * len(section.title))
            lines.append(section.content)
            lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# Factory Functions
# ============================================================================

def get_report_generator() -> ReportGeneratorEngine:
    """Get report generator engine."""
    return ReportGeneratorEngine()


def quick_report(
    data: pd.DataFrame = None,
    results: Dict[str, Any] = None
) -> str:
    """Quick report generation."""
    engine = ReportGeneratorEngine(verbose=False)
    report = engine.generate(data, results)
    return report.full_content
