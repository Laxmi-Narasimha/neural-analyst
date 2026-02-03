# AI Enterprise Data Analyst - Export Automation Engine
# PDF, PowerPoint, email report generation

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pathlib import Path
import json
from datetime import datetime
try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    PDF = "pdf"
    PPTX = "pptx"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class ReportSlide:
    title: str
    content: str = ""
    chart_path: str = None
    table_data: dict = None
    bullet_points: list[str] = field(default_factory=list)


@dataclass
class ReportDocument:
    title: str
    subtitle: str = ""
    author: str = "AI Data Analyst"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    slides: list[ReportSlide] = field(default_factory=list)
    executive_summary: str = ""


class HTMLExporter:
    """Generate HTML reports."""
    
    def export(self, doc: ReportDocument) -> str:
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{doc.title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 8px; }}
        .slide {{ background: #fff; padding: 30px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .slide h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        ul {{ line-height: 1.8; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
        .summary {{ background: #f0f7ff; padding: 20px; border-left: 4px solid #667eea; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{doc.title}</h1>
        <p>{doc.subtitle}</p>
        <p><small>{doc.author} | {doc.date}</small></p>
    </div>
"""
        if doc.executive_summary:
            html += f'<div class="summary"><h3>Executive Summary</h3><p>{doc.executive_summary}</p></div>'
        
        for slide in doc.slides:
            html += f'<div class="slide"><h2>{slide.title}</h2>'
            
            if slide.content:
                html += f'<p>{slide.content}</p>'
            
            if slide.bullet_points:
                html += '<ul>' + ''.join(f'<li>{bp}</li>' for bp in slide.bullet_points) + '</ul>'
            
            if slide.table_data:
                html += self._render_table(slide.table_data)
            
            if slide.chart_path:
                html += f'<img src="{slide.chart_path}" style="max-width:100%">'
            
            html += '</div>'
        
        html += '</body></html>'
        return html
    
    def _render_table(self, data: dict) -> str:
        if not data:
            return ""
        
        headers = list(data.keys())
        n_rows = len(list(data.values())[0]) if data else 0
        
        html = '<table><thead><tr>'
        for h in headers:
            html += f'<th>{h}</th>'
        html += '</tr></thead><tbody>'
        
        for i in range(min(n_rows, 20)):
            html += '<tr>'
            for h in headers:
                val = data[h][i] if i < len(data[h]) else ""
                html += f'<td>{val}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html


class MarkdownExporter:
    """Generate Markdown reports."""
    
    def export(self, doc: ReportDocument) -> str:
        md = f"# {doc.title}\n\n"
        md += f"**{doc.subtitle}**\n\n"
        md += f"*{doc.author} | {doc.date}*\n\n---\n\n"
        
        if doc.executive_summary:
            md += f"## Executive Summary\n\n{doc.executive_summary}\n\n---\n\n"
        
        for i, slide in enumerate(doc.slides, 1):
            md += f"## {i}. {slide.title}\n\n"
            
            if slide.content:
                md += f"{slide.content}\n\n"
            
            if slide.bullet_points:
                for bp in slide.bullet_points:
                    md += f"- {bp}\n"
                md += "\n"
            
            if slide.table_data:
                md += self._render_table(slide.table_data) + "\n"
        
        return md
    
    def _render_table(self, data: dict) -> str:
        if not data:
            return ""
        
        headers = list(data.keys())
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        n_rows = len(list(data.values())[0]) if data else 0
        for i in range(min(n_rows, 20)):
            row = [str(data[h][i]) if i < len(data[h]) else "" for h in headers]
            md += "| " + " | ".join(row) + " |\n"
        
        return md


class EmailDrafter:
    """Draft email reports."""
    
    def draft(self, doc: ReportDocument, recipient: str = "", tone: str = "professional") -> dict:
        if tone == "casual":
            greeting = "Hi,"
            closing = "Best,"
        else:
            greeting = "Dear Team,"
            closing = "Best regards,"
        
        subject = f"Report: {doc.title} - {doc.date}"
        
        body = f"""{greeting}

Please find below the key findings from the {doc.title} analysis.

**Executive Summary:**
{doc.executive_summary or 'Analysis complete. See details below.'}

**Key Highlights:**
"""
        for slide in doc.slides[:5]:
            body += f"\n• **{slide.title}**"
            if slide.bullet_points:
                body += ": " + slide.bullet_points[0]
        
        body += f"""

The full report is attached/linked for detailed review.

{closing}
{doc.author}
"""
        return {
            "to": recipient,
            "subject": subject,
            "body": body,
            "attachments": []
        }


class ExportEngine:
    """Unified export engine for all formats."""
    
    def __init__(self):
        self.html = HTMLExporter()
        self.markdown = MarkdownExporter()
        self.email = EmailDrafter()
    
    def create_report(
        self, title: str, findings: list[dict], 
        executive_summary: str = "", subtitle: str = ""
    ) -> ReportDocument:
        slides = []
        for finding in findings:
            slides.append(ReportSlide(
                title=finding.get("title", "Finding"),
                content=finding.get("content", ""),
                bullet_points=finding.get("bullets", []),
                table_data=finding.get("table"),
                chart_path=finding.get("chart")
            ))
        
        return ReportDocument(
            title=title,
            subtitle=subtitle,
            executive_summary=executive_summary,
            slides=slides
        )
    
    def export(
        self, doc: ReportDocument, format: ExportFormat, 
        output_path: str = None
    ) -> str:
        if format == ExportFormat.HTML:
            content = self.html.export(doc)
        elif format == ExportFormat.MARKDOWN:
            content = self.markdown.export(doc)
        else:
            content = self.html.export(doc)
        
        if output_path:
            Path(output_path).write_text(content, encoding='utf-8')
            logger.info(f"Report exported to {output_path}")
        
        return content
    
    def draft_email(self, doc: ReportDocument, recipient: str = "") -> dict:
        return self.email.draft(doc, recipient)
    
    def quick_export(
        self, title: str, data: dict, insights: list[str], 
        format: ExportFormat = ExportFormat.HTML
    ) -> str:
        doc = self.create_report(
            title=title,
            findings=[{
                "title": "Data Overview",
                "table": data
            }, {
                "title": "Key Insights",
                "bullets": insights
            }],
            executive_summary=insights[0] if insights else ""
        )
        return self.export(doc, format)


def get_export_engine() -> ExportEngine:
    return ExportEngine()
