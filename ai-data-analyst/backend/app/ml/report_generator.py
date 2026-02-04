# AI Enterprise Data Analyst - Report Generator Engine
# Minimal, test-backed report generation with Markdown + HTML outputs.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ReportType(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    DATA_PROFILE = "data_profile"
    FULL = "full"


@dataclass
class ReportSection:
    title: str
    content: str


@dataclass
class GeneratedReport:
    report_type: ReportType
    generated_at: datetime
    title: str
    sections: List[ReportSection] = field(default_factory=list)
    markdown: str = ""
    html: str = ""


class ReportGeneratorEngine:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def generate(
        self,
        data: Optional[pd.DataFrame] = None,
        report_type: ReportType = ReportType.FULL,
        analysis_results: Optional[Dict[str, Any]] = None,
        title: str = "Data Analysis Report",
    ) -> GeneratedReport:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame()
        analysis_results = analysis_results or {}

        sections: List[ReportSection] = []

        if report_type in (ReportType.EXECUTIVE_SUMMARY, ReportType.DATA_PROFILE, ReportType.FULL):
            sections.append(self._executive_summary(df, analysis_results))

        if report_type in (ReportType.DATA_PROFILE, ReportType.FULL):
            sections.append(self._data_overview(df))
            sections.append(self._schema_snapshot(df))

        if report_type == ReportType.FULL:
            sections.append(self._key_findings(analysis_results))
            sections.append(self._recommendations(analysis_results))

        md = self._render_markdown(title, sections)
        html = self._render_html(title, sections)

        return GeneratedReport(
            report_type=report_type,
            generated_at=datetime.utcnow(),
            title=title,
            sections=sections,
            markdown=md,
            html=html,
        )

    def _executive_summary(self, df: pd.DataFrame, results: Dict[str, Any]) -> ReportSection:
        lines: List[str] = []
        if df.empty:
            lines.append("- Dataset is empty (0 rows).")
        else:
            n_rows = int(len(df))
            n_cols = int(len(df.columns))
            missing = int(df.isna().sum().sum())
            total_cells = int(df.size) if df.size else 0
            missing_pct = (missing / total_cells * 100.0) if total_cells else 0.0
            n_num = int(len(df.select_dtypes(include=[np.number]).columns))
            n_cat = int(len(df.select_dtypes(include=["object", "category"]).columns))

            lines.append(f"- Rows: {n_rows:,}")
            lines.append(f"- Columns: {n_cols}")
            lines.append(f"- Missing values: {missing:,} ({missing_pct:.2f}%)")
            lines.append(f"- Numeric columns: {n_num}")
            lines.append(f"- Categorical columns: {n_cat}")

        for k, v in (results or {}).items():
            if isinstance(v, dict) and v.get("summary"):
                lines.append(f"- {k}: {v['summary']}")

        content = "\n".join(lines) or "- Analysis completed."
        return ReportSection(title="Executive Summary", content=content)

    def _data_overview(self, df: pd.DataFrame) -> ReportSection:
        if df.empty:
            return ReportSection(title="Data Overview", content="No rows available to profile.")

        n_rows = int(len(df))
        n_cols = int(len(df.columns))
        missing = int(df.isna().sum().sum())
        total_cells = int(df.size) if df.size else 0
        missing_pct = (missing / total_cells * 100.0) if total_cells else 0.0

        lines = [
            f"- Rows: {n_rows:,}",
            f"- Columns: {n_cols}",
            f"- Missing values: {missing:,} ({missing_pct:.2f}%)",
        ]
        return ReportSection(title="Data Overview", content="\n".join(lines))

    def _schema_snapshot(self, df: pd.DataFrame) -> ReportSection:
        if df.empty:
            return ReportSection(title="Schema Snapshot", content="No columns available.")

        rows: List[str] = []
        for col in df.columns[:50]:
            dtype = str(df[col].dtype)
            nulls = int(df[col].isna().sum())
            uniq = int(df[col].nunique(dropna=True))
            rows.append(f"- {col}: {dtype}, nulls={nulls:,}, unique={uniq:,}")
        if len(df.columns) > 50:
            rows.append(f"- ... ({len(df.columns) - 50} more columns)")
        return ReportSection(title="Schema Snapshot", content="\n".join(rows))

    def _key_findings(self, results: Dict[str, Any]) -> ReportSection:
        findings: List[str] = []
        for _, v in (results or {}).items():
            if isinstance(v, dict):
                if isinstance(v.get("key_insights"), list):
                    findings.extend([str(x) for x in v["key_insights"][:5]])
                if isinstance(v.get("top_findings"), list):
                    findings.extend([str(x) for x in v["top_findings"][:5]])
        if not findings:
            findings = ["No structured findings were provided by analysis modules."]
        content = "\n".join([f"{i+1}. {t}" for i, t in enumerate(findings[:10])])
        return ReportSection(title="Key Findings", content=content)

    def _recommendations(self, results: Dict[str, Any]) -> ReportSection:
        recs: List[str] = []
        for _, v in (results or {}).items():
            if isinstance(v, dict) and isinstance(v.get("recommendations"), list):
                recs.extend([str(x) for x in v["recommendations"][:5]])
        if not recs:
            recs = [
                "Review data quality (missingness, duplicates, outliers).",
                "Validate key metrics definitions and aggregation level.",
                "Iterate with segment/time slicing for deeper insights.",
            ]
        content = "\n".join([f"- {r}" for r in recs[:10]])
        return ReportSection(title="Recommendations", content=content)

    def _render_markdown(self, title: str, sections: List[ReportSection]) -> str:
        lines: List[str] = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        lines.append("")
        for s in sections:
            lines.append(f"## {s.title}")
            lines.append("")
            lines.append(s.content)
            lines.append("")
        return "\n".join(lines)

    def _render_html(self, title: str, sections: List[ReportSection]) -> str:
        parts: List[str] = []
        parts.append("<html><head>")
        parts.append(f"<title>{title}</title>")
        parts.append("</head><body>")
        parts.append(f"<h1>{title}</h1>")
        parts.append(f"<p><em>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</em></p>")
        for s in sections:
            parts.append(f"<h2>{s.title}</h2>")
            parts.append("<pre>")
            parts.append(self._escape_html(s.content))
            parts.append("</pre>")
        parts.append("</body></html>")
        return "\n".join(parts)

    def _escape_html(self, text: str) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


def get_report_generator() -> ReportGeneratorEngine:
    return ReportGeneratorEngine()

