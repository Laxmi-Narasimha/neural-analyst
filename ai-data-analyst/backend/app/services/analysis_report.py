from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.core.serialization import to_jsonable
from app.models import Analysis
from app.services.report_generator import Report, ReportType, ReportFormat, get_report_engine


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(to_jsonable(obj), indent=2, ensure_ascii=True, default=str)
    except Exception:
        try:
            return json.dumps(obj, indent=2, ensure_ascii=True, default=str)
        except Exception:
            return str(obj)


def _truncate_preview_rows(rows: list[dict[str, Any]], *, max_rows: int = 5, max_columns: int = 12) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[:max_rows])
    cols = df.columns.tolist()[:max_columns]
    return df[cols].copy()


def build_report_from_analysis(analysis: Analysis) -> Report:
    results = analysis.results or {}
    if not isinstance(results, dict):
        results = {}

    run_meta = results.get("run_meta") if isinstance(results.get("run_meta"), dict) else {}
    takeaways = results.get("takeaways") if isinstance(results.get("takeaways"), list) else []
    suggested_prompts = results.get("suggested_prompts") if isinstance(results.get("suggested_prompts"), list) else []
    steps = results.get("steps") if isinstance(results.get("steps"), list) else []
    insights = analysis.insights if isinstance(getattr(analysis, "insights", None), list) else []

    report = Report(
        title=f"{analysis.name} - Grounded Report",
        report_type=ReportType.EXECUTIVE,
        author="Neural Analyst",
    )
    report.metadata = {
        "analysis_id": str(analysis.id),
        "dataset_id": str(analysis.dataset_id),
        "analysis_type": getattr(analysis.analysis_type, "value", str(analysis.analysis_type)),
        "dataset_version": run_meta.get("dataset_version"),
        "confidence": run_meta.get("confidence"),
        "scanned_rows": run_meta.get("scanned_rows"),
        "dataset_rows": run_meta.get("dataset_rows"),
        "scan_ratio": run_meta.get("scan_ratio"),
    }

    report.add_section(
        "Run Metadata",
        "Scope and execution details for this grounded run.",
        "table",
        {
            "Analysis ID": str(analysis.id),
            "Dataset ID": str(analysis.dataset_id),
            "Dataset version": run_meta.get("dataset_version") or "unknown",
            "Confidence": run_meta.get("confidence") or "unknown",
            "Scanned rows": run_meta.get("scanned_rows"),
            "Dataset rows": run_meta.get("dataset_rows"),
            "Scan ratio": run_meta.get("scan_ratio"),
            "Status": getattr(analysis.status, "value", str(analysis.status)),
            "Started": analysis.started_at,
            "Completed": analysis.completed_at,
            "Duration (s)": analysis.duration_seconds,
        },
    )

    if takeaways:
        lines = []
        for i, t in enumerate(takeaways[:10]):
            lines.append(f"{i + 1}. {str(t)}")
        report.add_section("Top Takeaways", "\n".join(lines), "text")

    if suggested_prompts:
        lines = []
        for p in suggested_prompts[:15]:
            lines.append(f"- {str(p)}")
        report.add_section("Suggested Next Prompts", "\n".join(lines), "text")

    if insights:
        rows: list[dict[str, Any]] = []
        for i, ins in enumerate(insights[:50]):
            if not isinstance(ins, dict):
                continue
            aids = ins.get("artifact_ids") if isinstance(ins.get("artifact_ids"), list) else []
            rows.append(
                {
                    "rank": i + 1,
                    "kind": ins.get("kind"),
                    "score": ins.get("score"),
                    "title": ins.get("title"),
                    "detail": ins.get("detail"),
                    "artifact_ids": ", ".join([str(a) for a in aids[:5]]),
                }
            )
        if rows:
            report.add_section(
                "Insight Library",
                "Ranked insights derived deterministically from computed artifacts (no hallucinated values).",
                "table",
                pd.DataFrame(rows),
            )

    artifact_rows: list[dict[str, Any]] = []
    table_previews_added = 0
    max_table_previews = 10

    for i, step in enumerate(steps[:200]):
        if not isinstance(step, dict):
            continue
        op = str(step.get("operator") or "").strip() or f"step_{i + 1}"
        summary = step.get("summary")
        artifacts = step.get("artifacts") if isinstance(step.get("artifacts"), list) else []

        report.add_section(
            f"Step {i + 1}: {op}",
            "Operator summary (computed):\n\n" + _safe_json(summary),
            "code",
        )

        for a in artifacts:
            if not isinstance(a, dict):
                continue
            aid = a.get("artifact_id")
            atype = a.get("artifact_type")
            aname = a.get("name")
            artifact_rows.append(
                {
                    "step": i + 1,
                    "operator": op,
                    "artifact_type": atype,
                    "name": aname,
                    "artifact_id": aid,
                }
            )

            # Avoid embedding raw row previews by default.
            if op == "preview_rows":
                continue

            if table_previews_added >= max_table_previews:
                continue
            if str(atype or "").lower() != "table":
                continue

            preview = a.get("preview") if isinstance(a.get("preview"), dict) else {}
            preview_rows = preview.get("preview_rows") if isinstance(preview.get("preview_rows"), list) else []
            if not preview_rows:
                continue

            df = _truncate_preview_rows(preview_rows, max_rows=6, max_columns=14)
            if df.empty:
                continue

            report.add_section(
                f"Table Preview: {op} - {aname}",
                f"Artifact ID: {aid}",
                "table",
                df,
            )
            table_previews_added += 1

    if artifact_rows:
        df = pd.DataFrame(artifact_rows).head(500)
        report.add_section("Artifact Index", "Artifacts produced by this run (downloadable):", "table", df)

    return report


def export_report(report: Report, *, report_format: ReportFormat) -> str:
    engine = get_report_engine()
    return engine.export(report, format=report_format)
