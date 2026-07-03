from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _get_step_artifacts(step: dict[str, Any]) -> list[dict[str, Any]]:
    arts = step.get("artifacts")
    return arts if isinstance(arts, list) else []


def _first_table_preview(step: dict[str, Any]) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    for a in _get_step_artifacts(step):
        if str(a.get("artifact_type") or "").lower() != "table":
            continue
        preview = a.get("preview")
        if not isinstance(preview, dict):
            continue
        rows = preview.get("preview_rows")
        if isinstance(rows, list):
            return rows, _safe_str(a.get("artifact_id"))
    return None, None


@dataclass(frozen=True)
class Insight:
    kind: str
    score: float
    title: str
    detail: str
    artifact_ids: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "score": float(self.score),
            "title": self.title,
            "detail": self.detail,
            "artifact_ids": list(self.artifact_ids),
            "evidence": self.evidence,
        }


def extract_eda_insights(
    *,
    steps: list[dict[str, Any]],
    max_takeaways: int = 3,
    max_prompts: int = 10,
    max_actions: int = 10,
) -> dict[str, Any]:
    """
    Deterministic, evidence-first insight extraction from operator outputs.

    This intentionally avoids any LLM usage: insights are derived from artifacts/step summaries only.
    """

    insights: list[Insight] = []
    prompts: list[str] = []
    actions: list[dict[str, Any]] = []
    time_col: Optional[str] = None
    value_col: Optional[str] = None

    # Index steps by operator name (first occurrence).
    by_op: dict[str, dict[str, Any]] = {}
    for s in steps:
        op = str(s.get("operator") or "").strip()
        if not op or op in by_op:
            continue
        by_op[op] = s

    # Missingness hotspots
    miss = by_op.get("missingness_scan")
    if miss:
        rows, aid = _first_table_preview(miss)
        if rows:
            top = []
            for r in rows:
                col = _safe_str(r.get("column")) or ""
                pct = _safe_float(r.get("null_pct"))
                if col and pct is not None and pct > 0:
                    top.append((col, pct))
            top.sort(key=lambda x: x[1], reverse=True)
            top = top[:3]
            if top:
                detail = ", ".join([f"{c} ({p*100:.1f}%)" for c, p in top])
                score = float(top[0][1])
                insights.append(
                    Insight(
                        kind="missingness_hotspots",
                        score=min(1.0, max(0.0, score)),
                        title="Missingness hotspots",
                        detail=detail,
                        artifact_ids=[aid] if aid else [],
                        evidence={"top": [{"column": c, "null_pct": p} for c, p in top]},
                    )
                )
                prompts.append("Show missingness patterns and recommend fixes")

    # Key candidates / duplicates
    uniq = by_op.get("uniqueness_scan")
    if uniq:
        summary = uniq.get("summary") if isinstance(uniq.get("summary"), dict) else {}
        key_candidates = summary.get("key_candidates") if isinstance(summary.get("key_candidates"), list) else []
        keys = [str(k) for k in key_candidates if k]
        if keys:
            insights.append(
                Insight(
                    kind="key_candidates",
                    score=0.6,
                    title="Potential key columns",
                    detail=", ".join(keys[:5]),
                    artifact_ids=[],
                    evidence={"key_candidates": keys[:10]},
                )
            )
            prompts.append(f"Check duplicates and uniqueness for key candidates: {', '.join(keys[:3])}")

    # Strong correlations
    corr = by_op.get("correlation_matrix")
    if corr:
        rows, aid = _first_table_preview(corr)
        if rows:
            pairs = []
            for r in rows:
                a = _safe_str(r.get("column_a")) or ""
                b = _safe_str(r.get("column_b")) or ""
                v = _safe_float(r.get("corr"))
                if a and b and v is not None:
                    pairs.append((a, b, v, abs(v)))
            pairs.sort(key=lambda x: x[3], reverse=True)
            pairs = pairs[:3]
            if pairs and pairs[0][3] >= 0.6:
                detail = ", ".join([f"{a}~{b} (r={v:.2f})" for a, b, v, _ in pairs])
                score = min(1.0, pairs[0][3])
                insights.append(
                    Insight(
                        kind="correlation_highlights",
                        score=score,
                        title="Strong correlations",
                        detail=detail,
                        artifact_ids=[aid] if aid else [],
                        evidence={"pairs": [{"a": a, "b": b, "corr": v} for a, b, v, _ in pairs]},
                    )
                )
                prompts.append("Explain the strongest correlations (with caveats)")

    # Notable associations
    assoc = by_op.get("association_scan")
    if assoc:
        rows, aid = _first_table_preview(assoc)
        if rows:
            pairs = []
            for r in rows:
                a = _safe_str(r.get("column_a")) or ""
                b = _safe_str(r.get("column_b")) or ""
                t = _safe_str(r.get("association_type")) or ""
                v = _safe_float(r.get("score"))
                if a and b and t and v is not None:
                    pairs.append((a, b, t, v))
            pairs.sort(key=lambda x: x[3], reverse=True)
            pairs = pairs[:3]
            if pairs and pairs[0][3] >= 0.2:
                detail = ", ".join([f"{a} vs {b} ({t}, {v:.2f})" for a, b, t, v in pairs])
                score = min(1.0, max(0.0, float(pairs[0][3])))
                insights.append(
                    Insight(
                        kind="association_highlights",
                        score=score,
                        title="Notable associations",
                        detail=detail,
                        artifact_ids=[aid] if aid else [],
                        evidence={"pairs": [{"a": a, "b": b, "type": t, "score": v} for a, b, t, v in pairs]},
                    )
                )
                prompts.append("Show the strongest categorical associations")

    # Outliers
    outl = by_op.get("outlier_scan")
    if outl:
        rows, aid = _first_table_preview(outl)
        if rows:
            cols = []
            for r in rows:
                c = _safe_str(r.get("column")) or ""
                p = _safe_float(r.get("outlier_pct"))
                if c and p is not None:
                    cols.append((c, p))
            cols.sort(key=lambda x: x[1], reverse=True)
            cols = cols[:3]
            if cols and cols[0][1] > 0:
                detail = ", ".join([f"{c} ({p*100:.1f}%)" for c, p in cols])
                score = min(1.0, max(0.0, float(cols[0][1]) * 2.0))
                insights.append(
                    Insight(
                        kind="outlier_columns",
                        score=score,
                        title="Outlier-heavy columns (IQR)",
                        detail=detail,
                        artifact_ids=[aid] if aid else [],
                        evidence={"columns": [{"column": c, "outlier_pct": p} for c, p in cols]},
                    )
                )
                prompts.append("Review outlier bounds and decide on transformations or capping")

    # Dominant categories (if any column has a very high top-1 share)
    cat = by_op.get("categorical_topk")
    if cat:
        rows, aid = _first_table_preview(cat)
        if rows:
            best = None  # (col, value, pct)
            for r in rows:
                col = _safe_str(r.get("column")) or ""
                val = _safe_str(r.get("value")) or ""
                pct = _safe_float(r.get("pct"))
                if not col or pct is None:
                    continue
                if best is None or pct > best[2]:
                    best = (col, val, pct)
            if best and best[2] >= 0.8:
                col, val, pct = best
                insights.append(
                    Insight(
                        kind="dominant_category",
                        score=min(1.0, best[2]),
                        title="Highly imbalanced categorical feature",
                        detail=f"{col} is dominated by '{val}' (~{pct*100:.1f}% of rows in sample)",
                        artifact_ids=[aid] if aid else [],
                        evidence={"column": col, "value": val, "pct": pct},
                    )
                )
                prompts.append(f"Check if '{val}' dominance in {col} is expected or a data issue")

    # Time coverage hint (if resample ran and detected a time column)
    res = by_op.get("resample_aggregate")
    if res and isinstance(res.get("summary"), dict):
        time_col = _safe_str(res["summary"].get("time_column"))
        value_col = _safe_str(res["summary"].get("value_column"))
        if time_col:
            prompts.append(
                f"Show trend over time using {time_col}" + (f" for {value_col}" if value_col else "")
            )
            actions.append(
                {
                    "action_id": "trend",
                    "kind": "analysis",
                    "title": f"Deep dive time trend ({time_col})",
                    "detail": "Resample and aggregate with bounded output (auto time column).",
                    "params": {"time_column": time_col, **({"value_column": value_col} if value_col else {})},
                }
            )

    # Rank takeaways: prefer diversity; fall back to score order.
    kind_order = [
        "missingness_hotspots",
        "outlier_columns",
        "correlation_highlights",
        "association_highlights",
        "dominant_category",
        "key_candidates",
    ]
    kind_rank = {k: i for i, k in enumerate(kind_order)}
    insights_sorted = sorted(
        insights,
        key=lambda x: (-(x.score), kind_rank.get(x.kind, 999)),
    )

    picked: list[Insight] = []
    seen_kinds: set[str] = set()
    for ins in insights_sorted:
        if ins.kind in seen_kinds:
            continue
        picked.append(ins)
        seen_kinds.add(ins.kind)
        if len(picked) >= max_takeaways:
            break
    if len(picked) < max_takeaways:
        for ins in insights_sorted:
            if ins in picked:
                continue
            picked.append(ins)
            if len(picked) >= max_takeaways:
                break

    takeaways = [f"{i.title}: {i.detail}" for i in picked]

    # Suggested next actions (deterministic; no LLM).
    for ins in insights_sorted[:10]:
        if ins.kind == "missingness_hotspots":
            top = ins.evidence.get("top") if isinstance(ins.evidence, dict) else None
            cols = []
            if isinstance(top, list):
                for r in top[:2]:
                    if isinstance(r, dict) and r.get("column"):
                        cols.append(str(r["column"]))
            title = "Explain missingness patterns"
            if cols:
                title = f"Explain missingness patterns ({cols[0]})"
            actions.append(
                {
                    "action_id": "missingness_patterns",
                    "kind": "analysis",
                    "title": title,
                    "detail": "Break down missingness by category/time and surface likely causes (bounded).",
                    "params": {"columns": cols} if cols else {},
                }
            )
            if cols:
                actions.append(
                    {
                        "action_id": "transform_fill_missing",
                        "kind": "navigate",
                        "title": "Prepare a missing-value fix transform",
                        "detail": "Open Transform Builder prefilled for the most-missing columns (edit before applying).",
                        "params": {"columns": cols},
                    }
                )
        elif ins.kind == "outlier_columns":
            cols = []
            ev = ins.evidence.get("columns") if isinstance(ins.evidence, dict) else None
            if isinstance(ev, list):
                for r in ev[:2]:
                    if isinstance(r, dict) and r.get("column"):
                        cols.append(str(r["column"]))
            title = "Explain outliers"
            if cols:
                title = f"Explain outliers ({cols[0]})"
            actions.append(
                {
                    "action_id": "outlier_explain",
                    "kind": "analysis",
                    "title": title,
                    "detail": "Quantiles + bounds + (optional) outlier rate over time (bounded).",
                    "params": {"columns": cols} if cols else {},
                }
            )
        elif ins.kind == "key_candidates":
            ev = ins.evidence.get("key_candidates") if isinstance(ins.evidence, dict) else None
            keys: list[str] = []
            if isinstance(ev, list):
                for k in ev[:3]:
                    if k:
                        keys.append(str(k))
            if keys:
                actions.append(
                    {
                        "action_id": "transform_deduplicate",
                        "kind": "navigate",
                        "title": f"Prepare a deduplicate transform ({keys[0]})",
                        "detail": "Open Transform Builder prefilled to deduplicate on key candidates (edit before applying).",
                        "params": {"subset": keys},
                    }
                )
        elif ins.kind == "dominant_category":
            col = _safe_str(ins.evidence.get("column") if isinstance(ins.evidence, dict) else None)
            actions.append(
                {
                    "action_id": "segment_deep_dive",
                    "kind": "analysis",
                    "title": f"Segment deep dive{f' ({col})' if col else ''}",
                    "detail": "Compare top segments and how numeric features shift (bounded).",
                    "params": {"group_by": col} if col else {},
                }
            )
        elif ins.kind in {"correlation_highlights", "association_highlights"}:
            pairs = ins.evidence.get("pairs") if isinstance(ins.evidence, dict) else None
            a = b = None
            if isinstance(pairs, list) and pairs:
                top = pairs[0] if isinstance(pairs[0], dict) else None
                if isinstance(top, dict):
                    a = _safe_str(top.get("a") or top.get("column_a"))
                    b = _safe_str(top.get("b") or top.get("column_b"))
            if a and b:
                actions.append(
                    {
                        "action_id": "relationship_explain",
                        "kind": "analysis",
                        "title": f"Explain relationship ({a} vs {b})",
                        "detail": "Drill down into the strongest pair with bounded evidence tables (no raw rows).",
                        "params": {"column_a": a, "column_b": b},
                    }
                )

    seg = by_op.get("segment_summary")
    if seg and isinstance(seg.get("summary"), dict):
        gb = _safe_str(seg["summary"].get("group_by"))
        vc = _safe_str(seg["summary"].get("value_column"))
        if gb:
            actions.append(
                {
                    "action_id": "segment_deep_dive",
                    "kind": "analysis",
                    "title": f"Segment deep dive ({gb})",
                    "detail": "Compare top segments and how numeric features shift (bounded).",
                    "params": {"group_by": gb, **({"value_column": vc} if vc else {})},
                }
            )

    # If time exists, offer an anomaly/change-point scan (bounded).
    if time_col:
        actions.append(
            {
                "action_id": "time_anomaly_scan",
                "kind": "analysis",
                "title": f"Scan anomalies & change points ({time_col})",
                "detail": "Detect spikes/drops and biggest changes over time (bounded; auto metric).",
                "params": {"time_column": time_col, **({"value_column": value_col} if value_col else {})},
            }
        )

    actions.append(
        {
            "action_id": "privacy_risk_scan",
            "kind": "analysis",
            "title": "Review privacy & risk",
            "detail": "PII flags, identifiers, constants, and basic risk signals (schema-backed).",
            "params": {},
        }
    )
    actions.append(
        {
            "action_id": "export_report",
            "kind": "report",
            "title": "Generate a report artifact",
            "detail": "Export a cited report (markdown) from computed evidence.",
            "params": {"format": "markdown"},
        }
    )
    actions.append(
        {
            "action_id": "open_sql",
            "kind": "navigate",
            "title": "Run a safe SQL query",
            "detail": "Query your dataset with read-only, bounded SQL (DuckDB).",
            "params": {},
        }
    )

    # Deduplicate actions by action_id while preserving order.
    seen_actions: set[str] = set()
    actions_out: list[dict[str, Any]] = []
    for a in actions:
        aid = _safe_str(a.get("action_id")) if isinstance(a, dict) else None
        if not aid:
            continue
        key = aid.strip().lower()
        if key in seen_actions:
            continue
        seen_actions.add(key)
        actions_out.append(a)
        if len(actions_out) >= int(max_actions):
            break

    # Deduplicate prompts while preserving order.
    seen = set()
    prompts_out: list[str] = []
    for p in prompts:
        if not p:
            continue
        key = p.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        prompts_out.append(p)
        if len(prompts_out) >= max_prompts:
            break

    return {
        "insights": [i.to_dict() for i in insights_sorted[:25]],
        "takeaways": takeaways,
        "suggested_prompts": prompts_out,
        "suggested_actions": actions_out,
    }
