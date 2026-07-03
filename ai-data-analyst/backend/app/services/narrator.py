from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.core.config import NarratorMode, settings
from app.core.logging import LogContext, get_logger
from app.core.serialization import to_jsonable
from app.services.llm_service import Message as LLMMessage, get_llm_service

logger = get_logger(__name__)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(to_jsonable(obj), indent=2, ensure_ascii=True, default=str)
    except Exception:
        try:
            return json.dumps(obj, indent=2, ensure_ascii=True, default=str)
        except Exception:
            return str(obj)


def _format_ratio(scan_ratio: Optional[float]) -> str:
    try:
        if scan_ratio is None:
            return "unknown"
        return f"{float(scan_ratio) * 100.0:.1f}%"
    except Exception:
        return "unknown"


def build_deterministic_eda_narrative_markdown(
    *,
    analysis_name: str,
    run_meta: dict[str, Any],
    takeaways: list[str],
    suggested_prompts: list[str],
    insights: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    max_artifacts: int = 40,
) -> str:
    name = str(analysis_name or "Data Speaks").strip() or "Data Speaks"
    rm = _as_dict(run_meta)
    confidence = str(rm.get("confidence") or "unknown")
    dataset_version = str(rm.get("dataset_version") or "unknown")
    sample_rows = rm.get("sample_rows")
    scanned_rows = rm.get("scanned_rows")
    dataset_rows = rm.get("dataset_rows")
    scan_ratio = rm.get("scan_ratio")

    lines: list[str] = []
    lines.append("## Scope and confidence")
    lines.append(f"- Dataset version: `{dataset_version}`")
    if sample_rows is not None:
        lines.append(f"- Sample rows requested: `{sample_rows}`")
    if scanned_rows is not None and dataset_rows is not None:
        lines.append(f"- Rows scanned: `{scanned_rows}` out of `{dataset_rows}` ({_format_ratio(scan_ratio)})")
    elif scanned_rows is not None:
        lines.append(f"- Rows scanned: `{scanned_rows}`")
    lines.append(f"- Confidence: **{confidence}**")
    lines.append("")
    lines.append("> Evidence-first guarantee: any dataset-specific claim should be traceable to the artifacts below.")
    lines.append("")

    if takeaways:
        lines.append("## Key takeaways (computed)")
        for i, t in enumerate(takeaways[:10]):
            lines.append(f"{i + 1}. {str(t)}")
        lines.append("")

    if insights:
        lines.append("## Insight library (ranked)")
        for ins in insights[:8]:
            if not isinstance(ins, dict):
                continue
            title = str(ins.get("title") or "").strip()
            detail = str(ins.get("detail") or "").strip()
            score = ins.get("score")
            try:
                score_s = f"{float(score):.2f}"
            except Exception:
                score_s = ""
            if title:
                prefix = f"- **{title}**"
                if score_s:
                    prefix = f"- **{title}** (score {score_s})"
                lines.append(prefix + (f": {detail}" if detail else ""))
        lines.append("")

    artifacts: list[dict[str, Any]] = []
    for step in steps[:200]:
        if not isinstance(step, dict):
            continue
        op = str(step.get("operator") or "").strip()
        for a in _as_list(step.get("artifacts")):
            if not isinstance(a, dict):
                continue
            aid = str(a.get("artifact_id") or "").strip()
            if not aid:
                continue
            artifacts.append(
                {
                    "artifact_id": aid,
                    "artifact_type": str(a.get("artifact_type") or "").strip(),
                    "name": str(a.get("name") or "").strip(),
                    "operator": op,
                }
            )

    seen: set[str] = set()
    artifacts_unique: list[dict[str, Any]] = []
    for a in artifacts:
        aid = a.get("artifact_id")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        artifacts_unique.append(a)
        if len(artifacts_unique) >= max_artifacts:
            break

    if artifacts_unique:
        lines.append("## Evidence artifacts")
        for a in artifacts_unique:
            aid = a.get("artifact_id")
            at = a.get("artifact_type") or "artifact"
            an = a.get("name") or ""
            op = a.get("operator") or ""
            label = f"- `{aid}` ({at})"
            if an:
                label += f" - {an}"
            if op:
                label += f" _via {op}_"
            lines.append(label)
        lines.append("")

    if suggested_prompts:
        lines.append("## Suggested next prompts")
        for p in suggested_prompts[:12]:
            lines.append(f"- {str(p)}")
        lines.append("")

    if str(confidence).lower() in {"low", "medium"}:
        lines.append("## Notes / limitations")
        lines.append(
            "- This run used sampling or partial scans to stay responsive. "
            "For high-stakes decisions, re-run with a larger sample or full-scan operators where supported."
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


_HAS_DIGIT_RE = re.compile(r"[0-9]")


@dataclass(frozen=True)
class NarrativeOutput:
    markdown: str
    source: str
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens: Optional[int] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"markdown": self.markdown, "source": self.source}
        if self.model:
            out["model"] = self.model
        if self.latency_ms is not None:
            out["latency_ms"] = float(self.latency_ms)
        if self.tokens is not None:
            out["tokens"] = int(self.tokens)
        if self.warning:
            out["warning"] = str(self.warning)
        return out


class NarratorService:
    async def narrate_eda(
        self,
        *,
        analysis_name: str,
        run_meta: dict[str, Any],
        takeaways: list[str],
        suggested_prompts: list[str],
        insights: list[dict[str, Any]],
        steps: list[dict[str, Any]],
    ) -> NarrativeOutput:
        deterministic = build_deterministic_eda_narrative_markdown(
            analysis_name=analysis_name,
            run_meta=run_meta,
            takeaways=takeaways,
            suggested_prompts=suggested_prompts,
            insights=insights,
            steps=steps,
        )

        if settings.narrator.mode != NarratorMode.LLM:
            return NarrativeOutput(markdown=deterministic, source="deterministic")

        api_key = settings.openai.api_key.get_secret_value() if settings.openai.api_key else ""
        if not api_key:
            return NarrativeOutput(
                markdown=deterministic,
                source="deterministic",
                warning="NARRATOR_MODE=llm but OPENAI_API_KEY is missing; falling back to deterministic narrative.",
            )

        context = LogContext(component="NarratorService", operation="narrate_eda")

        payload = {
            "run_meta": _as_dict(run_meta),
            "takeaways": list(takeaways or [])[:10],
            "suggested_prompts": list(suggested_prompts or [])[:12],
            "insights": [
                {
                    "kind": ins.get("kind"),
                    "score": ins.get("score"),
                    "title": ins.get("title"),
                    "detail": ins.get("detail"),
                }
                for ins in (insights or [])[:10]
                if isinstance(ins, dict)
            ],
        }

        system = (
            "You are a premium AI data analyst narrator.\n"
            "- You must only use the provided evidence summary.\n"
            "- You MUST NOT invent numbers, dates, or counts.\n"
            "- You MUST NOT include any digits (0-9) anywhere in your output.\n"
            "- You may reference column names and concepts, but keep it grounded.\n"
            "Return markdown with short sections: Overview, What stands out, Risks/limitations, Next actions.\n"
        )
        user = (
            "Evidence summary (JSON):\n"
            f"{_safe_json(payload)}\n\n"
            "Write a concise narrative for a user who will read this after clicking 'Make the data speak'.\n"
        )

        llm = get_llm_service()

        t0 = time.perf_counter()
        try:
            resp = await llm.complete(
                messages=[LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
                model=settings.narrator.model,
                temperature=float(settings.narrator.temperature),
                max_tokens=int(settings.narrator.max_tokens),
            )
            dur_ms = (time.perf_counter() - t0) * 1000.0
            text = str(resp.content or "").strip()

            if not text:
                return NarrativeOutput(
                    markdown=deterministic,
                    source="deterministic",
                    warning="LLM narrator returned empty output; falling back to deterministic narrative.",
                )

            if _HAS_DIGIT_RE.search(text):
                return NarrativeOutput(
                    markdown=deterministic,
                    source="deterministic",
                    warning="LLM narrator violated digit policy; falling back to deterministic narrative.",
                )

            combined = (text.strip() + "\n\n---\n\n" + deterministic).strip() + "\n"
            return NarrativeOutput(
                markdown=combined,
                source="llm",
                model=str(resp.model or settings.narrator.model),
                latency_ms=float(dur_ms),
                tokens=int(getattr(resp.usage, "total_tokens", 0) or 0),
            )
        except Exception as e:
            logger.warning("LLM narrator failed; using deterministic narrative", context=context, error=str(e))
            return NarrativeOutput(
                markdown=deterministic,
                source="deterministic",
                warning="LLM narrator failed; using deterministic narrative.",
            )
