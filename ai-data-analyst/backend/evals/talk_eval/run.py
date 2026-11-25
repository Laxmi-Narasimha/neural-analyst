"""Talk to Your Data accuracy eval against real language models.

Usage (from ai-data-analyst/backend):
    .venv/Scripts/python.exe -m evals.talk_eval.run --models gpt-5.4-mini gpt-4.1-mini
    .venv/Scripts/python.exe -m evals.talk_eval.run --models gpt-5.4-mini --only r_total_west r_hinglish

Each case runs end to end through the real API: upload → profile → chat. The
rule planner answers what it can; the language model plans the rest. Results go
to evals/talk_eval/results/<timestamp>/ as JSON plus a Markdown summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from evals.talk_eval.cases import build_cases
from evals.talk_eval.datasets import DATASETS
from evals.talk_eval.harness import api_client, ask, upload_frame
from evals.talk_eval.scoring import score

RESULTS = Path(__file__).parent / "results"


async def run(models: list[str], only: list[str] | None, repeat: int) -> dict:
    os.environ["LLM_PLANNER"] = "on"
    cases = [c for c in build_cases(date.today()) if not only or c.id in only]
    report: dict = {"started": datetime.now().isoformat(timespec="seconds"), "today": date.today().isoformat(),
                    "models": {}, "cases": len(cases)}
    from app.services.llm_service import LLMService

    async with api_client() as (client, headers):
        ids = {}
        for name in sorted({c.dataset for c in cases}):
            ids[name] = await upload_frame(client, headers, name, DATASETS[name]())
        for model in models:
            os.environ["LLM_PLANNER_MODEL"] = model
            llm = LLMService()
            tokens0 = llm.total_usage.total_tokens
            rows = []
            for rep in range(repeat):
                for case in cases:
                    conv = None
                    t0 = time.perf_counter()
                    resp: dict = {}
                    for msg in case.messages:
                        resp = await ask(client, headers, ids[case.dataset], msg, conv)
                        conv = resp.get("conversation_id") or conv
                    s = score(case, resp)
                    rows.append({
                        "id": case.id, "rep": rep, "tags": list(case.tags), "kind": case.kind,
                        "question": case.messages[-1], "passed": s.passed, "strict": s.strict,
                        "route": s.route, "operators": s.operators, "reason": s.reason,
                        "seconds": round(time.perf_counter() - t0, 2),
                        "content": (resp.get("content") or "")[:1500],
                    })
                    mark = "PASS" if s.passed else "FAIL"
                    print(f"[{model}] {mark} {case.id:32s} {s.route:4s} {s.reason[:90]}", flush=True)
            n = len(rows)
            report["models"][model] = {
                "pass_rate": round(sum(r["passed"] for r in rows) / n, 3),
                "strict_rate": round(sum(r["strict"] for r in rows) / n, 3),
                "llm_routed": sum(r["route"] == "llm" for r in rows),
                "tokens": llm.total_usage.total_tokens - tokens0,
                "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
                "rows": rows,
            }
    return report


def write_report(report: dict) -> Path:
    out = RESULTS / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Talk to Your Data eval - {report['started']}", "",
             f"{report['cases']} cases, today = {report['today']}", "",
             "| model | pass | strict | LLM-routed | tokens | median s |", "|---|---|---|---|---|---|"]
    for m, r in report["models"].items():
        lines.append(f"| {m} | {r['pass_rate']:.0%} | {r['strict_rate']:.0%} | {r['llm_routed']} | {r['tokens']:,} | {r['median_seconds']} |")
    for m, r in report["models"].items():
        by_tag: dict = defaultdict(lambda: [0, 0])
        for row in r["rows"]:
            for t in row["tags"]:
                by_tag[t][0] += row["passed"]
                by_tag[t][1] += 1
        lines += ["", f"## {m}", "", "Failures:", ""]
        lines += [f"- `{row['id']}` ({row['route']}): {row['reason']}" for row in r["rows"] if not row["passed"]] or ["- none"]
        lines += ["", "By tag: " + ", ".join(f"{t} {p}/{n}" for t, (p, n) in sorted(by_tag.items()))]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[os.getenv("LLM_PLANNER_MODEL") or "gpt-5.4-mini"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    report = asyncio.run(run(args.models, args.only, args.repeat))
    out = write_report(report)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"Full results: {out}")


if __name__ == "__main__":
    main()
