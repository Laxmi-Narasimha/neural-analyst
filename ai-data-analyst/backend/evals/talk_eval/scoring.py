"""Score a chat response against a case. Pure functions, no I/O."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


@dataclass
class Score:
    passed: bool
    strict: bool
    answered: bool
    reason: str
    route: str = "rule"
    operators: list[str] = field(default_factory=list)


def numbers_in_text(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def _walk_numbers(node: Any) -> Iterable[float]:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        if math.isfinite(float(node)):
            yield float(node)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_numbers(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_numbers(v)


def table_rows(response: dict[str, Any]) -> list[list[dict[str, Any]]]:
    tables = []
    for action in response.get("agent_actions") or []:
        for art in action.get("artifacts") or []:
            if art.get("artifact_type") == "table":
                rows = (art.get("preview") or {}).get("preview_rows") or []
                if rows:
                    tables.append(rows)
    return tables


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.0051, 1e-4 * abs(b))


def _found(expected: float, candidates: Iterable[float]) -> bool:
    return any(close(c, expected) for c in candidates)


def _row_label(row: dict[str, Any]) -> str:
    for key in ("group", "period", "label", "category", "value_label", "bucket"):
        if key in row and row[key] is not None:
            return str(row[key])
    strings = [str(v) for v in row.values() if isinstance(v, str)]
    return strings[0] if strings else ""


def _label_match(expected: str, got: str) -> bool:
    e, g = expected.strip().casefold(), got.strip().casefold()
    return bool(e) and (e == g or g.startswith(e) or e in g)


def route_of(response: dict[str, Any]) -> str:
    content = response.get("content") or ""
    return "llm" if content.startswith("Interpreted your question as") else "rule"


def score(case: Any, response: dict[str, Any]) -> Score:
    content = response.get("content") or ""
    actions = response.get("agent_actions") or []
    ops = [str(a.get("operator")) for a in actions]
    answered = bool(actions)
    route = route_of(response)
    tables = table_rows(response)
    table_numbers = list(_walk_numbers(tables))
    all_numbers = numbers_in_text(content) + table_numbers

    def result(ok: bool, reason: str, strict: bool | None = None) -> Score:
        return Score(passed=ok, strict=ok if strict is None else (ok and strict), answered=answered,
                     reason=reason, route=route, operators=ops)

    if response.get("http_status") != 200:
        return result(False, f"HTTP {response.get('http_status')}")
    if any(f in content for f in case.forbid):  # exact: "operators" in a refusal is fine, "OPERATORS" is the prompt
        return result(False, "response contains forbidden text")

    if case.kind == "decline":
        return result(not answered, "declined" if not answered else f"answered with {ops} instead of declining")

    if not answered:
        return result(False, "no answer: " + content[:160].replace("\n", " "))

    if case.kind == "value":
        expected = case.expect if isinstance(case.expect, list) else [case.expect]
        if not any(_found(e, all_numbers) for e in expected):
            return result(False, f"expected {expected[0]:.4g} not in answer")
        single = all(len(t) <= 1 for t in tables)
        return result(True, "value found" + ("" if single else " (inside a multi-row table)"), strict=single)

    if case.kind == "top":
        firsts = [_row_label(t[0]) for t in tables if t]
        if any(_label_match(case.label, f) for f in firsts):
            ok_value = _found(case.expect, all_numbers)
            return result(True, "top group correct" + ("" if ok_value else " (value not shown)"), strict=ok_value)
        return result(False, f"first group {firsts[:2]} != {case.label}")

    if case.kind == "groups":
        if case.operators and not set(ops) & set(case.operators):
            return result(False, f"operators {ops} not in {case.operators}")
        missing = []
        for label, value in case.expect.items():
            hit = False
            for t in tables:
                for row in t:
                    if _label_match(label, _row_label(row)) and _found(value, _walk_numbers(row)):
                        hit = True
                        break
                if hit:
                    break
            if not hit:
                missing.append(label)
        if missing:
            return result(False, f"groups missing or wrong: {missing[:4]}")
        return result(True, "all groups correct")

    if case.kind == "operator":
        chosen = [a for a in actions if a.get("operator") in case.operators]
        if not chosen:
            return result(False, f"operators {ops} not in {case.operators}")
        for key, want in case.params.items():
            got = [str((a.get("summary") or {}).get(key) or "") for a in chosen]
            if want not in got and not any(_label_match(want, g) for g in got):
                return result(True, f"right operator, {key}={got} (wanted {want})", strict=False)
        return result(True, "right operator and parameters")

    return result(False, f"unknown case kind {case.kind}")
