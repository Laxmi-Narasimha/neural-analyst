"""Validate a proposed plan against the operator catalog and the dataset schema.

Nothing a planner proposes runs until it passes here. Column names are resolved
against the real schema (case, spacing and punctuation insensitive), types are
checked against what each parameter accepts, sensitive columns are refused and
numeric settings are kept within their declared ranges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.planning.catalog import ANY, CATALOG, NUMERIC, OperatorSpec, Param, column_kind

MAX_STEPS = 4


def _norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(name).casefold())


@dataclass
class ColumnInfo:
    name: str
    kind: str
    sensitive: bool = False


@dataclass
class ValidationResult:
    steps: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and not self.issues


class SchemaView:
    def __init__(self, schema: dict[str, Any] | None):
        self.columns: list[ColumnInfo] = []
        for c in (schema or {}).get("columns", []) or []:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            self.columns.append(
                ColumnInfo(
                    name=str(c["name"]),
                    kind=column_kind(c.get("inferred_type", "")),
                    sensitive=bool(c.get("is_potential_pii")),
                )
            )
        self._exact = {c.name: c for c in self.columns}
        self._folded: dict[str, list[ColumnInfo]] = {}
        for c in self.columns:
            self._folded.setdefault(_norm(c.name), []).append(c)

    def resolve(self, name: Any) -> Optional[ColumnInfo]:
        if name is None:
            return None
        text = str(name)
        if text in self._exact:
            return self._exact[text]
        matches = self._folded.get(_norm(text), [])
        return matches[0] if len(matches) == 1 else None

    def names(self) -> list[str]:
        return [c.name for c in self.columns]


def _check_param(spec: OperatorSpec, p: Param, value: Any, schema: SchemaView, result: ValidationResult) -> Any:
    label = f"{spec.name}.{p.name}"
    if p.kind == "column":
        col = schema.resolve(value)
        if col is None:
            result.issues.append(f"{label}: column '{value}' does not exist")
            return None
        if col.sensitive:
            result.issues.append(f"{label}: column '{col.name}' is marked sensitive and cannot be analysed")
            return None
        if ANY not in p.column_types and col.kind not in p.column_types and col.kind != ANY:
            wanted = " or ".join(p.column_types)
            result.issues.append(f"{label}: column '{col.name}' is {col.kind}, but this needs a {wanted} column")
            return None
        if col.name != str(value):
            result.notes.append(f"Read '{value}' as column '{col.name}'.")
        return col.name
    if p.kind == "enum":
        lookup = {c.casefold(): c for c in p.choices}
        chosen = lookup.get(str(value).strip().casefold())
        if chosen is None:
            result.issues.append(f"{label}: '{value}' is not one of {', '.join(p.choices)}")
        return chosen
    if p.kind == "int":
        try:
            number = int(value)
        except (TypeError, ValueError):
            result.issues.append(f"{label}: '{value}' is not a whole number")
            return None
        if p.min is not None and number < p.min:
            number = p.min
        if p.max is not None and number > p.max:
            number = p.max
        return number
    if p.kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().casefold()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        result.issues.append(f"{label}: '{value}' is not true or false")
        return None
    return value


def validate_plan(steps: Any, schema: dict[str, Any] | None) -> ValidationResult:
    """Return the cleaned, schema-bound steps, or the issues that block execution."""
    result = ValidationResult()
    view = SchemaView(schema)
    if not isinstance(steps, list) or not steps:
        result.issues.append("the plan has no steps")
        return result
    if len(steps) > MAX_STEPS:
        result.issues.append(f"the plan has {len(steps)} steps; at most {MAX_STEPS} are allowed")
        return result

    for raw in steps:
        if not isinstance(raw, dict):
            result.issues.append("a plan step is not an object")
            continue
        op = str(raw.get("operator") or "").strip()
        spec = CATALOG.get(op)
        if spec is None:
            result.issues.append(f"operator '{op}' is not available")
            continue
        given = raw.get("params") or {}
        if not isinstance(given, dict):
            result.issues.append(f"{op}: params must be an object")
            continue
        unknown = [k for k in given if spec.param(k) is None]
        if unknown:
            result.notes.append(f"Ignored unsupported settings for {op}: {', '.join(unknown)}.")
        params: dict[str, Any] = {}
        for p in spec.params:
            value = given.get(p.name)
            if value is None or value == "":
                if p.required:
                    result.issues.append(f"{op}.{p.name} is required")
                elif p.default is not None:
                    params[p.name] = p.default
                continue
            checked = _check_param(spec, p, value, view, result)
            if checked is not None:
                params[p.name] = checked
        # Cross-parameter rules.
        if op == "segment_summary" and params.get("sort_by", "count") != "count" and "value_column" not in params:
            result.issues.append("segment_summary: ranking by a value needs value_column")
        if op == "relationship_explain" and params.get("column_a") and params.get("column_a") == params.get("column_b"):
            result.issues.append("relationship_explain: column_a and column_b must differ")
        if op == "measure_aggregate" and params.get("group_by") and params.get("group_by") == params.get("value_column"):
            result.issues.append("measure_aggregate: group_by cannot be the measured column")
        result.steps.append({"operator": op, "params": params})

    if result.issues:
        result.steps = []
    return result


_VALUE_OPS = {"eq", "ne", "in", "not_in", "gt", "ge", "lt", "le"}
_ORDER_OPS = {"gt", "ge", "lt", "le"}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_FILTER_LEAVES = 12


def validate_filters(filters: Any, schema: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate typed row filters in the chat_scope condition format.

    Leaves: {"column", "op": eq|ne|in|not_in|gt|ge|lt|le, "value"},
            {"column", "op": "date", "start": "YYYY-MM-DD"|null, "end": "YYYY-MM-DD"|null} (end exclusive),
            {"column", "op": "null"|"present"}.
    Groups: {"op": "and"|"or", "children": [...]}.
    """
    view = SchemaView(schema)
    issues: list[str] = []
    leaves = 0

    def check(node: Any) -> Optional[dict[str, Any]]:
        nonlocal leaves
        if not isinstance(node, dict):
            issues.append("a filter is not an object")
            return None
        op = str(node.get("op") or "").strip().lower()
        if op in {"and", "or"}:
            children = [c for c in (check(ch) for ch in (node.get("children") or [])) if c]
            if len(children) < 2:
                if len(children) == 1:
                    return children[0]
                issues.append(f"an '{op}' group needs at least two conditions")
                return None
            return {"op": op, "children": children}
        leaves += 1
        col = view.resolve(node.get("column"))
        if col is None:
            issues.append(f"filter column '{node.get('column')}' does not exist")
            return None
        if col.sensitive:
            issues.append(f"filter column '{col.name}' is marked sensitive")
            return None
        if op == "date":
            if col.kind != "datetime":
                issues.append(f"date range on '{col.name}', which is not a date column")
                return None
            start, end = node.get("start"), node.get("end")
            for label, v in (("start", start), ("end", end)):
                if v is not None and not _ISO_DATE.match(str(v)):
                    issues.append(f"date filter {label} '{v}' must be YYYY-MM-DD")
                    return None
            if start is None and end is None:
                issues.append(f"date filter on '{col.name}' needs a start or an end")
                return None
            if start and end and str(start) >= str(end):
                issues.append(f"date filter on '{col.name}' starts on or after its end")
                return None
            return {"column": col.name, "op": "date", "start": start, "end": end}
        if op in {"null", "present"}:
            return {"column": col.name, "op": op}
        if op not in _VALUE_OPS:
            issues.append(f"filter operator '{op}' is not supported")
            return None
        value = node.get("value")
        if op in {"in", "not_in"}:
            values = value if isinstance(value, list) else [value]
            values = [v for v in values if v is not None and str(v) != ""]
            if not values:
                issues.append(f"'{op}' filter on '{col.name}' needs at least one value")
                return None
            return {"column": col.name, "op": op, "value": values}
        if value is None or isinstance(value, (list, dict)) or str(value) == "":
            issues.append(f"'{op}' filter on '{col.name}' needs one value")
            return None
        if op in _ORDER_OPS:
            if col.kind != NUMERIC:
                issues.append(f"'{op}' comparison on '{col.name}' needs a numeric column (use a date range for dates)")
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                issues.append(f"'{op}' filter on '{col.name}' needs a number, got '{value}'")
                return None
        return {"column": col.name, "op": op, "value": value}

    cleaned = [c for c in (check(f) for f in (filters or [])) if c] if isinstance(filters, list) else []
    if not isinstance(filters, list) and filters not in (None, []):
        issues.append("filters must be a list")
    if leaves > MAX_FILTER_LEAVES:
        issues.append(f"too many filter conditions ({leaves}); at most {MAX_FILTER_LEAVES}")
    return ([] if issues else cleaned), issues


def numeric_columns(schema: dict[str, Any] | None) -> list[str]:
    return [c.name for c in SchemaView(schema).columns if c.kind == NUMERIC and not c.sensitive]
