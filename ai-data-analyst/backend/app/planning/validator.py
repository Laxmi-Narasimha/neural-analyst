"""Validate a proposed plan against the operator catalog and the dataset schema.

Nothing a planner proposes runs until it passes here. Column names are resolved
against the real schema (case, spacing and punctuation insensitive), types are
checked against what each parameter accepts, sensitive columns are refused and
numeric settings are kept within their declared ranges.
"""

from __future__ import annotations

import difflib
import re
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Optional

from app.planning.catalog import ANY, CATALOG, NUMERIC, NUMERIC_AGGREGATIONS, OperatorSpec, Param, column_kind

MAX_STEPS = 4


def _norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(name).casefold())


@dataclass
class ColumnInfo:
    name: str
    kind: str
    sensitive: bool = False
    # Every distinct value, when the profile holds the complete list; else None.
    values: Optional[list[str]] = None
    identifier: bool = False


_ID_NAME = re.compile(r"(?:^|[^a-z])(?:id|ids|#|no|nbr|num|number|code|key|uuid|guid|sku)(?:[^a-z]|$)|id$", re.I)


def is_identifier(c: dict[str, Any]) -> bool:
    """The profiler marks every all-distinct column as unique; for numbers that alone is not an ID
    (a revenue column can have no repeated amounts), so numeric IDs also need an ID-like name."""
    if not c.get("is_unique_identifier"):
        return False
    if column_kind(c.get("inferred_type", "")) != NUMERIC:
        return True
    return bool(_ID_NAME.search(str(c.get("name") or "").strip()))


def complete_values(c: dict[str, Any]) -> Optional[list[str]]:
    """The column's full value list from its profile, or None when only a sample is known."""
    top = c.get("top_values")
    unique = c.get("unique_count")
    if isinstance(top, dict) and top and isinstance(unique, int) and len(top) >= unique:
        return [str(k) for k in top.keys()]
    return None


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def canonical_value(col: ColumnInfo, value: Any) -> tuple[Any, Optional[str], Optional[str]]:
    """Bind a user/model value to the column's real value: (value, note, issue)."""
    if col.kind == NUMERIC:
        if isinstance(value, bool):
            return value, None, None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value, None, f"'{value}' is not a number, but column '{col.name}' is numeric"
        return (int(number) if number.is_integer() else number), None, None
    if not col.values or not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return value, None, None
    text = str(value).strip()
    if text in col.values:
        return text, None, None
    folded = {v.strip().casefold(): v for v in col.values}
    if text.casefold() in folded:
        return folded[text.casefold()], None, None
    close = difflib.get_close_matches(text.casefold(), list(folded), n=2, cutoff=0.75)
    if close and (len(close) == 1 or _similarity(text.casefold(), close[0]) - _similarity(text.casefold(), close[1]) > 0.1):
        chosen = folded[close[0]]
        return chosen, f"Read '{text}' as {col.name} = '{chosen}'.", None
    shown = ", ".join(col.values[:20]) + (" ..." if len(col.values) > 20 else "")
    return value, None, f"'{text}' is not a value of column '{col.name}' (its values: {shown})"


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
                    values=None if c.get("is_potential_pii") else complete_values(c),
                    identifier=is_identifier(c),
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
    if p.kind == "date":
        text = str(value).strip()[:10]
        if not _ISO_DATE.match(text):
            result.issues.append(f"{label}: '{value}' must be a date as YYYY-MM-DD")
            return None
        try:
            date.fromisoformat(text)
        except ValueError:
            result.issues.append(f"{label}: '{value}' is not a valid calendar date")
            return None
        return text
    if p.kind == "value":
        if isinstance(value, (list, dict)):
            result.issues.append(f"{label}: give a single value")
            return None
        return value  # bound to its column in validate_plan
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
        if op == "measure_aggregate":
            agg = params.get("aggregation")
            col = view.resolve(params.get("value_column")) if params.get("value_column") else None
            if agg in NUMERIC_AGGREGATIONS:
                if col is None:
                    result.issues.append(f"measure_aggregate: {agg} needs a numeric value_column")
                elif col.kind != NUMERIC:
                    result.issues.append(f"measure_aggregate: {agg} needs a numeric column, but '{col.name}' is {col.kind}")
                elif col.identifier:
                    result.issues.append(f"measure_aggregate: '{col.name}' is an identifier; aggregate a measure instead")
            elif agg == "count":
                params.pop("value_column", None)
            elif agg == "distinct_count" and col is None:
                result.issues.append("measure_aggregate: distinct_count needs the column whose distinct values are counted")
        if op == "segment_summary" and params.get("value_column"):
            col = view.resolve(params["value_column"])
            if col is not None and col.identifier:
                result.issues.append(
                    f"segment_summary: '{col.name}' is an identifier, not a measure; omit value_column to rank by count")
        if op == "rate" and params.get("column") and "positive_value" in params:
            col = view.resolve(params["column"])
            if col is not None:
                bound, note, issue = canonical_value(col, params["positive_value"])
                if issue:
                    result.issues.append(f"rate.positive_value: {issue}")
                else:
                    params["positive_value"] = bound
                    if note:
                        result.notes.append(note)
        if op == "ratio" and params.get("numerator_column") and params.get("numerator_column") == params.get("denominator_column") \
                and not params.get("subtract_column"):
            result.issues.append("ratio: numerator and denominator are the same column, so the ratio is always 1")
        if op == "period_compare":
            if params.get("aggregation", "sum") != "count" and not params.get("value_column"):
                result.issues.append("period_compare: give value_column, or aggregation 'count' to compare row counts")
            if params.get("aggregation") == "count":
                params.pop("value_column", None)
            keys = ("current_start", "current_end", "previous_start", "previous_end")
            if all(params.get(k) for k in keys):
                cs, ce, ps, pe = (params[k] for k in keys)
                if cs >= ce or ps >= pe:
                    result.issues.append("period_compare: each period must start before it ends (ends are exclusive)")
                elif max(cs, ps) < min(ce, pe):
                    result.issues.append("period_compare: the two periods overlap")
        if op == "rate" and params.get("group_by") and params.get("group_by") == params.get("column"):
            result.issues.append("rate: group_by cannot be the outcome column")
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


def validate_filters(
    filters: Any, schema: dict[str, Any] | None, notes: Optional[list[str]] = None
) -> tuple[list[dict[str, Any]], list[str]]:
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

        def bind(v: Any) -> Any:
            bound, note, issue = canonical_value(col, v)
            if issue:
                issues.append(f"filter on '{col.name}': {issue}")
            elif note and notes is not None:
                notes.append(note)
            return bound

        if op in {"in", "not_in"}:
            values = value if isinstance(value, list) else [value]
            values = [v for v in values if v is not None and str(v) != ""]
            if not values:
                issues.append(f"'{op}' filter on '{col.name}' needs at least one value")
                return None
            if col.kind != NUMERIC:
                values = [bind(v) for v in values]
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
        elif col.kind != NUMERIC:
            value = bind(value)
        return {"column": col.name, "op": op, "value": value}

    cleaned = [c for c in (check(f) for f in (filters or [])) if c] if isinstance(filters, list) else []
    if not isinstance(filters, list) and filters not in (None, []):
        issues.append("filters must be a list")
    if leaves > MAX_FILTER_LEAVES:
        issues.append(f"too many filter conditions ({leaves}); at most {MAX_FILTER_LEAVES}")
    return ([] if issues else cleaned), issues


def numeric_columns(schema: dict[str, Any] | None) -> list[str]:
    return [c.name for c in SchemaView(schema).columns if c.kind == NUMERIC and not c.sensitive]
