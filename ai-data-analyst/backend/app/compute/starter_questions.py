"""Questions and executable steps derived from the current dataset profile."""

import re
from typing import Any


def _is_measure(column: dict[str, Any]) -> bool:
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", column["name"])
    tokens = set(re.findall(r"[a-z]+", name.lower()))
    return (
        column.get("inferred_type") in {"integer", "float"}
        and tokens.isdisjoint({"id", "code", "zip", "key", "identifier", "postcode"})
    )


def _ranked_mean(group: str, measure: str, *, limit: int = 5) -> dict[str, Any]:
    return {
        "operator": "segment_summary",
        "params": {"group_by": group, "value_column": measure, "sort_by": "value_mean", "limit": limit},
    }


def ranked_average_plan(message: str, schema: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Return an empty plan when the requested fields cannot be safely bound."""
    query = message.strip().rstrip("?.!").strip()
    match = re.fullmatch(r"(?:show (?:the )?)?(?:average|mean) (.+?) by (.+)", query, re.I)
    if match is None:
        return None
    measure_name, group_name = (name.strip().casefold() for name in match.groups())
    columns = [c for c in (schema or {}).get("columns") or [] if isinstance(c, dict) and c.get("name")]
    measures = [c for c in columns if c["name"].casefold() == measure_name and _is_measure(c) and not c.get("is_potential_pii")]
    groups = [c for c in columns if c["name"].casefold() == group_name and not c.get("is_potential_pii")
              and c.get("inferred_type") in {"string", "categorical", "boolean"}]
    if len(measures) != 1 or len(groups) != 1:
        return []
    return [_ranked_mean(groups[0]["name"], measures[0]["name"], limit=200)]


def measure_plan(message: str, schema: dict | None) -> list[dict] | None:
    match = re.fullmatch(
        r"(?:(?:show|calculate|what is|what's|give me)\s+)?(?:the\s+)?"
        r"(total|sum|average|mean|median|min|minimum|max|maximum)\s+(?:of\s+)?(.+)",
        message.strip(" .!?"), re.I,
    )
    if not match:
        return None
    agg = {"average": "mean", "total": "sum", "minimum": "min", "maximum": "max"}.get(match[1].lower(), match[1].lower())
    cols = [c for c in (schema or {}).get("columns", []) if c.get("name") and not c.get("is_potential_pii")]
    # Bind complete labels, including labels that contain 'by'.
    expr = match[2].casefold()
    choices = []
    for c in cols:
        name = c["name"]
        if not _is_measure(c):
            continue
        if expr == name.casefold():
            choices.append({"aggregation": agg, "value_column": name})
        elif expr.startswith(name.casefold() + " by "):
            group = expr[len(name) + 4:]
            choices.extend({"aggregation": agg, "value_column": name, "group_by": g["name"]}
                           for g in cols if g["name"].casefold() == group)
    return [{"operator": "measure_aggregate", "params": choices[0]}] if len(choices) == 1 else []


def starter_plans(schema: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    columns = [c for c in (schema or {}).get("columns") or [] if isinstance(c, dict) and c.get("name")]
    usable = [c for c in columns if not c.get("is_potential_pii") and not c.get("is_constant")]
    group = next((
        c["name"] for c in usable
        if c.get("inferred_type") in {"string", "categorical", "boolean"}
        and 2 <= (c.get("unique_count") or 0) <= 50 and not c.get("is_unique_identifier")
    ), None)
    measure = next((c["name"] for c in usable if _is_measure(c) and (c.get("unique_count") or 0) > 1), None)
    date = next((
        c["name"] for c in usable
        if c.get("inferred_type") in {"date", "datetime"} and (c.get("unique_count") or 0) > 1
    ), None)
    plans = {}
    if group and measure:
        plans[f"Which groups in {group} have the highest average {measure}?"] = _ranked_mean(group, measure)
    if date and measure:
        plans[f"What is the monthly trend for {measure} using {date}?"] = {
            "operator": "resample_aggregate",
            "params": {"time_column": date, "value_column": measure, "freq": "M", "max_points": 200},
        }
    if group and not measure:
        plans[f"Which values in {group} have the most rows?"] = {
            "operator": "segment_summary", "params": {"group_by": group, "limit": 5},
        }
    if measure and not (date or group):
        plans["What are the summary statistics for the numeric columns?"] = {
            "operator": "numeric_summary", "params": {"max_columns": 25},
        }
    if any(c.get("null_count") for c in columns):
        plans["Which columns have missing values?"] = {"operator": "missingness_scan", "params": {}}
    plans["How many rows and columns are in this dataset?"] = {"operator": "dataset_overview", "params": {}}
    plans["What are the column types?"] = {"operator": "schema_snapshot", "params": {}}
    return plans
