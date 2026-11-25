"""Bind chat row restrictions to a dataset schema, independently of the analysis operator."""
from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

import pandas as pd

from app.services.dataset_loader import LoadedDataset


from app.services.retail_calendar import PATTERNS, RetailCalendar, ScopeQuestion, retail_period, select_pattern


@dataclass
class ScopeEdit:
    question: str
    conditions: list[dict[str, Any]]
    remove: list[str]
    intersect: bool = False
    clear: bool = False
    union: bool = False
    retail_group: str = ""
    group_date: str = ""


def fiscal_period(text: str, today: date, bindings: dict):
    """Resolve month-based fiscal calendars. A single FY label can mean either end."""
    text = text.strip().lower().replace("financial", "fiscal").replace("current", "this")
    text = text.replace(chr(0x2013), "-").replace(chr(0x2014), "-")
    text = re.sub(r"fiscal quarter\s+([1-4])", r"fiscal q\1", text)
    retail = retail_period(text, today, bindings)
    if retail is not None:
        return retail
    if text in {"fytd", "fiscal year to date", "fqtd", "fiscal quarter to date"}:
        unit = "quarter" if text in {"fqtd", "fiscal quarter to date"} else "year"
        period = fiscal_period("this fiscal " + unit, today, bindings)
        return {**period, "end": (today + timedelta(days=1)).isoformat()}
    span = re.fullmatch(r"(?:between|from) (.+?) (to|through|until|~) (.+)", text)
    if span and re.search(r"\bfiscal\b|\bfy\s*\d", text):
        first = fiscal_period(span[1], today, bindings)
        if first is None:
            raise ScopeQuestion("Specify the fiscal year and optional quarter for the start of the range.")
        last = fiscal_period(span[3], today, {**bindings, "active_fiscal_year": first["fiscal_year_start"]})
        if last is None:
            raise ScopeQuestion("Specify the fiscal year and optional quarter for the end of the range.")
        end = last["start"] if span[2] == "until" else last["end"]
        if end <= first["start"]:
            raise ScopeQuestion("The fiscal range ends before it starts. Specify the fiscal year for each endpoint.")
        return {**first, "end": end, "fiscal_quarter": None, "fiscal_year_end": last["fiscal_year_start"]}
    relative = re.fullmatch(r"(this|last|previous) fiscal (year|quarter)", text)
    label = r"(?:fy\s*|fiscal year\s+)(\d{4}(?:[-/]\d{2,4})?)"
    quarter = r"(?:fiscal\s+)?q([1-4])"
    year_text, q = None, None
    if m := re.fullmatch(quarter + r"(?:\s+(?:of\s+)?" + label + r")?", text):
        q, year_text = int(m[1]), m[2]
        if not (text.startswith("fiscal") or year_text or bindings.get("active_fiscal_year") or bindings.get("fiscal_active")):
            return None
    elif m := re.fullmatch(label + r"(?:\s+" + quarter + r")?", text):
        year_text, q = m[1], int(m[2]) if m[2] else None
    elif m := re.fullmatch(r"fiscal q([1-4])\s+(\d{4}(?:[-/]\d{2,4})?)", text):
        q, year_text = int(m[1]), m[2]
    elif bindings.get("fiscal_active") and (m := re.fullmatch(r"q([1-4])\s+(\d{4})", text)):
        q, year_text = int(m[1]), m[2]
    elif not relative:
        return None
    month_name = bindings.get("fiscal_start")
    if not month_name:
        raise ScopeQuestion("Which month starts your fiscal year, or which retail week pattern do you use?", key="fiscal_start",
                            choices=(*calendar.month_name[1:], *PATTERNS))
    month = list(calendar.month_name).index(month_name)
    if relative:
        year = today.year - int(today.month < month)
        if relative[2] == "quarter":
            q = (today.month - month) % 12 // 3 + 1
            if relative[1] != "this":
                year -= int(q == 1)
                q = (q - 2) % 4 + 1
        elif relative[1] != "this":
            year -= 1
    elif not year_text and not bindings.get("fiscal_year") and bindings.get("active_fiscal_year"):
        year = int(bindings["active_fiscal_year"])
    else:
        year_text = year_text or bindings.get("fiscal_year")
        if not year_text:
            raise ScopeQuestion("Which fiscal year? Enter a label such as 2025-26 or 2026.", key="fiscal_year")
        span = re.split(r"[-/]", year_text)
        year = int(span[0])
        if len(span) == 2:
            if span[1] not in {str(year + 1), str(year + 1)[-2:]} or month == 1:
                raise ScopeQuestion("Use consecutive fiscal years for a spanning year, or a single year for a January start.")
        elif month != 1:
            convention = bindings.get("fiscal_label")
            if not convention:
                raise ScopeQuestion(f"Does FY{year} start or end in {year}?", key="fiscal_label",
                                    choices=("Starting year", "Ending year"))
            year -= int(convention == "Ending year")
    index = year * 12 + month - 1 + (3 * (q - 1) if q else 0)
    def boundary(offset):
        y, m = divmod(index + offset, 12)
        return date(y, m + 1, 1).isoformat()
    return {"start": boundary(0), "end": boundary(3 if q else 12),
            "fiscal_year_start": year, "fiscal_start_month": month, "fiscal_quarter": q}


def condition_leaves(conditions):
    for node in conditions:
        if "children" in node:
            yield from condition_leaves(node["children"])
        else:
            yield node


def _group(op, children):
    children = [c for c in children if c]
    if len(children) <= 1:
        return children[0] if children else None
    return {"op": op, "children": children}


def _period(text: str, today: date, bindings: dict) -> tuple[str, str] | None:
    """Half-open calendar interval. Relative dates are resolved once when requested."""
    text = text.casefold().strip()
    text = re.sub(r"^q([1-4])(?:\s+of|,)\s+(\d{4})$", r"q\1 \2", text)
    months = {name.casefold(): i for i in range(1, 13) for name in (calendar.month_name[i], calendar.month_abbr[i])}
    width = 1
    if text in {"this week", "last week", "previous week"}:
        monday = today - timedelta(days=today.weekday() + (0 if text == "this week" else 7))
        return monday.isoformat(), (monday + timedelta(days=7)).isoformat()
    if match := re.fullmatch(r"q([1-4])(?:\s+(\d{4}))?", text):
        year = match[2] or bindings.get("year")
        if not year:
            raise ScopeQuestion("Which calendar year should I use for this quarter?", key="year")
        year, month, width = int(year), 3 * int(match[1]) - 2, 3
    elif (match := re.fullmatch(r"([a-z]+)(?:\s+(\d{4}))?", text)) and match[1] in months:
        year = match[2] or bindings.get("year")
        if not year:
            raise ScopeQuestion("Which calendar year should I use for this month?", key="year")
        year, month = int(year), months[match[1]]
    elif re.fullmatch(r"\d{4}", text):
        year, month, width = int(text), 1, 12
    elif match := re.fullmatch(r"(\d{4})-(\d{2})", text):
        year, month = int(match[1]), int(match[2])
    elif match := re.fullmatch(r"(this|last|previous) (month|quarter|year)", text):
        width = {"month": 1, "quarter": 3, "year": 12}[match[2]]
        index = today.year * 12 + (today.month - 1) // width * width
        if match[1] != "this":
            index -= width
        year, zero = divmod(index, 12)
        month = zero + 1
    elif text in {"today", "yesterday"}:
        day = today - timedelta(days=int(text == "yesterday"))
        return day.isoformat(), (day + timedelta(days=1)).isoformat()
    elif match := re.fullmatch(r"(?:last|past) (\d{1,5}) days?", text):
        days = int(match[1])
        if not days:
            raise ScopeQuestion("The number of days must be positive.")
        return (today - timedelta(days=days - 1)).isoformat(), (today + timedelta(days=1)).isoformat()
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        day = date.fromisoformat(text)
        return day.isoformat(), (day + timedelta(days=1)).isoformat()
    else:
        return None
    start = date(year, month, 1)
    end_year, zero = divmod(year * 12 + month - 1 + width, 12)
    return start.isoformat(), date(end_year, zero + 1, 1).isoformat()


_ISO = r"\d{4}-\d{2}-\d{2}"
_DATE_TYPES = {"datetime", "timestamp", "date", "datetime64[ns]"}
_NUMBER_TYPES = {"integer", "float", "number", "numeric", "decimal", "int", "double"}
_OP = {"=": "eq", "==": "eq", "is": "eq", "equals": "eq", "!=": "ne", "<>": "ne",
       "is not": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le",
       "above": "gt", "below": "lt", "at least": "ge", "at most": "le",
       "greater than": "gt", "less than": "lt", "over": "gt", "under": "lt"}
_MARKER = re.compile(
    r"\b(?:where|whose|when|during|between|from|before|after|since|until|excluding|except|"
    r"only|filtered (?:to|by)|filter(?:ed)? (?:for|on)|with|without|for|in|on|"
    r"last|previous|past|today|yesterday|(?:this|current) (?=month|quarter|week|year|period|fiscal|financial|retail)|retail(?=\s+(?:fy|quarter|period|week|year|[qpw]\d))|financial|fiscal|fytd|fqtd|fy\s*\d{4}(?:[-/]\d{2,4})?|calendar (?=q[1-4])|q[1-4]|exclude|among|within)\b"
    r"|@c\d+(?=\s*(?:[=!<>]|\bis\b|\bequals\b))", re.I
)


def parse_scope(message: str, schema: dict, *, today: date, bindings: dict) -> ScopeEdit:
    columns = {c["name"]: c for c in (schema or {}).get("columns", []) if c.get("name")}
    names = sorted(columns, key=len, reverse=True)
    quoted: list[str] = []
    if re.search(r"@[cq]\d+", message):
        raise ScopeQuestion("Please rephrase the condition using the displayed column names.")
    # Protect quoted literals and column names before interpreting words such as 'and' or 'from'.
    name_pattern = "|".join(re.escape(n) for n in names)
    token_re = r'"(?:[^"]|"")*"|(?<!\w)\'(?:[^\']|\'\')*\''
    if names:
        token_re += rf"|(?<!\w)(?:{name_pattern})(?!\w)"
    lookup = {}
    for i, name in enumerate(names):
        lookup.setdefault(name.casefold(), []).append(i)

    def token(match):
        word = match[0]
        if word[0] in "'\"":
            quote = word[0]
            literal = word[1:-1].replace(quote * 2, quote)
            quoted.append(literal)
            return f"@q{len(quoted)-1}"
        # A schema may itself contain Year, Month or Week. Preserve the calendar
        # phrase in "last month", but still bind the column in "where Month = 4".
        if word.casefold() in {"year", "quarter", "month", "period", "week", "day", "days"}:
            if re.search(r"\b(?:this|last|previous|fiscal|retail|per|(?:last|past) \d+)\s+$", match.string[:match.start()], re.I):
                return word
            if word.casefold() in {"period", "week", "quarter"} and re.match(r"\s*\d", match.string[match.end():]):
                return word
        indices = lookup[word.casefold()]
        if len(indices) > 1 and word not in names:
            raise ScopeQuestion(f"Multiple columns match {word!r}. Use the exact column name and capitalization.")
        return f"@c{indices[0] if len(indices) == 1 else names.index(word)}"

    def restore(text):
        return re.sub(r"@([cq])(\d+)", lambda m: (names if m[1] == "c" else quoted)[int(m[2])], text)

    text = re.sub(token_re, token, message.strip().rstrip("?.!").strip(), flags=re.I)
    text = select_pattern(text, bindings)
    text = re.sub(r"^(?:(?:filter|restrict)(?: it| this)? (?:to|by)|(?:keep|show) only|just|add (?:a |the )?filter (?:for|on))\s+",
                  "only ", text, flags=re.I)
    months = "|".join(calendar.month_name[1:] + calendar.month_abbr[1:])

    def iso_day(match):
        month_name = match["month"] or match["month_after"]
        month = next(i for i in range(1, 13) if month_name.casefold() in
                     {calendar.month_name[i].casefold(), calendar.month_abbr[i].casefold()})
        try:
            return date(int(match["year"]), month, int(match["day"] or match["day_first"])).isoformat()
        except ValueError as exc:
            raise ScopeQuestion("Please use a valid calendar date.") from exc

    text = re.sub(rf"\b(?:(?P<month>{months})\s+(?P<day>\d{{1,2}})|(?P<day_first>\d{{1,2}})\s+"
                  rf"(?P<month_after>{months})),?\s+(?P<year>\d{{4}})\b", iso_day, text, flags=re.I)
    edit = ScopeEdit(restore(text), [], [])
    grouping = re.search(r"\bby retail (month|period|week|quarter|year)s?\b(?:\s+using\s+(@c\d+))?", text, re.I)
    if grouping:
        RetailCalendar.configured(bindings)
        edit.retail_group = "period" if grouping[1].lower() == "month" else grouping[1].lower()
        edit.group_date = restore(grouping[2]) if grouping[2] else ""
        text = (text[:grouping.start()] + text[grouping.end():]).strip()
    if re.fullmatch(r"(?:clear|reset|remove)(?: all)? (?:filters|conditions)|start over|all rows", text, re.I):
        return ScopeEdit("", [], [], clear=True)
    if re.fullmatch(r"(?:all dates|all time|(?:clear|remove|reset)(?: the)? date (?:filter|range))", text, re.I):
        return ScopeEdit("", [], ["@dates"])
    if m := re.fullmatch(r"(?:remove|clear|drop)(?: the)? (?:filter on )?(@c\d+)(?: filter| condition)?|all (@c\d+)", text, re.I):
        return ScopeEdit("", [], [restore(m[1] or m[2])])

    if m := re.fullmatch(r"(?:change|switch|set) (@c\d+) to (.+)", text, re.I):
        text = f"where {m[1]} = {m[2]}"
    if m := re.fullmatch(r"(?:change|switch|set)(?: the)? (?:date range|period) to (.+)", text, re.I):
        text = f"during {m[1]}"
    if m := re.fullmatch(r"((?:for|from|where|in|during|between|only) .+?),\s*((?:show|what|how|count|total|average|plot|give)\b.+)", text, re.I):
        text = f"{m[2]} {m[1]}"
    if m := re.fullmatch(r"([^,]+) only", text, re.I):
        if not re.search(r"\b(?:show|what|how|trend|rows|sum|total|average|for|where|only|plot|in)\b", m[1], re.I):
            text = f"only {m[1]}"

    follow = re.match(r"^(?:now\b|what about\b|how about\b|switch to\b|instead\b|also\b|same (?:analysis|question) but\b|and\b|or\b)\s*,?\s*", text, re.I)
    if follow:
        text = text[follow.end():]
        edit.union = follow[0].strip().casefold() == "or"
        edit.intersect = follow[0].strip().casefold() in {"and", "also"}
    text = re.sub(r"\s+instead$", "", text, flags=re.I)
    if edit.retail_group and not text.strip(" ,"):
        edit.question = ""
        return edit
    start = None
    for m in _MARKER.finditer(text):
        tail = text[m.end():].strip()
        if m[0].lower() == "between" and re.search(r"\b(?:correlations?|relationships?|association)\b", text[:m.start()], re.I) and not re.match(_ISO, tail):
            continue
        # These are operands of the existing analytical question.
        if m[0].lower() in {"for", "in", "on", "with"}:
            if re.match(r"^(?:this |my |the )?(?:dataset|data|file|sample)\b", tail, re.I):
                continue
            if re.match(r"^@c\d+(?:\s+(?:using|by|over time|for|in|where|from|during|only)\b|$)", tail, re.I) and not follow and text[:m.start()].strip():
                continue
        start = m.start()
        break
    if follow and (start is None or start > 0) and not re.search(r"\b(?:show|count|trend|average|mean|median|sum|total|min|max|minimum|maximum|forecast|compare|how many|summary|preview|plot)\b", text, re.I):
        start = 0
    if start is None:
        # A dangling qualifier must never be passed to the keyword router.
        if re.search(r"\b(?:last|previous|past|this)\s+(?:year|month|week|quarter)|\bq[1-4]\b|\b\d{4}-\d{2}-\d{2}\b|[<>=]|\b(?:excluding|restricted|filtered)\b", text, re.I):
            raise ScopeQuestion("Please put the restriction after the question, for example ‘where column = value’ or ‘in Q1 2026’.")
        edit.question = restore(text)
        return edit
    edit.question = restore(text[:start].strip(" ,"))
    tail = text[start:].strip()
    if tail.casefold() == "only" and edit.question:
        raise ScopeQuestion("Please identify the column and value to filter, for example ‘where column = value’.")
    # Keep the conjunction inside a range together; other ANDs delimit predicates.
    tail = re.sub(r"(\bbetween\s+(?:fy|[qpw]\d|retail|week|period|fiscal)\S*(?:\s+(?!and\b)[\w/-]+){0,4})\s+and\s+(?=fy|[qpw]\d|retail|week|period|fiscal)", r"\1 ~ ", tail, flags=re.I)
    tail = re.sub(r"(\bbetween\s+\S+)\s+and\s+(\S+)", r"\1 ~ \2", tail, flags=re.I)
    # A qualified date such as 'shipped in Q1' is one predicate. A subsequent
    # date after a value such as 'West in Q1' starts a separate predicate.
    next_date = rf"in\s+(?:retail\b|(?:[pw]|period\s*|week\s*)\d+\b|fiscal\b|fy\s*\d|calendar\s+q|q[1-4]\b|\d{{4}}\b|(?:{months})\s+\d{{4}})"
    tail = re.sub(rf"(@c\d+)\s+(?={next_date})", r"\1~", tail, flags=re.I)
    tail = re.sub(r"\bon or after\b", ">=", tail, flags=re.I)
    tail = re.sub(r"\bon or before\b", "<=", tail, flags=re.I)
    tail = re.sub(r"^(?:where|only|during|for|in)\s+(?=\()", "", tail, flags=re.I)
    # IN lists and quoted text are atoms. AND binds more tightly than OR.
    lists = []
    def protect_list(m):
        lists.append(m[0])
        return f"@list{len(lists)-1}"
    if re.search(r"@list\d|@implicit", tail, re.I):
        raise ScopeQuestion("Please quote this filter value.")
    tail = re.sub(r"\b(?:not\s+)?in\s*\([^()]*\)", protect_list, tail, flags=re.I)
    tail = re.sub(rf"(?<=\S)\s+(?=during\b|where\b|{next_date}|from\s+\d{{4}}-)", " @implicit ", tail, flags=re.I)
    tail = re.sub(r"\b(and|or)\s+@implicit\s+", r"\1 ", tail, flags=re.I)
    tokens = [t.strip() for t in re.split(r"(\(|\)|@implicit|\band(?:\s+also)?\b|\bor\b)", tail, flags=re.I) if t.strip()]
    def atom(part):
        part = re.sub(r"@list(\d+)", lambda m: lists[int(m[1])], part)
        found = []
        part = re.sub(r"(@c\d+)~", r"\1 ", part)
        if m := re.fullmatch(r"(?:without|excluding|except|exclude)\s+(@c\d+)\s*(?:=|is\s+)?\s*(.+)", part, re.I):
            part = f"where {m[1]} != {m[2]}"
        raw = re.sub(r"^(?:(?:where|whose|when|for|only|in|during|among|within|on(?! or)|with|filtered to|filtered by|filter for|filter on)\s+)+", "", part.strip(), flags=re.I)
        raw = re.sub(r"\s+only$", "", raw, flags=re.I).strip()
        if not raw:
            raise ScopeQuestion("Supply a condition on both sides of AND or OR.")
        col = None
        if m := re.match(r"^(@c\d+|@q\d+)\s*(?=[=!<>]|\s|$)", raw):
            candidate = restore(m[1])
            if candidate in columns:
                col = candidate
                raw = raw[m.end():].strip()
        typ = str(columns.get(col, {}).get("inferred_type", "")).lower()
        if col and not raw:
            raise ScopeQuestion(f"Which value should I select for {col}?")
        def dates(lo, hi):
            found.append({"column": col, "op": "date", "start": lo, "end": hi})
        try:
            expr = re.sub(r"^(?:in|during|on|is)\s+", "", raw, flags=re.I)
            fiscal = fiscal_period(expr, today, bindings) if not col or typ in _DATE_TYPES else None
            if fiscal:
                return {"column": col, "op": "date", **fiscal}
            expr = re.sub(r"^calendar\s+", "", expr, flags=re.I)
            period = _period(expr, today, bindings) if not col or typ in _DATE_TYPES else None
            if period:
                dates(*period)
                return _group("and", found)
            if m := re.fullmatch(rf"(?:between|from)\s+({_ISO})\s+(?:~|to|through|until)\s+({_ISO})", raw, re.I):
                lo, hi = date.fromisoformat(m[1]), date.fromisoformat(m[2])
                if hi < lo:
                    raise ScopeQuestion("The end date precedes the start date.")
                dates(lo.isoformat(), (hi + timedelta(days=int(" until " not in raw.lower()))).isoformat())
                return _group("and", found)
            if m := re.fullmatch(rf"(before|after|since|until|on or after|on or before|>=|<=|>|<|=)\s*({_ISO})", raw, re.I):
                op, day = m[1].lower(), date.fromisoformat(m[2])
                next_day = (day + timedelta(days=1)).isoformat()
                if op in {"since", ">=", "on or after", "after", ">"}:
                    dates(next_day if op in {"after", ">"} else day.isoformat(), None)
                elif op == "=":
                    dates(day.isoformat(), next_day)
                else:
                    dates(None, next_day if op in {"<=", "on or before"} else day.isoformat())
                return _group("and", found)
        except (OverflowError, ValueError) as error:
            if isinstance(error, ScopeQuestion):
                raise
            raise ScopeQuestion("Please give a valid calendar date or an inclusive ISO date range.") from error
        if col and (m := re.fullmatch(r"(?:is\s+)?(not null|null|missing|not missing)", raw, re.I)):
            found.append({"column": col, "op": "present" if "not" in m[1].lower() else "null"})
            return _group("and", found)
        if col and (m := re.fullmatch(r"(not in|in)\s*\((.+)\)", raw, re.I)):
            items = [restore(item.strip()) for item in m[2].split(",")]
            if not all(items):
                raise ScopeQuestion("Supply a value for each member of the IN list.")
            found.append({"column": col, "op": m[1].lower().replace(" ", "_"), "value": items})
            return _group("and", found)
        if col and (m := re.fullmatch(r"between\s+(\S+)\s+~\s+(\S+)", raw, re.I)):
            found.extend([{"column": col, "op": op, "value": restore(value)} for op, value in [("ge", m[1]), ("le", m[2])]])
            return _group("and", found)
        if col and (m := re.fullmatch(r"(>=|<=|!=|<>|==|>|<|=|is not\b|is\b|equals\b|above\b|below\b|at least\b|at most\b|greater than\b|less than\b|over\b|under\b)\s*(.+)", raw, re.I)):
            if re.search(r"\b(?:but|unless|except|with|without|where|before|after|for|excluding|in|during|since|until|only|q[1-4])\b|[<>=()]", m[2], re.I):
                raise ScopeQuestion("Please separate the conditions with AND and quote text values that contain condition words.")
            found.append({"column": col, "op": _OP[m[1].casefold()], "value": restore(m[2].strip())})
            return _group("and", found)
        if col:
            raise ScopeQuestion(f"Please specify how to filter {col}, for example ‘where {col} = value’.")
        op = "eq"
        if m := re.match(r"^(?:exclude|excluding|except|without)\s+", raw, re.I):
            op, raw = "ne", raw[m.end():]
        if not col and raw.lower().startswith("from ") and not re.search(r"\bto\b|\d{4}-|\bthrough\b", raw, re.I):
            raw = raw[5:].strip()
        if not col and re.search(r"[<>=()]|\b(?:before|after|between|from|until|since|last|quarter|fiscal|where|only|except|without|excluding)\b", raw, re.I):
            raise ScopeQuestion("I could not bind the complete restriction. Please use a column name and an explicit value or date range.")
        found.append({"column": col, "op": op, "value": restore(raw)})
        return _group("and", found)

    pos = 0
    def expression(precedence=0, depth=0, implied=None):
        nonlocal pos
        if depth > 32 or pos >= len(tokens):
            raise ScopeQuestion("Please check the grouping and supply a condition on both sides.")
        word = tokens[pos]
        pos += 1
        if word == "(":
            left = expression(depth=depth + 1)
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ScopeQuestion("Please close each filter group with a matching parenthesis.")
            pos += 1
        elif word == ")" or word.lower() in {"and", "or", "and also"}:
            raise ScopeQuestion("A filter value is missing.")
        else:
            left = atom(word)
            if implied and left.get("op") == "eq" and not left.get("column"):
                left["column"] = implied
        while pos < len(tokens):
            op = tokens[pos].lower()
            priority = {"and": 3, "and also": 3, "or": 2, "@implicit": 1}.get(op, 0)
            if priority <= precedence:
                break
            pos += 1
            op = "or" if op == "or" else "and"
            preceding = list(condition_leaves([left]))[-1]
            col = preceding.get("column") if op == "or" and preceding["op"] == "eq" else None
            right = expression(priority, depth + 1, col)
            # Flatten like operators to keep long lists shallow.
            children = left["children"] if left["op"] == op else [left]
            left = _group(op, [*children, right])
        return left

    root = expression()
    if pos != len(tokens):
        raise ScopeQuestion("Please check the parentheses and join conditions with AND or OR.")
    edit.conditions = root["children"] if root["op"] == "and" else [root]
    return edit


def verify_measurement(question: str, schema: dict, plan: list[dict]) -> None:
    """Decline residual qualifiers instead of letting a keyword match erase them."""
    if not question:
        return
    operator = plan[0]["operator"] if len(plan) == 1 else None
    text = question.strip(" .!?").casefold()
    if operator == "dataset_overview":
        if re.fullmatch(
            r"(?:(?:please |show (?:me )?|give me |what is |what's |tell me |get |the |a |an )*"
            r"(?:how many (?:rows|columns|records)(?: and (?:columns|rows))?|"
            r"(?:number of |count (?:of )?|total )?(?:rows|columns|records)(?: count)?|(?:exact )?row count|column count|"
            r"(?:dataset )?(?:size|overview|summary)|summary of (?:this|my|the) dataset)"
            r"(?: are (?:there|in)| in| of| does)?(?: (?:this|the|my) (?:dataset|file|data)(?: have)?)?)", text,
        ):
            return
        raise ScopeQuestion("Please specify the population as ‘how many rows where column = value’.")
    names = sorted((c["name"] for c in schema.get("columns", []) if c.get("name")), key=len, reverse=True)
    if names:
        text = re.sub(r"(?<!\w)(?:" + "|".join(map(re.escape, (n.casefold() for n in names))) + r")(?!\w)", "FIELD", text)
    if operator == "resample_aggregate" and not re.fullmatch(
        r"(?:(?:please|show|me|what|is|the|plot|display|give|a|of|for|by|using|on|over|time|series|"
        r"trend|trends|daily|weekly|monthly|quarterly|yearly|per|day|week|month|quarter|year|FIELD)\s*)+", text
    ):
        raise ScopeQuestion("I could not bind every part of the trend. Name the numeric and date columns, then add ‘where column = value’.")
    if operator in {"measure_aggregate", "segment_summary"} and re.match(r"^(?:show )?(?:average|mean) ", text):
        # The grouped-mean parser already bound the entire expression.
        return
    if operator == "measure_aggregate":
        return
    if re.search(r"\b(?:not|except|excluding|minus|without|versus|rather than|less than|more than|"
                 r"greater than|at least|at most|under|within|outside|restricted to|limited to)\b|[<>!=]=?", text, re.I):
        raise ScopeQuestion("I could not bind the entire restriction. Use ‘where column = value’ or explicit numeric or date bounds.")
    # Prevent unbound categories (e.g. 'find outliers among discharged patients')
    # or pasted literals from passing through to a broad dataset-wide operator.
    known = set("""what whats are is does the a an and or my this all data dataset file sample me show give please
        can you i want to get run compute do perform summarize summary overview analyze analyse analysis explore
        exploratory eda make speak speaks dataset size schema column columns types type names dtypes
        row rows count counts records number many how which have has there of for by on with from in over
        find check strongest top highest lowest most values value categories category groups group grouped
        variables variable between numeric numerical categorical missing missingness null nulls na nan
        unique uniqueness duplicates duplicate key candidate candidates cardinality stats statistics describe
        distribution text comments comment description notes review reviews correlations correlation corr
        association associations relationship relationships drivers dependence outliers outlier anomaly anomalies
        sentiment opinion positive negative tone scan trend trends time series timeseries using monthly weekly
        yearly quarterly daily day week month quarter year per forecast forecasting predict prediction next
        periods future projection train training fit build model models target machine learning regression
        classification baseline quick full automl tune tuning hyperparameter optimization optimize multiple
        algorithms feature importance xgboost random forest linear logistic for test ab b experiment
        variant control treatment cohort cohorts retention churn funnel conversion drop off rfm customer
        segments segment unsupervised cluster clusters clustering k means chart charts visualize visualization
        plot graph histogram heatmap exact full table duckdb sql aggregate without loading million
        head preview display schema snapshot first details ranked rank average mean totals total sum
        across breakdown break down percentile median populations privacy risk sample dataset field to your
        measures metric measure score returns calculate projections topk features comparison compare best
        strongest automatically detect identify as trend id correlations field
        these those it each every new saved trained existing latest apply use rescore scoring scored predictions
        predicted""".split())
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    if any(word.casefold() not in known for word in words):
        raise ScopeQuestion("Part of the question is unbound. Name the measurement and put restrictions after ‘where’, ‘for’, or ‘during’.")

def bind_conditions(edit: ScopeEdit, previous: list[dict], loaded: LoadedDataset,
                    plan: list[dict], bindings: dict, find_values=None) -> list[dict]:
    columns = {c["name"]: c for c in loaded.schema_info.get("columns", []) if c.get("name")}
    date_columns = {n for n, c in columns.items() if c.get("inferred_type") in _DATE_TYPES}
    def prune(nodes, predicate):
        kept = []
        for node in nodes:
            if "children" in node:
                node = _group(node["op"], prune(node["children"], predicate))
            elif predicate(node):
                continue
            if node:
                kept.append(node)
        return kept

    result = [] if edit.clear else prune(previous, lambda c: c["column"] in edit.remove or
                                          ((c["op"] == "date" or c["column"] in date_columns) and "@dates" in edit.remove))
    index = 0
    def bind(spec):
        nonlocal index
        if "children" in spec:
            return _group(spec["op"], [bind(child) for child in spec["children"]])
        item = dict(spec)
        col = item.get("column")
        key = f"field_{index}"
        index += 1
        if not col and key in bindings:
            col = bindings[key]
        if item["op"] == "date":
            candidates = [n for n, c in columns.items() if c.get("inferred_type") in _DATE_TYPES]
            if not col:
                # Prefer an explicitly bound analysis axis, then the last date restriction.
                axes = {step.get("params", {}).get("time_column") for step in plan} - {None}
                prior = {c["column"] for c in condition_leaves(result) if c["op"] == "date"}
                options = list(axes or prior or candidates)
                if not candidates:
                    raise ScopeQuestion("This dataset has no recognized date column. Convert the intended column to a date before applying a calendar range.")
                if len(options) != 1:
                    raise ScopeQuestion("Which date column defines this range?", key=key, choices=tuple(candidates))
                col = options[0]
            if col not in candidates:
                raise ScopeQuestion(f"{col} is not a recognized date column. Select a date column or convert its type.")
        elif not col:
            wanted = str(item["value"]).casefold()
            candidates = []
            for name, meta in columns.items():
                if name not in loaded.df or meta.get("is_potential_pii"):
                    continue
                if meta.get("inferred_type") in _NUMBER_TYPES and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", wanted):
                    continue
                if meta.get("inferred_type") in _DATE_TYPES:
                    continue
                if (find_values(name, wanted) if find_values else loaded.df[name].astype("string").str.casefold().eq(wanted).any()):
                    candidates.append(name)
            if len(candidates) != 1:
                if candidates:
                    raise ScopeQuestion(f"Which column should I match to {item['value']}?", key=key, choices=tuple(candidates))
                raise ScopeQuestion(f"I couldn't locate the value {item['value']!r}. Please specify the condition as ‘where column = value’.")
            col = candidates[0]
        if col not in loaded.df:
            raise ScopeQuestion(f"The active dataset has no column named {col}.")
        if item["op"] in {"ne", "not_in"} and columns.get(col, {}).get("inferred_type") not in _NUMBER_TYPES | _DATE_TYPES:
            values = item["value"] if isinstance(item.get("value"), list) else [item.get("value")]
            present = lambda v: (find_values(col, str(v).casefold()) if find_values
                                 else loaded.df[col].astype("string").str.casefold().eq(str(v).casefold()).any())
            missing = [v for v in values if not present(v)]
            if missing:
                # 'excluding returned orders' read as Returned != 'orders' would keep every row.
                raise ScopeQuestion(f"{col} has no value {missing[0]!r}, so excluding it would change nothing. "
                                    f"Specify the value to exclude, for example ‘where {col} != value’.")
        item["column"] = col
        return item

    updates = [bind(spec) for spec in edit.conditions]
    if edit.union:
        if not result or not updates:
            raise ScopeQuestion("Start with a filter before adding an OR alternative.")
        new_fields = {c["column"] for c in condition_leaves(updates)}
        def fields(node):
            return {c["column"] for c in condition_leaves([node])}
        candidates = [i for i, old in enumerate(result) if fields(old) & new_fields]
        if not candidates:
            # Keep independent date ranges in force on both alternatives.
            candidates = [i for i, old in enumerate(result) if not fields(old) <= date_columns]
        if len(result) == 1:
            candidates = [0]
        if len(candidates) != 1:
            raise ScopeQuestion("Which condition should gain the alternative? Write both sides as a parenthesized OR group.")
        i = candidates[0]
        result[i] = _group("or", [result[i], _group("and", updates)])
        return result


    # A later equality selects a new value for that dimension. Opposite bounds
    # and exclusions accumulate. Conditions from one turn always apply together.
    if not edit.intersect:
        additions = []
        for item in updates:
            nested = set(c["column"] for c in condition_leaves([item]))
            if "children" in item:
                # Replace a restated group as a unit. A date or equality outside
                # the group remains a separate restriction, even if mentioned inside.
                for old in result:
                    fields = {c["column"] for c in condition_leaves([old])}
                    if "children" in old and fields & nested and fields != nested:
                        raise ScopeQuestion("The new group overlaps an existing OR. Use AND to add it, or restate the full group.")
                result = [old for old in result if not (
                    "children" in old and {c["column"] for c in condition_leaves([old])} == nested)]
                additions.append(item)
                continue
            # A replacement inside a saved OR must stay in its original branch.
            def substitute(node):
                if "children" in node:
                    return _group(node["op"], [substitute(c) for c in node["children"]])
                return item if node["column"] == item["column"] and node["op"] == item["op"] else node
            nested_hits = [c for n in result if "children" in n for c in condition_leaves([n])
                           if c["column"] == item["column"] and c["op"] == item["op"]]
            if nested_hits:
                if len(nested_hits) > 1 or item["op"] == "date" and not (item.get("start") and item.get("end")):
                    raise ScopeQuestion("This column occurs in a compound filter. Identify the full OR group to change.")
                result = [substitute(n) for n in result]
                continue
            additions.append(item)
            op = item["op"]
            if op in {"ne", "not_in"}:
                continue
            same = ({"eq", "in", "null", "present"} if op in {"eq", "in", "null", "present"} else
                    {"le", "lt"} if op in {"le", "lt"} else {"ge", "gt"} if op in {"ge", "gt"} else {op})
            retained = []
            for old in result:
                if old.get("column") != item["column"] or old["op"] not in same:
                    retained.append(old)
                elif op == "date" and not (item["start"] and item["end"]):
                    # Keep the opposite endpoint when editing a single bound.
                    opposite = "start" if item["end"] else "end"
                    if old.get(opposite):
                        retained.append({**old, ("end" if opposite == "start" else "start"): None})
            result = retained
        updates = additions
    for item in updates:
        if item not in result:
            result.append(item)
    return sorted(result, key=lambda c: json.dumps(c, sort_keys=True, ensure_ascii=False))


def scoped_dataset(loaded: LoadedDataset, conditions: list[dict]) -> LoadedDataset:
    frame = loaded.df
    date_columns = {c["name"] for c in loaded.schema_info.get("columns", []) if c.get("inferred_type") in _DATE_TYPES}
    def evaluate(c):
        if "children" in c:
            children = iter(c["children"])
            mask = evaluate(next(children))
            for child in children:
                if c["op"] == "or":
                    mask |= evaluate(child)
                else:
                    mask &= evaluate(child)
            return mask
        s = frame[c["column"]]
        op = c["op"]
        if op == "date":
            dates = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
            keep = dates.notna()
            if c["start"]:
                keep &= dates.ge(pd.Timestamp(c["start"], tz="UTC"))
            if c["end"]:
                keep &= dates.lt(pd.Timestamp(c["end"], tz="UTC"))
        elif op in {"null", "present"}:
            keep = s.isna() if op == "null" else s.notna()
        else:
            values = c["value"] if isinstance(c["value"], list) else [c["value"]]
            if c["column"] in date_columns:
                try:
                    values = [date.fromisoformat(v).isoformat() for v in values]
                except (TypeError, ValueError) as e:
                    raise ScopeQuestion(f"Use calendar dates in YYYY-MM-DD format for {c['column']}.") from e
                s = pd.to_datetime(s, errors="coerce", utc=True, format="mixed").dt.strftime("%Y-%m-%d")
            elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                try:
                    values = list(pd.to_numeric(values, errors="raise"))
                except (ValueError, TypeError) as e:
                    raise ScopeQuestion(f"Use a numeric value for {c['column']}.") from e
            else:
                if op in {"lt", "le", "gt", "ge"}:
                    raise ScopeQuestion(f"Use a numeric comparison or an ISO date range for {c['column']}.")
                s = s.astype("string").str.casefold()
                values = [str(v).casefold() for v in values]
            if op in {"in", "not_in", "eq", "ne"}:
                keep = s.isin(values)
                if op in {"ne", "not_in"}:
                    keep = ~keep
            else:
                keep = getattr(s, op)(values[0])
            keep &= s.notna()
        return keep.fillna(False)
    mask = pd.Series(True, index=frame.index)
    for c in conditions:
        mask &= evaluate(c)
    subset = frame.loc[mask]
    # Never hand a scoped operator the full file path or full-data profile statistics.
    schema = {"columns": [
        {k: c[k] for k in ("name", "inferred_type", "is_potential_pii") if k in c}
        for c in loaded.schema_info.get("columns", [])
    ]}
    schema.update({k: loaded.schema_info[k] for k in ("worksheets", "worksheet_column") if k in loaded.schema_info})
    return replace(loaded, df=subset, file_path="", schema_info=schema,
                   profile_report={"row_count": len(subset), "column_count": len(subset.columns)})


def describe_scope(conditions: list[dict], join=" AND ", nested=False) -> str:
    labels = []
    symbols = {"eq": "=", "ne": "≠", "ge": "≥", "le": "≤", "gt": ">", "lt": "<", "not_in": "not in"}
    for c in conditions:
        if "children" in c:
            labels.append("(" + describe_scope(c["children"], " " + c["op"].upper() + " ", True) + ")")
            continue
        col = c["column"]
        if c.get("retail_pattern"):
            labels.append(f"{c['retail_pattern']} FY{c['retail_fy']} {c['retail_unit']} (year ends {c['retail_year_end']})")
        if c.get("fiscal_start_month"):
            labels.append(f"FY starting {calendar.month_name[c['fiscal_start_month']]} {c['fiscal_year_start']}" +
                          (f" Q{c['fiscal_quarter']}" if c.get("fiscal_quarter") else ""))
        if c["op"] == "date":
            if c["start"]:
                labels.append(f"{col} ≥ {c['start']}")
            if c["end"]:
                labels.append(f"{col} < {c['end']}")
        else:
            labels.append(f"{col} {symbols.get(c['op'], c['op'])} {c.get('value', '')}".strip())
    suffix = " (UTC calendar dates)" if not nested and any(c["op"] == "date" for c in condition_leaves(conditions)) else ""
    return join.join(labels) + suffix
