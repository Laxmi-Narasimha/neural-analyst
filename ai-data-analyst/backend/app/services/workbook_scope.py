"""Versioned worksheet catalogs and per-turn workbook views for grounded chat."""
import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import update

from app.core.serialization import to_jsonable
from app.models import Dataset
from app.services import dataset_batches
from app.services.chat_scope import ScopeQuestion, _DATE_TYPES, _NUMBER_TYPES, condition_leaves
from app.services.data_ingestion import FileFormat, get_ingestion_service
from app.services.object_store import get_object_store


def profile_workbook(path, fmt, filename, version):
    ingestion = get_ingestion_service()
    catalog = []
    first = None
    for name, batches in dataset_batches.workbook_sheets(path, fmt, "all"):
        frame, count = dataset_batches.sample_batches(batches, fmt)
        profile = ingestion._generate_profile(frame, filename, FileFormat(fmt), path.stat().st_size, version)
        if first is None:
            first = (frame, profile, count)
        columns = []
        for col in profile.columns:
            info = col.to_dict()
            columns.append({key: info[key] for key in (
                "name", "inferred_type", "is_potential_pii", "is_unique_identifier", "is_constant", "null_count",
                "unique_count", "unique_percentage"
            ) if key in info} | {"has_values": col.null_count < len(frame)})
        catalog.append({"name": name, "rows": count, "columns": columns})
    if first is None:
        raise ValueError("This workbook contains no worksheets.")
    return (*first, to_jsonable(catalog))


def _read_catalog(record):
    path = Path(get_object_store().ensure_local_path(record.file_path, expected_size_bytes=record.file_size_bytes,
                                                     filename_hint=record.original_filename))
    return profile_workbook(path, record.file_format, record.original_filename,
                            (record.profile_report or {}).get("file_hash", ""))[3]


async def workbook_catalog(record, session):
    schema = record.schema_info or {}
    if catalog := schema.get("worksheet_catalog"):
        return catalog
    catalog = await asyncio.to_thread(_read_catalog, record)
    # Upgrade a legacy profile once. A concurrent version activation wins.
    await session.execute(update(Dataset).where(
        Dataset.id == record.id, Dataset.file_path == record.file_path, Dataset.updated_at == record.updated_at
    ).values(schema_info={**schema, "worksheet_catalog": catalog}).execution_options(synchronize_session=False))
    return catalog


def type_family(kind):
    if kind in _NUMBER_TYPES:
        return "number"
    if kind in _DATE_TYPES:
        return "date"
    return "boolean" if kind == "boolean" else "text"


def worksheet_view(record, catalog, requested):
    names = [sheet["name"] for sheet in catalog]
    if requested is None:
        requested = names[:1]
    if not isinstance(requested, list) or not requested or any(not isinstance(n, str) or n not in names for n in requested):
        raise ScopeQuestion("Choose one or more worksheets from this workbook: " + ", ".join(names))
    selected = [s for s in catalog if s["name"] in requested]
    fields = {}
    for sheet in selected:
        for col in sheet["columns"]:
            name = col["name"]
            prior = fields.get(name)
            if prior is None:
                fields[name] = dict(col)
                continue
            kind = col["inferred_type"]
            if prior["has_values"] and col["has_values"] and type_family(prior["inferred_type"]) != type_family(kind):
                raise ScopeQuestion(f"Column {name!r} has different types across these worksheets. Select sheets with compatible types or analyze them separately.")
            if col["has_values"] and (not prior["has_values"] or kind in {"float", "double", "decimal"}):
                prior["inferred_type"] = kind
            prior["has_values"] |= col["has_values"]
            prior["is_potential_pii"] |= col.get("is_potential_pii", False)
    # Stable even when switching between a single sheet and multiple sheets.
    reserved = {c["name"].casefold() for s in catalog for c in s["columns"]}
    source = "Worksheet"
    while source.casefold() in reserved:
        source = "Source " + source
    chosen = [s["name"] for s in selected]
    fields[source] = {"name": source, "inferred_type": "categorical", "unique_count": len(chosen)}
    rows = sum(s["rows"] for s in selected)
    schema = {"columns": list(fields.values()), "worksheets": chosen, "worksheet_column": source}
    return SimpleNamespace(id=record.id, name=record.name, file_format=record.file_format, file_path=record.file_path,
                           file_size_bytes=record.file_size_bytes, original_filename=record.original_filename,
                           schema_info=schema, profile_report={"file_hash": (record.profile_report or {}).get("file_hash")},
                           row_count=rows, column_count=len(fields))


def check_carried_columns(schema, last, plan, edit):
    fields = {c["name"]: c for c in schema["columns"]}
    required = set()
    if not edit.clear:
        for leaf in condition_leaves(last.get("filters", [])):
            if leaf["column"] not in edit.remove and not (leaf["op"] == "date" and "@dates" in edit.remove):
                required.add(leaf["column"])
    for step in plan:
        params = step.get("params", {})
        for key in ("value_column", "group_by", "time_column", "target_column"):
            name = params.get(key)
            if name:
                required.add(name)
                if fields.get(name, {}).get("is_potential_pii"):
                    raise ScopeQuestion(f"Column {name!r} is marked as potentially sensitive on these worksheets. Choose another field for this calculation.")
    for name in sorted(required):
        old = last.get("column_types", {}).get(name)
        if name not in fields or (old and old != type_family(fields[name]["inferred_type"])):
            raise ScopeQuestion(f"The selected worksheets cannot carry forward column {name!r}. Select compatible sheets, or send ‘start over’ on the new sheets and specify a new calculation.")


def sheet_command(message, catalog):
    """Extract an explicit trailing sheet clause, ignoring quoted data literals."""
    pattern = r""" "(?:[^"]|"")*" | (?<!\w)'(?:[^']|'')*' | (?P<clause>
        (?:\b(?:on|across|from|using)\s+|^(?:use|switch\s+to)\s+)(?P<all>all\s+)?(?:work)?sheets?\b
    ) """
    for match in re.finditer(pattern, message, re.I | re.X):
        if not match.group("clause"):
            continue
        text = message[match.end():].strip().rstrip("?").strip()
        names = [sheet["name"] for sheet in catalog]
        if match.group("all") and not text:
            return message[:match.start()].strip(), names
        found = []
        while text:
            for name in sorted(names, key=len, reverse=True):
                literals = ['"' + name.replace('"', '""') + '"', "'" + name.replace("'", "''") + "'", name]
                token = next((v for v in literals if text[:len(v)].casefold() == v.casefold() and
                              re.match(r"(?:\s*,|\s+and\s+|$)", text[len(v):], re.I)), None)
                if token is not None:
                    found.append(name)
                    text = re.sub(r"^(?:\s*,\s*(?:and\s+)?|\s+and\s+)", "", text[len(token):], count=1, flags=re.I)
                    break
            else:
                raise ScopeQuestion("Choose a worksheet by its full name: " + ", ".join(names))
        if found:
            return message[:match.start()].strip(), found
    return message, None
