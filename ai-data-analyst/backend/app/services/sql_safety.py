from __future__ import annotations

import re
from dataclasses import dataclass


class UnsafeSQLError(ValueError):
    pass


_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "call",
    "execute",
    "vacuum",
    "attach",
    "detach",
    "pragma",
)

_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", flags=re.IGNORECASE)
_LEADING_RE = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/\s*)*", flags=re.DOTALL)
_START_RE = re.compile(r"^(select|with)\b", flags=re.IGNORECASE)

# DuckDB can read arbitrary local files inside a SELECT via table functions like read_csv_auto().
# For dataset-scoped queries, we must prevent exfiltration by blocking these functions and
# the `FROM 'path'` shorthand.
_DUCKDB_FORBIDDEN_TABLE_FUNCS = (
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "parquet_scan",
    "csv_scan",
    "json_scan",
    "sqlite_scan",
    "postgres_scan",
    "mysql_scan",
    "iceberg_scan",
    "delta_scan",
)
_DUCKDB_FORBIDDEN_FUNCS_RE = re.compile(
    r"\b(" + "|".join(_DUCKDB_FORBIDDEN_TABLE_FUNCS) + r")\b",
    flags=re.IGNORECASE,
)
_DUCKDB_FROM_FILE_LITERAL_RE = re.compile(r"\b(from|join)\s+['\"]", flags=re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise UnsafeSQLError("Query must be a string")
    s = sql.strip()
    # Strip a trailing semicolon (but reject multiple statements).
    if ";" in s[:-1]:
        raise UnsafeSQLError("Multiple statements are not allowed")
    if s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def validate_readonly_sql(sql: str) -> str:
    s = normalize_sql(sql)
    # Remove leading comments for start keyword check only.
    leading_stripped = _LEADING_RE.sub("", s).lstrip()
    if not _START_RE.match(leading_stripped):
        raise UnsafeSQLError("Only SELECT/CTE queries are allowed")
    if _FORBIDDEN_RE.search(leading_stripped):
        raise UnsafeSQLError("Query contains a forbidden keyword")
    return s


def validate_dataset_sql(sql: str) -> str:
    """
    Validate SQL intended to run against an uploaded dataset via DuckDB.

    In addition to readonly validation, block common file/table functions and
    file-literal scans to prevent reading arbitrary server-side files.
    """

    s = validate_readonly_sql(sql)
    # Check the user query itself for obvious file scan/exfil patterns.
    leading_stripped = _LEADING_RE.sub("", s).lstrip()
    if _DUCKDB_FORBIDDEN_FUNCS_RE.search(leading_stripped):
        raise UnsafeSQLError("Query contains a forbidden table function")
    if _DUCKDB_FROM_FILE_LITERAL_RE.search(leading_stripped):
        raise UnsafeSQLError("Direct file reads are not allowed in dataset queries")
    return s


def enforce_row_limit(sql: str, limit_rows: int) -> str:
    limit = int(limit_rows)
    if limit <= 0:
        raise ValueError("limit_rows must be positive")
    s = normalize_sql(sql)
    # Always enforce at the outer level to cap results even if the user included a larger LIMIT.
    return f"SELECT * FROM ({s}) AS __na_q LIMIT {limit}"


_IDENT_PART_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(identifier: str, dialect: str) -> str:
    # Conservative identifier quoting: allow simple names and dotted schema.table.
    # We do NOT allow arbitrary expressions here.
    if not isinstance(identifier, str):
        raise ValueError("identifier must be a string")
    parts = [p for p in identifier.strip().split(".") if p]
    if not parts:
        raise ValueError("identifier is empty")
    for p in parts:
        if not _IDENT_PART_RE.match(p):
            raise ValueError("identifier contains invalid characters")

    if dialect in ("postgresql", "sqlite"):
        q = '"'
    elif dialect in ("mysql",):
        q = "`"
    else:
        q = '"'
    return ".".join([f"{q}{p}{q}" for p in parts])


@dataclass(frozen=True)
class QueryGuardrails:
    max_rows: int
    timeout_seconds: int
