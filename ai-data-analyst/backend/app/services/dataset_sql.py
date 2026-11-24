"""Run a user's SQL against one dataset, with no way to reach any other file.

The query text is checked by sql_safety first, but a pattern check alone cannot stop DuckDB from
reading files: it has dozens of file functions (glob, read_text, read_blob, ...), and new versions
add more. So the query never runs where files are reachable:

1. The dataset file is copied once into its own DuckDB database file as a native table
   named ``dataset`` (cached by file path, size and modification time).
2. The user's query runs on a read-only connection to that database, after turning off
   external access and locking the configuration, so no statement can turn it back on.

DuckDB then refuses every file read (and network read) itself, whatever the query is.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED = {"csv": "csv", "tsv": "csv", "txt": "csv", "parquet": "parquet", "pq": "parquet"}


def _quote(value: str) -> str:
    return "'" + str(value or "").replace("\\", "/").replace("'", "''") + "'"


def _cache_dir() -> Path:
    from app.core.config import settings

    folder = Path(settings.object_store.cache_dir) / "dataset_sql"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _cache_path(file_path: str, kind: str) -> Path:
    st = os.stat(file_path)
    key = f"{Path(file_path).resolve()}|{st.st_size}|{st.st_mtime_ns}|{kind}"
    return _cache_dir() / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}.duckdb"


def _source_note(target: Path) -> Path:
    return target.with_suffix(".source")


def prune_orphans() -> int:
    """Delete cached copies whose source file is gone (a purged dataset) or has changed since."""
    removed = 0
    for note in _cache_dir().glob("*.source"):
        try:
            source = note.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        try:
            if source and note.stem in {_cache_path(source, k).stem for k in ("csv", "parquet")}:
                continue
        except OSError:
            pass  # the source is gone
        for path in (note.with_suffix(".duckdb"), note.with_suffix(".duckdb.wal"), note):
            try:
                path.unlink(missing_ok=True)
                removed += path.suffix == ".duckdb"
            except OSError:
                pass  # still open by a running query; the next prune removes it
    return removed


def _build(file_path: str, kind: str, target: Path) -> None:
    import duckdb

    tmp = target.with_name(f"{target.stem}.{uuid4().hex}.building")
    reader = "read_parquet" if kind == "parquet" else "read_csv_auto"
    try:
        con = duckdb.connect(str(tmp), config={"threads": "2"})
        try:
            con.execute(f"CREATE TABLE dataset AS SELECT * FROM {reader}({_quote(file_path)})")
            con.execute("CHECKPOINT")
        finally:
            con.close()
        try:
            os.replace(tmp, target)
        except PermissionError:
            # Another request built and opened it first (Windows keeps open files locked).
            if not target.exists():
                raise
    finally:
        for leftover in (tmp, tmp.with_name(tmp.name + ".wal")):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass


def sandboxed_database(file_path: str, file_format: str) -> Path:
    kind = _SUPPORTED.get(str(file_format or "").lower().strip())
    if kind is None:
        raise ValueError("Dataset SQL is supported for CSV/TSV/Parquet only")
    target = _cache_path(file_path, kind)
    if not target.exists():
        # Written first, so a crash mid-build still leaves a note that lets purge find the copy.
        _source_note(target).write_text(str(Path(file_path).resolve()), encoding="utf-8")
        _build(file_path, kind, target)
    return target


def open_sandboxed(db_path: Path) -> Any:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
    except Exception:
        con.close()
        raise
    return con


def run_dataset_sql(*, file_path: str, file_format: str, sql: str) -> tuple[pd.DataFrame, float]:
    """Run already-validated SQL against table ``dataset``. Returns (rows, milliseconds)."""
    con = open_sandboxed(sandboxed_database(file_path, file_format))
    try:
        t0 = time.perf_counter()
        df = con.execute(str(sql)).df()
        return df, (time.perf_counter() - t0) * 1000.0
    finally:
        try:
            con.close()
        except Exception:
            pass
