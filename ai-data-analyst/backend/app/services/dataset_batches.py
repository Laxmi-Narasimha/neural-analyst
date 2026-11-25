"""Bounded readers shared by upload profiling and conversational analysis."""
from __future__ import annotations

from itertools import islice
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from app.services.data_ingestion import CSVParser

BATCH_ROWS = 8192
LARGE_BYTES = 16 * 1024 * 1024
LARGE_ROWS = 50_000


def large_file(path: Path, fmt: str) -> bool:
    if path.stat().st_size >= LARGE_BYTES:
        return True
    if fmt == "xls":
        # BIFF row counts are available only after decoding the sheet.
        return path.stat().st_size >= 256 * 1024
    if fmt == "xlsx":
        with ZipFile(path) as archive:
            return sum(i.file_size for i in archive.infolist()) >= LARGE_BYTES
    if fmt == "parquet":
        import pyarrow.parquet as pq
        metadata = pq.ParquetFile(path).metadata
        return (metadata.num_rows > LARGE_ROWS or
                sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)) >= LARGE_BYTES)
    return False


def frames(path: Path, fmt: str, schema=None):
    if fmt in {"csv", "tsv"}:
        parser = CSVParser()
        with path.open("rb") as source:
            encoding = parser._detect_encoding(source)
            sep = "\t" if fmt == "tsv" else parser._detect_delimiter(source, encoding)
            text = {c["name"]: "string" for c in (schema or {}).get("columns", [])
                    if c.get("inferred_type") in {"string", "categorical", "text"}}
            with pd.read_csv(source, sep=sep, encoding=encoding, dtype=text, chunksize=BATCH_ROWS) as reader:
                yield from reader
    elif fmt == "parquet":
        import pyarrow.parquet as pq
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(batch_size=BATCH_ROWS):
                yield batch.to_pandas()
    elif fmt in {"xls", "xlsx"}:
        selected = (schema or {}).get("worksheets")
        source_column = (schema or {}).get("worksheet_column")
        columns = [c["name"] for c in (schema or {}).get("columns", [])]
        for name, batches in workbook_sheets(path, fmt, selected):
            for frame in batches:
                if frame.empty:
                    continue
                if source_column:
                    frame[source_column] = name
                yield frame.reindex(columns=columns) if selected else frame
    elif fmt == "json":
        import ijson
        with path.open("rb") as source:
            first = source.read(1)
            while first and first.isspace():
                first = source.read(1)
            source.seek(0)
            records = ijson.items(source, "item" if first == b"[" else "",
                                  multiple_values=first != b"[", use_float=True)
            names = [c["name"] for c in (schema or {}).get("columns", [])]
            while batch := list(islice(records, BATCH_ROWS)):
                if any(not isinstance(row, dict) or any(isinstance(v, (dict, list)) for v in row.values()) for row in batch):
                    raise ValueError("Large JSON uploads need flat records in an array or one record per line.")
                frame = pd.DataFrame.from_records(batch)
                if names and set(frame) - set(names):
                    raise ValueError("New JSON fields were found. Reprocess the dataset to update its schema.")
                yield frame.reindex(columns=names) if names else frame
    else:
        raise ValueError(f"Unsupported format for batched processing: {fmt}")


def xls_sample_limit(frame, limit):
    """Keep wide binary-sheet samples proportional to decoded bytes as well as row count."""
    size = int(frame.memory_usage(index=False, deep=True).sum())
    return min(limit, max(1, (8 * 1024 * 1024 * len(frame)) // max(1, size)))


def profile_sample(path, fmt):
    """Uniform priority sample across batches with an exact count of input rows."""
    return sample_batches(frames(path, fmt), fmt)


def sample_batches(batches, fmt):
    rng = np.random.default_rng(42)
    sample = pd.DataFrame()
    ranks = np.empty(0)
    count = 0
    limit = BATCH_ROWS
    for batch in batches:
        count += len(batch)
        if not len(batch):
            if not count:
                sample = batch
            continue
        candidates = pd.concat([sample, batch], ignore_index=True) if len(sample) else batch.reset_index(drop=True)
        priorities = np.concatenate([ranks, rng.random(len(batch))])
        if fmt == "xls":
            limit = xls_sample_limit(batch, limit)
        if len(priorities) > limit:
            take = np.argpartition(priorities, limit)[:limit]
            sample, ranks = candidates.iloc[take], priorities[take]
        else:
            sample, ranks = candidates, priorities
    return sample.reset_index(drop=True), count


def workbook_sheets(path, fmt, selected):
    """Read the first sheet by default, every sheet for profiling, or a named selection."""
    if fmt == "xls":
        from app.services.legacy_excel import xls_sheets
        yield from xls_sheets(path, BATCH_ROWS, selected)
        return
    import openpyxl
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for index, sheet in enumerate(book.worksheets):
            if selected == "all" or (selected is None and index == 0) or (selected and sheet.title in selected):
                yield sheet.title, _xlsx_batches(sheet)
    finally:
        book.close()


def _xlsx_batches(sheet):
    rows = sheet.values
    header = next(rows, ())
    names = []
    for i, val in enumerate(header):
        base = str(val) if val is not None else f"Unnamed: {i}"
        name, suffix = base, 1
        while name in names:
            name = f"{base}.{suffix}"
            suffix += 1
        names.append(name)
    yield pd.DataFrame(columns=names)
    while batch := list(islice(rows, BATCH_ROWS)):
        # Write-only exporters can omit the worksheet dimensions and emit ragged rows.
        width = max(map(len, batch))
        while len(names) < width:
            name = f"Unnamed: {len(names)}"
            while name in names:
                name += ".1"
            names.append(name)
        yield pd.DataFrame.from_records(batch).reindex(columns=range(len(names))).set_axis(names, axis=1)
