from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logging import get_logger
from app.services.data_ingestion import DataIngestionService, FileFormat

logger = get_logger(__name__)


def _reservoir_sample_frames(frames: list[pd.DataFrame], k: int, *, random_state: int = 42) -> pd.DataFrame:
    """Combine chunked frames with reservoir sampling to cap memory."""
    if not frames:
        return pd.DataFrame()
    if k <= 0:
        return pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(random_state)
    reservoir: list[pd.Series] = []
    seen = 0

    for frame in frames:
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            seen += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, seen))
                if j < k:
                    reservoir[j] = row

    if not reservoir:
        return pd.DataFrame()
    return pd.DataFrame(reservoir).reset_index(drop=True)


def _read_csv_chunks(path: Path, *, chunk_rows: int) -> list[pd.DataFrame]:
    return list(pd.read_csv(path, chunksize=int(chunk_rows), low_memory=False))


def _read_parquet_sample(path: Path, *, sample_rows: int, random_state: int) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        df = pd.read_parquet(path)
        if len(df) <= sample_rows:
            return df
        return df.sample(n=int(sample_rows), random_state=random_state)

    pf = pq.ParquetFile(path)
    frames: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=min(100_000, sample_rows)):
        frames.append(batch.to_pandas())
        if sum(len(f) for f in frames) >= sample_rows * 2:
            break
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(combined) <= sample_rows:
        return combined
    return combined.sample(n=int(sample_rows), random_state=random_state)


def load_representative_sample(
    file_path: str | Path,
    *,
    filename: str,
    file_format: str | None,
    sample_rows: int,
    total_rows: Optional[int] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a statistically representative sample for ML without loading full tables.

    For files larger than sample_rows, uses reservoir sampling over CSV chunks or
    batched parquet reads. Falls back to head(n) when chunking is unavailable.
    """
    path = Path(file_path)
    sample_rows = max(1_000, int(sample_rows))
    chunk_rows = int(settings.ml.reservoir_chunk_rows)
    meta: dict[str, Any] = {
        "requested_sample_rows": sample_rows,
        "sampling_method": "full_load",
        "total_rows_hint": int(total_rows) if total_rows is not None else None,
    }

    ingestion = DataIngestionService()
    fmt: FileFormat | None = None
    if file_format:
        try:
            fmt = FileFormat(str(file_format).lower())
        except Exception:
            fmt = None

    if total_rows is not None and int(total_rows) <= sample_rows:
        with path.open("rb") as f:
            df = ingestion.parse_file(f, filename, file_format=fmt)
        meta["sampling_method"] = "full_load"
        meta["loaded_rows"] = int(len(df))
        return df, meta

    suffix = path.suffix.lower()
    is_csv = suffix in {".csv", ".tsv", ".txt"} or fmt == FileFormat.CSV
    is_parquet = suffix in {".parquet", ".pq"} or fmt == FileFormat.PARQUET

    if is_csv:
        try:
            frames = _read_csv_chunks(path, chunk_rows=chunk_rows)
            if len(frames) == 1 and len(frames[0]) <= sample_rows:
                meta["sampling_method"] = "single_chunk"
                meta["loaded_rows"] = int(len(frames[0]))
                return frames[0], meta
            df = _reservoir_sample_frames(frames, sample_rows, random_state=random_state)
            meta["sampling_method"] = "reservoir_csv"
            meta["loaded_rows"] = int(len(df))
            meta["chunks_scanned"] = len(frames)
            return df, meta
        except Exception as e:
            logger.warning(f"Reservoir CSV sampling failed, falling back: {e}")

    if is_parquet:
        try:
            df = _read_parquet_sample(path, sample_rows=sample_rows, random_state=random_state)
            meta["sampling_method"] = "parquet_sample"
            meta["loaded_rows"] = int(len(df))
            return df, meta
        except Exception as e:
            logger.warning(f"Parquet sampling failed, falling back: {e}")

    with path.open("rb") as f:
        try:
            df = ingestion.parse_file(f, filename, file_format=fmt, nrows=sample_rows)
        except TypeError:
            f.seek(0)
            df = ingestion.parse_file(f, filename, file_format=fmt)
            if len(df) > sample_rows:
                df = df.sample(n=int(sample_rows), random_state=random_state)
    meta["sampling_method"] = "head_or_random"
    meta["loaded_rows"] = int(len(df))
    return df, meta


async def aload_representative_sample(**kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
    return await asyncio.to_thread(load_representative_sample, **kwargs)