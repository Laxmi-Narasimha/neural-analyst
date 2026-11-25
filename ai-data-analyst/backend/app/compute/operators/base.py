from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import pandas as pd


@dataclass(frozen=True)
class OperatorContext:
    dataset_id: UUID
    dataset_version: str
    df: pd.DataFrame
    profile_report: dict[str, Any]
    schema_info: dict[str, Any]
    source_file_path: str | None = None
    source_file_format: str | None = None
    # The signed-in owner the operator runs for; operators that persist anything (models) need it.
    owner_id: UUID | None = None
    # Shown to people (for example in the name of a model trained on this dataset).
    dataset_name: str | None = None


@dataclass(frozen=True)
class OperatorResult:
    tables: dict[str, pd.DataFrame]
    metrics: dict[str, Any]
    charts: dict[str, dict[str, Any]]
    summary: dict[str, Any]


class Operator(Protocol):
    name: str

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult: ...
