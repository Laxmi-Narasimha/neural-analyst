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


@dataclass(frozen=True)
class OperatorResult:
    tables: dict[str, pd.DataFrame]
    metrics: dict[str, Any]
    charts: dict[str, dict[str, Any]]
    summary: dict[str, Any]


class Operator(Protocol):
    name: str

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult: ...
