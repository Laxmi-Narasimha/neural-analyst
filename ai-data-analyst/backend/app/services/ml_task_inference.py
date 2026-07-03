from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.dataset_loader import DatasetLoaderService


_ID_NAME_RE = re.compile(r"(?:^|_)(id|uuid|guid|token|hash|key)(?:$|_)", re.IGNORECASE)
_TARGET_HINT_RE = re.compile(
    r"(?:^|_)(target|label|class|outcome|churn|fraud|revenue|sales|amount|score)(?:$|_)",
    re.IGNORECASE,
)
_TIME_NAME_RE = re.compile(r"(date|time|timestamp|created|updated|event)", re.IGNORECASE)


@dataclass(frozen=True)
class LeakageWarning:
    column: str
    warning_type: str
    detail: str
    severity: str


@dataclass(frozen=True)
class TargetCandidate:
    column: str
    inferred_task: str
    score: float
    non_null_rate: float
    unique_ratio: float
    reasons: list[str]
    leakage_warnings: list[LeakageWarning]


@dataclass(frozen=True)
class TaskInferenceResult:
    dataset_id: UUID
    dataset_version: str
    sample_rows: int
    split_strategy: str
    split_time_column: Optional[str]
    selected_target: Optional[str]
    selected_task: Optional[str]
    candidates: list[TargetCandidate]
    warnings: list[str]


def _safe_unique_ratio(series: pd.Series) -> float:
    n = int(series.shape[0] or 0)
    if n <= 0:
        return 0.0
    try:
        return float(series.nunique(dropna=True)) / float(max(n, 1))
    except Exception:
        return 0.0


def _detect_time_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        name = str(c)
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            cols.append(name)
            continue
        if _TIME_NAME_RE.search(name):
            try:
                parsed = pd.to_datetime(s.dropna().head(200), errors="coerce")
                ratio = float(parsed.notna().sum()) / float(max(parsed.shape[0], 1))
                if ratio >= 0.6:
                    cols.append(name)
            except Exception:
                pass
    return cols


def _is_id_like(name: str, series: pd.Series, unique_ratio: float) -> bool:
    if _ID_NAME_RE.search(name):
        return True
    if unique_ratio >= 0.995:
        return True
    if unique_ratio >= 0.98 and pd.api.types.is_numeric_dtype(series):
        return True
    return False


def _is_categorical_candidate(series: pd.Series, n_rows: int) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        uniq = int(series.nunique(dropna=True))
        return 2 <= uniq <= max(50, int(0.2 * max(n_rows, 1)))
    return False


def _score_candidate(name: str, series: pd.Series, non_null_rate: float, unique_ratio: float, n_rows: int) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    n_unique = int(series.nunique(dropna=True))

    if _is_categorical_candidate(series, n_rows):
        inferred_task = "classification"
        score += 0.45
        reasons.append("categorical distribution looks suitable for supervised classification")
    elif pd.api.types.is_numeric_dtype(series):
        if 2 <= n_unique <= 20:
            inferred_task = "classification"
            score += 0.45
            reasons.append("low-cardinality numeric target suggests supervised classification")
        else:
            inferred_task = "regression"
            score += 0.4
            reasons.append("numeric target candidate supports supervised regression")
    else:
        inferred_task = "classification"
        score += 0.2
        reasons.append("non-numeric target defaults to classification")

    score += 0.25 * max(0.0, min(1.0, non_null_rate))
    if non_null_rate < 0.75:
        reasons.append("lower completeness reduces confidence")

    if unique_ratio <= 0.98:
        score += 0.15
    else:
        score -= 0.2
        reasons.append("very high uniqueness suggests an identifier-like field")

    if _TARGET_HINT_RE.search(name):
        score += 0.15
        reasons.append("column name hints at a target/outcome")

    if _is_id_like(name, series, unique_ratio):
        score -= 0.5
        reasons.append("identifier-like pattern detected")

    return inferred_task, max(0.0, min(score, 1.0)), reasons


def _leakage_warnings(df: pd.DataFrame, target: str) -> list[LeakageWarning]:
    warnings: list[LeakageWarning] = []
    if target not in df.columns:
        return warnings

    target_series = df[target]
    for c in df.columns:
        if c == target:
            continue
        name = str(c)
        s = df[c]

        # Exact copy leakage check.
        try:
            aligned = pd.concat([target_series, s], axis=1).dropna()
            if aligned.shape[0] >= 20:
                same_ratio = float((aligned.iloc[:, 0] == aligned.iloc[:, 1]).mean())
                if same_ratio >= 0.995:
                    warnings.append(
                        LeakageWarning(
                            column=name,
                            warning_type="exact_copy",
                            detail=f"{name} matches target values almost perfectly",
                            severity="high",
                        )
                    )
                    continue
        except Exception:
            pass

        # Numeric near-perfect correlation leakage check.
        try:
            if pd.api.types.is_numeric_dtype(target_series) and pd.api.types.is_numeric_dtype(s):
                aligned = pd.concat([target_series, s], axis=1).dropna()
                if aligned.shape[0] >= 30:
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    if math.isfinite(corr) and abs(corr) >= 0.995:
                        warnings.append(
                            LeakageWarning(
                                column=name,
                                warning_type="high_correlation",
                                detail=f"{name} has near-perfect correlation with target ({corr:.3f})",
                                severity="medium",
                            )
                        )
                        continue
        except Exception:
            pass

        # Name-based leakage hints.
        low_target = str(target).lower()
        low_name = name.lower()
        if low_target and low_target in low_name and low_name != low_target:
            warnings.append(
                LeakageWarning(
                    column=name,
                    warning_type="name_pattern",
                    detail=f"{name} appears derived from target naming pattern",
                    severity="medium",
                )
            )

    return warnings


def infer_target_candidates(
    df: pd.DataFrame,
    *,
    preferred_target: Optional[str] = None,
    max_candidates: int = 8,
) -> tuple[list[TargetCandidate], Optional[str], Optional[str], list[str]]:
    warnings: list[str] = []
    n_rows = int(df.shape[0] or 0)
    if n_rows <= 0 or int(df.shape[1] or 0) <= 0:
        return [], None, None, ["dataset has no rows or no columns to infer targets from"]

    candidates: list[TargetCandidate] = []
    for c in df.columns:
        name = str(c)
        s = df[c]
        non_null_rate = 1.0 - (float(s.isna().mean()) if n_rows > 0 else 1.0)
        unique_ratio = _safe_unique_ratio(s)
        inferred_task, score, reasons = _score_candidate(name, s, non_null_rate, unique_ratio, n_rows)
        if score <= 0.05:
            continue
        candidates.append(
            TargetCandidate(
                column=name,
                inferred_task=inferred_task,
                score=round(float(score), 4),
                non_null_rate=round(float(non_null_rate), 4),
                unique_ratio=round(float(unique_ratio), 4),
                reasons=reasons,
                leakage_warnings=[],
            )
        )

    if not candidates:
        return [], None, None, ["no viable target candidates were detected"]

    candidates.sort(key=lambda c: (c.score, c.non_null_rate), reverse=True)
    candidates = candidates[: int(max(1, max_candidates))]

    selected: Optional[TargetCandidate] = None
    if preferred_target:
        for c in candidates:
            if c.column == preferred_target:
                selected = c
                break
        if selected is None:
            warnings.append(f"preferred target '{preferred_target}' was not selected as a top candidate")
    if selected is None:
        selected = candidates[0]

    leak = _leakage_warnings(df, selected.column)
    candidates = [
        TargetCandidate(
            column=c.column,
            inferred_task=c.inferred_task,
            score=c.score,
            non_null_rate=c.non_null_rate,
            unique_ratio=c.unique_ratio,
            reasons=c.reasons,
            leakage_warnings=leak if c.column == selected.column else [],
        )
        for c in candidates
    ]
    return candidates, selected.column, selected.inferred_task, warnings


class MLTaskInferenceService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._loader = DatasetLoaderService(session)

    async def infer(
        self,
        *,
        dataset_id: UUID,
        owner_id: UUID,
        preferred_target: Optional[str] = None,
        sample_rows: int = 100_000,
    ) -> TaskInferenceResult:
        loaded = await self._loader.load_dataset(
            dataset_id,
            owner_id=owner_id,
            require_ready=True,
            sample_rows=max(1000, min(int(sample_rows), 500_000)),
        )
        df = loaded.df

        candidates, selected_target, selected_task, warnings = infer_target_candidates(
            df,
            preferred_target=preferred_target,
            max_candidates=8,
        )
        time_columns = _detect_time_columns(df)
        split_time_column = time_columns[0] if time_columns else None
        split_strategy = "time_split" if split_time_column else "random_split"

        return TaskInferenceResult(
            dataset_id=dataset_id,
            dataset_version=str(loaded.version_hash),
            sample_rows=int(df.shape[0] or 0),
            split_strategy=split_strategy,
            split_time_column=split_time_column,
            selected_target=selected_target,
            selected_task=selected_task,
            candidates=candidates,
            warnings=warnings,
        )
