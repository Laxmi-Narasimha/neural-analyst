"""Score a frame with a saved ModelBundle and say whether the result can be trusted.

Shared by the batch scoring API and the chat ``ml_score`` operator, so both refuse the same
inputs and report the same checks:
- every row is scored (callers enforce the row cap; nothing here samples),
- input columns the model needs are checked; when half or more of them are missing the file is
  refused, because predictions made mostly from imputed values look confident and are not,
- each input is compared with the training data (PSI drift), and the mix of predictions with the
  held-out predictions at training time,
- when the file holds the outcome, how the model actually did on it, and whether those rows may be
  the ones it was trained on (which makes that number optimistic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.ml.drift import prediction_shift

# Refuse when this share of the model's inputs is absent from the file.
MAX_MISSING_INPUT_SHARE = 0.5


@dataclass
class ScoredFrame:
    table: pd.DataFrame
    prediction_column: str
    prediction_summary: dict[str, Any]
    drift: Optional[dict[str, Any]]
    prediction_shift: Optional[dict[str, Any]]
    realised: Optional[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "rows_scored": int(len(self.table)),
            "prediction_column": self.prediction_column,
            "prediction_summary": self.prediction_summary,
            "drift": self.drift,
            "prediction_shift": self.prediction_shift,
            "realised_performance": self.realised,
            "warnings": list(self.warnings),
        }


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def realised_performance(bundle: Any, df: pd.DataFrame, preds: list[Any]) -> Optional[dict[str, Any]]:
    target = bundle.target_column
    if target not in df.columns:
        return None
    actual = df[target]
    known = actual.notna().to_numpy()
    if known.sum() < 10:
        return None
    held_out = (bundle.training_info or {}).get("model_test_metrics") or {}
    if bundle.is_classifier:
        a = np.asarray([_norm(x) for x in actual[known]])
        p = np.asarray([_norm(x) for x in np.asarray(preds, dtype=object)[known]])
        labels = sorted(set(a) | set(p))
        recalls = [float((p[a == c] == c).mean()) for c in labels if (a == c).any()]
        return {"rows_with_outcome": int(known.sum()), "accuracy": round(float((a == p).mean()), 4),
                "balanced_accuracy": round(float(np.mean(recalls)), 4),
                "held_out_accuracy_at_training": held_out.get("accuracy"),
                "held_out_balanced_accuracy_at_training": held_out.get("balanced_accuracy")}
    a = pd.to_numeric(actual[known], errors="coerce").to_numpy(dtype=float)
    p = np.asarray(preds, dtype=float)[known]
    ok = np.isfinite(a)
    if ok.sum() < 10:
        return None
    return {"rows_with_outcome": int(ok.sum()), "mae": round(float(np.abs(a[ok] - p[ok]).mean()), 4),
            "held_out_mae_at_training": held_out.get("mae")}


def missing_inputs(bundle: Any, columns: Any) -> tuple[list[str], list[str]]:
    """(columns the model uses, those absent from ``columns``)."""
    from app.ml.tabular_pipeline import dedupe_column_names

    present = set(dedupe_column_names(list(columns)))
    needed = list(getattr(bundle, "used_columns", None) or bundle.input_columns)
    return needed, [c for c in needed if c not in present]


def score_frame(bundle: Any, df: pd.DataFrame) -> ScoredFrame:
    """Score every row of ``df``. Raises ValueError when the file cannot be scored honestly."""
    from app.ml.tabular_pipeline import ModelBundle

    if not isinstance(bundle, ModelBundle):
        raise ValueError("This model was saved without its preprocessing; retrain it to score files")
    if df.empty:
        raise ValueError("The file has no rows to score")
    needed, missing = missing_inputs(bundle, df.columns)
    if needed and len(missing) >= MAX_MISSING_INPUT_SHARE * len(needed):
        raise ValueError(
            f"This file lacks {len(missing)} of the {len(needed)} columns the model uses "
            f"({', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}). Predictions would be mostly guesses, "
            "so nothing was scored. Score a file with the same columns as the training data."
        )
    warnings = list(bundle.check_input(df))
    preds = bundle.predict(df)
    proba = bundle.predict_proba(df)

    out = df.copy()
    pred_col = f"predicted_{bundle.target_column}"
    out[pred_col] = preds
    if proba is not None and bundle.classes is not None:
        arr = np.asarray(proba, dtype=float)
        for i, cls in enumerate(bundle.classes):
            out[f"probability_{cls}"] = np.round(arr[:, i], 6)

    if bundle.classes is not None:
        counts = pd.Series([str(p) for p in preds]).value_counts()
        prediction_summary: dict[str, Any] = {"counts": {str(k): int(v) for k, v in counts.items()}}
    else:
        vals = np.asarray(preds, dtype=float)
        prediction_summary = {"mean": float(np.nanmean(vals)), "min": float(np.nanmin(vals)),
                              "max": float(np.nanmax(vals))}

    drift = bundle.drift_report(df)
    if drift is None:
        warnings.append("This model was trained before drift references were stored; retrain it to get a drift check.")
    elif drift["verdict"] != "stable":
        warnings.append(drift["message"])
    return ScoredFrame(
        table=out,
        prediction_column=pred_col,
        prediction_summary=prediction_summary,
        drift=drift,
        prediction_shift=prediction_shift(getattr(bundle, "prediction_reference", None), preds, proba, bundle.classes),
        realised=realised_performance(bundle, df, preds),
        warnings=warnings,
    )


def training_overlap_warning(tags: dict[str, Any] | None, *, dataset_id: Any, dataset_version: Any,
                             realised: Optional[dict[str, Any]]) -> Optional[str]:
    """Warn when the scored rows may include the model's own training rows."""
    tags = tags or {}
    if not realised or str(tags.get("dataset_id") or "") != str(dataset_id):
        return None
    same = str(tags.get("dataset_version") or "") == str(dataset_version or "")
    which = "the same version of the dataset" if same else "an earlier version of this dataset"
    return (f"The model was trained on {which}, so many scored rows were seen in training and the accuracy "
            "on them is optimistic. The held-out figure from training is the honest one.")
