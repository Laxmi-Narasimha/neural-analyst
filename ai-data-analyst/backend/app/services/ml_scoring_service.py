"""Score a whole new dataset with a saved model, and say whether the result can be trusted.

Every row is scored (no sampling); a file larger than ML_MAX_SCORING_ROWS is refused with a clear
message. The answer carries:
- a table artifact: the input rows plus the prediction (and class probabilities),
- a drift report: each input the model uses compared with its training data (PSI),
- how the mix of predictions compares with the held-out predictions at training time,
- when the file already holds the outcome, how the model actually did on it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.serialization import to_jsonable


def _realised_performance(bundle: Any, df: pd.DataFrame, preds: list[Any]) -> Optional[dict[str, Any]]:
    target = bundle.target_column
    if target not in df.columns:
        return None
    actual = df[target]
    known = actual.notna().to_numpy()
    if known.sum() < 10:
        return None
    if bundle.is_classifier:
        a = actual[known].astype(str).str.strip().str.casefold().to_numpy()
        p = np.asarray([str(x).strip().casefold() for x in np.asarray(preds, dtype=object)[known]])
        labels = sorted(set(a) | set(p))
        recalls = [float((p[a == c] == c).mean()) for c in labels if (a == c).any()]
        held_out = (bundle.training_info or {}).get("model_test_metrics") or {}
        return {"rows_with_outcome": int(known.sum()), "accuracy": round(float((a == p).mean()), 4),
                "balanced_accuracy": round(float(np.mean(recalls)), 4),
                "held_out_accuracy_at_training": held_out.get("accuracy"),
                "held_out_balanced_accuracy_at_training": held_out.get("balanced_accuracy")}
    a = pd.to_numeric(actual[known], errors="coerce").to_numpy(dtype=float)
    p = np.asarray(preds, dtype=float)[known]
    ok = np.isfinite(a)
    if ok.sum() < 10:
        return None
    held_out = (bundle.training_info or {}).get("model_test_metrics") or {}
    return {"rows_with_outcome": int(ok.sum()), "mae": round(float(np.abs(a[ok] - p[ok]).mean()), 4),
            "held_out_mae_at_training": held_out.get("mae")}


class MLScoringService:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def score_dataset(
        self,
        *,
        owner_id: UUID,
        model: Any,
        model_name: str,
        model_version: Optional[int],
        dataset_id: UUID,
    ) -> dict[str, Any]:
        from app.compute.artifacts import ArtifactStore
        from app.ml.drift import prediction_shift
        from app.ml.tabular_pipeline import ModelBundle
        from app.services.artifact_index import ArtifactIndexService
        from app.services.dataset_loader import DatasetLoaderService

        if not isinstance(model, ModelBundle):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="This model was saved without its preprocessing; retrain it to score files")
        cap = int(settings.ml.max_scoring_rows)
        try:
            loaded = await DatasetLoaderService(self._session).load_dataset(dataset_id, owner_id=owner_id, max_rows=cap)
        except HTTPException:
            raise
        except Exception as e:
            if "too large" in str(e).lower():
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    detail=f"Batch scoring covers every row and is limited to {cap:,} rows; "
                                           "split the file or raise ML_MAX_SCORING_ROWS") from e
            raise
        df = loaded.df

        def _score() -> dict[str, Any]:
            warnings = model.check_input(df)
            preds = model.predict(df)
            proba = model.predict_proba(df)
            return {"warnings": warnings, "preds": preds, "proba": proba,
                    "drift": model.drift_report(df), "realised": _realised_performance(model, df, preds)}

        scored = await asyncio.to_thread(_score)
        preds, proba = scored["preds"], scored["proba"]

        out = df.copy()
        pred_col = f"predicted_{model.target_column}"
        out[pred_col] = preds
        if proba is not None and model.classes is not None:
            arr = np.asarray(proba, dtype=float)
            for i, cls in enumerate(model.classes):
                out[f"probability_{cls}"] = np.round(arr[:, i], 6)

        version = model_version
        drift = scored["drift"]
        shift = prediction_shift(getattr(model, "prediction_reference", None), preds, proba, model.classes)
        if model.classes is not None:
            counts = pd.Series([str(p) for p in preds]).value_counts()
            prediction_summary: dict[str, Any] = {"counts": {str(k): int(v) for k, v in counts.items()}}
        else:
            vals = np.asarray(preds, dtype=float)
            prediction_summary = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}

        warnings = list(scored["warnings"])
        if drift is None:
            warnings.append("This model was trained before drift references were stored; retrain it to get a drift check.")
        elif drift["verdict"] != "stable":
            warnings.append(drift["message"])

        ref = ArtifactStore().write_table(
            name=f"predictions_{model_name}",
            df=out,
            dataset_id=dataset_id,
            dataset_version=loaded.version_hash,
            operator_name="ml_score",
            operator_params={"model_name": model_name, "model_version": version},
            operator_summary={"rows_scored": int(len(out)), "drift": (drift or {}).get("verdict")},
        )
        await ArtifactIndexService(self._session).index_many(owner_id=owner_id, refs=[ref])
        return to_jsonable({
            "artifact_id": str(ref.artifact_id),
            "model_name": model_name,
            "model_version": version,
            "dataset_id": str(dataset_id),
            "rows_scored": int(len(out)),
            "prediction_column": pred_col,
            "prediction_summary": prediction_summary,
            "decision_threshold": getattr(model, "decision_threshold", None),
            "calibrated": bool(getattr(model, "calibrator", None)),
            "drift": drift,
            "prediction_shift": shift,
            "realised_performance": scored["realised"],
            "warnings": warnings,
        })
