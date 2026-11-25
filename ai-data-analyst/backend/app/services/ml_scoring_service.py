"""Score a whole new dataset with a saved model, and say whether the result can be trusted.

Every row is scored (no sampling); a file larger than ML_MAX_SCORING_ROWS is refused with a clear
message. The checks themselves live in app.ml.scoring and are shared with the chat ``ml_score``
operator. The answer carries:
- a table artifact: the input rows plus the prediction (and class probabilities),
- a drift report: each input the model uses compared with its training data (PSI),
- how the mix of predictions compares with the held-out predictions at training time,
- when the file already holds the outcome, how the model actually did on it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.serialization import to_jsonable


def too_large_message(cap: int) -> str:
    return (f"Batch scoring covers every row and is limited to {cap:,} rows; "
            "split the file or raise ML_MAX_SCORING_ROWS")


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
        model_tags: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        from app.compute.artifacts import ArtifactStore
        from app.ml.scoring import score_frame, training_overlap_warning
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
                                    detail=too_large_message(cap)) from e
            raise
        try:
            scored = await asyncio.to_thread(score_frame, model, loaded.df)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
        overlap = training_overlap_warning(model_tags, dataset_id=dataset_id, dataset_version=loaded.version_hash,
                                           realised=scored.realised)
        if overlap:
            scored.warnings.append(overlap)

        ref = ArtifactStore().write_table(
            name=f"predictions_{model_name}",
            df=scored.table,
            dataset_id=dataset_id,
            dataset_version=loaded.version_hash,
            operator_name="ml_score",
            operator_params={"model_name": model_name, "model_version": model_version},
            operator_summary={"rows_scored": int(len(scored.table)), "drift": (scored.drift or {}).get("verdict")},
        )
        await ArtifactIndexService(self._session).index_many(owner_id=owner_id, refs=[ref])
        return to_jsonable({
            "artifact_id": str(ref.artifact_id),
            "model_name": model_name,
            "model_version": model_version,
            "dataset_id": str(dataset_id),
            **scored.summary(),
            "decision_threshold": getattr(model, "decision_threshold", None),
            "calibrated": bool(getattr(model, "calibrator", None)),
        })
