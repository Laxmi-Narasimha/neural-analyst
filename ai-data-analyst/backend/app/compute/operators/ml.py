from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.compute.chart_helpers import importance_bar_chart
from app.compute.operators.base import OperatorContext, OperatorResult
from app.core.config import settings
from app.ml.training_run import run_training
from app.services.ml_task_inference import infer_target_candidates


@dataclass(frozen=True)
class MLTaskInferenceOperator:
    name: str = "ml_task_inference"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        preferred = params.get("preferred_target") or params.get("target_column")
        candidates, selected_target, selected_task, warnings = infer_target_candidates(
            ctx.df,
            preferred_target=str(preferred) if preferred else None,
        )

        rows = []
        for c in candidates:
            rows.append(
                {
                    "column": c.column,
                    "inferred_task": c.inferred_task,
                    "score": round(float(c.score), 4),
                    "non_null_rate": round(float(c.non_null_rate), 4),
                    "unique_ratio": round(float(c.unique_ratio), 4),
                    "reasons": "; ".join(c.reasons[:3]),
                    "leakage_count": len(c.leakage_warnings),
                }
            )

        table = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["column", "inferred_task", "score", "non_null_rate", "unique_ratio", "reasons", "leakage_count"]
        )

        return OperatorResult(
            tables={"target_candidates": table},
            metrics={
                "candidate_count": len(rows),
                "selected_target": selected_target,
                "selected_task": selected_task,
            },
            charts={},
            summary={
                "selected_target": selected_target,
                "selected_task": selected_task,
                "warnings": warnings,
                "candidates": [
                    {
                        "column": c.column,
                        "inferred_task": c.inferred_task,
                        "score": c.score,
                        "reasons": c.reasons,
                    }
                    for c in candidates[:8]
                ],
            },
        )


@dataclass(frozen=True)
class MLTrainBaselineOperator:
    name: str = "ml_train_baseline"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        target = str(params.get("target_column") or params.get("target") or "").strip()
        if not target:
            raise ValueError("target_column is required for ml_train_baseline")
        if target not in ctx.df.columns:
            raise ValueError(f"Target column '{target}' not found")

        optimize = bool(params.get("optimize", True))
        max_models = int(params.get("max_models") or settings.ml.baseline_max_models)
        trials = int(settings.ml.baseline_optimization_trials) if optimize else 0
        exclude = params.get("exclude_columns") or None

        run = run_training(
            ctx.df,
            target_column=target,
            registered_name=f"ds_{str(ctx.dataset_id)[:8]}_{target}",
            owner_id=ctx.owner_id,
            task=params.get("task"),
            optimize=optimize and trials > 0,
            max_models=max_models,
            optimization_trials=max(5, trials),
            optimization_timeout=min(600, int(settings.ml.automl_time_budget)),
            exclude_columns=list(exclude) if exclude else None,
            time_column=params.get("time_column") or None,
            split=params.get("split"),
            test_start=params.get("test_start") or None,
            description=f"Chat baseline training on {target}",
            tags={
                "dataset_id": str(ctx.dataset_id),
                "dataset_version": str(ctx.dataset_version or ""),
                "mode": "chat_baseline",
            },
            random_state=int(settings.ml.default_random_state),
        )

        summary = run.summary()
        summary["sample_rows"] = int(ctx.df.shape[0])
        tables = {"model_comparison": pd.DataFrame(summary["models"])}
        importance = summary.get("feature_importance") or {}
        charts: dict[str, Any] = {}
        if importance:
            tables["feature_importance"] = pd.DataFrame(
                [{"feature": k, "importance": float(v)} for k, v in importance.items()]
            )
            positive = {k: float(v) for k, v in importance.items() if float(v) > 0}
            if positive:
                charts["feature_importance"] = importance_bar_chart(
                    positive, title="Permutation importance on held-out rows"
                )

        return OperatorResult(
            tables=tables,
            metrics={
                "best_model": summary["best_model"],
                "registered_model": summary["registered_model"],
                "registered_version": summary["registered_version"],
                "target_column": target,
                "task": summary["task"],
                "cv_mean": summary.get("cv_mean"),
                "beats_baseline": summary.get("beats_baseline"),
            },
            charts=charts,
            summary=summary,
        )


@dataclass(frozen=True)
class MLScoreOperator:
    """Score every row in scope with one of the owner's saved models, with drift and trust checks.

    ``model_name`` and ``model_version`` are bound by code against the owner's registry before this
    runs (see app.services.ml_model_choice); ``model_fingerprint`` only keys the result cache so a
    retrained model never reuses old predictions.
    """

    name: str = "ml_score"

    def run(self, ctx: OperatorContext, params: dict[str, Any]) -> OperatorResult:
        from app.ml.model_registry import get_model_registry
        from app.ml.scoring import score_frame, training_overlap_warning

        if ctx.owner_id is None:
            raise ValueError("Scoring needs a signed-in owner")
        name = str(params.get("model_name") or "").strip()
        if not name:
            raise ValueError("model_name is required for ml_score")
        registry = get_model_registry(ctx.owner_id)
        info = next((m for m in registry.list_models() if m["name"] == name), None)
        if info is None:
            raise ValueError(f"Model '{name}' not found")
        version = params.get("model_version")
        version = int(version) if version not in (None, "") else int(info.get("production_version") or info["latest_version"])
        bundle = registry.load_model(name, version=version)
        scored = score_frame(bundle, ctx.df)
        overlap = training_overlap_warning(info.get("tags"), dataset_id=ctx.dataset_id,
                                           dataset_version=ctx.dataset_version, realised=scored.realised)
        if overlap:
            scored.warnings.append(overlap)

        tables: dict[str, pd.DataFrame] = {"predictions": scored.table}
        charts: dict[str, Any] = {}
        drift_rows = (scored.drift or {}).get("columns") or []
        if drift_rows:
            tables["drift"] = pd.DataFrame([
                {"column": r["column"], "level": r.get("level"), "psi": r.get("psi"), "note": r.get("note")}
                for r in drift_rows
            ])
            psi = {r["column"]: float(r["psi"]) for r in drift_rows if r.get("psi") is not None}
            if psi:
                charts["drift"] = importance_bar_chart(
                    psi, title="Drift from training data per input (PSI; 0.10 moderate, 0.25 major)")
        training_info = getattr(bundle, "training_info", None) or {}
        summary = {
            "model_name": name,
            "model_version": version,
            "target_column": bundle.target_column,
            "task": bundle.task,
            "classes": list(bundle.classes) if bundle.classes is not None else None,
            "decision_threshold": getattr(bundle, "decision_threshold", None),
            "calibrated": bool(getattr(bundle, "calibrator", None)),
            "held_out_metrics_at_training": training_info.get("model_test_metrics"),
            **scored.summary(),
        }
        return OperatorResult(
            tables=tables,
            metrics={
                "rows_scored": summary["rows_scored"],
                "model_name": name,
                "model_version": version,
                "drift_verdict": (scored.drift or {}).get("verdict"),
            },
            charts=charts,
            summary=summary,
        )


P0_ML_OPERATORS = [
    MLTaskInferenceOperator(),
    MLTrainBaselineOperator(),
    MLScoreOperator(),
]