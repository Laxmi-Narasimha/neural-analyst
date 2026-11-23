"""One training run, shared by the chat operator, the API and background jobs.

It chooses the evaluation split, trains through the leakage-safe engine,
registers a ModelBundle (fitted pipeline + schema + evaluation record) and
returns a JSON-safe summary that callers render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from app.ml.auto_ml_engine import (
    AdvancedAutoMLEngine,
    MLTask,
    ModelStatus,
    ModelTrainingResult,
    OptimizationStrategy,
    _round_or_none,
)
from app.ml.model_registry import _jsonable, get_model_registry
from app.services.ml_task_inference import resolve_split_time_column


def map_task(task: Optional[str]) -> Optional[MLTask]:
    """Map a user/API task string to an engine task (None = auto-detect)."""
    if not task:
        return None
    t = str(task).strip().lower()
    if t in {"classification", "binary_classification", "binary", "multiclass_classification", "multiclass"}:
        # The engine settles binary vs multiclass from the number of classes.
        return MLTask.BINARY_CLASSIFICATION
    if t == "regression":
        return MLTask.REGRESSION
    return None


@dataclass
class TrainingRun:
    engine: AdvancedAutoMLEngine
    results: list[ModelTrainingResult]
    best: ModelTrainingResult
    registered_name: str
    registered_version: int
    report: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def model_rows(self) -> list[dict[str, Any]]:
        rows = []
        baseline = self.engine._baseline[1] if self.engine._baseline else None
        for r in ([baseline] if baseline else []) + list(self.results):
            test = r.test_metrics or {}
            rows.append(
                {
                    "model_name": r.model_name,
                    "display_name": r.display_name,
                    "is_baseline": bool(r.is_baseline),
                    "status": r.status.value,
                    "cv_scoring": r.cv_scoring,
                    "cv_mean": _round_or_none(r.cv_mean),
                    "cv_std": _round_or_none(r.cv_std),
                    "test_accuracy": _round_or_none(test.get("accuracy")),
                    "test_roc_auc": _round_or_none(test.get("roc_auc", test.get("roc_auc_ovr"))),
                    "test_f1": _round_or_none(test.get("f1")),
                    "test_r2": _round_or_none(test.get("r2")),
                    "test_rmse": _round_or_none(test.get("rmse")),
                    "test_mae": _round_or_none(test.get("mae")),
                    "training_sec": round(float(r.training_time_sec or 0.0), 2),
                    "error": r.errors[0] if r.errors else None,
                }
            )
        return rows

    def summary(self) -> dict[str, Any]:
        best = self.best
        return _jsonable(
            {
                "best_model": best.model_name,
                "best_model_display": best.display_name,
                "registered_model": self.registered_name,
                "registered_version": int(self.registered_version),
                "target_column": self.report.get("target_column"),
                "task": best.task.value,
                "classes": self.report.get("classes"),
                "test_metrics": {k: _round_or_none(v) for k, v in best.test_metrics.items()},
                "cv_mean": _round_or_none(best.cv_mean),
                "cv_std": _round_or_none(best.cv_std),
                "cv_scoring": best.cv_scoring,
                "best_params": best.best_params,
                "feature_importance": dict(list(best.feature_importance.items())[:15]),
                "importance_method": self.report.get("importance_method"),
                "baseline": self.report.get("baseline"),
                "beats_baseline": self.report.get("beats_baseline"),
                "split": self.report.get("split"),
                "cross_validation": self.report.get("cross_validation"),
                "excluded_leaky_columns": self.report.get("excluded_leaky_columns"),
                "preprocessing": self.report.get("preprocessing"),
                "rows_used": self.report.get("rows_used"),
                "rows_dropped_missing_target": self.report.get("rows_dropped_missing_target"),
                "warnings": list(dict.fromkeys(self.notes + list(self.report.get("warnings") or []))),
                "models": self.model_rows(),
            }
        )


def run_training(
    df: pd.DataFrame,
    *,
    target_column: str,
    registered_name: str,
    task: Optional[str] = None,
    optimize: bool = False,
    max_models: int = 4,
    optimization_strategy: Optional[OptimizationStrategy] = None,
    optimization_trials: int = 10,
    optimization_timeout: int = 300,
    exclude_columns: Optional[list[str]] = None,
    time_column: Optional[str] = None,
    split: Optional[str] = None,
    description: str = "",
    tags: Optional[dict[str, str]] = None,
    extra_params: Optional[dict[str, Any]] = None,
    random_state: int = 42,
) -> TrainingRun:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    chosen_time, notes = resolve_split_time_column(
        df, target_column=target_column, time_column=time_column, split=split
    )
    strategy = optimization_strategy
    if strategy is None:
        strategy = OptimizationStrategy.BAYESIAN if optimize else OptimizationStrategy.NONE
    engine = AdvancedAutoMLEngine(
        optimization_strategy=strategy,
        optimization_trials=max(1, int(optimization_trials)),
        optimization_timeout=int(optimization_timeout),
        random_state=random_state,
        verbose=False,
    )
    results = engine.auto_train(
        df,
        target_column=target_column,
        task=map_task(task),
        optimize=bool(optimize),
        max_models=int(max_models),
        exclude_columns=exclude_columns,
        time_column=chosen_time,
    )
    best = next(r for r in results if r.status == ModelStatus.COMPLETED)
    bundle = engine.build_bundle(best.model_name)

    metrics = {k: v for k, v in best.test_metrics.items()}
    if best.cv_mean is not None:
        metrics[f"cv_{best.cv_scoring}"] = best.cv_mean
    registry = get_model_registry()
    version = registry.register_model(
        name=registered_name,
        model=bundle,
        metrics=metrics,
        params={
            "target_column": target_column,
            "task": best.task.value,
            "best_model_name": best.model_name,
            "best_params": best.best_params,
            "input_columns": bundle.input_columns,
            "classes": bundle.classes,
            "training_report": engine.training_report,
            **(extra_params or {}),
        },
        description=description,
        tags=tags or {},
    )
    return TrainingRun(
        engine=engine,
        results=results,
        best=best,
        registered_name=registered_name,
        registered_version=int(version.version),
        report=engine.training_report,
        notes=notes,
    )
