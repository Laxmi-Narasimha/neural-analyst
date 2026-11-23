from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.compute.chart_helpers import importance_bar_chart
from app.compute.operators.base import OperatorContext, OperatorResult
from app.core.config import settings
from app.ml.auto_ml_engine import AdvancedAutoMLEngine, MLTask, OptimizationStrategy
from app.ml.model_registry import get_model_registry
from app.services.ml_task_inference import infer_target_candidates


def _map_task(task: str | None) -> MLTask | None:
    if not task:
        return None
    t = str(task).strip().lower()
    if t in {"classification", "binary_classification", "multiclass_classification"}:
        return MLTask.BINARY_CLASSIFICATION
    if t == "regression":
        return MLTask.REGRESSION
    return None


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
        task = _map_task(params.get("task"))

        trials = int(settings.ml.baseline_optimization_trials) if optimize else 0
        strategy = OptimizationStrategy.BAYESIAN if optimize and trials > 0 else OptimizationStrategy.NONE

        engine = AdvancedAutoMLEngine(
            optimization_strategy=strategy,
            optimization_trials=max(5, trials),
            optimization_timeout=min(600, int(settings.ml.automl_time_budget)),
            verbose=False,
        )

        results = engine.auto_train(
            ctx.df,
            target_column=target,
            task=task,
            optimize=optimize and trials > 0,
            max_models=max_models,
        )

        best_name, best_model, best_result = engine.get_best_model()
        registered_name = f"ds_{str(ctx.dataset_id)[:8]}_{target}"
        registry = get_model_registry()
        version = registry.register_model(
            name=registered_name,
            model=best_model,
            metrics=best_result.test_metrics,
            params={
                "target_column": target,
                "task": best_result.task.value,
                "best_model_name": best_name,
                "best_params": best_result.best_params,
            },
            description=f"Chat baseline training on {target}",
            tags={"dataset_id": str(ctx.dataset_id), "mode": "chat_baseline"},
        )

        model_rows = []
        for r in results:
            try:
                d = r.to_dict()
            except Exception:
                continue
            model_rows.append(
                {
                    "model_name": d.get("model_name"),
                    "display_name": d.get("display_name"),
                    "task": d.get("task"),
                    "cv_mean": (d.get("metrics") or {}).get("cv_mean"),
                    "test_accuracy": (d.get("metrics") or {}).get("test", {}).get("accuracy"),
                    "test_r2": (d.get("metrics") or {}).get("test", {}).get("r2"),
                    "test_rmse": (d.get("metrics") or {}).get("test", {}).get("rmse"),
                    "training_sec": (d.get("timing") or {}).get("training_sec"),
                }
            )

        importance_rows = []
        for name, score in sorted(best_result.feature_importance.items(), key=lambda x: -x[1])[:25]:
            importance_rows.append({"feature": name, "importance": round(float(score), 6)})

        tables = {"model_comparison": pd.DataFrame(model_rows)}
        if importance_rows:
            tables["feature_importance"] = pd.DataFrame(importance_rows)

        charts: dict[str, Any] = {}
        shap_importance: dict[str, float] = {}
        try:
            from app.ml.explainability import get_explainability_engine

            feature_cols = [c for c in ctx.df.columns if c != target and pd.api.types.is_numeric_dtype(ctx.df[c])]
            if feature_cols:
                X_exp = ctx.df[feature_cols].head(300).fillna(0)
                y_exp = ctx.df[target].head(300)
                explainer = get_explainability_engine()
                shap_importance = explainer.get_feature_importance(best_model, X_exp, y_exp)
                if shap_importance:
                    tables["shap_importance"] = pd.DataFrame(
                        [{"feature": k, "importance": round(float(v), 6)} for k, v in shap_importance.items()]
                    ).sort_values("importance", ascending=False)
                    charts["shap_importance"] = importance_bar_chart(
                        shap_importance,
                        title="SHAP / feature importance",
                    )
        except Exception:
            shap_importance = dict(sorted(best_result.feature_importance.items(), key=lambda x: -x[1])[:15])

        if not charts and best_result.feature_importance:
            charts["feature_importance"] = importance_bar_chart(
                dict(sorted(best_result.feature_importance.items(), key=lambda x: -x[1])[:15]),
                title="Feature importance",
            )

        return OperatorResult(
            tables=tables,
            metrics={
                "best_model": best_name,
                "registered_model": registered_name,
                "registered_version": int(version.version),
                "target_column": target,
                "task": best_result.task.value,
            },
            charts=charts,
            summary={
                "best_model": best_name,
                "registered_model": registered_name,
                "registered_version": int(version.version),
                "target_column": target,
                "task": best_result.task.value,
                "test_metrics": best_result.test_metrics,
                "cv_mean": best_result.cv_mean,
                "cv_std": best_result.cv_std,
                "best_params": best_result.best_params,
                "feature_importance": dict(
                    sorted(best_result.feature_importance.items(), key=lambda x: -x[1])[:15]
                ),
                "shap_importance": shap_importance,
                "models": model_rows,
                "sample_rows": int(ctx.df.shape[0]),
            },
        )


P0_ML_OPERATORS = [
    MLTaskInferenceOperator(),
    MLTrainBaselineOperator(),
]