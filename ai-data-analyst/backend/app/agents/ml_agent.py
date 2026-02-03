# AI Enterprise Data Analyst - ML Agent
# Agent for machine learning with AutoML capabilities

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

import pandas as pd
import numpy as np

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
)
from app.ml.ml_engine import MLEngine, MLTask, get_ml_engine
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


class MLAgent(BaseAgent[dict[str, Any]]):
    """
    Machine Learning Agent with AutoML capabilities.
    
    Capabilities:
    - Automatic problem type detection
    - Feature engineering
    - Model selection and training
    - Hyperparameter optimization
    - Model comparison and evaluation
    - Predictions
    """
    
    name: str = "MLAgent"
    description: str = "Machine learning with AutoML for classification, regression, and clustering"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.engine = get_ml_engine()
    
    def _register_tools(self) -> None:
        """Register ML tools."""
        
        self.register_tool(AgentTool(
            name="auto_train",
            description="Automatically train and compare multiple ML models",
            function=self._auto_train,
            parameters={
                "data": {"type": "object", "description": "Training data as dict"},
                "target_column": {"type": "string"},
                "task": {"type": "string", "enum": ["classification", "regression"]},
                "algorithms": {"type": "array", "items": {"type": "string"}},
                "optimize": {"type": "boolean", "default": True}
            },
            required_params=["data", "target_column"]
        ))
        
        self.register_tool(AgentTool(
            name="predict",
            description="Make predictions using trained model",
            function=self._predict,
            parameters={
                "data": {"type": "object"},
                "model_name": {"type": "string"}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="explain_model",
            description="Explain model predictions and feature importance",
            function=self._explain_model,
            parameters={
                "model_name": {"type": "string"}
            },
            required_params=["model_name"]
        ))
        
        self.register_tool(AgentTool(
            name="recommend_algorithm",
            description="Recommend best algorithm for the data",
            function=self._recommend_algorithm,
            parameters={
                "data_info": {"type": "object"},
                "task": {"type": "string"}
            },
            required_params=["data_info", "task"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute ML analysis based on context."""
        # Use LLM to understand the ML task
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are an ML expert. Analyze the request and suggest the best approach."
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "available_algorithms": self.engine.compare_models([])
        }
    
    async def _auto_train(
        self,
        data: dict,
        target_column: str,
        task: str = "classification",
        algorithms: list[str] = None,
        optimize: bool = True
    ) -> dict[str, Any]:
        """Auto-train multiple models and compare."""
        df = pd.DataFrame(data)
        ml_task = MLTask(task)
        
        results = self.engine.train(
            df=df,
            target_column=target_column,
            task=ml_task,
            algorithms=algorithms,
            optimize_hyperparams=optimize
        )
        
        comparison = self.engine.compare_models(results)
        
        return {
            "status": "success",
            "results": [r.to_dict() for r in results],
            "comparison": comparison,
            "best_model": results[0].algorithm if results else None,
            "best_score": results[0].cv_mean if results else 0
        }
    
    async def _predict(
        self,
        data: dict,
        model_name: str = None
    ) -> dict[str, Any]:
        """Make predictions."""
        df = pd.DataFrame(data)
        
        if model_name is None:
            model_name, _ = self.engine.get_best_model()
        
        predictions = self.engine.predict(model_name, df)
        
        return {
            "status": "success",
            "model_used": model_name,
            "predictions": predictions.tolist()
        }
    
    async def _explain_model(
        self,
        model_name: str
    ) -> dict[str, Any]:
        """Explain model and feature importance."""
        if model_name not in self.engine._trained_models:
            return {"error": f"Model {model_name} not found"}
        
        model, _ = self.engine._trained_models[model_name]
        
        explanation = {
            "model_type": type(model).__name__,
            "parameters": model.get_params()
        }
        
        if hasattr(model, 'feature_importances_'):
            explanation["has_feature_importance"] = True
        
        return explanation
    
    async def _recommend_algorithm(
        self,
        data_info: dict,
        task: str
    ) -> dict[str, Any]:
        """Recommend algorithms based on data characteristics."""
        n_samples = data_info.get("n_samples", 0)
        n_features = data_info.get("n_features", 0)
        n_classes = data_info.get("n_classes", 2)
        has_categoricals = data_info.get("has_categoricals", False)
        
        recommendations = []
        
        if task == "classification":
            if n_samples < 1000:
                recommendations.append({
                    "algorithm": "logistic_regression",
                    "reason": "Good for small datasets, fast training"
                })
            
            if has_categoricals:
                recommendations.append({
                    "algorithm": "lightgbm_classifier",
                    "reason": "Handles categoricals natively, very fast"
                })
            
            recommendations.append({
                "algorithm": "xgboost_classifier",
                "reason": "Best overall performance for most datasets"
            })
            
            recommendations.append({
                "algorithm": "random_forest_classifier",
                "reason": "Robust, handles missing values, good baseline"
            })
        
        else:  # regression
            recommendations.append({
                "algorithm": "xgboost_regressor",
                "reason": "Best for most regression tasks"
            })
            
            if n_features > n_samples // 2:
                recommendations.append({
                    "algorithm": "ridge_regression",
                    "reason": "Good when features > samples"
                })
        
        return {
            "task": task,
            "recommendations": recommendations[:5],
            "data_summary": data_info
        }


# Factory function
def get_ml_agent() -> MLAgent:
    """Get ML agent instance."""
    return MLAgent()
