# AI Enterprise Data Analyst - Causal Inference Agent
# Specialized agent for causal analysis and A/B testing

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
from app.ml.causal_inference import (
    CausalInferenceEngine,
    CausalMethod,
    get_causal_inference_engine,
)
from app.ml.ab_testing import (
    ABTestingEngine,
    TestType,
    get_ab_testing_engine,
)
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger

logger = get_logger(__name__)


class CausalInferenceAgent(BaseAgent[dict[str, Any]]):
    """
    Causal Inference Agent for treatment effect estimation and experiments.
    
    Capabilities:
    - Propensity Score Matching
    - Inverse Probability Weighting
    - Doubly Robust estimation
    - Difference-in-Differences
    - A/B test analysis (Frequentist, Bayesian, Sequential)
    - Experiment design and sample size calculation
    """
    
    name: str = "CausalInferenceAgent"
    description: str = "Estimate causal effects and analyze experiments"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.causal_engine = get_causal_inference_engine()
        self.ab_engine = get_ab_testing_engine()
    
    def _register_tools(self) -> None:
        """Register causal inference tools."""
        
        self.register_tool(AgentTool(
            name="estimate_treatment_effect",
            description="Estimate causal treatment effect using propensity score methods",
            function=self._estimate_effect,
            parameters={
                "data": {"type": "object"},
                "treatment_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "covariates": {"type": "array"},
                "method": {"type": "string", "default": "matching"}
            },
            required_params=["data", "treatment_col", "outcome_col", "covariates"]
        ))
        
        self.register_tool(AgentTool(
            name="analyze_ab_test",
            description="Analyze A/B experiment results",
            function=self._analyze_ab_test,
            parameters={
                "data": {"type": "object"},
                "variant_col": {"type": "string"},
                "metric_col": {"type": "string"},
                "test_type": {"type": "string", "default": "frequentist"},
                "is_binary": {"type": "boolean", "default": False}
            },
            required_params=["data", "variant_col", "metric_col"]
        ))
        
        self.register_tool(AgentTool(
            name="calculate_sample_size",
            description="Calculate required sample size for experiment",
            function=self._calculate_sample_size,
            parameters={
                "baseline": {"type": "number"},
                "mde": {"type": "number"},
                "is_binary": {"type": "boolean", "default": True},
                "power": {"type": "number", "default": 0.8}
            },
            required_params=["baseline", "mde"]
        ))
        
        self.register_tool(AgentTool(
            name="diff_in_diff",
            description="Estimate effect using Difference-in-Differences",
            function=self._diff_in_diff,
            parameters={
                "data": {"type": "object"},
                "treatment_col": {"type": "string"},
                "time_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "pre_period": {"type": "any"},
                "post_period": {"type": "any"}
            },
            required_params=["data", "treatment_col", "time_col", "outcome_col"]
        ))
        
        self.register_tool(AgentTool(
            name="interpret_results",
            description="Get AI interpretation of causal analysis results",
            function=self._interpret_results,
            parameters={
                "results": {"type": "object"}
            },
            required_params=["results"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute causal analysis."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="""You are a causal inference expert. Help design experiments and 
                    interpret treatment effects. Consider confounders and assumptions."""
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "methods_available": ["PSM", "IPW", "DiD", "Doubly Robust"]
        }
    
    async def _estimate_effect(
        self,
        data: dict,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str],
        method: str = "matching"
    ) -> dict[str, Any]:
        """Estimate treatment effect."""
        df = pd.DataFrame(data)
        
        if method == "matching":
            result = self.causal_engine.estimate_ate_matching(
                df, treatment_col, outcome_col, covariates
            )
        elif method == "ipw":
            result = self.causal_engine.estimate_ate_ipw(
                df, treatment_col, outcome_col, covariates
            )
        elif method == "doubly_robust":
            result = self.causal_engine.estimate_ate_doubly_robust(
                df, treatment_col, outcome_col, covariates
            )
        else:
            result = self.causal_engine.estimate_ate_matching(
                df, treatment_col, outcome_col, covariates
            )
        
        return result.to_dict()
    
    async def _analyze_ab_test(
        self,
        data: dict,
        variant_col: str,
        metric_col: str,
        test_type: str = "frequentist",
        is_binary: bool = False
    ) -> dict[str, Any]:
        """Analyze A/B test."""
        df = pd.DataFrame(data)
        test_type_enum = TestType(test_type)
        
        result = self.ab_engine.analyze_experiment(
            df=df,
            variant_col=variant_col,
            metric_col=metric_col,
            test_type=test_type_enum,
            is_binary=is_binary
        )
        
        return result.to_dict()
    
    async def _calculate_sample_size(
        self,
        baseline: float,
        mde: float,
        is_binary: bool = True,
        power: float = 0.8
    ) -> dict[str, int]:
        """Calculate experiment sample size."""
        return self.ab_engine.calculate_sample_size(
            baseline=baseline,
            mde=mde,
            is_binary=is_binary,
            power=power
        )
    
    async def _diff_in_diff(
        self,
        data: dict,
        treatment_col: str,
        time_col: str,
        outcome_col: str,
        pre_period: Any = None,
        post_period: Any = None
    ) -> dict[str, Any]:
        """Run Difference-in-Differences analysis."""
        df = pd.DataFrame(data)
        
        # Auto-detect periods if not provided
        if pre_period is None or post_period is None:
            periods = sorted(df[time_col].unique())
            if len(periods) >= 2:
                pre_period = periods[0]
                post_period = periods[-1]
            else:
                return {"error": "Need at least 2 time periods for DiD"}
        
        result = self.causal_engine.estimate_diff_in_diff(
            df, treatment_col, time_col, outcome_col, pre_period, post_period
        )
        
        return result.to_dict()
    
    async def _interpret_results(
        self,
        results: dict
    ) -> dict[str, Any]:
        """Get AI interpretation of causal results."""
        prompt = f"""Interpret this causal analysis result for a business stakeholder:

Effect Estimate: {results.get('estimate', 'N/A')}
P-value: {results.get('p_value', 'N/A')}
Confidence Interval: {results.get('confidence_interval', 'N/A')}
Is Significant: {results.get('is_significant', 'N/A')}
Method: {results.get('method', 'N/A')}

Consider:
1. Practical significance vs statistical significance
2. Key assumptions and limitations
3. Actionable recommendations
4. Confidence level for decision-making"""

        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a senior data scientist explaining causal analysis to stakeholders."
                ),
                LLMMessage(role="user", content=prompt)
            ]
        )
        
        return {
            "interpretation": response.content,
            "raw_results": results
        }


# Factory function
def get_causal_inference_agent() -> CausalInferenceAgent:
    """Get causal inference agent instance."""
    return CausalInferenceAgent()
