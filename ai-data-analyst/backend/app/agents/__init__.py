# AI Enterprise Data Analyst - Agents Package
"""AI agent implementations for autonomous data analysis."""

from __future__ import annotations

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentState,
    AgentAction,
    AgentTool,
    AgentMemory,
    ActionStatus,
)
from app.agents.orchestrator import OrchestratorAgent, get_orchestrator


def _lazy_call(module_name: str, fn_name: str, *args, **kwargs):
    module = __import__(module_name, fromlist=[fn_name])
    fn = getattr(module, fn_name)
    return fn(*args, **kwargs)


# Class aliases are intentionally not eagerly imported to avoid importing heavy optional
# ML/NLP stacks during API startup in lightweight environments.
StatisticalAgent = None
MLAgent = None
EDAAgent = None
VisualizationAgent = None
NLPAgent = None
TimeSeriesAgent = None
NL2SQLAgent = None
DataQualityAgent = None
CausalInferenceAgent = None


def get_statistical_agent(*args, **kwargs):
    return _lazy_call("app.agents.statistical_agent", "get_statistical_agent", *args, **kwargs)


def get_ml_agent(*args, **kwargs):
    return _lazy_call("app.agents.ml_agent", "get_ml_agent", *args, **kwargs)


def get_eda_agent(*args, **kwargs):
    return _lazy_call("app.agents.eda_agent", "get_eda_agent", *args, **kwargs)


def get_visualization_agent(*args, **kwargs):
    return _lazy_call("app.agents.visualization_agent", "get_visualization_agent", *args, **kwargs)


def get_nlp_agent(*args, **kwargs):
    return _lazy_call("app.agents.nlp_agent", "get_nlp_agent", *args, **kwargs)


def get_timeseries_agent(*args, **kwargs):
    return _lazy_call("app.agents.timeseries_agent", "get_timeseries_agent", *args, **kwargs)


def get_nl2sql_agent(*args, **kwargs):
    return _lazy_call("app.agents.nl2sql_agent", "get_nl2sql_agent", *args, **kwargs)


def get_data_quality_agent(*args, **kwargs):
    return _lazy_call("app.agents.data_quality_agent", "get_data_quality_agent", *args, **kwargs)


def get_causal_inference_agent(*args, **kwargs):
    return _lazy_call("app.agents.causal_agent", "get_causal_inference_agent", *args, **kwargs)


__all__ = [
    # Base
    "BaseAgent",
    "AgentRole",
    "AgentContext",
    "AgentState",
    "AgentAction",
    "AgentTool",
    "AgentMemory",
    "ActionStatus",
    # Orchestrator
    "OrchestratorAgent",
    "get_orchestrator",
    # Specialized Agents
    "StatisticalAgent",
    "get_statistical_agent",
    "MLAgent",
    "get_ml_agent",
    "EDAAgent",
    "get_eda_agent",
    "VisualizationAgent",
    "get_visualization_agent",
    "NLPAgent",
    "get_nlp_agent",
    "TimeSeriesAgent",
    "get_timeseries_agent",
    "NL2SQLAgent",
    "get_nl2sql_agent",
    "DataQualityAgent",
    "get_data_quality_agent",
    "CausalInferenceAgent",
    "get_causal_inference_agent",
]
