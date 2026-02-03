# AI Enterprise Data Analyst - Data Quality Agent
# Specialized agent integrating data quality engine with ReAct pattern

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

import pandas as pd

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
)
from app.ml.data_quality import (
    DataQualityEngine,
    DatasetQualityReport,
    get_data_quality_engine,
)
from app.ml.imputation import SmartImputationEngine, get_smart_imputation_engine
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger

logger = get_logger(__name__)


class DataQualityAgent(BaseAgent[dict[str, Any]]):
    """
    Data Quality Agent for comprehensive data assessment and remediation.
    
    Capabilities:
    - Full data quality profiling
    - Missing value analysis and imputation
    - Outlier detection and treatment
    - Validation rule checking
    - Quality improvement recommendations
    """
    
    name: str = "DataQualityAgent"
    description: str = "Assess and improve data quality for analytical reliability"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.quality_engine = get_data_quality_engine()
        self.imputation_engine = get_smart_imputation_engine()
    
    def _register_tools(self) -> None:
        """Register data quality tools."""
        
        self.register_tool(AgentTool(
            name="profile_data_quality",
            description="Generate comprehensive data quality report with scores and issue identification",
            function=self._profile_quality,
            parameters={
                "data": {"type": "object", "description": "DataFrame as dict"},
                "dataset_name": {"type": "string", "description": "Name for the dataset"}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="analyze_missing_values",
            description="Analyze missing data patterns (MCAR/MAR/MNAR) and suggest remediation",
            function=self._analyze_missing,
            parameters={
                "data": {"type": "object"}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="detect_outliers",
            description="Detect outliers using multiple methods",
            function=self._detect_outliers,
            parameters={
                "data": {"type": "object"},
                "columns": {"type": "array"},
                "method": {"type": "string", "default": "iqr"}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="impute_missing",
            description="Intelligently impute missing values using appropriate methods",
            function=self._impute_missing,
            parameters={
                "data": {"type": "object"},
                "strategy": {"type": "string", "default": "auto"}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="validate_data",
            description="Validate data against custom rules",
            function=self._validate_data,
            parameters={
                "data": {"type": "object"},
                "rules": {"type": "array"}
            },
            required_params=["data", "rules"]
        ))
        
        self.register_tool(AgentTool(
            name="get_cleaning_recommendations",
            description="Get AI-powered recommendations for data cleaning",
            function=self._get_recommendations,
            parameters={
                "quality_report": {"type": "object"}
            },
            required_params=["quality_report"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute data quality assessment."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="""You are a data quality expert. Analyze data quality issues and 
                    provide actionable recommendations for improving data reliability."""
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "recommendations": ["Profile data quality", "Check for missing patterns"]
        }
    
    async def _profile_quality(
        self,
        data: dict,
        dataset_name: str = "dataset"
    ) -> dict[str, Any]:
        """Generate comprehensive data quality report."""
        df = pd.DataFrame(data)
        report = self.quality_engine.analyze(df, dataset_name)
        
        return report.to_dict()
    
    async def _analyze_missing(
        self,
        data: dict
    ) -> dict[str, Any]:
        """Analyze missing data patterns."""
        df = pd.DataFrame(data)
        analysis = self.quality_engine.missing_analyzer.analyze(df)
        
        return analysis
    
    async def _detect_outliers(
        self,
        data: dict,
        columns: list[str] = None,
        method: str = "iqr"
    ) -> dict[str, Any]:
        """Detect outliers in data."""
        from app.ml.data_quality import OutlierMethod
        
        df = pd.DataFrame(data)
        method_enum = OutlierMethod(method)
        
        result = self.quality_engine.outlier_detector.detect(
            df, columns, method_enum
        )
        
        return result
    
    async def _impute_missing(
        self,
        data: dict,
        strategy: str = "auto"
    ) -> dict[str, Any]:
        """Impute missing values."""
        df = pd.DataFrame(data)
        
        imputed_df = self.imputation_engine.analyze_and_impute(df, strategy)
        results = self.imputation_engine.get_results()
        
        return {
            "imputed_data": imputed_df.to_dict(orient="records"),
            "imputation_summary": results
        }
    
    async def _validate_data(
        self,
        data: dict,
        rules: list[dict]
    ) -> dict[str, Any]:
        """Validate data against rules."""
        from app.ml.data_quality import ValidationRule, ValidationType
        
        df = pd.DataFrame(data)
        validator = self.quality_engine.validator
        
        # Add rules
        for rule in rules:
            val_type = ValidationType(rule.get("type", "range"))
            validator.add_rule(ValidationRule(
                name=rule.get("name", "custom_rule"),
                validation_type=val_type,
                column=rule["column"],
                params=rule.get("params", {}),
                error_message=rule.get("message", "Validation failed")
            ))
        
        # Validate
        results = validator.validate(df)
        
        return results
    
    async def _get_recommendations(
        self,
        quality_report: dict
    ) -> dict[str, Any]:
        """Get AI-powered cleaning recommendations."""
        # Use LLM to generate contextual recommendations
        prompt = f"""Based on this data quality report, provide specific actionable recommendations:

Quality Score: {quality_report.get('overall_metrics', {}).get('quality_score', 'N/A')}
Critical Issues: {len(quality_report.get('critical_issues', []))}
Warnings: {len(quality_report.get('warnings', []))}

Recommendations from analysis:
{quality_report.get('recommendations', [])}

Provide 3-5 prioritized, specific recommendations with estimated impact."""

        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a senior data engineer specializing in data quality."
                ),
                LLMMessage(role="user", content=prompt)
            ]
        )
        
        return {
            "ai_recommendations": response.content,
            "automated_recommendations": quality_report.get("recommendations", [])
        }


# Factory function
def get_data_quality_agent() -> DataQualityAgent:
    """Get data quality agent instance."""
    return DataQualityAgent()
