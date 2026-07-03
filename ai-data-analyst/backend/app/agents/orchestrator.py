# AI Enterprise Data Analyst - Orchestrator Agent
# Master agent that coordinates specialized agents for comprehensive data analysis

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
    AgentAction,
    ActionStatus,
)
from app.services.llm_service import get_llm_service, Message as LLMMessage, LLMResponse
from app.core.logging import get_logger, LogContext
from app.compute.executor import ComputeExecutor, ExecutionResult
from app.core.exceptions import AgentException
from app.services.database import db_manager

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent[dict[str, Any]]):
    """
    Master orchestrator agent using ReAct pattern.
    
    Coordinates specialized agents (EDA, Statistical, ML, etc.)
    to provide comprehensive data analysis capabilities.
    
    Implements:
    - ReAct (Reason + Act) loop
    - Dynamic tool selection
    - Memory management
    - Multi-step planning
    """
    
    name: str = "OrchestratorAgent"
    description: str = "Master agent orchestrating data analysis tasks"
    role: AgentRole = AgentRole.ORCHESTRATOR
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self._active_user_id: UUID | None = None
        self._active_dataset_id: UUID | None = None
        self._session = None
        self._compute: ComputeExecutor | None = None

    def _require_compute(self, dataset_id: str) -> tuple[UUID, ComputeExecutor, UUID]:
        if self._active_user_id is None:
            raise AgentException(message="Orchestrator tool calls require an authenticated user")

        try:
            ds = UUID(str(dataset_id))
        except Exception as e:
            raise AgentException(message=f"Invalid dataset_id: {dataset_id}") from e

        if self._active_dataset_id is not None and ds != self._active_dataset_id:
            raise AgentException(message="Tool dataset_id does not match the active dataset context")

        if self._compute is None:
            raise AgentException(message="Compute engine is not initialized")

        return ds, self._compute, self._active_user_id

    def _execution_result_to_dict(self, r: ExecutionResult) -> dict[str, Any]:
        return {
            "operator": r.operator_name,
            "summary": r.summary,
            "artifacts": [
                {
                    "artifact_id": str(a.artifact_id),
                    "artifact_type": getattr(a.artifact_type, "value", None),
                    "name": a.name,
                    "created_at": a.created_at,
                    "storage_path": a.storage_path,
                    "preview": a.preview,
                    "dataset_id": str(a.dataset_id) if a.dataset_id else None,
                    "dataset_version": a.dataset_version,
                    "operator_name": a.operator_name,
                    "operator_params": a.operator_params or {},
                }
                for a in r.artifacts
            ],
        }
    
    def _register_tools(self) -> None:
        """Register orchestrator tools."""
        
        # Data exploration tool
        self.register_tool(AgentTool(
            name="analyze_data_summary",
            description="Get a summary of the dataset including shape, types, and basic statistics",
            function=self._analyze_data_summary,
            parameters={
                "dataset_id": {"type": "string", "description": "Dataset UUID"},
                "columns": {"type": "array", "items": {"type": "string"}, "description": "Specific columns to analyze"}
            },
            required_params=["dataset_id"]
        ))
        
        # Statistical analysis tool
        self.register_tool(AgentTool(
            name="run_statistical_analysis",
            description="Perform statistical analysis like hypothesis tests, correlations, or distributions",
            function=self._run_statistical_analysis,
            parameters={
                "dataset_id": {"type": "string", "description": "Dataset UUID"},
                "analysis_type": {"type": "string", "enum": ["correlation", "hypothesis_test", "distribution", "anova"]},
                "columns": {"type": "array", "items": {"type": "string"}},
                "config": {"type": "object", "description": "Analysis configuration"}
            },
            required_params=["dataset_id", "analysis_type"]
        ))
        
        # ML model tool
        self.register_tool(AgentTool(
            name="train_ml_model",
            description="Train a machine learning model for prediction or classification",
            function=self._train_ml_model,
            parameters={
                "dataset_id": {"type": "string", "description": "Dataset UUID"},
                "model_type": {"type": "string", "enum": ["classification", "regression", "clustering"]},
                "target_column": {"type": "string", "description": "Target variable"},
                "features": {"type": "array", "items": {"type": "string"}},
                "algorithm": {"type": "string", "description": "Specific algorithm"}
            },
            required_params=["dataset_id", "model_type"]
        ))
        
        # Visualization tool
        self.register_tool(AgentTool(
            name="create_visualization",
            description="Create a data visualization like charts, plots, or dashboards",
            function=self._create_visualization,
            parameters={
                "dataset_id": {"type": "string", "description": "Dataset UUID"},
                "viz_type": {"type": "string", "enum": ["histogram", "scatter", "line", "bar", "heatmap", "box"]},
                "x_column": {"type": "string"},
                "y_column": {"type": "string"},
                "color_by": {"type": "string"}
            },
            required_params=["dataset_id", "viz_type"]
        ))
        
        # SQL generation tool
        self.register_tool(AgentTool(
            name="generate_sql",
            description="Generate SQL query from natural language question",
            function=self._generate_sql,
            parameters={
                "dataset_id": {"type": "string", "description": "Dataset UUID"},
                "question": {"type": "string", "description": "Natural language question"}
            },
            required_params=["dataset_id", "question"]
        ))
        
        # Data quality tool
        self.register_tool(AgentTool(
            name="assess_data_quality",
            description="Assess data quality including missing values, duplicates, and anomalies",
            function=self._assess_data_quality,
            parameters={
                "dataset_id": {"type": "string", "description": "Dataset UUID"},
                "checks": {"type": "array", "items": {"type": "string"}}
            },
            required_params=["dataset_id"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """
        Core execution using ReAct loop.
        
        1. Understand the user's request
        2. Plan analysis approach
        3. Execute tools iteratively
        4. Synthesize results
        5. Generate response
        """
        log_context = LogContext(
            component=self.name,
            operation="execute",
            request_id=str(context.request_id)
        )
        
        logger.info(
            f"Starting orchestration: {context.task_description[:100]}",
            context=log_context
        )
        
        self._active_user_id = context.user_id
        self._active_dataset_id = context.dataset_id
        self._compute = None

        try:
            async with db_manager.session() as session:
                self._session = session
                self._compute = ComputeExecutor(session)

                # Build initial messages for ReAct loop
                messages = self._build_initial_messages(context)

                # ReAct loop
                max_iterations = 10
                results: dict[str, Any] = {
                    "actions": [],
                    "insights": [],
                    "visualizations": [],
                    "final_answer": "",
                }

                for iteration in range(max_iterations):
                    if self._state:
                        self._state.current_step = iteration

                    # Get LLM response with function calling
                    response = await self._llm_client.complete(
                        messages=messages,
                        tools=[tool.to_openai_function() for tool in self._tools.values()],
                        tool_choice="auto",
                    )

                    # Check if we need to call a tool
                    if response.tool_calls:
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["function"]["name"]
                            tool_args = json.loads(tool_call["function"]["arguments"])

                            logger.debug(
                                f"Executing tool: {tool_name}",
                                context=log_context,
                                args=tool_args,
                            )

                            # Execute tool
                            try:
                                tool_result = await self.execute_tool(
                                    tool_name,
                                    thought=f"Executing {tool_name} for analysis",
                                    **tool_args,
                                )

                                results["actions"].append(
                                    {
                                        "tool": tool_name,
                                        "args": tool_args,
                                        "result": tool_result,
                                    }
                                )

                                # Add tool result to messages
                                messages.append(
                                    LLMMessage(
                                        role="assistant",
                                        content=None,
                                        tool_calls=[tool_call],
                                    )
                                )
                                messages.append(
                                    LLMMessage(
                                        role="tool",
                                        content=json.dumps(tool_result),
                                        tool_call_id=tool_call["id"],
                                    )
                                )

                            except Exception as e:
                                logger.error(
                                    f"Tool execution failed: {e}",
                                    context=log_context,
                                )
                                messages.append(
                                    LLMMessage(
                                        role="tool",
                                        content=json.dumps({"error": str(e)}),
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                    else:
                        # No more tools to call - we have the final answer
                        results["final_answer"] = response.content

                        if response.content:
                            # Extract any insights mentioned
                            results["insights"] = self._extract_insights(response.content)

                        break

                # Add to memory
                self._memory.add_to_episodic_memory(
                    content=f"Completed analysis: {results['final_answer'][:200]}",
                    memory_type="analysis_result",
                    metadata={"actions_count": len(results["actions"])},
                )

                logger.info(
                    f"Orchestration complete with {len(results['actions'])} actions",
                    context=log_context,
                )

                return results
        finally:
            self._session = None
            self._compute = None
            self._active_user_id = None
            self._active_dataset_id = None
    
    def _build_initial_messages(self, context: AgentContext) -> list[LLMMessage]:
        """Build initial messages for ReAct loop."""
        system_prompt = """You are the Orchestrator Agent for an AI Enterprise Data Analyst system.

Your role is to:
1. Understand user requests about data analysis
2. Plan the appropriate analysis steps
3. Use available tools to execute the analysis
4. Synthesize results into clear insights

Available capabilities:
- Data exploration and summarization
- Statistical analysis (hypothesis tests, correlations, distributions)
- Machine learning (classification, regression, clustering)
- Data visualization
- SQL query generation
- Data quality assessment

When analyzing:
1. First understand what the user wants to achieve
2. Check data quality and structure
3. Perform appropriate analysis
4. Provide clear, actionable insights

Always explain your reasoning and methodology.

Non-negotiable grounding rule:
- Never fabricate numeric values or dataset facts.
- If the user asks for dataset-specific numbers, you MUST call tools and cite the tool outputs.
- If a tool returns an error or insufficient evidence, say you cannot compute it yet and ask a minimal clarification.
"""
        
        messages = [
            LLMMessage(role="system", content=system_prompt)
        ]
        
        # Add dataset context if available
        if context.dataset_id:
            messages.append(LLMMessage(
                role="system",
                content=f"Active Dataset: {context.dataset_id}\nDataset Info: {json.dumps(context.dataset_info)}"
            ))
        
        # Add conversation history
        for msg in context.conversation_history:
            messages.append(LLMMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))
        
        # Add current task
        messages.append(LLMMessage(
            role="user",
            content=context.task_description
        ))
        
        return messages
    
    def _extract_insights(self, content: str) -> list[dict[str, Any]]:
        """Extract insights from response content."""
        insights = []
        
        # Simple extraction - can be enhanced with NLP
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                insights.append({
                    "type": "finding",
                    "content": line[2:]
                })
            elif any(word in line.lower() for word in ['significant', 'correlation', 'pattern', 'trend']):
                insights.append({
                    "type": "statistical",
                    "content": line
                })
        
        return insights[:10]  # Limit to top 10
    
    # ========================================================================
    # Tool Implementations
    # ========================================================================
    
    async def _analyze_data_summary(
        self,
        dataset_id: str,
        columns: list[str] = None
    ) -> dict[str, Any]:
        """Analyze dataset and return summary statistics."""
        ds, compute, owner_id = self._require_compute(dataset_id)

        plan = [
            {"operator": "dataset_overview", "params": {}},
            {"operator": "schema_snapshot", "params": {}},
            {"operator": "missingness_scan", "params": {}},
            {"operator": "numeric_summary", "params": {"max_columns": 25}},
        ]
        if columns:
            # Columns are not yet supported as a first-class operator parameter in P0;
            # keep the API stable and ignore for now rather than hallucinating.
            pass

        results = await compute.run_plan(dataset_id=ds, plan=plan, owner_id=owner_id, sample_rows=200_000)
        return {
            "status": "success",
            "dataset_id": str(ds),
            "runs": [self._execution_result_to_dict(r) for r in results],
        }
    
    async def _run_statistical_analysis(
        self,
        dataset_id: str,
        analysis_type: str,
        columns: list[str] = None,
        config: dict = None
    ) -> dict[str, Any]:
        """Run statistical analysis on dataset."""
        ds, compute, owner_id = self._require_compute(dataset_id)
        t = str(analysis_type or "").strip().lower()

        if t == "correlation":
            plan = [{"operator": "correlation_matrix", "params": {"max_columns": 25}}]
        elif t == "distribution":
            plan = [
                {"operator": "numeric_summary", "params": {"max_columns": 25}},
                {"operator": "categorical_topk", "params": {"k": 10, "max_columns": 10}},
            ]
        else:
            raise AgentException(message=f"Unsupported analysis_type for P0: {analysis_type}")

        results = await compute.run_plan(dataset_id=ds, plan=plan, owner_id=owner_id, sample_rows=200_000)
        return {
            "status": "success",
            "analysis_type": t,
            "dataset_id": str(ds),
            "runs": [self._execution_result_to_dict(r) for r in results],
        }
    
    async def _train_ml_model(
        self,
        dataset_id: str,
        model_type: str,
        target_column: str = None,
        features: list[str] = None,
        algorithm: str = None
    ) -> dict[str, Any]:
        """Train ML model on dataset."""
        raise AgentException(
            message="ML training is not available in P0. Use the modeling workflows (P2) once implemented."
        )
    
    async def _create_visualization(
        self,
        dataset_id: str,
        viz_type: str,
        x_column: str = None,
        y_column: str = None,
        color_by: str = None
    ) -> dict[str, Any]:
        """Create data visualization."""
        raise AgentException(
            message="Chart operators are not yet implemented in the safe compute layer. "
            "Use table-based operators (EDA) for P0, or implement chart operators in P1."
        )
    
    async def _generate_sql(
        self,
        dataset_id: str,
        question: str
    ) -> dict[str, Any]:
        """Generate SQL from natural language."""
        # Use LLM for SQL generation
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="Generate a SQL query for the given question. Return only the SQL, no explanation."
                ),
                LLMMessage(role="user", content=question)
            ],
            temperature=0.0
        )
        
        return {
            "status": "success",
            "sql": response.content,
            "explanation": f"Query to answer: {question}"
        }
    
    async def _assess_data_quality(
        self,
        dataset_id: str,
        checks: list[str] = None
    ) -> dict[str, Any]:
        """Assess data quality."""
        ds, compute, owner_id = self._require_compute(dataset_id)
        plan = [
            {"operator": "missingness_scan", "params": {}},
            {"operator": "uniqueness_scan", "params": {"max_columns": 200}},
            {"operator": "outlier_scan", "params": {"max_columns": 25}},
        ]
        results = await compute.run_plan(dataset_id=ds, plan=plan, owner_id=owner_id, sample_rows=200_000)
        return {
            "status": "success",
            "dataset_id": str(ds),
            "runs": [self._execution_result_to_dict(r) for r in results],
        }


# Factory function
def get_orchestrator() -> OrchestratorAgent:
    """Get orchestrator agent instance."""
    return OrchestratorAgent()
