# AI Enterprise Data Analyst - Base Agent Architecture
# Enterprise-grade agent system with ReAct pattern, memory, and tool execution

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Generic, Optional, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.core.logging import get_logger, LogContext
from app.core.exceptions import (
    AgentException,
    AgentExecutionException,
    AgentTimeoutException
)

logger = get_logger(__name__)


# ============================================================================
# Agent Types and Enums
# ============================================================================

class AgentRole(str, Enum):
    """Roles that agents can assume."""
    ORCHESTRATOR = "orchestrator"
    SPECIALIST = "specialist"
    UTILITY = "utility"


class ActionStatus(str, Enum):
    """Status of an agent action."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ThoughtType(str, Enum):
    """Types of agent thoughts in ReAct pattern."""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    PLANNING = "planning"
    DECISION = "decision"
    REFLECTION = "reflection"


# ============================================================================
# Agent Memory Classes
# ============================================================================

class MemoryEntry(BaseModel):
    """Single memory entry for agent short/long-term memory."""
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: Optional[list[float]] = None


class AgentMemory:
    """
    Agent memory system with short-term and long-term storage.
    
    Implements a memory architecture inspired by cognitive science:
    - Working memory: Recent context and active processing
    - Episodic memory: Specific events and experiences
    - Semantic memory: General knowledge and facts
    """
    
    def __init__(
        self,
        max_working_memory: int = 10,
        max_episodic_memory: int = 100
    ) -> None:
        self._working_memory: list[MemoryEntry] = []
        self._episodic_memory: list[MemoryEntry] = []
        self._semantic_memory: dict[str, Any] = {}
        self._max_working = max_working_memory
        self._max_episodic = max_episodic_memory
    
    def add_to_working_memory(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None
    ) -> MemoryEntry:
        """Add entry to working memory with automatic cleanup."""
        entry = MemoryEntry(
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance
        )
        
        self._working_memory.append(entry)
        
        # Maintain size limit (FIFO with importance consideration)
        while len(self._working_memory) > self._max_working:
            # Remove least important old entry
            min_idx = min(
                range(len(self._working_memory) - 1),  # Don't remove newest
                key=lambda i: self._working_memory[i].importance
            )
            removed = self._working_memory.pop(min_idx)
            
            # Move to episodic if important enough
            if removed.importance >= 0.7:
                self._episodic_memory.append(removed)
        
        return entry
    
    def add_to_episodic_memory(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> MemoryEntry:
        """Add entry to episodic (long-term) memory."""
        entry = MemoryEntry(
            memory_type=memory_type,
            content=content,
            metadata=metadata or {}
        )
        
        self._episodic_memory.append(entry)
        
        # Maintain size limit
        while len(self._episodic_memory) > self._max_episodic:
            self._episodic_memory.pop(0)
        
        return entry
    
    def set_semantic(self, key: str, value: Any) -> None:
        """Store fact in semantic memory."""
        self._semantic_memory[key] = value
    
    def get_semantic(self, key: str, default: Any = None) -> Any:
        """Retrieve fact from semantic memory."""
        return self._semantic_memory.get(key, default)
    
    def get_working_memory(self) -> list[MemoryEntry]:
        """Get all working memory entries."""
        return list(self._working_memory)
    
    def get_recent_context(self, n: int = 5) -> str:
        """Get recent context as formatted string."""
        recent = self._working_memory[-n:]
        return "\n".join(
            f"[{e.memory_type}] {e.content}"
            for e in recent
        )
    
    def search_episodic(
        self,
        query: str,
        limit: int = 5
    ) -> list[MemoryEntry]:
        """Search episodic memory (simple text match for now)."""
        query_lower = query.lower()
        matches = [
            e for e in self._episodic_memory
            if query_lower in e.content.lower()
        ]
        return matches[:limit]
    
    def clear_working_memory(self) -> None:
        """Clear working memory."""
        self._working_memory.clear()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize memory state."""
        return {
            "working_memory": [e.model_dump() for e in self._working_memory],
            "episodic_memory": [e.model_dump() for e in self._episodic_memory],
            "semantic_memory": self._semantic_memory
        }


# ============================================================================
# Agent Action and Tool Classes
# ============================================================================

@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action_type: str = ""
    tool_name: Optional[str] = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    output: Optional[Any] = None
    error: Optional[str] = None
    status: ActionStatus = ActionStatus.PENDING
    duration_ms: Optional[float] = None
    thought: Optional[str] = None  # ReAct thought before action
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "thought": self.thought
        }


@dataclass
class AgentTool:
    """
    Represents a tool that an agent can use.
    
    Tools are the actions that agents can take to interact
    with the system and external services.
    """
    
    name: str
    description: str
    function: Callable[..., Coroutine[Any, Any, Any]]
    parameters: dict[str, Any] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)
    return_type: str = "any"
    is_async: bool = True
    
    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }


# ============================================================================
# Agent State and Context
# ============================================================================

class AgentState(BaseModel):
    """Current state of an agent execution."""
    
    agent_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    status: ActionStatus = ActionStatus.PENDING
    current_step: int = 0
    max_steps: int = 50
    
    # Goal and progress
    goal: str = ""
    sub_goals: list[str] = Field(default_factory=list)
    completed_sub_goals: list[str] = Field(default_factory=list)
    
    # Execution trace
    actions: list[dict[str, Any]] = Field(default_factory=list)
    thoughts: list[dict[str, Any]] = Field(default_factory=list)
    
    # Results
    final_result: Optional[Any] = None
    error_message: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentContext(BaseModel):
    """
    Context passed to agents for execution.
    
    Contains all information needed for an agent to perform its task.
    """
    
    # Request info
    request_id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    
    # Task info
    task_description: str
    task_type: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    
    # Data context
    dataset_id: Optional[UUID] = None
    dataset_info: dict[str, Any] = Field(default_factory=dict)
    
    # Conversation context
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    
    # Additional context
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Constraints
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    max_llm_calls: int = Field(default=20, ge=1, le=100)


# ============================================================================
# Base Agent Class
# ============================================================================

ResultT = TypeVar("ResultT")


class BaseAgent(ABC, Generic[ResultT]):
    """
    Abstract base class for all agents.
    
    Implements the Template Method pattern where the execute() method
    defines the algorithm skeleton, and subclasses implement specific steps.
    
    The ReAct (Reason + Act) pattern is used for agent execution:
    1. Observe the current state
    2. Think/Reason about what to do
    3. Decide on an action
    4. Execute the action
    5. Observe the result
    6. Repeat until goal is achieved
    """
    
    # Class attributes
    name: str = "BaseAgent"
    description: str = "Base agent class"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client: Any = None) -> None:
        """
        Initialize agent with optional LLM client.
        
        Args:
            llm_client: LLM client for reasoning (OpenAI, etc.)
        """
        self._llm_client = llm_client
        self._memory = AgentMemory()
        self._tools: dict[str, AgentTool] = {}
        self._state: Optional[AgentState] = None
        self._logger = get_logger(f"agent.{self.name}")
        
        # Register tools
        self._register_tools()
    
    @property
    def memory(self) -> AgentMemory:
        """Get agent memory."""
        return self._memory
    
    @property
    def state(self) -> Optional[AgentState]:
        """Get current agent state."""
        return self._state
    
    @property
    def tools(self) -> dict[str, AgentTool]:
        """Get registered tools."""
        return self._tools
    
    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool for the agent."""
        self._tools[tool.name] = tool
        self._logger.debug(f"Registered tool: {tool.name}")
    
    @abstractmethod
    def _register_tools(self) -> None:
        """Register agent-specific tools. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _execute_core(self, context: AgentContext) -> ResultT:
        """
        Core execution logic. Must be implemented by subclasses.
        
        This is where the main agent logic lives.
        """
        pass
    
    async def execute(self, context: AgentContext) -> ResultT:
        """
        Execute the agent with the given context.
        
        This is the Template Method that defines the execution flow.
        Subclasses should not override this - override _execute_core instead.
        """
        log_context = LogContext(
            component=self.name,
            operation="execute",
            request_id=str(context.request_id)
        )
        
        # Initialize state
        self._state = AgentState(
            agent_name=self.name,
            goal=context.task_description,
            started_at=datetime.utcnow()
        )
        
        self._logger.info(
            f"Starting agent execution",
            context=log_context,
            goal=context.task_description[:100]
        )
        
        try:
            # Pre-execution hook
            await self._before_execute(context)
            
            # Core execution
            self._state.status = ActionStatus.RUNNING
            result = await self._execute_core(context)
            
            # Post-execution hook
            await self._after_execute(context, result)
            
            # Update state
            self._state.status = ActionStatus.COMPLETED
            self._state.final_result = result
            self._state.completed_at = datetime.utcnow()
            
            self._logger.info(
                f"Agent execution completed",
                context=log_context,
                steps=self._state.current_step
            )
            
            return result
            
        except Exception as e:
            self._state.status = ActionStatus.FAILED
            self._state.error_message = str(e)
            self._state.completed_at = datetime.utcnow()
            
            self._logger.error(
                f"Agent execution failed: {e}",
                context=log_context,
                exc_info=True
            )
            
            raise AgentExecutionException(
                agent_name=self.name,
                step=f"step_{self._state.current_step}",
                cause=e
            )
    
    async def _before_execute(self, context: AgentContext) -> None:
        """Hook called before execution. Override for custom behavior."""
        # Add task to working memory
        self._memory.add_to_working_memory(
            content=f"Task: {context.task_description}",
            memory_type="goal",
            importance=1.0
        )
    
    async def _after_execute(self, context: AgentContext, result: ResultT) -> None:
        """Hook called after execution. Override for custom behavior."""
        # Store result in episodic memory
        self._memory.add_to_episodic_memory(
            content=f"Completed task: {context.task_description[:100]}",
            memory_type="task_completion",
            metadata={"result_type": type(result).__name__}
        )
    
    async def execute_tool(
        self,
        tool_name: str,
        thought: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Execute a registered tool with tracking.
        
        Args:
            tool_name: Name of the tool to execute
            thought: Optional reasoning for the action (ReAct pattern)
            **kwargs: Arguments to pass to the tool
        
        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            raise AgentExecutionException(
                agent_name=self.name,
                cause=ValueError(f"Unknown tool: {tool_name}")
            )
        
        tool = self._tools[tool_name]
        
        # Create action record
        action = AgentAction(
            action_type="tool_call",
            tool_name=tool_name,
            tool_input=kwargs,
            thought=thought
        )
        
        start_time = datetime.utcnow()
        
        try:
            action.status = ActionStatus.RUNNING
            
            # Execute tool
            if tool.is_async:
                result = await tool.function(**kwargs)
            else:
                result = tool.function(**kwargs)
            
            action.output = result
            action.status = ActionStatus.COMPLETED
            
        except Exception as e:
            action.error = str(e)
            action.status = ActionStatus.FAILED
            raise
            
        finally:
            action.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Track in state
            if self._state:
                self._state.actions.append(action.to_dict())
                self._state.current_step += 1
            
            # Add to memory
            self._memory.add_to_working_memory(
                content=f"Executed {tool_name}: {str(action.output)[:200] if action.output else action.error}",
                memory_type="action_result",
                importance=0.6
            )
        
        return result
    
    def get_execution_trace(self) -> list[dict[str, Any]]:
        """Get the full execution trace."""
        if self._state:
            return self._state.actions
        return []
    
    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get list of available tools in OpenAI function format."""
        return [tool.to_openai_function() for tool in self._tools.values()]
