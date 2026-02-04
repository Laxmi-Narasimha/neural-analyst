# AI Enterprise Data Analyst - Chat API Routes
# Production-grade REST API for conversational AI interactions

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.api.schemas import (
    APIResponse,
    PaginatedResponse,
    PaginationMeta,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    MessageRole,
    ConversationResponse,
)
from app.core.logging import get_logger, LogContext
from app.core.serialization import to_jsonable
from app.compute.executor import ComputeExecutor
from app.compute.plans import eda_p0_plan
from app.services.database import get_db_session
from app.services.dataset_loader import DatasetLoaderService
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.api.routes.auth import require_permission
from app.services.auth_service import AuthUser, Permission

logger = get_logger(__name__)

router = APIRouter()

def _artifact_to_dict(a: Any) -> dict[str, Any]:
    return {
        "artifact_id": str(getattr(a, "artifact_id", "")),
        "artifact_type": getattr(getattr(a, "artifact_type", None), "value", None),
        "name": getattr(a, "name", None),
        "created_at": getattr(a, "created_at", None),
        "storage_path": getattr(a, "storage_path", None),
        "preview": getattr(a, "preview", None),
        "dataset_id": str(getattr(a, "dataset_id", None)) if getattr(a, "dataset_id", None) else None,
        "dataset_version": getattr(a, "dataset_version", None),
        "operator_name": getattr(a, "operator_name", None),
        "operator_params": getattr(a, "operator_params", None) or {},
    }


# ============================================================================
# Conversation Repository
# ============================================================================

class ConversationRepository:
    """Repository for Conversation and Message CRUD operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def create_conversation(
        self,
        user_id: UUID,
        title: str,
        dataset_id: Optional[UUID] = None,
    ):
        """Create new conversation."""
        from app.models import Conversation
        
        conversation = Conversation(
            title=title,
            user_id=user_id,
            active_dataset_id=dataset_id,
            context={}
        )
        
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        
        return conversation
    
    async def get_conversation(self, conversation_id: UUID):
        """Get conversation by ID."""
        from app.models import Conversation
        
        query = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.is_deleted == False
        )
        result = await self.session.execute(query)
        return result.scalars().first()
    
    async def list_conversations(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ):
        """List conversations for user."""
        from app.models import Conversation, Message
        
        query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.is_deleted == False
        ).order_by(Conversation.updated_at.desc()).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        conversations = result.scalars().all()
        
        # Get total count
        count_query = select(func.count()).where(
            Conversation.user_id == user_id,
            Conversation.is_deleted == False
        )
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0
        
        return conversations, total
    
    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        agent_actions: list[dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        """Add message to conversation."""
        from app.models import Message, Conversation
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            agent_actions=agent_actions or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        self.session.add(message)
        
        # Update conversation timestamp
        conv = await self.get_conversation(conversation_id)
        if conv:
            conv.updated_at = datetime.utcnow()
        
        await self.session.commit()
        await self.session.refresh(message)
        
        return message
    
    async def get_messages(
        self,
        conversation_id: UUID,
        limit: int = 50,
    ):
        """Get messages for conversation."""
        from app.models import Message
        
        query = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.asc()).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def delete_conversation(self, conversation_id: UUID):
        """Soft delete conversation."""
        from app.models import Conversation
        
        conv = await self.get_conversation(conversation_id)
        if conv:
            conv.soft_delete()
            await self.session.commit()
            return True
        return False


# ============================================================================
# Chat Service
# ============================================================================

class ChatService:
    """
    Service for handling chat interactions with AI.
    
    Orchestrates the conversation flow, agent execution,
    and response generation.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> None:
        self._session = session
        self.repo = ConversationRepository(session)
        self.executor = ComputeExecutor(session)
        self.dataset_loader = DatasetLoaderService(session)
        self.llm = get_llm_service()
        self.user_id = user_id
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[UUID] = None,
        dataset_id: Optional[UUID] = None,
        context: dict[str, Any] = None,
    ) -> ChatResponse:
        """
        Process chat message and generate response.
        
        Args:
            message: User's message
            conversation_id: Existing conversation ID or None for new
            dataset_id: Dataset context for analysis
            context: Additional context
        
        Returns:
            ChatResponse with AI-generated response
        """
        context = context or {}
        log_context = LogContext(
            component="ChatService",
            operation="chat"
        )
        
        # Get or create conversation
        if conversation_id:
            conversation = await self.repo.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            if conversation.user_id != self.user_id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        else:
            # Generate title from first message
            title = message[:50] + "..." if len(message) > 50 else message
            conversation = await self.repo.create_conversation(
                user_id=self.user_id,
                title=title,
                dataset_id=dataset_id
            )

        # Update active dataset if user explicitly sets it.
        if dataset_id is not None and conversation.active_dataset_id != dataset_id:
            conversation.active_dataset_id = dataset_id
            await self._session.commit()

        active_dataset_id = dataset_id or conversation.active_dataset_id
        
        # Store user message
        await self.repo.add_message(
            conversation_id=conversation.id,
            role=MessageRole.USER.value,
            content=message
        )
        
        # If we have a dataset, prefer compute-grounded responses (no hallucinated numbers).
        if active_dataset_id is not None:
            try:
                await self.dataset_loader.get_dataset_record(
                    active_dataset_id, owner_id=self.user_id, require_ready=True
                )
            except Exception as e:
                content = (
                    "Your dataset isn't ready for analysis yet. "
                    "Please process it first via the Datasets API, then retry.\n\n"
                    f"Details: {e}"
                )
                assistant_msg = await self.repo.add_message(
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT.value,
                    content=content,
                    agent_actions=[],
                )
                return ChatResponse(
                    conversation_id=conversation.id,
                    message_id=assistant_msg.id,
                    content=content,
                    role=MessageRole.ASSISTANT,
                    agent_actions=[],
                    suggestions=["Run dataset processing", "Run Data Speaks (EDA)"],
                    metadata={"dataset_id": str(active_dataset_id)},
                )

            plan = self._select_plan(message)
            sample_rows = 200_000
            results = await self.executor.run_plan(
                dataset_id=active_dataset_id,
                plan=plan,
                owner_id=self.user_id,
                sample_rows=sample_rows,
            )

            agent_actions = to_jsonable(
                [
                    {
                        "type": "operator_run",
                        "operator": r.operator_name,
                        "summary": r.summary,
                        "artifacts": [_artifact_to_dict(a) for a in r.artifacts],
                    }
                    for r in results
                ]
            )

            content = self._format_grounded_response(
                user_message=message,
                dataset_id=active_dataset_id,
                sample_rows=sample_rows,
                results=results,
            )
            suggestions = self._generate_suggestions(message, content)

            assistant_msg = await self.repo.add_message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT.value,
                content=content,
                agent_actions=agent_actions,
            )

            return ChatResponse(
                conversation_id=conversation.id,
                message_id=assistant_msg.id,
                content=content,
                role=MessageRole.ASSISTANT,
                agent_actions=agent_actions,
                suggestions=suggestions,
                metadata={
                    "dataset_id": str(active_dataset_id),
                    "sample_rows": sample_rows,
                },
            )

        # Otherwise, fall back to an LLM-only assistant (no dataset context).
        messages = await self.repo.get_messages(conversation.id, limit=20)
        llm_messages = self._build_llm_messages(messages, None, context)

        logger.info("Generating response", context=log_context, conversation_id=str(conversation.id))

        try:
            response = await self.llm.complete(messages=llm_messages, temperature=0.1)
            agent_actions: list[dict[str, Any]] = []
            suggestions = self._generate_suggestions(message, response.content)

            assistant_msg = await self.repo.add_message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT.value,
                content=response.content,
                agent_actions=agent_actions,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            return ChatResponse(
                conversation_id=conversation.id,
                message_id=assistant_msg.id,
                content=response.content,
                role=MessageRole.ASSISTANT,
                agent_actions=agent_actions,
                suggestions=suggestions,
                metadata={
                    "model": response.model,
                    "tokens": response.usage.total_tokens,
                    "latency_ms": response.latency_ms,
                },
            )
        except Exception as e:
            logger.error(f"Chat failed: {e}", context=log_context, exc_info=True)
            raise

    def _select_plan(self, user_message: str) -> list[dict[str, Any]]:
        msg = (user_message or "").lower()

        if any(k in msg for k in ["eda", "analyze", "analysis", "overview", "summary"]):
            return eda_p0_plan()

        plan: list[dict[str, Any]] = []
        if any(k in msg for k in ["schema", "columns", "dtypes"]):
            plan.append({"operator": "schema_snapshot", "params": {}})
        if any(k in msg for k in ["preview", "head", "sample"]):
            plan.append({"operator": "preview_rows", "params": {"limit": 25}})
        if any(k in msg for k in ["missing", "null", "na", "nan"]):
            plan.append({"operator": "missingness_scan", "params": {}})
        if any(k in msg for k in ["correlation", "corr"]):
            plan.append({"operator": "correlation_matrix", "params": {"max_columns": 25}})
        if "outlier" in msg:
            plan.append({"operator": "outlier_scan", "params": {"max_columns": 25}})

        return plan or eda_p0_plan()

    def _format_grounded_response(
        self,
        *,
        user_message: str,
        dataset_id: UUID,
        sample_rows: int,
        results: list[Any],
    ) -> str:
        lines: list[str] = []
        lines.append("I ran safe data operators to answer your question.")
        lines.append(f"- dataset_id: {dataset_id}")
        lines.append(f"- sampled_rows: {sample_rows}")
        lines.append("")

        # Highlights: missingness (if present in this run).
        for r in results:
            op = getattr(r, "operator_name", "")
            if op != "missingness_scan":
                continue

            lines.append("Missingness (top columns):")
            preview_rows: list[dict[str, Any]] = []
            for a in getattr(r, "artifacts", []):
                if getattr(getattr(a, "artifact_type", None), "value", "") != "table":
                    continue
                pr = (getattr(a, "preview", {}) or {}).get("preview_rows")
                if isinstance(pr, list):
                    preview_rows = pr
                    break

            top: list[dict[str, Any]] = []
            for row in preview_rows:
                try:
                    if float(row.get("null_pct", 0.0)) > 0:
                        top.append(row)
                except Exception:
                    continue
            top = top[:3]

            if top:
                for row in top:
                    col = row.get("column")
                    pct = float(row.get("null_pct", 0.0)) * 100.0
                    lines.append(f"- {col}: {pct:.2f}% missing")
            else:
                lines.append("- No missing values detected in the scanned sample.")
            lines.append("")

        lines.append("Artifacts (downloadable):")
        for r in results:
            op = getattr(r, "operator_name", "")
            for a in getattr(r, "artifacts", []):
                aid = getattr(a, "artifact_id", None)
                if not aid:
                    continue
                atype = getattr(getattr(a, "artifact_type", None), "value", "artifact")
                aname = getattr(a, "name", op)
                lines.append(f"- {op} -> {atype}: {aname} (artifact_id={aid})")

        lines.append("")
        lines.append("Tell me what to run next (e.g., 'show correlations', 'top missing columns', 'outliers').")
        return "\n".join(lines)
    
    def _build_llm_messages(
        self,
        messages,
        dataset_id: Optional[UUID],
        context: dict[str, Any]
    ) -> list[LLMMessage]:
        """Build LLM message list with system context."""
        llm_messages = []
        
        # System message with context
        system_content = self._build_system_message(dataset_id, context)
        llm_messages.append(LLMMessage(role="system", content=system_content))
        
        # History
        for msg in messages:
            llm_messages.append(LLMMessage(
                role=msg.role,
                content=msg.content
            ))
        
        return llm_messages
    
    def _build_system_message(
        self,
        dataset_id: Optional[UUID],
        context: dict[str, Any]
    ) -> str:
        """Build system message with relevant context."""
        system = """You are an AI Enterprise Data Analyst assistant. You help users:
- Analyze and understand their data
- Perform statistical analysis and hypothesis testing
- Build and evaluate machine learning models
- Create visualizations and dashboards
- Generate insights and recommendations
- Write and explain SQL queries
- Perform feature engineering

Guidelines:
- Be precise and data-driven in your responses
- Explain your reasoning and methodology
- Suggest appropriate visualizations
- Warn about data quality issues
- Recommend best practices

"""
        
        if dataset_id:
            system += f"\nActive Dataset ID: {dataset_id}\n"
        
        if context:
            system += f"\nAdditional Context: {context}\n"
        
        return system
    
    def _generate_suggestions(self, user_message: str, response: str) -> list[str]:
        """Generate follow-up suggestions."""
        # Simple rule-based suggestions (can be enhanced with LLM)
        suggestions = []
        
        keywords = user_message.lower()
        
        if "summary" in keywords or "overview" in keywords:
            suggestions.append("Show me the correlation matrix")
            suggestions.append("What are the outliers in this data?")
        
        if "distribution" in keywords:
            suggestions.append("Test for normality")
            suggestions.append("Suggest appropriate transformations")
        
        if "predict" in keywords or "model" in keywords:
            suggestions.append("Compare multiple algorithms")
            suggestions.append("Show feature importance")
        
        if "missing" in keywords or "null" in keywords:
            suggestions.append("Suggest imputation strategies")
            suggestions.append("Analyze missing data patterns")
        
        # Default suggestions
        if not suggestions:
            suggestions = [
                "Generate a data quality report",
                "Perform exploratory data analysis",
                "Suggest analysis approaches"
            ]
        
        return suggestions[:3]


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "",
    response_model=APIResponse[ChatResponse],
    summary="Send chat message",
    description="Send a message to the AI data analyst and receive a response"
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.RUN_ANALYSIS)),
):
    """
    Send a chat message and receive AI response.
    
    The AI can:
    - Answer questions about data
    - Perform analysis
    - Generate SQL queries
    - Create visualizations
    - Provide recommendations
    """
    service = ChatService(session=db, user_id=user.user_id)
    
    response = await service.chat(
        message=request.message,
        conversation_id=request.conversation_id,
        dataset_id=request.dataset_id,
        context=request.context
    )
    
    return APIResponse.success(data=response)


@router.get(
    "/conversations",
    response_model=PaginatedResponse[ConversationResponse],
    summary="List conversations",
    description="Get paginated list of chat conversations"
)
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    """List all conversations for the current user."""
    repo = ConversationRepository(db)
    skip = (page - 1) * page_size
    
    conversations, total = await repo.list_conversations(
        user_id=user.user_id,
        skip=skip,
        limit=page_size
    )
    
    total_pages = (total + page_size - 1) // page_size
    
    response_data = [
        ConversationResponse(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=len(c.messages) if hasattr(c, 'messages') else 0,
            active_dataset_id=c.active_dataset_id
        )
        for c in conversations
    ]
    
    return PaginatedResponse(
        status="success",
        data=response_data,
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=APIResponse[dict],
    summary="Get conversation",
    description="Get conversation with message history"
)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.READ_DATA)),
):
    """Get conversation with full message history."""
    repo = ConversationRepository(db)
    
    conversation = await repo.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    if conversation.user_id != user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    
    messages = await repo.get_messages(conversation_id, limit=100)
    
    return APIResponse.success(data={
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "active_dataset_id": str(conversation.active_dataset_id) if conversation.active_dataset_id else None,
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "agent_actions": m.agent_actions
            }
            for m in messages
        ]
    })


@router.delete(
    "/conversations/{conversation_id}",
    response_model=APIResponse[None],
    summary="Delete conversation",
    description="Delete a conversation and its messages"
)
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(require_permission(Permission.WRITE_DATA)),
):
    """Delete a conversation."""
    repo = ConversationRepository(db)

    conversation = await repo.get_conversation(conversation_id)
    if not conversation or conversation.user_id != user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    success = await repo.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return APIResponse.success(data=None, message="Conversation deleted")
