# AI Enterprise Data Analyst - Chat API Routes
# Production-grade REST API for conversational AI interactions

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

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
from app.services.database import get_db_session
from app.services.llm_service import get_llm_service, Message as LLMMessage

logger = get_logger(__name__)

router = APIRouter()


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
        self.repo = ConversationRepository(session)
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
        else:
            # Generate title from first message
            title = message[:50] + "..." if len(message) > 50 else message
            conversation = await self.repo.create_conversation(
                user_id=self.user_id,
                title=title,
                dataset_id=dataset_id
            )
        
        # Store user message
        await self.repo.add_message(
            conversation_id=conversation.id,
            role=MessageRole.USER.value,
            content=message
        )
        
        # Get conversation history
        messages = await self.repo.get_messages(conversation.id, limit=20)
        
        # Build LLM messages
        llm_messages = self._build_llm_messages(messages, dataset_id, context)
        
        # Generate response
        logger.info(
            f"Generating response",
            context=log_context,
            conversation_id=str(conversation.id)
        )
        
        try:
            response = await self.llm.complete(
                messages=llm_messages,
                temperature=0.1,
            )
            
            # Parse for any agent actions (simplified for now)
            agent_actions = []
            suggestions = self._generate_suggestions(message, response.content)
            
            # Store assistant message
            assistant_msg = await self.repo.add_message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT.value,
                content=response.content,
                agent_actions=agent_actions,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            
            logger.info(
                f"Response generated",
                context=log_context,
                tokens=response.usage.total_tokens
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
                    "latency_ms": response.latency_ms
                }
            )
            
        except Exception as e:
            logger.error(f"Chat failed: {e}", context=log_context, exc_info=True)
            raise
    
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
    # user_id: UUID = Depends(get_current_user_id),
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
    # For demo, using a fixed user ID
    user_id = uuid4()
    
    service = ChatService(session=db, user_id=user_id)
    
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
):
    """List all conversations for the current user."""
    user_id = uuid4()
    
    repo = ConversationRepository(db)
    skip = (page - 1) * page_size
    
    conversations, total = await repo.list_conversations(
        user_id=user_id,
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
):
    """Get conversation with full message history."""
    repo = ConversationRepository(db)
    
    conversation = await repo.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
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
):
    """Delete a conversation."""
    repo = ConversationRepository(db)
    
    success = await repo.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return APIResponse.success(data=None, message="Conversation deleted")
