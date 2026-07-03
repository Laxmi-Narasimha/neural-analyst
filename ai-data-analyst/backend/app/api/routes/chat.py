# AI Enterprise Data Analyst - Chat API Routes
# Production-grade REST API for conversational AI interactions

from __future__ import annotations

import re
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
from app.services.subscription_service import SubscriptionService, UsageAction
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
        self.subscription = SubscriptionService(session)

    def _merge_recent_user_messages(self, messages: list[Any], current: str, *, limit: int = 4) -> str:
        snippets: list[str] = []
        for msg in reversed(messages or []):
            role = str(getattr(msg, "role", "") or "").lower()
            if role != "user":
                continue
            text = str(getattr(msg, "content", "") or "").strip()
            if text:
                snippets.append(text)
            if len(snippets) >= limit:
                break
        snippets.reverse()
        if not snippets:
            return current
        if snippets[-1] == current:
            snippets = snippets[:-1]
        if not snippets:
            return current
        return " | ".join(snippets[-3:] + [current])

    async def _persist_turn_memory(
        self,
        conversation: Any,
        *,
        user_message: str,
        operators: list[str],
        dataset_record: Any | None,
    ) -> None:
        conv_ctx_raw = conversation.context if isinstance(getattr(conversation, "context", None), dict) else {}
        conv_ctx: dict[str, Any] = dict(conv_ctx_raw or {})
        history = conv_ctx.get("turn_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "user": user_message[:500],
                "operators": operators[:12],
                "at": datetime.utcnow().isoformat(),
            }
        )
        conv_ctx["turn_history"] = history[-20:]
        if dataset_record is not None:
            schema_info = getattr(dataset_record, "schema_info", None) or {}
            cols = schema_info.get("columns") if isinstance(schema_info, dict) else []
            col_names = [str(c.get("name")) for c in cols if isinstance(c, dict) and c.get("name")][:40]
            conv_ctx["dataset_memory"] = {
                "dataset_id": str(getattr(dataset_record, "id", "")),
                "row_count": getattr(dataset_record, "row_count", None),
                "column_count": getattr(dataset_record, "column_count", None),
                "columns": col_names,
            }
        conversation.context = conv_ctx
        await self._session.commit()
    
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
                dataset_record = await self.dataset_loader.get_dataset_record(
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

            conv_ctx_raw = conversation.context if isinstance(getattr(conversation, "context", None), dict) else {}
            conv_ctx: dict[str, Any] = dict(conv_ctx_raw or {})

            effective_message = message
            pending = conv_ctx.get("pending_clarification")
            if isinstance(pending, dict) and str(pending.get("dataset_id") or "") == str(active_dataset_id):
                resolved = self._resolve_pending_clarification(
                    pending=pending,
                    user_message=message,
                    request_context=context,
                    dataset_record=dataset_record,
                )
                if resolved is None:
                    prompt, clarification_payload = self._format_clarification_prompt(pending)
                    assistant_msg = await self.repo.add_message(
                        conversation_id=conversation.id,
                        role=MessageRole.ASSISTANT.value,
                        content=prompt,
                        agent_actions=[],
                    )
                    return ChatResponse(
                        conversation_id=conversation.id,
                        message_id=assistant_msg.id,
                        content=prompt,
                        role=MessageRole.ASSISTANT,
                        agent_actions=[],
                        suggestions=["Reply with the number or column name"],
                        clarification=clarification_payload,
                        metadata={
                            "dataset_id": str(active_dataset_id),
                            "clarification_required": True,
                            "question_id": str(clarification_payload.get("question_id") or ""),
                        },
                    )

                conv_ctx.pop("pending_clarification", None)
                pinned_raw = conv_ctx.get("pinned")
                pinned: dict[str, Any] = dict(pinned_raw) if isinstance(pinned_raw, dict) else {}
                pinned[str(resolved["param_key"])] = str(resolved["value"])
                conv_ctx["pinned"] = pinned
                conversation.context = conv_ctx
                await self._session.commit()

                plan = resolved["plan"]
                effective_message = str(pending.get("original_user_message") or effective_message)
            else:
                pinned_raw = conv_ctx.get("pinned")
                pinned = dict(pinned_raw) if isinstance(pinned_raw, dict) else {}

                plan = self._select_plan(message, dataset_record=dataset_record)
                plan, pending_new = self._maybe_clarify_plan(
                    plan=plan,
                    user_message=message,
                    dataset_record=dataset_record,
                    pinned=pinned,
                )
                if pending_new is not None:
                    conv_ctx["pending_clarification"] = pending_new
                    conversation.context = conv_ctx
                    await self._session.commit()

                    prompt, clarification_payload = self._format_clarification_prompt(pending_new)
                    assistant_msg = await self.repo.add_message(
                        conversation_id=conversation.id,
                        role=MessageRole.ASSISTANT.value,
                        content=prompt,
                        agent_actions=[],
                    )
                    return ChatResponse(
                        conversation_id=conversation.id,
                        message_id=assistant_msg.id,
                        content=prompt,
                        role=MessageRole.ASSISTANT,
                        agent_actions=[],
                        suggestions=["Reply with the number or column name"],
                        clarification=clarification_payload,
                        metadata={
                            "dataset_id": str(active_dataset_id),
                            "clarification_required": True,
                            "question_id": str(clarification_payload.get("question_id") or ""),
                        },
                    )

            operator_names = [str(step.get("operator") or "") for step in plan if isinstance(step, dict)]
            gate = await self.subscription.assert_can_run(
                self.user_id,
                action=UsageAction.TALK_PREVIEW,
                operators=operator_names,
            )
            plan = self.subscription.filter_plan_for_free(
                plan,
                preview_only=bool(gate.get("preview_only")),
            )

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
                user_message=effective_message,
                dataset_id=active_dataset_id,
                sample_rows=sample_rows,
                results=results,
            )
            suggestions = self._generate_suggestions(effective_message, content)

            assistant_msg = await self.repo.add_message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT.value,
                content=content,
                agent_actions=agent_actions,
            )

            await self.subscription.record_usage(
                self.user_id,
                action=UsageAction.TALK_PREVIEW,
                operators=operator_names,
            )
            await self._persist_turn_memory(
                conversation,
                user_message=effective_message,
                operators=operator_names,
                dataset_record=dataset_record,
            )

            return ChatResponse(
                conversation_id=conversation.id,
                message_id=assistant_msg.id,
                content=content,
                role=MessageRole.ASSISTANT,
                agent_actions=agent_actions,
                suggestions=suggestions,
                clarification=None,
                metadata={
                    "dataset_id": str(active_dataset_id),
                    "sample_rows": sample_rows,
                    "plan": gate.get("plan"),
                    "preview_only": gate.get("preview_only", False),
                },
            )

        # No dataset: block data-analysis intents from LLM hallucination (reuse router keywords).
        data_plan = self._select_plan(message, dataset_record=None)
        if data_plan:
            content = (
                "I don't have a dataset attached to this conversation yet. "
                "Upload or select a dataset from **Datasets**, then ask your question again.\n\n"
                "I only run compute-backed analysis when a dataset is linked — "
                "I won't guess row counts or statistics without one."
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
                suggestions=["Upload a dataset", "Open Datasets", "Run Data Speaks (EDA)"],
                metadata={"no_dataset": True, "blocked_plan": data_plan},
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

    def _schema_columns(self, dataset_record: Any | None) -> list[dict[str, Any]]:
        if dataset_record is None:
            return []
        schema_info = getattr(dataset_record, "schema_info", None) or {}
        cols = schema_info.get("columns")
        if not isinstance(cols, list):
            return []
        return [c for c in cols if isinstance(c, dict)]

    def _column_candidates(self, dataset_record: Any | None, *, kind: str) -> list[str]:
        cols = self._schema_columns(dataset_record)
        out: list[tuple[str, int]] = []

        def _t(col: dict[str, Any]) -> str:
            return str(col.get("inferred_type") or col.get("type") or col.get("dtype") or "").lower()

        for c in cols:
            name = str(c.get("name") or "").strip()
            if not name:
                continue
            t = _t(c)
            if kind == "time":
                if any(k in t for k in ["datetime", "date", "time", "timestamp"]):
                    out.append((name, int(c.get("unique_count") or 0)))
            elif kind == "categorical":
                if not any(k in t for k in ["categorical", "string", "text", "boolean", "email", "url", "phone"]):
                    continue
                if bool(c.get("is_unique_identifier")):
                    continue
                try:
                    uniq_pct = float(c.get("unique_percentage") or 0.0)
                except Exception:
                    uniq_pct = 0.0
                if uniq_pct >= 0.95:
                    continue
                try:
                    uniq = int(c.get("unique_count") or 0)
                except Exception:
                    uniq = 0
                if uniq < 2:
                    continue
                # Prefer "segmentable" categoricals (bounded cardinality like the operator heuristic).
                score = uniq if uniq <= 50 else 0
                out.append((name, int(score)))
            elif kind == "numeric":
                if any(k in t for k in ["int", "float", "double", "number", "numeric", "decimal"]):
                    out.append((name, int(c.get("unique_count") or 0)))

        out.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in out]

    def _message_mentions_column(self, message: str, column: str) -> bool:
        msg = (message or "").lower()
        col = (column or "").lower()
        if not msg or not col:
            return False
        if col in msg:
            return True
        try:
            msg_norm = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
            col_norm = re.sub(r"[^a-z0-9]+", " ", column.lower()).strip()
            return bool(col_norm) and f" {col_norm} " in f" {msg_norm} "
        except Exception:
            return False

    def _maybe_clarify_plan(
        self,
        *,
        plan: list[dict[str, Any]],
        user_message: str,
        dataset_record: Any,
        pinned: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        plan_out: list[dict[str, Any]] = []
        pinned_group = str(pinned.get("group_by") or "").strip() or None
        pinned_time = str(pinned.get("time_column") or "").strip() or None

        schema_cols = [str(c.get("name") or "") for c in self._schema_columns(dataset_record) if c.get("name")]
        schema_set = {c for c in schema_cols if c}

        cat_candidates = self._column_candidates(dataset_record, kind="categorical")[:8]
        time_candidates = self._column_candidates(dataset_record, kind="time")[:8]

        for step in plan or []:
            if not isinstance(step, dict):
                continue
            op = str(step.get("operator") or "").strip()
            params_raw = step.get("params") or {}
            params = dict(params_raw) if isinstance(params_raw, dict) else {}

            if op == "segment_summary":
                group_by = str(params.get("group_by") or "").strip() or None
                if group_by and group_by in schema_set:
                    params["group_by"] = group_by
                elif pinned_group and pinned_group in schema_set:
                    params["group_by"] = pinned_group
                else:
                    mentioned = [c for c in cat_candidates if self._message_mentions_column(user_message, c)]
                    if len(mentioned) == 1:
                        params["group_by"] = mentioned[0]
                    elif len(cat_candidates) >= 2:
                        pending = self._build_pending_clarification(
                            dataset_id=str(getattr(dataset_record, "id", "") or ""),
                            original_user_message=user_message,
                            operator=op,
                            param_key="group_by",
                            options=cat_candidates,
                            reason=(
                                "Your question needs a grouping column, and multiple categorical columns look plausible."
                            ),
                            plan_template=plan_out + [{"operator": op, "params": params}] + [
                                s for s in plan[len(plan_out) + 1 :] if isinstance(s, dict)
                            ],
                        )
                        return plan, pending

            if op == "resample_aggregate":
                time_col = str(params.get("time_column") or "").strip() or None
                if time_col and time_col in schema_set:
                    params["time_column"] = time_col
                elif pinned_time and pinned_time in schema_set:
                    params["time_column"] = pinned_time
                else:
                    mentioned = [c for c in time_candidates if self._message_mentions_column(user_message, c)]
                    if len(mentioned) == 1:
                        params["time_column"] = mentioned[0]
                    elif len(time_candidates) >= 2:
                        pending = self._build_pending_clarification(
                            dataset_id=str(getattr(dataset_record, "id", "") or ""),
                            original_user_message=user_message,
                            operator=op,
                            param_key="time_column",
                            options=time_candidates,
                            reason=(
                                "Your question asks for a time trend, but I need to know which date/time column to use."
                            ),
                            plan_template=plan_out + [{"operator": op, "params": params}] + [
                                s for s in plan[len(plan_out) + 1 :] if isinstance(s, dict)
                            ],
                        )
                        return plan, pending

            plan_out.append({"operator": op, "params": params})

        return plan_out, None

    def _build_pending_clarification(
        self,
        *,
        dataset_id: str,
        original_user_message: str,
        operator: str,
        param_key: str,
        options: list[str],
        reason: str,
        plan_template: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from uuid import uuid4

        opts = [{"value": str(o), "label": str(o)} for o in options if o]
        return {
            "question_id": uuid4().hex,
            "dataset_id": str(dataset_id),
            "operator": str(operator),
            "param_key": str(param_key),
            "options": opts,
            "reason": str(reason),
            "original_user_message": str(original_user_message or ""),
            "plan": to_jsonable(plan_template or []),
            "created_at": datetime.utcnow().isoformat(),
        }

    def _format_clarification_prompt(self, pending: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        options = pending.get("options") if isinstance(pending.get("options"), list) else []
        reason = str(pending.get("reason") or "I need one clarification to proceed.").strip()

        lines: list[str] = ["Clarification needed.", reason, ""]
        if options:
            lines.append("Choose one:")
            for i, opt in enumerate(options, start=1):
                if not isinstance(opt, dict):
                    continue
                label = opt.get("label") or opt.get("value")
                if label:
                    lines.append(f"{i}. {label}")
            lines.append("")
            lines.append("Reply with the number or the column name.")
        else:
            lines.append("Reply with the column name.")

        payload = {
            "question_id": str(pending.get("question_id") or ""),
            "kind": "clarification_required",
            "dataset_id": str(pending.get("dataset_id") or ""),
            "operator": str(pending.get("operator") or ""),
            "param_key": str(pending.get("param_key") or ""),
            "reason": reason,
            "options": options,
        }
        return "\n".join(lines), payload

    def _resolve_pending_clarification(
        self,
        *,
        pending: dict[str, Any],
        user_message: str,
        request_context: dict[str, Any],
        dataset_record: Any,
    ) -> dict[str, Any] | None:
        options = pending.get("options") if isinstance(pending.get("options"), list) else []
        if not options:
            return None

        param_key = str(pending.get("param_key") or "").strip()
        operator = str(pending.get("operator") or "").strip()
        if not param_key or not operator:
            return None

        answer_raw: Any = None
        ctx_cl = request_context.get("clarification") if isinstance(request_context, dict) else None
        if isinstance(ctx_cl, dict) and str(ctx_cl.get("question_id") or "") == str(pending.get("question_id") or ""):
            answer_raw = ctx_cl.get("answer") or ctx_cl.get(param_key)

        msg = str(user_message or "").strip()
        if answer_raw is None and msg.isdigit():
            idx = int(msg) - 1
            if 0 <= idx < len(options):
                opt = options[idx] if isinstance(options[idx], dict) else {}
                answer_raw = opt.get("value") or opt.get("label")

        if answer_raw is None:
            answer_raw = msg

        ans = str(answer_raw or "").strip()
        if not ans:
            return None

        option_map: dict[str, str] = {}
        for opt in options:
            if not isinstance(opt, dict):
                continue
            v = str(opt.get("value") or "").strip()
            l = str(opt.get("label") or "").strip()
            if v:
                option_map[v.lower()] = v
            if l:
                option_map[l.lower()] = v or l

        chosen = option_map.get(ans.lower())
        if chosen is None:
            # Try substring match for convenience (user may answer "use order_date")
            low = ans.lower()
            for k, v in option_map.items():
                if k and k in low:
                    chosen = v
                    break

        if chosen is None:
            return None

        # Validate column exists on dataset schema.
        schema_set = {str(c.get("name") or "") for c in self._schema_columns(dataset_record) if c.get("name")}
        if chosen not in schema_set:
            return None

        plan_raw = pending.get("plan") if isinstance(pending.get("plan"), list) else []
        plan_out: list[dict[str, Any]] = []
        for step in plan_raw:
            if not isinstance(step, dict):
                continue
            op = str(step.get("operator") or "").strip()
            params_raw = step.get("params") or {}
            params = dict(params_raw) if isinstance(params_raw, dict) else {}
            if op == operator:
                params[param_key] = chosen
            plan_out.append({"operator": op, "params": params})

        return {"plan": plan_out, "param_key": param_key, "value": chosen}

    def _select_plan(self, user_message: str, *, dataset_record: Any | None = None) -> list[dict[str, Any]]:
        msg_raw = user_message or ""
        msg = msg_raw.lower()

        schema_cols: list[str] = []
        col_types: dict[str, str] = {}
        if dataset_record is not None:
            schema_info = getattr(dataset_record, "schema_info", None) or {}
            cols = schema_info.get("columns")
            if isinstance(cols, list):
                for c in cols:
                    if not isinstance(c, dict):
                        continue
                    name = c.get("name")
                    if not name:
                        continue
                    n = str(name)
                    schema_cols.append(n)
                    inferred = c.get("inferred_type") or c.get("type") or c.get("dtype")
                    if inferred:
                        col_types[n] = str(inferred).lower()

        def _norm(value: Any) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

        msg_norm = " " + _norm(msg_raw) + " "
        mentioned: list[str] = []
        for col in schema_cols:
            cn = _norm(col)
            if cn and f" {cn} " in msg_norm:
                mentioned.append(col)
                continue
            if col.lower() in msg:
                mentioned.append(col)

        seen: set[str] = set()
        mentioned_cols: list[str] = []
        for c in mentioned:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            mentioned_cols.append(c)

        top_k: int | None = None
        m_top = re.search(r"\btop\s+(\d{1,3})\b", msg)
        if m_top:
            try:
                top_k = int(m_top.group(1))
            except Exception:
                top_k = None

        freq: str | None = None
        if any(k in msg for k in ["daily", "per day"]):
            freq = "D"
        elif any(k in msg for k in ["weekly", "per week"]):
            freq = "W"
        elif any(k in msg for k in ["monthly", "per month"]):
            freq = "M"

        def _first_by_type(substrings: list[str]) -> str | None:
            for c in mentioned_cols:
                t = col_types.get(c, "")
                if any(s in t for s in substrings):
                    return c
            return None

        time_col = _first_by_type(["datetime", "date", "timestamp", "time"])
        value_col = _first_by_type(["int", "float", "double", "number", "numeric", "decimal"])

        group_by: str | None = None
        if any(k in msg for k in ["group by", "grouped by", "by "]):
            if mentioned_cols:
                group_by = mentioned_cols[0]

        if any(
            k in msg
            for k in [
                "how many rows",
                "row count",
                "number of rows",
                "column count",
                "number of columns",
                "dataset overview",
                "dataset size",
            ]
        ):
            return [{"operator": "dataset_overview", "params": {}}]

        if any(k in msg for k in ["eda", "analyze", "analysis"]) and not any(k in msg for k in ["top", "group by"]):
            return eda_p0_plan()

        plan: list[dict[str, Any]] = []
        if any(k in msg for k in ["overview", "summarize", "summary"]):
            plan.append({"operator": "dataset_overview", "params": {}})

        if any(k in msg for k in ["schema", "columns", "dtypes"]):
            plan.append({"operator": "schema_snapshot", "params": {}})
        if any(k in msg for k in ["preview", "head", "sample rows", "show rows"]):
            plan.append({"operator": "preview_rows", "params": {"limit": 25}})
        if any(k in msg for k in ["missing", "null", "na", "nan"]):
            plan.append({"operator": "missingness_scan", "params": {}})
        if any(k in msg for k in ["unique", "uniqueness", "duplicate", "duplicates", "cardinality", "key candidate"]):
            plan.append({"operator": "uniqueness_scan", "params": {"max_columns": 200}})
        if any(k in msg for k in ["distribution", "stats", "statistics", "describe"]):
            plan.append({"operator": "numeric_summary", "params": {"max_columns": 25}})
        if any(k in msg for k in ["text", "comment", "comments", "description", "notes", "review", "reviews"]):
            plan.append({"operator": "text_summary", "params": {"max_columns": 25}})
        if any(k in msg for k in ["correlation", "corr"]):
            plan.append({"operator": "correlation_matrix", "params": {"max_columns": 25}})
        if any(k in msg for k in ["association", "associations", "relationships", "relationship", "drivers", "dependence"]):
            plan.append(
                {
                    "operator": "association_scan",
                    "params": {"max_categorical_columns": 20, "max_numeric_columns": 20, "max_pairs": 200},
                }
            )
        if any(k in msg for k in ["trend", "over time", "time series", "timeseries", "daily", "weekly", "monthly"]):
            params: dict[str, Any] = {"freq": freq or "M", "max_points": 200}
            if time_col:
                params["time_column"] = time_col
            if value_col:
                params["value_column"] = value_col
            plan.append({"operator": "resample_aggregate", "params": params})
        if "group by" in msg or any(
            k in msg for k in ["segment", "segments", "by segment", "per segment", "top values", "value counts", "top categories"]
        ):
            params: dict[str, Any] = {"limit": int(top_k) if top_k else 50}
            if group_by:
                params["group_by"] = group_by
            elif mentioned_cols:
                params["group_by"] = mentioned_cols[0]
            plan.append({"operator": "segment_summary", "params": params})
        if "outlier" in msg or "anomaly" in msg:
            plan.append({"operator": "outlier_scan", "params": {"max_columns": 25}})

        if len(plan) > 4:
            plan = plan[:4]

        return plan or eda_p0_plan()

    def _format_grounded_response(
        self,
        *,
        user_message: str,
        dataset_id: UUID,
        sample_rows: int,
        results: list[Any],
    ) -> str:
        """Format compute results into natural, conversational language."""

        def _first_table_preview(res: Any) -> tuple[list[dict[str, Any]], Any | None]:
            for a in getattr(res, "artifacts", []):
                if getattr(getattr(a, "artifact_type", None), "value", "") != "table":
                    continue
                preview = getattr(a, "preview", {}) or {}
                pr = preview.get("preview_rows")
                if isinstance(pr, list):
                    return pr, a
            return [], None

        def _fmt_pct(value: Any) -> str:
            try:
                return f"{float(value) * 100.0:.1f}%"
            except Exception:
                return "n/a"

        def _fmt_number(value: Any) -> str:
            try:
                v = float(value)
                if v == int(v) and abs(v) < 1e15:
                    return f"{int(v):,}"
                return f"{v:,.2f}"
            except Exception:
                return str(value) if value is not None else "n/a"

        sections: list[str] = []

        for r in results:
            op = str(getattr(r, "operator_name", "") or "")
            preview_rows, _ = _first_table_preview(r)
            summary = getattr(r, "summary", {}) or {}

            if op == "dataset_overview" and preview_rows:
                row = preview_rows[0] if isinstance(preview_rows[0], dict) else {}
                rows_count = _fmt_number(row.get('rows'))
                cols_count = _fmt_number(row.get('columns'))
                parts = [f"Your dataset contains **{rows_count} rows** across **{cols_count} columns**."]
                mem = row.get("sample_memory_bytes")
                try:
                    if mem is not None and float(mem) > 0:
                        mb = float(mem) / (1024.0 * 1024.0)
                        parts.append(f"The sample uses approximately {mb:.1f} MB in memory.")
                except Exception:
                    pass
                sections.append(" ".join(parts))

            elif op == "schema_snapshot" and preview_rows:
                names = []
                for row in preview_rows[:15]:
                    if not isinstance(row, dict):
                        continue
                    n = row.get("name") or row.get("column")
                    dtype = row.get("dtype") or row.get("type") or ""
                    if n:
                        names.append(f"**{n}** ({dtype})" if dtype else f"**{n}**")
                if names:
                    sections.append(f"The columns include: {', '.join(names)}.")

            elif op == "missingness_scan" and preview_rows:
                top = []
                for row in preview_rows:
                    if not isinstance(row, dict):
                        continue
                    try:
                        pct = float(row.get("null_pct", 0.0))
                    except Exception:
                        continue
                    if pct > 0:
                        top.append(row)
                top = top[:5]
                if not top:
                    sections.append("\u2705 **No missing values** were detected in the scanned sample \u2014 your data looks complete.")
                else:
                    items = [f"**{row.get('column')}** ({_fmt_pct(row.get('null_pct'))} missing)" for row in top]
                    sections.append(f"\u26a0\ufe0f Some columns have missing values: {', '.join(items)}.")

            elif op == "uniqueness_scan":
                key_candidates = (summary.get("key_candidates") or []) if isinstance(summary, dict) else []
                keys = [f"**{k}**" for k in key_candidates if k][:5]
                if keys:
                    sections.append(f"Potential primary key columns: {', '.join(keys)}.")

            elif op == "segment_summary" and preview_rows:
                group_by = summary.get("group_by") if isinstance(summary, dict) else None
                header = f"Top groups by **{group_by}**" if group_by else "Top groups"
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    items.append(f"  \u2022 **{row.get('group')}**: {_fmt_number(row.get('count'))} rows")
                if items:
                    sections.append(f"{header}:\n" + "\n".join(items))

            elif op == "correlation_matrix" and preview_rows:
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    try:
                        v = float(row.get("corr"))
                        strength = "strong" if abs(v) > 0.7 else "moderate" if abs(v) > 0.4 else "weak"
                        direction = "positive" if v > 0 else "negative"
                        items.append(f"  \u2022 **{row.get('column_a')}** \u2194 **{row.get('column_b')}**: r = {v:.3f} ({strength} {direction})")
                    except Exception:
                        continue
                if items:
                    sections.append(f"Top correlations found:\n" + "\n".join(items))

            elif op == "association_scan" and preview_rows:
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    try:
                        v = float(row.get("score"))
                        items.append(f"  \u2022 **{row.get('column_a')}** vs **{row.get('column_b')}** ({row.get('association_type')}): score = {v:.3f}")
                    except Exception:
                        continue
                if items:
                    sections.append(f"Key associations discovered:\n" + "\n".join(items))

            elif op == "outlier_scan" and preview_rows:
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    pct = _fmt_pct(row.get('outlier_pct'))
                    items.append(f"  \u2022 **{row.get('column')}**: {pct} outliers")
                if items:
                    sections.append(f"Outlier analysis (IQR method):\n" + "\n".join(items))

            elif op == "resample_aggregate" and preview_rows:
                time_col = summary.get("time_column") if isinstance(summary, dict) else None
                value_col = summary.get("value_column") if isinstance(summary, dict) else None
                freq = summary.get("freq") if isinstance(summary, dict) else None
                freq_label = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "yearly"}.get(freq or "", freq or "")
                desc = f"Here's the {freq_label} trend"
                if time_col:
                    desc += f" over **{time_col}**"
                if value_col:
                    desc += f" for **{value_col}**"
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    items.append(f"  \u2022 {row.get('period')}: count = {_fmt_number(row.get('count'))}, sum = {_fmt_number(row.get('value_sum'))}")
                if items:
                    sections.append(f"{desc}:\n" + "\n".join(items))

            elif op == "numeric_summary" and preview_rows:
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    mean = _fmt_number(row.get("mean"))
                    median = _fmt_number(row.get("50%") if row.get("50%") is not None else row.get("median"))
                    items.append(f"  \u2022 **{row.get('column')}**: mean = {mean}, median = {median}")
                if items:
                    sections.append(f"Numeric summary:\n" + "\n".join(items))

            elif op == "text_summary" and preview_rows:
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    items.append(f"  \u2022 **{row.get('column')}**: {_fmt_number(row.get('unique_count'))} unique values")
                if items:
                    sections.append(f"Text columns:\n" + "\n".join(items))

            elif op == "preview_rows" and preview_rows:
                sections.append(f"Here are the first {len(preview_rows)} rows of your data (see the table artifact for the full preview).")

        if not sections:
            sections.append("I've analyzed your data. Check the artifacts below for detailed results.")

        return "\n\n".join(sections)
    
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
        """Generate context-aware follow-up suggestions based on what was analyzed."""
        suggestions: list[str] = []
        msg = user_message.lower()
        resp = response.lower()

        # Based on what was just shown, suggest logical next steps
        if "overview" in resp or "rows" in msg and "columns" in msg:
            suggestions.extend(["Show me the column types and schema", "Check for missing values", "Show correlations between numeric columns"])
        elif "missing" in resp or "missing" in msg:
            suggestions.extend(["Show me the outlier analysis", "What are the top correlations?", "Show distribution statistics"])
        elif "correlation" in resp or "corr" in msg:
            suggestions.extend(["Show outliers in the most correlated columns", "What are the top segments?", "Analyze trends over time"])
        elif "outlier" in resp:
            suggestions.extend(["Show the data distribution", "Group by the top categories", "Show trends over time"])
        elif "segment" in resp or "group" in msg:
            suggestions.extend(["Show correlations", "Check for outliers", "Analyze time trends"])
        elif "trend" in resp or "time" in msg:
            suggestions.extend(["Show segment breakdown", "Check for outliers", "Show the full schema"])
        elif "schema" in resp or "columns" in msg:
            suggestions.extend(["Show me a summary of the dataset", "Check for missing values", "Show distribution statistics"])
        elif "predict" in msg or "model" in msg:
            suggestions.extend(["Compare multiple algorithms", "Show feature importance", "Check data quality first"])
        else:
            suggestions.extend(["Show me an overview of the dataset", "Check for missing values", "Show correlations between columns"])

        # Deduplicate and limit
        seen: set[str] = set()
        unique: list[str] = []
        for s in suggestions:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique.append(s)
        return unique[:3]


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
