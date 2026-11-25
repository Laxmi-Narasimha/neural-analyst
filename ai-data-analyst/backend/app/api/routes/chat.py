# AI Enterprise Data Analyst - Chat API Routes
# Production-grade REST API for conversational AI interactions

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update

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
from app.compute.starter_questions import measure_plan, ranked_average_plan, starter_plans
from app.core.config import settings
from app.models import Conversation, JobType
from app.services.job_service import JobService
from app.workers.dispatcher import enqueue_ml_training
from app.services.database import get_db_session
from app.services.dataset_loader import DatasetLoaderService, file_version
from app.services.retail_calendar import PATTERNS, SETTING_KEYS, RetailCalendar
from app.services.workbook_scope import check_carried_columns, sheet_command, type_family, workbook_catalog, worksheet_view
from app.services.chat_disk import needs_scan, prepare_scan
from app.services.chat_scope import condition_leaves, ScopeEdit, ScopeQuestion, bind_conditions, describe_scope, parse_scope, scoped_dataset, verify_measurement
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.services.subscription_service import PREVIEW_OPERATORS, SubscriptionService, UsageAction
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

def _training_quality_lines(summary: dict[str, Any]) -> list[str]:
    """Imbalance, cut-off, calibration and weakest segment, in the user's terms."""
    lines: list[str] = []
    balance = summary.get("class_balance") or {}
    if balance.get("imbalanced") and summary.get("selection_metric_reason"):
        lines.append(f"**Rare outcome:** {summary['selection_metric_reason']}")
    thr = summary.get("decision_threshold") or {}
    if thr.get("threshold") is not None:
        chosen, default = thr.get("held_out_at_chosen") or {}, thr.get("held_out_at_default_0_5") or {}
        lines.append(
            f"**Decision cut-off:** predicts **{thr.get('positive_class')}** when its probability is at least "
            f"{float(thr['threshold']):.3f} (chosen on training folds). On held-out rows it finds "
            f"{float(chosen.get('recall') or 0):.0%} of them with {float(chosen.get('precision') or 0):.0%} precision; "
            f"the usual 0.5 would find {float(default.get('recall') or 0):.0%}."
        )
    cal = summary.get("calibration") or {}
    before = (cal.get("held_out_before") or {}).get("ece")
    after = (cal.get("held_out_after") or {}).get("ece")
    if cal.get("applied") and before is not None and after is not None:
        lines.append(f"**Probabilities:** calibrated ({cal.get('method')}); held-out calibration error "
                     f"{float(before):.3f} → {float(after):.3f}.")
    elif before is not None:
        lines.append(f"**Probabilities:** held-out calibration error {float(before):.3f} "
                     "(average gap between predicted and actual rates).")
    seg = summary.get("error_by_segment") or {}
    weak = [s for s in seg.get("worst_segments") or [] if s.get("significant")]
    if weak:
        s, metric = weak[0], seg.get("metric", "error_rate")
        lines.append(
            f"**Weakest segment:** {s['column']} = {s['segment']}: {metric.replace('_', ' ')} "
            f"{_fmt_number(s.get(metric))} vs {_fmt_number(seg.get('overall'))} overall ({int(s['rows']):,} held-out rows)."
        )
    split = summary.get("split") or {}
    if split.get("time_column_source") == "inferred":
        lines.append(f"**Prediction moment:** assumed to be **{split.get('time_column')}**; tell me if predictions "
                     "are made at a different time, or give a backtest date.")
    return lines


_METRIC_NAMES = {"roc_auc": "ROC AUC", "average_precision": "PR AUC", "f1_macro": "macro F1", "f1_weighted": "weighted F1",
                 "accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "r2": "R²",
                 "neg_mean_absolute_error": "negative MAE", "neg_root_mean_squared_error": "negative RMSE"}


def _held_out_line(summary: dict[str, Any]) -> str:
    """Held-out results as a data scientist reads them: ranking quality, then the outcome class."""
    m = {k: v for k, v in (summary.get("test_metrics") or {}).items() if v is not None}
    classes = summary.get("classes") or []

    def num(key: str, label: str, pct: bool = False) -> list[str]:
        if key not in m:
            return []
        return [f"{label} {float(m[key]):.0%}" if pct else f"{label} {float(m[key]):.3f}"]

    if len(classes) == 2:
        positive = ((summary.get("decision_threshold") or {}).get("positive_class")) or classes[1]
        head = num("roc_auc", "ROC AUC") + num("average_precision", "PR AUC") + num("balanced_accuracy", "balanced accuracy")
        tail = num("precision", "precision", True) + num("recall", "recall", True)
        text = "**Held-out results:** " + ", ".join(head)
        return text + (f"; for **{positive}**: " + ", ".join(tail) if tail else "") + "."
    if classes:
        bits = num("accuracy", "accuracy") + num("balanced_accuracy", "balanced accuracy") + num("f1", "weighted F1") \
            + num("roc_auc_ovr", "ROC AUC (one-vs-rest)")
        return "**Held-out results:** " + ", ".join(bits) + "."
    bits = num("r2", "R²") + [f"{label} {_fmt_number(m[key])}" for key, label in (("mae", "MAE"), ("rmse", "RMSE")) if key in m]
    return "**Held-out results:** " + ", ".join(bits) + "."


def _scoring_lines(summary: dict[str, Any]) -> list[str]:
    """A batch scoring result in the user's terms: what was predicted, and how far to trust it."""
    rows = int(summary.get("rows_scored") or 0)
    target = summary.get("target_column")
    lines = [f"I scored **all {rows:,} rows** with your model **{summary.get('model_name')}** "
             f"v{summary.get('model_version')} (predicts **{target}**)."]
    ps = summary.get("prediction_summary") or {}
    if ps.get("counts"):
        total = sum(int(v) for v in ps["counts"].values()) or 1
        mix = ", ".join(f"**{k}** {int(v):,} ({int(v) / total:.0%})"
                        for k, v in sorted(ps["counts"].items(), key=lambda kv: -int(kv[1]))[:6])
        lines.append(f"**Predictions:** {mix}.")
    elif ps.get("mean") is not None:
        lines.append(f"**Predictions:** average {_fmt_number(ps['mean'])}, from {_fmt_number(ps.get('min'))} "
                     f"to {_fmt_number(ps.get('max'))}.")
    thr = summary.get("decision_threshold") or {}
    classes = summary.get("classes") or []
    if thr.get("threshold") is not None and classes:
        pos = classes[int(thr.get("positive_index") or 0)]
        lines.append(f"**Decision cut-off:** a row is predicted **{pos}** when its probability is at least "
                     f"{float(thr['threshold']):.3f}, the cut-off tuned at training. Sort by **probability_{pos}** "
                     "to apply a different one.")
    if summary.get("calibrated"):
        lines.append("**Probabilities:** calibrated at training, so 0.30 means about 30 in 100 such rows.")
    drift = summary.get("drift") or {}
    if drift:
        shifted = [c for c in drift.get("columns") or [] if c.get("level") in {"major", "moderate", "missing_column"}][:4]
        detail = "; ".join(
            f"{c['column']} " + (f"PSI {float(c['psi']):.2f}" if c.get("psi") is not None else "missing")
            for c in shifted)
        lines.append(f"**Drift check:** {drift.get('message')}" + (f" ({detail})" if detail else ""))
    shift = summary.get("prediction_shift") or {}
    if shift.get("kind") == "classes":
        train, new = shift.get("training") or {}, shift.get("new") or {}
        bits = ", ".join(f"{k} {float(train.get(k, 0)):.0%} → {float(v):.0%}" for k, v in new.items())
        lines.append(f"**Prediction mix vs held-out rows at training:** {bits} (PSI {float(shift.get('psi') or 0):.2f}, "
                     f"{shift.get('level')}).")
    elif shift.get("kind") == "values":
        lines.append(f"**Predictions vs training:** average {_fmt_number(shift.get('new_mean'))} here, "
                     f"{_fmt_number(shift.get('training_mean'))} on held-out rows at training.")
    real = summary.get("realised_performance") or {}
    if real.get("accuracy") is not None:
        ref = real.get("held_out_balanced_accuracy_at_training")
        lines.append(f"**How it did here:** {target} is known for {int(real['rows_with_outcome']):,} rows: accuracy "
                     f"{float(real['accuracy']):.1%}, balanced accuracy {float(real['balanced_accuracy']):.1%}"
                     + (f" (held-out at training: {float(ref):.1%})" if ref is not None else "") + ".")
    elif real.get("mae") is not None:
        ref = real.get("held_out_mae_at_training")
        lines.append(f"**How it did here:** {target} is known for {int(real['rows_with_outcome']):,} rows: mean absolute "
                     f"error {_fmt_number(real['mae'])}" + (f" (held-out at training: {_fmt_number(ref)})" if ref is not None else "")
                     + ".")
    for w in [w for w in summary.get("warnings") or [] if w != drift.get("message")][:4]:
        lines.append(f"⚠️ {w}")
    lines.append(f"Every row with **{summary.get('prediction_column')}**"
                 + (" and the probability columns" if classes else "")
                 + " is in the **predictions** table artifact, ready to open or download.")
    return lines


def _fmt_number(value: Any) -> str:
    """Numbers as users read them: thousands separators, 2 decimals unless whole; missing -> n/a."""
    if value is None or isinstance(value, bool):
        return "n/a" if value is None else str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v or v in (float("inf"), float("-inf")):
        return "n/a"
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    return f"{v:,.2f}"


def _dropped_label(condition: Any) -> str:
    """How a rule-detected filter the planner dropped is named to the user."""
    if not isinstance(condition, dict):
        return ""
    if condition.get("op") == "date":
        start, end = condition.get("start"), condition.get("end")
        return f"the date range {start or '…'} to {end or '…'}" if (start or end) else ""
    value = condition.get("value")
    if value is None or str(value).strip() == "":
        return ""
    return repr(value)


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

    def _turn_memory(
        self,
        conv_ctx: dict[str, Any],
        *,
        user_message: str,
        operators: list[str],
        dataset_record: Any | None,
    ) -> dict[str, Any]:
        history = conv_ctx.get("turn_history")
        history = list(history) if isinstance(history, list) else []
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
        return conv_ctx

    async def _save_scope_context(self, conversation, state: dict, *, dataset_id: UUID | None = None) -> None:
        # Claim this turn before committing its answer. Concurrent clients must
        # retry against the winning scope instead of silently dropping its edits.
        revision = (conversation.context or {}).get("scope_revision")
        result = await self._session.execute(
            update(Conversation).where(
                Conversation.id == conversation.id,
                Conversation.active_dataset_id == conversation.active_dataset_id,
                Conversation.context["scope_revision"].as_string() == revision,
            ).values(context={**state, "scope_revision": uuid4().hex},
                     active_dataset_id=dataset_id or conversation.active_dataset_id).execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            await self._session.rollback()
            raise HTTPException(409, "This conversation changed during analysis. Retry your question against the latest scope.")

    async def _scope_reply(self, conversation, dataset_id, reason, pending=None) -> ChatResponse:
        payload = None
        content = reason
        if pending:
            if pending.get("scope_key") in {"year", "fiscal_year"}:
                payload = {"question_id": pending["question_id"], "param_key": pending["scope_key"], "options": [], "reason": reason}
            else:
                content, payload = self._format_clarification_prompt(pending)
                if reason != pending.get("reason"):
                    content = reason + "\n\n" + content
            if pending.get("worksheets"):
                content = "Worksheets: " + ", ".join(pending["worksheets"]) + ".\n\n" + content
        reply = await self.repo.add_message(conversation_id=conversation.id, role=MessageRole.ASSISTANT.value,
                                            content=content, agent_actions=[])
        return ChatResponse(conversation_id=conversation.id, message_id=reply.id, content=content,
                            clarification=payload, metadata={"dataset_id": str(dataset_id),
                            "worksheets": (pending or {}).get("worksheets"), "clarification_required": bool(pending), "unsupported_question": not pending})

    async def chat(
        self,
        message: str,
        conversation_id: Optional[UUID] = None,
        dataset_id: Optional[UUID] = None,
        context: dict[str, Any] = None,
        background_tasks: BackgroundTasks | None = None,
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

            conv_ctx = dict(conversation.context or {}) if conversation.active_dataset_id == active_dataset_id else {}
            version = str((dataset_record.profile_report or {}).get("file_hash") or "")
            if not version:
                version = await asyncio.to_thread(file_version, dataset_record)
            last = conv_ctx.get("analysis_scope") or {}
            pending = conv_ctx.get("pending_clarification") or {}
            if pending.get("dataset_id") != str(active_dataset_id):
                pending = {}
            reset = message.strip().lower().rstrip(".!?") in {"clear filters", "clear all filters", "reset filters", "start over"}
            changed_version = last and last.get("dataset_version") != version
            if pending and message.strip().lower() in {"cancel", "cancel clarification"}:
                await self._save_scope_context(conversation, {**conv_ctx, "pending_clarification": None}, dataset_id=active_dataset_id)
                return await self._scope_reply(conversation, active_dataset_id, "Clarification cancelled. Your last completed analysis is still active.")
            if reset or (pending and pending.get("dataset_version", version) != version):
                pending = {}
                conv_ctx.pop("pending_clarification", None)
            if changed_version:
                conv_ctx.pop("pinned", None)
                if not reset:
                    return await self._scope_reply(conversation, active_dataset_id,
                        "The dataset version changed. Send ‘start over’ to analyze the current version with fresh filters.")
                last = {}

            workbook = dataset_record.file_format in {"xls", "xlsx"}
            sheets = None
            requested = (context or {}).get("worksheets")
            if workbook:
                try:
                    catalog = await workbook_catalog(dataset_record, self._session)
                    # Conversations created before sheet selection always used the first tab.
                    if last and not last.get("worksheets"):
                        last = {**last, "worksheets": [catalog[0]["name"]],
                                "column_types": {c["name"]: type_family(c["inferred_type"]) for c in catalog[0]["columns"]}}
                    if pending and not pending.get("worksheets"):
                        pending = {**pending, "worksheets": [catalog[0]["name"]]}
                    message, named = sheet_command(message, catalog)
                    if named is not None:
                        requested = named
                    saved = pending.get("worksheets") or last.get("worksheets")
                    dataset_record = worksheet_view(dataset_record, catalog, saved if requested is None else requested)
                    sheets = dataset_record.schema_info["worksheets"]
                    if pending and pending.get("worksheets", sheets) != sheets:
                        return await self._scope_reply(conversation, active_dataset_id,
                            "This clarification belongs to worksheets " + ", ".join(pending["worksheets"]) +
                            ". Answer on those sheets or send ‘cancel clarification’ before switching.", pending)
                    if message.strip().lower() == "start over":
                        last = {}
                except ScopeQuestion as error:
                    return await self._scope_reply(conversation, active_dataset_id, str(error))
            elif requested is not None:
                return await self._scope_reply(conversation, active_dataset_id, "Worksheet selection is available for Excel workbooks.")
            effective_message = message
            bindings = dict(conv_ctx.get("fiscal_calendar") or {})
            pinned = dict(conv_ctx.get("pinned") or {})
            if sheets and sheets != last.get("worksheets"):
                pinned = {}
            resolved_plan = None
            if pending and not reset:
                clicked = (context or {}).get("clarification")
                if not isinstance(clicked, dict):
                    clicked = {}
                if clicked.get("question_id") and clicked["question_id"] != pending["question_id"]:
                    return await self._scope_reply(conversation, active_dataset_id, pending["reason"], pending)
                effective_message = pending["original_user_message"]
                bindings = dict(pending.get("scope_bindings") or {})
                if pending.get("scope_key"):
                    choices = [o["value"] for o in pending.get("options", [])]
                    answer = str(clicked.get("answer", message)).strip()
                    if not clicked and answer.isdecimal() and len(answer) <= 6 and choices and 1 <= int(answer) <= len(choices):
                        answer = choices[int(answer) - 1]
                    matches = [c for c in choices if c.casefold() == answer.casefold()]
                    value = answer if answer in choices else None
                    if value is None and len(matches) == 1:
                        value = matches[0]
                    if pending["scope_key"] == "year" and re.fullmatch(r"[1-9]\d{3}", answer):
                        value = answer
                    if pending["scope_key"] == "fiscal_year" and re.fullmatch(r"(?:FY\s*)?([1-9]\d{3}(?:[-/]\d{2,4})?)", answer, re.I):
                        value = re.sub(r"^FY\s*", "", answer, flags=re.I)
                    if value is None:
                        return await self._scope_reply(conversation, active_dataset_id, pending["reason"], pending)
                    key = pending["scope_key"]
                    if key == "fiscal_start" and value in PATTERNS:
                        key = "retail_pattern"
                    bindings[key] = value
                    resolved_plan = pending.get("plan") or None
                else:
                    resolved = self._resolve_pending_clarification(
                        pending=pending, user_message=message, request_context=context, dataset_record=dataset_record,
                    )
                    if resolved is None:
                        return await self._scope_reply(conversation, active_dataset_id, pending["reason"], pending)
                    pinned[resolved["param_key"]] = resolved["value"]
                    resolved_plan = resolved["plan"]

            schema = dataset_record.schema_info or {}
            retail_years = {year for c in condition_leaves(last.get("filters", [])) if "retail_anchor" in c
                            for year in (c["retail_anchor"], c.get("retail_last_anchor", c["retail_anchor"]))}
            if len(retail_years) == 1:
                bindings["active_retail_anchor"] = retail_years.pop()
            fiscal_years = {year for c in condition_leaves(last.get("filters", [])) if c.get("fiscal_start_month")
                            for year in (c["fiscal_year_start"], c.get("fiscal_year_end", c["fiscal_year_start"]))}
            if len(fiscal_years) == 1:
                bindings["active_fiscal_year"] = fiscal_years.pop()
                bindings["fiscal_active"] = True
            elif fiscal_years:
                bindings["fiscal_active"] = True
            if "year" not in bindings:
                # A follow-up quarter or month can reuse a previously confirmed calendar year.
                prior_years = {
                    (c["start"][:4], (datetime.fromisoformat(c["end"]) - timedelta(days=1)).strftime("%Y"))
                    for c in condition_leaves(last.get("filters", [])) if c["op"] == "date" and not c.get("fiscal_start_month") and not c.get("retail_pattern") and c.get("start") and c.get("end")
                }
                if len(prior_years) == 1:
                    first, final = prior_years.pop()
                    if first == final:
                        bindings["year"] = first
            plan = resolved_plan
            loaded = None
            conditions = []
            matched_rows = None
            disk = None
            planner_note: str | None = None
            force_llm = False
            # "with my fraud model" names a model, not a row filter: keep it away from the filter parser.
            scope_message, model_ref = self._split_model_reference(effective_message)
            try:
                if not effective_message.strip() and sheets:
                    edit = ScopeEdit("", [], [])
                elif effective_message.strip() in starter_plans(schema):
                    edit = ScopeEdit(effective_message.strip(), [], [])
                else:
                    try:
                        edit = parse_scope(scope_message, schema, today=datetime.now(timezone.utc).date(), bindings=bindings)
                    except ScopeQuestion as error:
                        # The rule parser could not read the restriction; the validated
                        # planner gets the whole message instead of the user getting a dead end.
                        if plan is not None or not self._llm_can_recover(error):
                            raise
                        edit = ScopeEdit(effective_message.strip(), [], [])
                        force_llm = True
                if not edit.question and not edit.clear and not last and conv_ctx.get("turn_history") and plan is None:
                    raise ScopeQuestion("This older conversation has no saved calculation or filters. Restate the calculation with its conditions once to continue.")
                if plan is None and model_ref is not None:
                    plan = [{"operator": "ml_score", "params": {"model_hint": model_ref}}]
                if plan is None:
                    if edit.question and not force_llm:
                        plan = self._select_plan(edit.question, dataset_record=dataset_record)
                    elif edit.question:
                        plan = []
                    else:
                        plan = last.get("steps") or [{"operator": "dataset_overview", "params": {}}]
                        if edit.clear and any(s["operator"].startswith("ml_") for s in plan):
                            plan = [{"operator": "dataset_overview", "params": {}}]
                if not plan and edit.question:
                    plan, planner_note = await self._plan_with_llm(edit, effective_message, schema, last)
                if not plan:
                    raise ScopeQuestion("I couldn't match the complete question to a supported calculation. Specify the measurement or comparison and its conditions.")
                if sheets and sheets != last.get("worksheets") and last:
                    check_carried_columns(schema, last, plan, edit)
                if planner_note is None:
                    # Rule plans bind keywords, so residual qualifiers must be refused here.
                    # LLM plans report unbound restrictions themselves (see app/planning).
                    try:
                        verify_measurement(edit.question, schema, plan)
                    except ScopeQuestion as error:
                        if resolved_plan is not None or not edit.question or not self._llm_can_recover(error):
                            raise
                        plan, planner_note = await self._plan_with_llm(edit, effective_message, schema, last)
                        if not plan:
                            raise
                if edit.retail_group:
                    if len(plan) != 1 or plan[0]["operator"] != "measure_aggregate":
                        raise ScopeQuestion("Name a numeric measure, for example ‘sum Sales by retail period using Posted’.")
                    retail = RetailCalendar.configured(bindings)
                    choices = [c["name"] for c in schema.get("columns", []) if c.get("inferred_type") in {"date", "datetime"}
                               and not c.get("is_potential_pii")]
                    axis = edit.group_date or bindings.get("retail_axis")
                    if not axis:
                        def axes(conditions):
                            return {c.get("column") for c in condition_leaves(conditions)
                                    if c["op"] == "date" and c.get("column") in choices}
                        restricted = axes(edit.conditions) or axes(last.get("filters", [])) or {
                            s["params"]["time_column"] for s in last.get("steps", [])
                            if s.get("params", {}).get("time_column") in choices}
                        if len(restricted) == 1:
                            axis = restricted.pop()
                        elif len(choices) == 1:
                            axis = choices[0]
                        elif choices:
                            raise ScopeQuestion("Which date column defines these retail groups?", key="retail_axis", choices=tuple(choices))
                    if axis not in choices:
                        raise ScopeQuestion("Choose a recognized date column for the retail groups.")
                    plan = [{"operator": "measure_aggregate", "params": {**plan[0]["params"], "time_column": axis,
                             "retail_unit": edit.retail_group, "retail_calendar": asdict(retail)}}]
                elif any(step.get("params", {}).get("retail_unit") for step in plan):
                    spec = asdict(RetailCalendar.configured(bindings))
                    plan = [{**s, "params": {**s["params"], "retail_calendar": spec}}
                            if s.get("params", {}).get("retail_unit") else s for s in plan]
                plan, pending_new = self._maybe_clarify_plan(
                    plan=plan, user_message=edit.question, dataset_record=dataset_record, pinned=pinned,
                )
                if pending_new:
                    pending_new.update(original_user_message=effective_message, scope_bindings=bindings, dataset_version=version, worksheets=sheets)
                    await self._save_scope_context(conversation, {**conv_ctx, "pending_clarification": pending_new}, dataset_id=active_dataset_id)
                    return await self._scope_reply(conversation, active_dataset_id, pending_new["reason"], pending_new)
                expanded = self._expand_drivers(plan, effective_message, dataset_record)
                if expanded is not plan and planner_note:
                    dims = ", ".join(str(s["params"]["group_by"]) for s in expanded)
                    planner_note = re.sub(r", change by [^.;]+", f", change compared by {dims}", planner_note, count=1)
                plan = expanded
                operator_names = [step["operator"] for step in plan]
                scoring = "ml_score" in operator_names
                if scoring and sheets and sheets != [catalog[0]["name"]]:
                    raise ScopeQuestion("Scoring uses the dataset's first worksheet. Save the selected worksheet as its "
                                        "own dataset to score it.")
                gate = await self.subscription.assert_can_run(
                    self.user_id, action=UsageAction.TALK_PREVIEW, operators=operator_names,
                )
                if sheets and any(s["operator"] == "ml_train_baseline" for s in plan) and self._detect_ml_mode(effective_message) == "automl" and sheets != [catalog[0]["name"]]:
                    raise ScopeQuestion("Full AutoML jobs use the uploaded dataset’s first worksheet. Use a quick baseline to train on the selected worksheets.")
                if not scoring and (sheets or await asyncio.to_thread(needs_scan, dataset_record)):
                    try:
                        disk = await asyncio.to_thread(prepare_scan, dataset_record, edit, last.get("filters", []), plan, bindings)
                    except ScopeQuestion as error:
                        if (planner_note is not None or resolved_plan is not None or not edit.question
                                or not self._llm_can_recover(error)):
                            raise
                        replan, planner_note = await self._plan_with_llm(edit, effective_message, schema, last)
                        if not replan:
                            raise
                        plan = replan
                        disk = await asyncio.to_thread(prepare_scan, dataset_record, edit, last.get("filters", []), plan, bindings)
                    loaded, conditions = disk.loaded, disk.filters
                    matched_rows = disk.rows if conditions else None
                elif edit.conditions or (last.get("filters") and not edit.clear):
                    loaded = await (self._load_for_scoring(active_dataset_id) if scoring else
                                    self.dataset_loader.load_dataset(active_dataset_id, owner_id=self.user_id))
                    if version and version != loaded.version_hash:
                        raise ScopeQuestion("The dataset changed while planning. Please retry on its current version.")
                    try:
                        conditions = await asyncio.to_thread(bind_conditions, edit, last.get("filters", []), loaded, plan, bindings)
                    except ScopeQuestion as error:
                        # A rule-detected value that matches no column (e.g. 'the West Branch'):
                        # let the planner review the detected filters, then bind its typed ones.
                        if (planner_note is not None or resolved_plan is not None or not edit.question
                                or not self._llm_can_recover(error)):
                            raise
                        replan, planner_note = await self._plan_with_llm(edit, effective_message, schema, last)
                        if not replan:
                            raise
                        plan = replan
                        conditions = await asyncio.to_thread(bind_conditions, edit, last.get("filters", []), loaded, plan, bindings)
                    if conditions:
                        if any(s["operator"] in {"ml_train_baseline", "ml_task_inference"} for s in plan):
                            raise ScopeQuestion("This modeling workflow needs a versioned filtered dataset. Save the filtered rows as a dataset before training.")
                        loaded = await asyncio.to_thread(scoped_dataset, loaded, conditions)
                        matched_rows = len(loaded.df)
                    elif not scoring:
                        loaded = None
                if scoring and loaded is None:
                    loaded = await self._load_for_scoring(active_dataset_id)
            except ScopeQuestion as error:
                pending_new = None
                if error.key and (error.choices or error.key in {"year", "fiscal_year"}):
                    pending_new = {
                        "dataset_id": str(active_dataset_id), "dataset_version": version,
                        "question_id": uuid4().hex, "scope_key": error.key, "param_key": error.key,
                        "reason": str(error), "options": [{"label": c, "value": c} for c in error.choices],
                        "original_user_message": effective_message, "scope_bindings": bindings, "plan": plan, "worksheets": sheets,
                    }
                await self._save_scope_context(conversation, {**conv_ctx, "pending_clarification": pending_new}, dataset_id=active_dataset_id)
                return await self._scope_reply(conversation, active_dataset_id, str(error), pending_new)

            plan = self.subscription.filter_plan_for_free(
                plan,
                preview_only=bool(gate.get("preview_only")),
            )

            ml_mode = self._detect_ml_mode(effective_message)
            target_col = (
                str(pinned.get("target_column") or "").strip()
                or self._extract_target_column(effective_message, dataset_record=dataset_record)
            )

            if (ml_mode == "automl" and target_col and background_tasks is not None
                    and any(step.get("operator") == "ml_train_baseline" for step in plan)):
                await self.subscription.assert_can_run(
                    self.user_id,
                    action=UsageAction.ML_TRAIN,
                )
                jobs = JobService(self._session)
                job = await jobs.create(
                    owner_id=self.user_id,
                    job_type=JobType.ML_TRAINING,
                    dataset_id=active_dataset_id,
                    payload={
                        "target_column": target_col,
                        "mode": "automl",
                        "source": "chat",
                        "sample_rows": settings.ml.automl_training_sample_rows,
                    },
                    status_message="Queued AutoML from chat",
                )
                enqueue_ml_training(
                    background_tasks=background_tasks,
                    job_id=job.id,
                    owner_id=self.user_id,
                    dataset_id=active_dataset_id,
                    target_column=target_col,
                    sample_rows=settings.ml.automl_training_sample_rows,
                    max_models=settings.ml.automl_max_models,
                    optimize=True,
                )
                content = (
                    f"I've started a **full AutoML job** to train and tune models predicting **{target_col}**.\n\n"
                    f"- **Job ID:** `{job.id}`\n"
                    f"- **Sample size:** up to {settings.ml.automl_training_sample_rows:,} representative rows "
                    f"(scales to multi-million row datasets via reservoir sampling)\n"
                    f"- **Includes:** model comparison, cross-validation, and hyperparameter tuning\n\n"
                    f"Track progress under **Jobs** or poll `/api/v1/ml/jobs/{job.id}`."
                )
                await self._save_scope_context(conversation, {
                    **conv_ctx, "pending_clarification": None, "pinned": pinned,
                    "analysis_scope": {"dataset_id": str(active_dataset_id), "dataset_version": version,
                                       "steps": plan, "filters": [], "matched_rows": None, "worksheets": sheets},
                }, dataset_id=active_dataset_id)
                assistant_msg = await self.repo.add_message(
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT.value,
                    content=content,
                    agent_actions=[{"type": "ml_job", "job_id": str(job.id), "target_column": target_col}],
                )
                return ChatResponse(
                    conversation_id=conversation.id,
                    message_id=assistant_msg.id,
                    content=content,
                    role=MessageRole.ASSISTANT,
                    agent_actions=[{"type": "ml_job", "job_id": str(job.id)}],
                    suggestions=["Show feature importance when the job completes", "Run a quick baseline model now"],
                    metadata={"dataset_id": str(active_dataset_id), "ml_job_id": str(job.id)},
                )

            sample_rows = 200_000
            if conditions or "measure_aggregate" in operator_names or "ml_score" in operator_names:
                sample_rows = None
            elif {"ml_train_baseline", "ml_task_inference"}.intersection(operator_names):
                sample_rows = settings.ml.chat_training_sample_rows
            if disk:
                sample_rows = len(loaded.df) if disk.sampled_ops else None
            if loaded is None:
                loaded = await self.dataset_loader.load_dataset(active_dataset_id, owner_id=self.user_id, sample_rows=sample_rows)
            if version != loaded.version_hash:
                return await self._scope_reply(conversation, active_dataset_id, "The dataset version changed during this turn. Please retry.")
            results = await self.executor.run_plan(
                dataset_id=active_dataset_id,
                plan=plan if matched_rows != 0 else [{"operator": "dataset_overview", "params": {}}],
                owner_id=self.user_id,
                sample_rows=sample_rows,
                loaded=loaded,
                row_scope=conditions,
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
            scope = {"dataset_id": str(active_dataset_id), "dataset_version": version, "steps": plan,
                     "filters": conditions, "matched_rows": matched_rows, "analyzed_rows": disk.rows if disk and not disk.sampled_ops else len(loaded.df)}
            if conditions:
                content = (f"Applied filters: {describe_scope(conditions)}. Matched {matched_rows:,} rows.\n\n" +
                           (content if matched_rows else "No rows match these conditions."))
            elif edit.clear or edit.remove:
                content = "All rows are in scope.\n\n" + content
            for result in results:
                if result.summary.get("undated_rows"):
                    content += f"\n\nRetail grouping excluded {result.summary['undated_rows']:,} rows without a valid date."
            if disk:
                scope.update(execution="batched", scanned_matching_rows=disk.rows, sampled_operators=disk.sampled_ops)
                if disk.sampled_ops:
                    content = (f"Sample-based analysis ({', '.join(disk.sampled_ops)}): {len(loaded.df):,} rows sampled "
                               f"from {disk.rows:,} matching rows.\n\n" + content)
                if any(r.summary.get("groups_truncated") for r in results):
                    content += "\n\nThe evidence table is limited to 1,000 groups."
            if not disk and not conditions and dataset_record.row_count is not None and len(loaded.df) < dataset_record.row_count:
                content += f"\n\nCompute input: {len(loaded.df):,} sampled rows out of {dataset_record.row_count:,}."
            if planner_note:
                scope["planned_by"] = "llm"
                content = planner_note + "\n\n" + content
            if sheets:
                scope.update(worksheets=sheets, worksheet_column=schema["worksheet_column"],
                             column_types={c["name"]: type_family(c["inferred_type"]) for c in schema["columns"]})
                note = "Worksheets: " + ", ".join(sheets) + "."
                if len(sheets) > 1:
                    note += f" Rows are combined by column name. Group by {schema['worksheet_column']} to compare worksheets."
                    layouts = [{c["name"] for c in entry["columns"]} for entry in catalog if entry["name"] in sheets]
                    if any(layout != layouts[0] for layout in layouts[1:]):
                        note += " Columns absent from a worksheet are treated as blank."
                content = note + "\n\n" + content
            state = self._turn_memory(
                {**conv_ctx, "pinned": pinned, "pending_clarification": None, "analysis_scope": scope,
                 "fiscal_calendar": {k: bindings[k] for k in ("fiscal_start", "fiscal_label", *SETTING_KEYS) if k in bindings}},
                user_message=effective_message, operators=operator_names, dataset_record=dataset_record,
            )
            await self._save_scope_context(conversation, state, dataset_id=active_dataset_id)
            used = {r.operator_name for r in results}
            suggestions = [
                question for question, step in starter_plans(dataset_record.schema_info).items()
                if step["operator"] not in used and (not gate.get("preview_only") or step["operator"] in PREVIEW_OPERATORS)
            ][:3]

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
                    "analysis_scope": scope,
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

        # Hosted chat is scoped to datasets; avoid open-ended, unmetered LLM calls.
        if self.subscription.is_enforcement_enabled():
            content = "Upload or select a dataset to start talking to your data."
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
                suggestions=["Upload a dataset", "Open Datasets"],
                metadata={"no_dataset": True},
            )

        # Self-hosted mode can use the general LLM assistant without a dataset.
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

    _WHY_RE = re.compile(r"\b(?:why|drove|drives|driver|drivers|caused|cause|explain|explains|behind|reason)\b", re.I)

    def _expand_drivers(self, plan: list[dict[str, Any]], message: str, dataset_record: Any) -> list[dict[str, Any]]:
        """'Why did sales drop': check every plausible dimension, not only the one the planner chose.

        The planner names one group_by and its choice varies between runs; the answer must not. Code
        compares the same two periods by up to four low-cardinality dimensions and the answer leads
        with the one where the change is most concentrated.
        """
        steps = [s for s in plan if s.get("operator") == "period_compare"]
        if not steps or len(steps) != len(plan) or not self._WHY_RE.search(message or ""):
            return plan
        base = dict(steps[0].get("params") or {})
        if base.get("aggregation", "sum") not in {"sum", "count"}:
            return plan  # contributions add up only for additive measures
        used = [str(s["params"]["group_by"]) for s in steps if (s.get("params") or {}).get("group_by")]
        by_name = {str(c.get("name")): c for c in self._schema_columns(dataset_record)}
        candidates = [
            c for c in self._column_candidates(dataset_record, kind="categorical")
            if 2 <= int((by_name.get(c) or {}).get("unique_count") or 0) <= 20
            and not (by_name.get(c) or {}).get("is_potential_pii")
        ]
        candidates.sort(key=lambda c: int((by_name.get(c) or {}).get("unique_count") or 0))
        dims = list(dict.fromkeys(used + candidates))[:6]
        if len(dims) < 2:
            return plan
        return [{"operator": "period_compare", "params": {**base, "group_by": d}} for d in dims]

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
        pinned_target = str(pinned.get("target_column") or "").strip() or None

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

            if op == "ml_train_baseline":
                target = str(params.get("target_column") or "").strip() or None
                if target and target in schema_set:
                    params["target_column"] = target
                elif pinned_target and pinned_target in schema_set:
                    params["target_column"] = pinned_target
                else:
                    mentioned = [
                        c
                        for c in schema_cols
                        if self._message_mentions_column(user_message, c)
                    ]
                    if len(mentioned) == 1:
                        params["target_column"] = mentioned[0]
                    else:
                        numeric_or_cat = self._column_candidates(dataset_record, kind="categorical")[:6]
                        numeric_cols = self._column_candidates(dataset_record, kind="numeric")[:6]
                        options = list(dict.fromkeys(numeric_or_cat + numeric_cols + schema_cols[:8]))[:8]
                        if len(options) >= 2:
                            pending = self._build_pending_clarification(
                                dataset_id=str(getattr(dataset_record, "id", "") or ""),
                                original_user_message=user_message,
                                operator=op,
                                param_key="target_column",
                                options=options,
                                reason="I need to know which column to predict (the target variable).",
                                plan_template=plan_out
                                + [{"operator": op, "params": params}]
                                + [s for s in plan[len(plan_out) + 1 :] if isinstance(s, dict)],
                            )
                            return plan, pending

            if op == "ml_score":
                from app.services.ml_model_choice import choose_model

                hint = str(params.pop("model_hint", "") or "")
                outcome = choose_model(self.user_id, columns=schema_cols,
                                       requested=str(params.get("model_name") or "").strip() or None,
                                       message=f"{user_message} {hint}".strip())
                choice = outcome.choice
                if choice is None and outcome.options:
                    # A model chosen earlier in this conversation settles a tie between fitting models.
                    choice = next((m for m in outcome.options if m.name == pinned.get("model_name")), None)
                if choice is None and outcome.options:
                    pending = self._build_pending_clarification(
                        dataset_id=str(getattr(dataset_record, "id", "") or ""),
                        original_user_message=user_message,
                        operator=op,
                        param_key="model_name",
                        options=[m.name for m in outcome.options],
                        labels={m.name: m.label for m in outcome.options},
                        reason=outcome.reason,
                        plan_template=plan_out + [{"operator": op, "params": params}] + [
                            s for s in plan[len(plan_out) + 1 :] if isinstance(s, dict)
                        ],
                    )
                    return plan, pending
                if choice is None:
                    raise ScopeQuestion(outcome.reason)
                params.update(model_name=choice.name, model_version=choice.version,
                              model_fingerprint=choice.fingerprint)

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
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        from uuid import uuid4

        opts = [{"value": str(o), "label": str((labels or {}).get(o) or o)} for o in options if o]
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
        if answer_raw is None and msg.isdecimal() and len(msg) <= 6:
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
            chosen = option_map.get(re.sub(r"^(?:please )?use\s+", "", ans.lower()))

        if chosen is None:
            return None

        if param_key == "model_name":
            # Re-bound against the owner's registry when the plan runs; nothing else to check here.
            pass
        else:
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

    _SCORE_RE = re.compile(
        r"\b(?:re)?score\s+(?:this|these|that|the|every|all|each|it|new|my|our)\b"
        r"|\b(?:apply|run|use)\s+(?:my|our|the|that)\s+(?:[\w-]+\s+){0,3}model\b"
        r"|\bwith\s+(?:my|our|the)\s+(?:saved\s+|trained\s+|existing\s+)?(?:[\w-]+\s+){0,3}model\b"
        r"|\bbatch\s+(?:scor|predict)",
        re.I,
    )

    _MODEL_REF_RE = re.compile(
        r"\b(?:with|using|via|from)\s+(?:my|our|the|that|a)\s+(?:[\w-]+\s+){0,3}?model\b"
        r"|\b(?:apply|run|use)\s+(?:my|our|the|that)\s+(?:[\w-]+\s+){0,3}?model\b(?:\s+(?:on|to|for)\b)?",
        re.I,
    )

    def _split_model_reference(self, message: str) -> tuple[str, str | None]:
        """(message for the filter parser, the words naming a model) when this is a scoring request."""
        if not self._is_score_intent(message):
            return message, None
        refs: list[str] = []

        def cut(match: re.Match) -> str:
            refs.append(match[0])
            return "score" if re.match(r"(?:apply|run|use)\b", match[0], re.I) else ""

        text = self._MODEL_REF_RE.sub(cut, message)
        try:
            from app.ml.model_registry import get_model_registry

            for name in sorted((m["name"] for m in get_model_registry(self.user_id).list_models()), key=len, reverse=True):
                pattern = rf"\b(?:with|using)?\s*(?<![\w-]){re.escape(name)}(?![\w-])"
                if re.search(pattern, text, re.I):
                    refs.append(name)
                    text = re.sub(pattern, "", text, flags=re.I)
        except Exception:
            pass
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"^score\s+score\b", "score", text, flags=re.I)
        return text or "score this file", " ".join(refs)

    def _is_score_intent(self, user_message: str) -> bool:
        """Applying a saved model to this data, as opposed to training a new one."""
        msg = user_message or ""
        if re.search(r"\b(?:train|retrain|build|fit|create|tune)\b", msg, re.I):
            return False
        return bool(self._SCORE_RE.search(msg))

    def _is_ml_intent(self, user_message: str) -> bool:
        msg = (user_message or "").lower()
        keys = [
            "train a model",
            "train model",
            "machine learning",
            "automl",
            "hyperparameter",
            "hyper parameter",
            "predict ",
            "prediction model",
            "classification model",
            "regression model",
            "feature importance",
            "xgboost",
            "random forest",
            " tune ",
            "tuning",
            "build a model",
            "model to predict",
        ]
        return any(k in msg for k in keys) or (
            any(k in msg for k in [" train ", " model ", " ml "])
            and any(k in msg for k in ["predict", "target", "churn", "classify", "regression", "forecast"])
        )

    def _detect_ml_mode(self, user_message: str) -> str | None:
        if not self._is_ml_intent(user_message):
            return None
        msg = (user_message or "").lower()
        if any(
            k in msg
            for k in [
                "automl",
                "hyperparameter",
                "hyper parameter",
                "full train",
                "compare models",
                "tune ",
                "tuning",
                "optimize model",
            ]
        ):
            return "automl"
        return "baseline"

    def _extract_target_column(self, user_message: str, *, dataset_record: Any | None) -> str | None:
        msg_raw = user_message or ""
        msg = msg_raw.lower()
        for col in self._schema_columns(dataset_record):
            name = str(col.get("name") or "")
            if not name:
                continue
            if self._message_mentions_column(msg_raw, name):
                return name
        m = re.search(r"(?:predict|forecast|target|for)\s+([a-zA-Z_][a-zA-Z0-9_]*)", msg)
        if m:
            candidate = m.group(1)
            for col in self._schema_columns(dataset_record):
                if str(col.get("name") or "").lower() == candidate.lower():
                    return str(col.get("name"))
        return None

    async def _load_for_scoring(self, dataset_id: UUID):
        """Every row, up to the scoring cap; a larger file is refused rather than sampled."""
        from app.services.ml_scoring_service import too_large_message

        cap = int(settings.ml.max_scoring_rows)
        try:
            return await self.dataset_loader.load_dataset(dataset_id, owner_id=self.user_id, max_rows=cap)
        except Exception as e:
            if "too large" in str(e).lower():
                raise ScopeQuestion(too_large_message(cap) + ". The scoring API takes the same limit.") from e
            raise

    def _llm_can_recover(self, error: ScopeQuestion) -> bool:
        """A rule-stage refusal the planner may resolve: not a pending choice, and a model is configured."""
        from app.planning.llm_planner import planner_enabled

        if error.key or error.choices or not planner_enabled() or self.llm is None:
            return False
        try:
            return bool(getattr(self.llm, "is_available", True))
        except Exception:
            return False

    def _model_menu(self, schema: dict[str, Any]) -> list[dict[str, Any]]:
        """The owner's saved models, so the planner can name one; code binds the choice later."""
        from app.services.ml_model_choice import list_scoring_models

        try:
            cols = [str(c.get("name")) for c in (schema or {}).get("columns") or [] if isinstance(c, dict) and c.get("name")]
            return [{"name": m.name, "predicts": m.target, "task": m.task, "fits_this_file": m.usable}
                    for m in list_scoring_models(self.user_id, cols)[:20]]
        except Exception:
            return []

    async def _plan_with_llm(
        self, edit: ScopeEdit, original_message: str, schema: dict[str, Any], last: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Plan a question the rule planner could not bind, via the validated LLM planner.

        The model also reviews the filters the rule parser detected (dropping
        misreadings such as "brings in the most money") and may add typed filters
        with explicit columns; ``edit.conditions`` is updated in place and then
        bound against the data by the normal scope path.

        Returns (steps, note shown above the answer). Raises ScopeQuestion when the
        model asks for clarification or says the question is out of reach.
        """
        from app.planning.llm_planner import describe_plan, plan_question

        outcome = await plan_question(
            edit.question,
            schema,
            llm=self.llm,
            model=os.getenv("LLM_PLANNER_MODEL") or settings.llm_planner_model or None,
            context={
                "previous_steps": (last or {}).get("steps") or [],
                "active_filters": (last or {}).get("filters") or [],
                "detected_filters": list(edit.conditions),
                "original_message": original_message,
                "today": datetime.now(timezone.utc).date().isoformat(),
                "models": self._model_menu(schema),
            },
        )
        if outcome.status == "plan":
            if outcome.keep_detected is not None:
                dropped = [c for i, c in enumerate(edit.conditions) if i not in set(outcome.keep_detected)]
                edit.conditions = [edit.conditions[i] for i in outcome.keep_detected]
            else:
                dropped = []
            edit.conditions = list(edit.conditions) + list(outcome.filters)
            # Written from the validated plan, never from the model's own wording.
            note = f"Interpreted your question as: {describe_plan(outcome.steps)}."
            # Only worth saying when nothing replaced them: typed filters or period parameters that
            # re-express the same words are shown in the answer itself.
            if dropped and not outcome.filters and not any(s.get("operator") == "period_compare" for s in outcome.steps):
                labels = [_dropped_label(c) for c in dropped]
                labels = [x for x in labels if x]
                if labels:
                    note += " Not treated as filters: " + ", ".join(labels) + "."
            if outcome.notes:
                note += " " + " ".join(outcome.notes)
            return outcome.steps, note
        if outcome.status == "clarify":
            text = outcome.question
            if outcome.options:
                text += " Options: " + ", ".join(outcome.options) + "."
            raise ScopeQuestion(text)
        if outcome.status == "unsupported":
            raise ScopeQuestion(
                "This question is outside what I can calculate reliably on this dataset"
                + (f": {outcome.reason}" if outcome.reason else ".")
            )
        return [], None

    def _select_plan(self, user_message: str, *, dataset_record: Any | None = None) -> list[dict[str, Any]]:
        msg_raw = user_message or ""
        if re.fullmatch(r"(?:count(?: of)?|total|how many|number of) (?:rows|records)", msg_raw.strip(" .!?"), re.I):
            return [{"operator": "dataset_overview", "params": {}}]
        if self._is_score_intent(msg_raw):
            return [{"operator": "ml_score", "params": {}}]
        if dataset_record is not None:
            schema = getattr(dataset_record, "schema_info", None)
            for question, step in starter_plans(schema).items():
                if question.casefold().rstrip("?") == msg_raw.strip().casefold().rstrip("?"):
                    return [step]
            ranking = ranked_average_plan(msg_raw, schema)
            if ranking is not None:
                return ranking
            metric = measure_plan(msg_raw, schema)
            if metric is not None:
                return metric
        msg = msg_raw.lower()

        if self._is_ml_intent(msg_raw):
            target = self._extract_target_column(msg_raw, dataset_record=dataset_record)
            mode = self._detect_ml_mode(msg_raw)
            optimize = mode != "baseline" or any(k in msg for k in ["tune", "tuning", "hyperparameter"])
            if not target:
                return [{"operator": "ml_task_inference", "params": {}}]
            return [
                {
                    "operator": "ml_train_baseline",
                    "params": {
                        "target_column": target,
                        "optimize": optimize,
                    },
                }
            ]

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
        elif any(k in msg for k in ["quarterly", "per quarter"]):
            freq = "Q"
        elif any(k in msg for k in ["yearly", "per year"]):
            freq = "Y"

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

        plan: list[dict[str, Any]] = []
        if any(k in msg for k in ["overview", "summarize", "summary"]):
            plan.append({"operator": "dataset_overview", "params": {}})

        if any(k in msg for k in ["schema", "columns", "dtypes"]):
            plan.append({"operator": "schema_snapshot", "params": {}})
        if any(k in msg for k in ["preview", "head", "sample rows", "show rows"]):
            plan.append({"operator": "preview_rows", "params": {"limit": 25}})
        if re.search(r"\b(?:missing|nulls?|na|n/a|nan)\b", msg):
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

        if any(k in msg for k in ["sentiment", "opinion", "positive", "negative review", "text tone"]):
            plan.append({"operator": "sentiment_scan", "params": {"max_columns": 3}})
        if any(k in msg for k in ["forecast", "predict next", "projection", "future trend"]) and not self._is_ml_intent(msg_raw):
            plan.append({"operator": "forecast_scan", "params": {"horizon": 8, "method": "ensemble"}})
        if any(k in msg for k in ["a/b", "ab test", "a b test", "experiment", "variant", "control group"]):
            plan.append({"operator": "ab_test_scan", "params": {}})
        if any(k in msg for k in ["cohort", "retention", "churn cohort"]):
            plan.append({"operator": "cohort_scan", "params": {"max_periods": 8}})
        if any(k in msg for k in ["funnel", "conversion", "drop-off", "drop off"]):
            plan.append({"operator": "funnel_scan", "params": {}})
        if any(k in msg for k in ["rfm", "customer segment", "customer segments", "monetary value"]):
            plan.append({"operator": "rfm_scan", "params": {}})
        if any(k in msg for k in ["cluster", "clustering", "k-means", "k means", "segments unsupervised"]):
            plan.append({"operator": "clustering_scan", "params": {"n_clusters": 4}})
        if any(k in msg for k in ["chart", "visualize", "visualization", "plot", "graph", "histogram"]):
            plan.append({"operator": "smart_viz_scan", "params": {"max_charts": 3}})
        if any(k in msg for k in ["heatmap", "correlation heatmap", "corr heatmap"]):
            plan.append({"operator": "correlation_heatmap", "params": {}})
        if "distribution" in msg and any(k in msg for k in ["chart", "plot", "histogram", "visualize"]):
            plan.append({"operator": "distribution_chart", "params": {}})
        if any(
            k in msg
            for k in [
                "million rows",
                "full table count",
                "exact row count",
                "duckdb",
                "sql aggregate",
                "without loading",
            ]
        ):
            plan.append({"operator": "duckdb_profile", "params": {}})

        if len(plan) > 4:
            plan = plan[:4]

        if plan:
            return plan
        if dataset_record is None or re.fullmatch(
            r"(?:run |do |perform |start )?(?:an? )?(?:eda|exploratory (?:data )?analysis|data speaks)"
            r"|(?:please )?(?:analy[sz]e|explore|summari[sz]e) (?:this|my|the) (?:dataset|data|file)"
            r"|make (?:the|my) data speak", msg.strip(" .!?")
        ):
            return eda_p0_plan()
        return []

    def _format_grounded_response(
        self,
        *,
        user_message: str,
        dataset_id: UUID,
        sample_rows: int | None,
        results: list[Any],
    ) -> str:
        """Format compute results into natural, conversational language."""

        def _first_table_preview(res: Any) -> tuple[list[dict[str, Any]], Any | None]:
            for a in getattr(res, "artifacts", []):
                if getattr(getattr(a, "artifact_type", None), "value", "") != "table":
                    continue
                if str(getattr(a, "name", "")).endswith(":overall"):
                    continue  # the grouped table is the evidence; the overall row is in the summary
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

        sections: list[str] = []

        # Several period comparisons of the same periods by different dimensions ("why did X drop"):
        # lead with the dimension where the change is most concentrated, summarise the rest.
        driver_top, driver_skip, driver_note = None, set(), ""
        compared = [r for r in results if getattr(r, "operator_name", "") == "period_compare"
                    and (getattr(r, "summary", {}) or {}).get("group_by")]
        if len(compared) >= 2:
            def _concentration(res: Any) -> tuple[float, float, float, dict[str, Any]] | None:
                # A group explains a change when its share of the change far exceeds its usual share of
                # the measure: South being 22% of sales but 130% of the drop beats Online being 62% of
                # sales and 130% of the drop. Rank by that excess.
                rows, _ = _first_table_preview(res)
                overall = (res.summary or {}).get("overall_change") or 0
                base = (res.summary or {}).get("overall_previous") or 0
                best = None
                for row in rows:
                    share, change, prev = row.get("contribution_pct"), row.get("change"), row.get("previous")
                    if not (isinstance(share, (int, float)) and share == share and isinstance(change, (int, float))
                            and isinstance(prev, (int, float)) and overall and base and (change > 0) == (overall > 0)):
                        continue
                    weight = prev / base * 100
                    if best is None or share - weight > best[0]:
                        best = (share - weight, float(share), weight, row)
                return best

            ranked = sorted(((c, res) for res in compared if (c := _concentration(res))), key=lambda x: -x[0][0])
            if ranked:
                (_, share, weight, row), top = ranked[0]
                driver_top = id(top)
                driver_skip = {id(res) for res in compared if id(res) != driver_top}
                pct = row.get("change_pct")
                driver_note = (f"**Where the change is concentrated:** {top.summary['group_by']} = **{row.get('group')}** "
                               f"accounts for {share:.0f}% of the total change while it was {weight:.0f}% of the "
                               f"comparison period" + (f" ({pct:+.1f}% in that group)" if isinstance(pct, (int, float)) else "")
                               + ".")
                others = []
                for (_, o_share, o_weight, o_row), res in ranked[1:]:
                    others.append(f"{res.summary['group_by']} ({o_row.get('group')}: {o_share:.0f}% of the change, "
                                  f"{o_weight:.0f}% of the base)")
                if others:
                    driver_note += " Also checked, less concentrated: " + ", ".join(others) + "."

        for r in results:
            op = str(getattr(r, "operator_name", "") or "")
            if id(r) in driver_skip:
                continue
            preview_rows, _ = _first_table_preview(r)
            summary = getattr(r, "summary", {}) or {}

            if op == "period_compare" and (preview_rows or summary.get("overall_current") is not None):
                agg = summary.get("aggregation") or "sum"
                what = ("**Number of rows**" if agg == "count" else
                        f"{'Average' if agg == 'mean' else 'Total'} **{summary.get('value_column')}**")
                def _last(day):
                    try:
                        return (date.fromisoformat(str(day)[:10]) - timedelta(days=1)).isoformat()
                    except ValueError:
                        return str(day)
                cur = f"{summary['current_start']} to {_last(summary['current_end'])}"
                prev = f"{summary['previous_start']} to {_last(summary['previous_end'])}"
                def _chg(change, pct, sep=" ("):
                    if change is None:
                        return "n/a"
                    text = ("+" if change > 0 else "") + _fmt_number(change)
                    if pct is None:
                        return text
                    return text + f"{sep}{'+' if pct > 0 else ''}{pct:.2f}%" + (")" if sep == " (" else "")
                sections.append(
                    f"{what}, {cur} vs {prev}: {_fmt_number(summary.get('overall_current'))} vs "
                    f"{_fmt_number(summary.get('overall_previous'))}, change "
                    f"{_chg(summary.get('overall_change'), summary.get('overall_change_pct'))}.")
                if summary.get("aligned_to_data"):
                    sections.append(
                        f"The current period is still in progress and the data runs to {_last(summary['current_end'])}, "
                        f"so both periods are compared over the same number of days.")
                if summary.get("data_ends"):
                    sections.append(f"The data ends on {summary['data_ends']}, before the current period does; "
                                    "days after that have no rows.")
                if not summary.get("overall_previous_rows"):
                    sections.append("No rows fall in the comparison period, so there is no meaningful % change.")
                if not summary.get("overall_current_rows"):
                    sections.append("No rows fall in the current period.")
                if driver_note and id(r) == driver_top:
                    sections.append(driver_note)
                if summary.get("group_by") and preview_rows:
                    items = []
                    for row in preview_rows[:8]:
                        share = row.get("contribution_pct")
                        if isinstance(share, (int, float)) and share == share:
                            share = 0.0 if abs(share) < 0.05 else share
                            tail = f"; {share:.1f}% of the total change"
                        else:
                            tail = ""
                        items.append(f"- {row.get('group')}: {_fmt_number(row.get('previous'))} → "
                                     f"{_fmt_number(row.get('current'))} ({_chg(row.get('change'), row.get('change_pct'), ', ')}{tail})")
                    sections.append(f"Change by **{summary['group_by']}** (largest change first):\n" + "\n".join(items))
                    if any(isinstance(r.get("contribution_pct"), (int, float)) and r["contribution_pct"] < 0 for r in preview_rows):
                        sections.append("Contributions add up to 100% of the total change; groups that moved the other way have negative shares.")
            elif op == "ratio" and preview_rows:
                top = f"**{summary.get('numerator_column')}**"
                if summary.get("subtract_column"):
                    top = f"(**{summary.get('numerator_column')}** - **{summary['subtract_column']}**)"
                label = f"Total {top} / total **{summary.get('denominator_column')}**"
                unit = "%" if summary.get("as_percent") else ""
                def _val(row):
                    v = row.get("value")
                    if v is None or (isinstance(v, float) and v != v):
                        return "n/a (denominator is zero)"
                    return f"{_fmt_number(v)}{unit}"
                if summary.get("group_by"):
                    items = [f"- {row.get('group')}: {_val(row)}" for row in preview_rows[:10]]
                    sections.append(f"{label} by **{summary['group_by']}**:\n" + "\n".join(items))
                else:
                    sections.append(f"{label}: {_val(preview_rows[0])}.")
                if summary.get("rows_excluded_missing"):
                    sections.append(f"{_fmt_number(summary['rows_excluded_missing'])} rows with a missing value in these columns are left out.")
            elif op == "share_of_total" and preview_rows:
                what = f"**{summary['value_column']}**" if summary.get("value_column") else "rows"
                items = [f"- {row.get('group')}: {row['share_pct']:.2f}% ({_fmt_number(row.get('value'))})"
                         if isinstance(row.get("share_pct"), (int, float)) else f"- {row.get('group')}: n/a"
                         for row in preview_rows[:10]]
                sections.append(f"Share of {what} by **{summary['group_by']}** (total {_fmt_number(summary.get('total'))}):\n"
                                + "\n".join(items))
                if summary.get("negative_values"):
                    sections.append("Some groups are negative, so shares do not read as parts of a whole.")
            elif op == "rate" and preview_rows:
                what = f"**{summary.get('column')}** = {summary.get('positive_value')}"
                def _pct(row):
                    rate = row.get("rate")
                    if rate is None:
                        return "n/a (no rows with a value)"
                    return f"{float(rate) * 100:.2f}% ({_fmt_number(row.get('matching'))} of {_fmt_number(row.get('rows'))})"
                if summary.get("group_by"):
                    items = [f"- {row.get('group')}: {_pct(row)}" for row in preview_rows[:10]]
                    sections.append(f"Share of rows where {what}, by **{summary['group_by']}**:\n" + "\n".join(items))
                else:
                    sections.append(f"Share of rows where {what}: {_pct(preview_rows[0])}.")
                if summary.get("rows_missing_value"):
                    sections.append(f"{_fmt_number(summary['rows_missing_value'])} rows with an empty {summary.get('column')} are not counted.")
            elif op == "measure_aggregate" and preview_rows:
                agg = summary.get("aggregation")
                if agg == "count":
                    label = "**Number of rows**"
                elif agg == "distinct_count":
                    label = f"Number of distinct **{summary.get('value_column')}**"
                else:
                    label = f"{str(agg).title()} of **{summary['value_column']}**"
                if summary.get("group_by"):
                    items = [f"- {row.get('group')}: {_fmt_number(row.get('value'))}" for row in preview_rows[:10]]
                    more = "\nShowing 10 groups; the full table is attached below." if len(preview_rows) > 10 else ""
                    sections.append(f"{label} by **{summary['group_by']}**:\n" + "\n".join(items) + more)
                else:
                    sections.append(f"{label}: {_fmt_number(preview_rows[0].get('value'))}.")
            elif op == "dataset_overview" and preview_rows:
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

            elif op == "segment_summary" and (preview_rows or summary.get("sort_by") == "value_mean"):
                group_by = summary.get("group_by") if isinstance(summary, dict) else None
                if summary.get("sort_by") == "value_mean":
                    metric = summary.get("value_column")
                    lines = [
                        f"  • **{row['group'] if row['group'] is not None else '(missing)'}**: {_fmt_number(row['value_mean'])}"
                        for row in preview_rows[:5] if row.get("value_mean") is not None
                    ]
                    sections.append(
                        f"Highest average **{metric}** by **{group_by}** in the analyzed rows:\n" + "\n".join(lines)
                        if lines else f"No group had a usable value for **{metric}** in the analyzed rows."
                    )
                    continue
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
                if summary.get("highest"):
                    unit = "" if summary.get("measure") == "value_sum" else " rows"
                    points = int(summary.get("points") or 0)
                    span = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}.get(freq or "", "period")
                    what = f"total **{value_col}**" if summary.get("measure") == "value_sum" else "the number of rows"
                    desc = (f"{freq_label.capitalize()} trend of {what} by **{time_col}**, {summary['first_label']} to "
                            f"{summary['last_label']} ({points} {span}{'' if points == 1 else 's'})")
                    hi, lo = summary["highest"], summary["lowest"]
                    if points <= 1:
                        items.append(f"- Only one {span} has data, so there is no trend yet: **{hi['label']}**, "
                                     f"{_fmt_number(hi['value'])}{unit}.")
                    else:
                        items.append(f"- Highest: **{hi['label']}** ({_fmt_number(hi['value'])}{unit}); "
                                     f"lowest: **{lo['label']}** ({_fmt_number(lo['value'])}{unit}).")
                    ye = summary.get("year_earlier")
                    if points <= 1:
                        pass
                    elif ye:
                        sign = "+" if ye["change_pct"] >= 0 else ""
                        items.append(f"- Latest, **{summary['last_label']}**: {_fmt_number(summary['last_value'])}{unit}, "
                                     f"vs {_fmt_number(ye['value'])} in {ye['label']} ({sign}{ye['change_pct']:.1f}% year over year).")
                    else:
                        items.append(f"- Latest, **{summary['last_label']}**: {_fmt_number(summary['last_value'])}{unit}.")
                    items.append("The chart and the full table are attached below.")
                else:
                    for row in preview_rows[:5]:
                        if not isinstance(row, dict):
                            continue
                        items.append(f"  \u2022 {row.get('label') or row.get('period')}: count = {_fmt_number(row.get('count'))}, "
                                     f"sum = {_fmt_number(row.get('value_sum'))}")
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

            elif op == "ml_task_inference" and preview_rows:
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    items.append(
                        f"  • **{row.get('column')}** — {row.get('inferred_task')} "
                        f"(score {float(row.get('score') or 0):.2f})"
                    )
                selected = summary.get("selected_target") if isinstance(summary, dict) else None
                if items:
                    lead = (
                        f"Top target candidates (recommended: **{selected}**):" if selected else "Top target candidates:"
                    )
                    sections.append(f"{lead}\n" + "\n".join(items))
                if isinstance(summary, dict) and summary.get("warnings"):
                    sections.append("Warnings: " + "; ".join(str(w) for w in summary.get("warnings")[:3]))

            elif op == "ml_train_baseline":
                best = summary.get("best_model") if isinstance(summary, dict) else None
                target = summary.get("target_column") if isinstance(summary, dict) else None
                task = summary.get("task") if isinstance(summary, dict) else None
                metrics = summary.get("test_metrics") if isinstance(summary, dict) else {}
                cv_mean = summary.get("cv_mean") if isinstance(summary, dict) else None
                registered = summary.get("registered_model") if isinstance(summary, dict) else None
                summary = summary if isinstance(summary, dict) else {}
                best_display = summary.get("best_model_display") or best
                scoring = _METRIC_NAMES.get(str(summary.get("cv_scoring") or ""), summary.get("cv_scoring") or "score")
                task_text = {"binary_classification": "two possible outcomes",
                             "multiclass_classification": "several categories",
                             "regression": "a number"}.get(str(task), str(task))
                compared = len([m for m in summary.get("models") or [] if not m.get("is_baseline")])
                lines = [
                    f"I trained and compared {compared or 'several'} models to predict **{target}** ({task_text}).",
                    f"**Best model:** {best_display}",
                ]
                split = summary.get("split") if isinstance(summary.get("split"), dict) else {}
                if split.get("strategy") == "time_ordered":
                    test_period = split.get("test_period") or ["?", "?"]
                    lines.append(
                        f"**Evaluation:** trained on earlier rows, tested on the {int(split.get('test_rows') or 0):,} "
                        f"most recent by **{split.get('time_column')}** ({str(test_period[0])[:10]} to {str(test_period[1])[:10]})."
                    )
                elif split:
                    lines.append(
                        f"**Evaluation:** {int(split.get('test_rows') or 0):,} randomly held-out rows "
                        "never seen during training or model selection."
                    )
                if cv_mean is not None:
                    lines.append(f"**Cross-validated {scoring}** on the training rows: {float(cv_mean):.3f}")
                else:
                    lines.append("**Cross-validation:** could not be computed, so this model is unvalidated.")
                baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else {}
                if baseline.get("cv_mean") is not None:
                    verdict = summary.get("beats_baseline")
                    verdict_text = (
                        "beats it" if verdict else "does **not** meaningfully beat it" if verdict is False else "compared"
                    )
                    lines.append(
                        f"**Naive baseline** (no learning) {scoring}: {float(baseline['cv_mean']):.3f}; the best model {verdict_text}."
                    )
                if isinstance(metrics, dict) and metrics:
                    lines.append(_held_out_line(summary))
                lines.extend(_training_quality_lines(summary))
                leaky = summary.get("excluded_leaky_columns") or []
                if leaky:
                    lines.append(
                        "**Excluded because they give away the answer:** "
                        + ", ".join(f"{i.get('column')} ({i.get('reason')})" for i in leaky if isinstance(i, dict))
                    )
                if registered:
                    lines.append(
                        f"Model registered as **{registered}** v{summary.get('registered_version')} "
                        "with its preprocessing, ready to score new rows."
                    )
                importance = summary.get("feature_importance") or {}
                if isinstance(importance, dict) and importance:
                    top_feats = [(k, v) for k, v in importance.items() if v is not None][:5]
                    feat_lines = [f"- **{k}**: {float(v):.4f}" for k, v in top_feats]
                    lines.append(
                        f"**Most influential columns** (drop in {scoring} when shuffled on held-out rows):\n"
                        + "\n".join(feat_lines)
                    )
                shown = ("Excluded columns that give away", "Predictions use a cut-off", "The model is clearly worse",
                         "The rarer class is under", "At least one class is under", "Evaluated on the most recent rows")
                said = ("was taken as the moment a prediction is made",)
                for w in [w for w in (summary.get("warnings") or [])
                          if not str(w).startswith(shown) and not any(s in str(w) for s in said)][:4]:
                    lines.append(f"⚠️ {w}")
                sampled = summary.get("sample_rows")
                if sampled:
                    lines.append(f"Trained on **{int(sampled):,}** representative rows from your dataset.")
                sections.append("\n".join(lines))

            elif op == "ml_score" and summary.get("rows_scored") is not None:
                sections.append("\n".join(_scoring_lines(summary)))

            elif op == "sentiment_scan" and preview_rows:
                items = []
                for row in preview_rows[:5]:
                    if not isinstance(row, dict):
                        continue
                    items.append(
                        f"  • **{row.get('column')}**: "
                        f"{float(row.get('positive_pct') or 0):.1f}% positive, "
                        f"{float(row.get('negative_pct') or 0):.1f}% negative "
                        f"(avg polarity {float(row.get('avg_score') or 0):.3f})"
                    )
                if items:
                    sections.append("Sentiment scan:\n" + "\n".join(items))

            elif op == "forecast_scan":
                time_col = summary.get("time_column") if isinstance(summary, dict) else None
                value_col = summary.get("value_column") if isinstance(summary, dict) else None
                method = summary.get("method") if isinstance(summary, dict) else None
                horizon = summary.get("horizon") if isinstance(summary, dict) else None
                lead = f"Forecast for **{value_col}** over **{time_col}**"
                if method:
                    lead += f" using **{method}**"
                if horizon:
                    lead += f" ({horizon} periods ahead)"
                items = []
                for row in preview_rows[:6]:
                    if not isinstance(row, dict):
                        continue
                    if row.get("note"):
                        items.append(f"  • {row.get('note')}")
                        continue
                    period = row.get("date") or row.get("period") or row.get("ds")
                    value = row.get("yhat") or row.get("forecast") or row.get("value")
                    if period is not None or value is not None:
                        items.append(f"  • {period}: {_fmt_number(value)}")
                metrics = summary.get("metrics") if isinstance(summary, dict) else {}
                if isinstance(metrics, dict) and metrics:
                    metric_bits = [f"{k}={float(v):.4f}" for k, v in list(metrics.items())[:3] if v is not None]
                    if metric_bits:
                        lead += f"\nBacktest metrics: {', '.join(metric_bits)}"
                if items:
                    sections.append(f"{lead}:\n" + "\n".join(items))
                elif lead:
                    sections.append(lead)

            elif op == "ab_test_scan" and preview_rows:
                row = preview_rows[0] if isinstance(preview_rows[0], dict) else {}
                if row:
                    sig = "significant" if row.get("significant") else "not significant"
                    sections.append(
                        f"A/B test on **{row.get('metric')}**: "
                        f"**{row.get('variant_a')}** vs **{row.get('variant_b')}** — "
                        f"p = {float(row.get('p_value') or 0):.4f} ({sig}), "
                        f"lift {float(row.get('lift_pct') or 0):+.2f}%."
                    )

            elif op == "duckdb_profile" and preview_rows:
                row = preview_rows[0] if isinstance(preview_rows[0], dict) else {}
                engine = summary.get("engine") if isinstance(summary, dict) else row.get("engine")
                count = row.get("row_count") or row.get("rows") or summary.get("row_count")
                cols = row.get("column_count") or row.get("columns") or summary.get("column_count")
                parts = []
                if count is not None:
                    parts.append(f"**{_fmt_number(count)} rows**")
                if cols is not None:
                    parts.append(f"**{_fmt_number(cols)} columns**")
                if parts:
                    label = "Full-dataset profile (DuckDB pushdown)" if engine == "duckdb_pushdown" else "Dataset profile"
                    sections.append(f"{label}: {' across '.join(parts)}.")

            elif op == "cohort_scan" and isinstance(summary, dict):
                sections.append(
                    f"Cohort retention: **{summary.get('n_customers', 'n/a')}** customers across "
                    f"**{summary.get('n_cohorts', 'n/a')}** cohorts. "
                    f"Period-1 retention ~ **{float(summary.get('avg_retention_period_1', 0)) * 100:.1f}%**."
                )

            elif op == "funnel_scan" and preview_rows:
                items = []
                for row in preview_rows[:6]:
                    if not isinstance(row, dict):
                        continue
                    items.append(f"  • **{row.get('step')}**: {_fmt_number(row.get('count'))} ({row.get('overall_rate')}% overall)")
                if items:
                    conv = summary.get("overall_conversion_pct") if isinstance(summary, dict) else None
                    lead = f"Funnel conversion **{conv}%** end-to-end:" if conv is not None else "Funnel steps:"
                    sections.append(f"{lead}\n" + "\n".join(items))

            elif op == "rfm_scan" and isinstance(summary, dict):
                sections.append(
                    f"RFM segmentation: **{summary.get('n_customers', 'n/a')}** customers in "
                    f"**{summary.get('segments', 0)}** segments."
                )

            elif op == "clustering_scan" and isinstance(summary, dict):
                sections.append(
                    f"K-Means clustering (k={summary.get('n_clusters')}) on features: "
                    f"{', '.join('**' + str(f) + '**' for f in (summary.get('features') or [])[:5])}."
                )

            elif op in {"smart_viz_scan", "correlation_heatmap", "distribution_chart"} and isinstance(summary, dict):
                n = summary.get("charts_generated") or summary.get("columns") or summary.get("column")
                sections.append(f"Generated interactive chart(s) — see chart artifacts below. ({n})")

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
    background_tasks: BackgroundTasks,
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
        context=request.context,
        background_tasks=background_tasks,
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
        "worksheets": ((conversation.context or {}).get("pending_clarification") or {}).get("worksheets") or
                      ((conversation.context or {}).get("analysis_scope") or {}).get("worksheets"),
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
