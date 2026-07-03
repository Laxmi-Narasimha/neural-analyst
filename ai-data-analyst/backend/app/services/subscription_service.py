"""Subscription plans, usage metering, and freemium enforcement."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import DeploymentMode, get_settings
from app.core.exceptions import BaseApplicationException, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)

PREVIEW_OPERATORS = frozenset({
    "dataset_overview",
    "schema_snapshot",
    "preview_rows",
})


class SubscriptionPlan(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    BYOK = "byok"


class UsageAction(str, Enum):
    TALK_PREVIEW = "talk_preview"
    COMPUTE_SESSION = "compute_session"
    DATA_SPEAKS = "data_speaks"
    QUALITY_RUN = "quality_run"
    ML_TRAIN = "ml_train"


PLAN_LIMITS: dict[str, dict[str, Any]] = {
    SubscriptionPlan.FREE.value: {
        "talk_preview_sessions": 1,
        "compute_sessions": 0,
        "max_upload_mb": 10,
        "monthly_queries": 100,
        "features": {"preview_ops", "quality_preview"},
    },
    SubscriptionPlan.PRO.value: {
        "talk_preview_sessions": None,
        "compute_sessions": None,
        "max_upload_mb": 1024,
        "monthly_queries": None,
        "features": {"all"},
    },
    SubscriptionPlan.ENTERPRISE.value: {
        "talk_preview_sessions": None,
        "compute_sessions": None,
        "max_upload_mb": 2048,
        "monthly_queries": None,
        "features": {"all", "api", "sso"},
    },
    SubscriptionPlan.BYOK.value: {
        "talk_preview_sessions": None,
        "compute_sessions": None,
        "max_upload_mb": 2048,
        "monthly_queries": None,
        "features": {"all"},
    },
}


def _month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _default_usage_state() -> dict[str, Any]:
    return {
        "month_key": _month_key(),
        "talk_preview_used": 0,
        "compute_sessions_used": 0,
        "monthly_queries": 0,
        "last_action_at": None,
    }


def _default_subscription_settings() -> dict[str, Any]:
    return {
        "plan": SubscriptionPlan.FREE.value,
        "status": "active",
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "period_end": None,
        "usage": _default_usage_state(),
    }


class QuotaExceededError(BaseApplicationException):
    def __init__(self, message: str, *, upgrade_url: str = "/pricing") -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.QUOTA_EXCEEDED,
            recovery_hint="Upgrade to Pro, self-host with your own keys, or wait for the next billing period.",
            http_status_code=402,
        )
        self.upgrade_url = upgrade_url


class SubscriptionService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.settings = get_settings()

    def is_enforcement_enabled(self) -> bool:
        return self.settings.deployment_mode == DeploymentMode.SAAS

    async def get_user_record(self, user_id: UUID):
        from app.models import User

        result = await self.session.execute(
            select(User).where(User.id == user_id, User.is_deleted == False)  # noqa: E712
        )
        return result.scalars().first()

    def get_subscription_state(self, user: Any) -> dict[str, Any]:
        raw = user.settings if isinstance(getattr(user, "settings", None), dict) else {}
        sub = raw.get("subscription")
        if not isinstance(sub, dict):
            sub = _default_subscription_settings()
        merged = deepcopy(_default_subscription_settings())
        merged.update(sub)
        usage = merged.get("usage")
        if not isinstance(usage, dict):
            usage = _default_usage_state()
        if usage.get("month_key") != _month_key():
            usage = _default_usage_state()
        merged["usage"] = usage
        return merged

    async def persist_subscription_state(self, user: Any, state: dict[str, Any]) -> None:
        raw = dict(user.settings or {})
        raw["subscription"] = state
        user.settings = raw
        await self.session.commit()

    def effective_plan(self, user: Any, sub_state: dict[str, Any]) -> str:
        if getattr(user, "is_superuser", False):
            return SubscriptionPlan.ENTERPRISE.value
        plan = str(sub_state.get("plan") or SubscriptionPlan.FREE.value).lower()
        status = str(sub_state.get("status") or "active").lower()
        if plan in {SubscriptionPlan.PRO.value, SubscriptionPlan.ENTERPRISE.value} and status not in {"active", "trialing"}:
            return SubscriptionPlan.FREE.value
        return plan

    def plan_limits(self, plan: str) -> dict[str, Any]:
        return PLAN_LIMITS.get(plan, PLAN_LIMITS[SubscriptionPlan.FREE.value])

    async def get_status(self, user_id: UUID) -> dict[str, Any]:
        user = await self.get_user_record(user_id)
        if not user:
            raise QuotaExceededError("User not found")
        sub = self.get_subscription_state(user)
        plan = self.effective_plan(user, sub)
        limits = self.plan_limits(plan)
        return {
            "enforcement_enabled": self.is_enforcement_enabled(),
            "deployment_mode": self.settings.deployment_mode.value,
            "plan": plan,
            "status": sub.get("status"),
            "period_end": sub.get("period_end"),
            "stripe_customer_id": sub.get("stripe_customer_id"),
            "usage": sub.get("usage"),
            "limits": limits,
            "self_host_url": "https://github.com/Laxmi-Narasimha/neural-analyst",
        }

    async def assert_can_run(
        self,
        user_id: UUID,
        *,
        action: UsageAction,
        operators: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if not self.is_enforcement_enabled():
            return {"allowed": True, "plan": "self_host"}

        user = await self.get_user_record(user_id)
        if not user:
            raise QuotaExceededError("User not found")

        sub = self.get_subscription_state(user)
        plan = self.effective_plan(user, sub)
        limits = self.plan_limits(plan)
        usage = sub["usage"]

        if plan in {SubscriptionPlan.PRO.value, SubscriptionPlan.ENTERPRISE.value, SubscriptionPlan.BYOK.value}:
            return {"allowed": True, "plan": plan}

        ops = operators or []
        uses_compute = any(op not in PREVIEW_OPERATORS for op in ops) if ops else action != UsageAction.TALK_PREVIEW

        if action in {UsageAction.COMPUTE_SESSION, UsageAction.DATA_SPEAKS, UsageAction.ML_TRAIN} or uses_compute:
            if int(limits.get("compute_sessions") or 0) == 0:
                preview_left = max(0, int(limits.get("talk_preview_sessions") or 0) - int(usage.get("talk_preview_used") or 0))
                if preview_left <= 0:
                    raise QuotaExceededError(
                        "Your free Talk-to-Your-Data preview is used. "
                        "Upgrade to Pro for full compute, or self-host with your own API keys."
                    )
                raise QuotaExceededError(
                    "Compute is not included on the free hosted plan. "
                    "You can run one preview session (schema + row overview). "
                    "Upgrade to Pro or self-host for full analysis."
                )

        if action == UsageAction.TALK_PREVIEW:
            cap = limits.get("talk_preview_sessions")
            if cap is not None and int(usage.get("talk_preview_used") or 0) >= int(cap):
                raise QuotaExceededError(
                    "Free preview limit reached. Subscribe for unlimited analysis or install locally."
                )

        monthly_cap = limits.get("monthly_queries")
        if monthly_cap is not None and int(usage.get("monthly_queries") or 0) >= int(monthly_cap):
            raise QuotaExceededError("Monthly query limit reached on the Free plan.")

        return {"allowed": True, "plan": plan, "preview_only": not uses_compute and plan == SubscriptionPlan.FREE.value}

    def filter_plan_for_free(self, plan_steps: list[dict[str, Any]], *, preview_only: bool) -> list[dict[str, Any]]:
        if not preview_only:
            return plan_steps
        filtered = [s for s in plan_steps if str(s.get("operator") or "") in PREVIEW_OPERATORS]
        if filtered:
            return filtered
        return [{"operator": "dataset_overview", "params": {}}]

    async def record_usage(
        self,
        user_id: UUID,
        *,
        action: UsageAction,
        operators: Optional[list[str]] = None,
    ) -> None:
        if not self.is_enforcement_enabled():
            return

        user = await self.get_user_record(user_id)
        if not user:
            return

        sub = self.get_subscription_state(user)
        plan = self.effective_plan(user, sub)
        if plan != SubscriptionPlan.FREE.value:
            return

        usage = sub["usage"]
        ops = operators or []
        uses_compute = any(op not in PREVIEW_OPERATORS for op in ops) if ops else action != UsageAction.TALK_PREVIEW

        usage["monthly_queries"] = int(usage.get("monthly_queries") or 0) + 1
        if action == UsageAction.TALK_PREVIEW or not uses_compute:
            usage["talk_preview_used"] = int(usage.get("talk_preview_used") or 0) + 1
        if uses_compute:
            usage["compute_sessions_used"] = int(usage.get("compute_sessions_used") or 0) + 1
        usage["last_action_at"] = datetime.now(timezone.utc).isoformat()
        sub["usage"] = usage
        await self.persist_subscription_state(user, sub)

    async def set_plan(
        self,
        user_id: UUID,
        *,
        plan: str,
        status: str = "active",
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> dict[str, Any]:
        user = await self.get_user_record(user_id)
        if not user:
            raise QuotaExceededError("User not found")
        sub = self.get_subscription_state(user)
        sub["plan"] = plan
        sub["status"] = status
        if stripe_customer_id:
            sub["stripe_customer_id"] = stripe_customer_id
        if stripe_subscription_id:
            sub["stripe_subscription_id"] = stripe_subscription_id
        if period_end:
            sub["period_end"] = period_end
        await self.persist_subscription_state(user, sub)
        return await self.get_status(user_id)