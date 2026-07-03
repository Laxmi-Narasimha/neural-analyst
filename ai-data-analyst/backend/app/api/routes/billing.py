"""Billing and subscription routes (Stripe when configured)."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.api.schemas import APIResponse
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.auth_service import AuthUser, Permission
from app.services.database import get_db_session
from app.services.subscription_service import SubscriptionPlan, SubscriptionService

logger = get_logger(__name__)
router = APIRouter()


class CheckoutRequest(BaseModel):
    plan: str = Field(..., pattern="^(pro|enterprise)$")
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class PortalRequest(BaseModel):
    return_url: Optional[str] = None


def _stripe_headers(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}", "Content-Type": "application/x-www-form-urlencoded"}


@router.get("/status", summary="Subscription status")
async def subscription_status(
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(get_current_user),
):
    svc = SubscriptionService(db)
    return APIResponse.success(data=await svc.get_status(user.user_id))


@router.post("/checkout", summary="Create Stripe checkout session")
async def create_checkout(
    body: CheckoutRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(get_current_user),
):
    settings = get_settings()
    secret = settings.billing.stripe_secret_key.get_secret_value() if settings.billing.stripe_secret_key else ""
    price_id = (
        settings.billing.stripe_price_pro
        if body.plan == SubscriptionPlan.PRO.value
        else settings.billing.stripe_price_enterprise
    )

    if not secret or not price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Billing is not configured. Self-host the project or contact support.",
        )

    svc = SubscriptionService(db)
    state = await svc.get_status(user.user_id)
    customer_id = state.get("stripe_customer_id")

    base_frontend = settings.billing.frontend_url.rstrip("/")
    success = body.success_url or f"{base_frontend}/app/settings?billing=success"
    cancel = body.cancel_url or f"{base_frontend}/pricing?billing=cancelled"

    form: dict[str, str] = {
        "mode": "subscription",
        "success_url": success,
        "cancel_url": cancel,
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": "1",
        "client_reference_id": str(user.user_id),
        "metadata[user_id]": str(user.user_id),
        "metadata[plan]": body.plan,
    }
    if customer_id:
        form["customer"] = str(customer_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.stripe.com/v1/checkout/sessions",
            headers=_stripe_headers(secret),
            data=form,
        )
    if resp.status_code >= 400:
        logger.error("Stripe checkout failed", status=resp.status_code, body=resp.text[:500])
        raise HTTPException(status_code=502, detail="Could not create checkout session")

    payload = resp.json()
    return APIResponse.success(
        data={
            "checkout_url": payload.get("url"),
            "session_id": payload.get("id"),
        }
    )


@router.post("/portal", summary="Stripe customer portal")
async def customer_portal(
    body: PortalRequest,
    db: AsyncSession = Depends(get_db_session),
    user: AuthUser = Depends(get_current_user),
):
    settings = get_settings()
    secret = settings.billing.stripe_secret_key.get_secret_value() if settings.billing.stripe_secret_key else ""
    if not secret:
        raise HTTPException(status_code=503, detail="Billing portal is not configured")

    svc = SubscriptionService(db)
    state = await svc.get_status(user.user_id)
    customer_id = state.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="No billing account linked yet")

    return_url = body.return_url or f"{settings.billing.frontend_url.rstrip('/')}/app/settings"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.stripe.com/v1/billing_portal/sessions",
            headers=_stripe_headers(secret),
            data={"customer": str(customer_id), "return_url": return_url},
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail="Could not open billing portal")
    return APIResponse.success(data={"portal_url": resp.json().get("url")})


@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db_session)):
    settings = get_settings()
    webhook_secret = (
        settings.billing.stripe_webhook_secret.get_secret_value()
        if settings.billing.stripe_webhook_secret
        else ""
    )
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    if not _verify_stripe_signature(payload, sig, webhook_secret):
        raise HTTPException(status_code=400, detail="Invalid signature")

    event = json.loads(payload.decode("utf-8"))
    event_type = event.get("type")
    data_obj = (event.get("data") or {}).get("object") or {}

    svc = SubscriptionService(db)

    if event_type == "checkout.session.completed":
        user_id_raw = data_obj.get("metadata", {}).get("user_id") or data_obj.get("client_reference_id")
        plan = data_obj.get("metadata", {}).get("plan") or SubscriptionPlan.PRO.value
        if user_id_raw:
            await svc.set_plan(
                UUID(str(user_id_raw)),
                plan=str(plan),
                status="active",
                stripe_customer_id=data_obj.get("customer"),
                stripe_subscription_id=data_obj.get("subscription"),
            )
    elif event_type in {"customer.subscription.updated", "customer.subscription.deleted"}:
        meta_uid = (data_obj.get("metadata") or {}).get("user_id")
        sub_status = str(data_obj.get("status") or "active")
        plan = SubscriptionPlan.PRO.value
        if sub_status in {"canceled", "unpaid", "incomplete_expired"}:
            plan = SubscriptionPlan.FREE.value
            sub_status = "canceled"
        if meta_uid:
            await svc.set_plan(
                UUID(str(meta_uid)),
                plan=plan,
                status=sub_status,
                stripe_customer_id=data_obj.get("customer"),
                stripe_subscription_id=data_obj.get("id"),
                period_end=str(data_obj.get("current_period_end") or ""),
            )

    return {"received": True}


def _verify_stripe_signature(payload: bytes, header: str, secret: str) -> bool:
    try:
        parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
        timestamp = parts.get("t", "")
        v1 = parts.get("v1", "")
        signed = f"{timestamp}.{payload.decode('utf-8')}".encode("utf-8")
        expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, v1)
    except Exception:
        return False