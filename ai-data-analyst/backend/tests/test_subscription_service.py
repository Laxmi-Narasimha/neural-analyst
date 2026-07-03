from __future__ import annotations

from uuid import uuid4

import pytest
import pytest_asyncio

from app.core.config import DeploymentMode, settings
from app.services.subscription_service import SubscriptionService, UsageAction, QuotaExceededError


@pytest_asyncio.fixture
async def sub_session(db_session):
    old_mode = settings.deployment_mode
    settings.deployment_mode = DeploymentMode.SAAS
    try:
        yield db_session
    finally:
        settings.deployment_mode = old_mode


@pytest.mark.asyncio
async def test_free_plan_blocks_compute_after_preview(sub_session):
    from app.models import User
    from app.services.auth_service import PasswordHasher

    user = User(
        id=uuid4(),
        email=f"free-{uuid4().hex[:8]}@example.com",
        hashed_password=PasswordHasher.hash("Str0ngPassw0rd!"),
        full_name="Free User",
        is_active=True,
        settings={"subscription": {"plan": "free", "status": "active", "usage": {"talk_preview_used": 1}}},
    )
    sub_session.add(user)
    await sub_session.commit()

    svc = SubscriptionService(sub_session)
    with pytest.raises(QuotaExceededError):
        await svc.assert_can_run(
            user.id,
            action=UsageAction.COMPUTE_SESSION,
            operators=["correlation_matrix"],
        )


@pytest.mark.asyncio
async def test_self_host_skips_enforcement(db_session):
    old_mode = settings.deployment_mode
    settings.deployment_mode = DeploymentMode.SELF_HOST
    try:
        svc = SubscriptionService(db_session)
        assert svc.is_enforcement_enabled() is False
        gate = await svc.assert_can_run(uuid4(), action=UsageAction.DATA_SPEAKS, operators=["correlation_matrix"])
        assert gate["allowed"] is True
    finally:
        settings.deployment_mode = old_mode