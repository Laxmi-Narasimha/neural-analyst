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


@pytest.mark.asyncio
async def test_hosted_free_upload_size_and_monthly_count(sub_session):
    from app.models import User
    from app.services.auth_service import PasswordHasher

    user = User(
        id=uuid4(),
        email=f"uploads-{uuid4().hex[:8]}@example.com",
        hashed_password=PasswordHasher.hash("Str0ngPassw0rd!"),
        full_name="Upload User",
        is_active=True,
        settings={},
    )
    sub_session.add(user)
    await sub_session.commit()
    svc = SubscriptionService(sub_session)

    with pytest.raises(QuotaExceededError):
        await svc.assert_can_upload(user.id, file_size_bytes=11 * 1024 * 1024)

    for _ in range(3):
        await svc.assert_can_upload(user.id, file_size_bytes=1024)
        await svc.record_upload(user.id)

    with pytest.raises(QuotaExceededError):
        await svc.assert_can_upload(user.id, file_size_bytes=1024)
    assert (await svc.get_status(user.id))["usage"]["monthly_uploads"] == 3


@pytest.mark.asyncio
async def test_hosted_pro_upload_respects_file_size_limit(sub_session):
    from app.models import User
    from app.services.auth_service import PasswordHasher

    user = User(
        id=uuid4(),
        email=f"pro-upload-{uuid4().hex[:8]}@example.com",
        hashed_password=PasswordHasher.hash("Str0ngPassw0rd!"),
        full_name="Pro Upload User",
        is_active=True,
        settings={"subscription": {"plan": "pro", "status": "active"}},
    )
    sub_session.add(user)
    await sub_session.commit()
    svc = SubscriptionService(sub_session)

    await svc.assert_can_upload(user.id, file_size_bytes=11 * 1024 * 1024)
    with pytest.raises(QuotaExceededError):
        await svc.assert_can_upload(user.id, file_size_bytes=1025 * 1024 * 1024)


@pytest.mark.asyncio
async def test_beta_compute_is_metered_and_training_is_blocked(sub_session):
    from app.models import User
    from app.services.auth_service import PasswordHasher

    user = User(
        id=uuid4(),
        email=f"beta-{uuid4().hex[:8]}@example.com",
        hashed_password=PasswordHasher.hash("Str0ngPassw0rd!"),
        full_name="Beta User",
        is_active=True,
        settings={},
    )
    sub_session.add(user)
    await sub_session.commit()
    svc = SubscriptionService(sub_session)
    await svc.set_plan(user.id, plan="beta")

    with pytest.raises(QuotaExceededError):
        await svc.assert_can_run(user.id, action=UsageAction.ML_TRAIN)

    for _ in range(10):
        await svc.assert_can_run(user.id, action=UsageAction.COMPUTE_SESSION)
        await svc.record_usage(user.id, action=UsageAction.COMPUTE_SESSION)

    with pytest.raises(QuotaExceededError):
        await svc.assert_can_run(user.id, action=UsageAction.COMPUTE_SESSION)
    assert (await svc.get_status(user.id))["usage"]["compute_sessions_used"] == 10
