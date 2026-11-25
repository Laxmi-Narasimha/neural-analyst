"""Private beta signups require the configured invite code."""

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from app.api.routes import auth
from app.core.config import DeploymentMode
from app.services.subscription_service import SubscriptionService


class _AuthService:
    def __init__(self):
        self.calls = 0

    async def register_user(self, **kwargs):
        self.calls += 1
        return {"email": kwargs["email"], "user_id": str(uuid4())}


@pytest.mark.asyncio
async def test_private_beta_registration_requires_invitation(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_settings",
        lambda: SimpleNamespace(
            registration_invite_code=SecretStr("beta-secret"),
            deployment_mode=DeploymentMode.SELF_HOST,
        ),
    )
    service = _AuthService()
    base = {"email": "beta@example.com", "password": "strong-password"}

    for invite_code in (None, "wrong"):
        with pytest.raises(HTTPException) as exc:
            await auth.register(
                auth.RegisterRequest(**base, invite_code=invite_code),
                db=None,
                auth_service=service,
            )
        assert exc.value.status_code == 403
    assert service.calls == 0

    result = await auth.register(
        auth.RegisterRequest(**base, invite_code="beta-secret"),
        db=None,
        auth_service=service,
    )
    assert result["status"] == "success"
    assert service.calls == 1


@pytest.mark.asyncio
async def test_invited_hosted_user_gets_metered_beta_plan(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_settings",
        lambda: SimpleNamespace(
            registration_invite_code=SecretStr("beta-secret"),
            deployment_mode=DeploymentMode.SAAS,
        ),
    )
    assigned = []

    async def fake_set_plan(self, user_id, *, plan):
        assigned.append((user_id, plan))

    monkeypatch.setattr(SubscriptionService, "set_plan", fake_set_plan)
    result = await auth.register(
        auth.RegisterRequest(email="invited@example.com", password="strong-password", invite_code="beta-secret"),
        db=None,
        auth_service=_AuthService(),
    )
    assert assigned == [(UUID(result["user"]["user_id"]), "beta")]
