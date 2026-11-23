import pytest
from pydantic import SecretStr

from app.core.config import AuthMode, Environment, Settings, settings
from app.main import validate_startup_security


def test_settings_coerce_release_debug_flag_to_false():
    settings = Settings(debug="release")
    assert settings.debug is False


def test_settings_coerce_debug_alias_to_true():
    settings = Settings(debug="debug")
    assert settings.debug is True


def test_production_jwt_rejects_development_secret():
    old_environment = settings.environment
    old_auth_mode = settings.auth_mode
    old_secret = settings.security.secret_key
    try:
        settings.environment = Environment.PRODUCTION
        settings.auth_mode = AuthMode.JWT
        settings.security.secret_key = SecretStr("dev-secret-change-in-production")
        with pytest.raises(RuntimeError, match="SECURITY_SECRET_KEY"):
            validate_startup_security()
        settings.security.secret_key = SecretStr("s" * 32)
        validate_startup_security()
    finally:
        settings.environment = old_environment
        settings.auth_mode = old_auth_mode
        settings.security.secret_key = old_secret
