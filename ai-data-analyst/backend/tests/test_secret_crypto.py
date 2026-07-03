import pytest

from app.services.secret_crypto import decrypt_secret, encrypt_secret


def test_encrypt_decrypt_roundtrip():
    token = encrypt_secret("super-secret")
    assert isinstance(token, str)
    assert token != "super-secret"
    assert decrypt_secret(token) == "super-secret"


def test_encrypt_none_is_none():
    assert encrypt_secret(None) is None
    assert decrypt_secret(None) is None


def test_decrypt_invalid_raises():
    with pytest.raises(ValueError):
        decrypt_secret("not-a-valid-token")

