from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def _derive_fernet_key(passphrase: str) -> bytes:
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    # Prefer explicit connections key; fallback to SECURITY_SECRET_KEY to avoid
    # breaking dev environments. Both are acceptable as long as the secret is
    # stable and not checked into source control.
    raw: Optional[str] = None
    if settings.connections.encryption_key is not None:
        raw = settings.connections.encryption_key.get_secret_value() or None
    if not raw:
        raw = settings.security.secret_key.get_secret_value() or None
    if not raw:
        raise RuntimeError("No encryption secret configured")

    # Accept either a full Fernet key or a passphrase.
    try:
        key = raw.encode("utf-8")
        Fernet(key)
        return Fernet(key)
    except Exception:
        return Fernet(_derive_fernet_key(raw))


def encrypt_secret(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value == "":
        return ""
    f = _get_fernet()
    return f.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    if token == "":
        return ""
    f = _get_fernet()
    try:
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("Invalid encrypted secret") from e

