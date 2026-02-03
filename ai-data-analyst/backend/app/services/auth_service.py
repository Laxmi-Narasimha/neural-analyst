# AI Enterprise Data Analyst - Authentication Service
# JWT-based authentication with role-based access control

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
import hashlib
import secrets

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

from app.core.logging import get_logger
from app.core.exceptions import AuthenticationException, AuthorizationException
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


# ============================================================================
# Auth Types
# ============================================================================

class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API = "api"


class Permission(str, Enum):
    """Granular permissions."""
    READ_DATA = "read:data"
    WRITE_DATA = "write:data"
    DELETE_DATA = "delete:data"
    RUN_ANALYSIS = "run:analysis"
    MANAGE_MODELS = "manage:models"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p for p in Permission],
    UserRole.ANALYST: [
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.RUN_ANALYSIS,
        Permission.MANAGE_MODELS,
    ],
    UserRole.VIEWER: [Permission.READ_DATA],
    UserRole.API: [
        Permission.READ_DATA,
        Permission.RUN_ANALYSIS,
    ],
}


@dataclass
class TokenPayload:
    """JWT token payload."""
    
    sub: str  # User ID
    email: str
    role: UserRole
    permissions: list[Permission]
    exp: datetime
    iat: datetime = field(default_factory=datetime.utcnow)
    jti: str = field(default_factory=lambda: str(uuid4()))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.sub,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "exp": int(self.exp.timestamp()),
            "iat": int(self.iat.timestamp()),
            "jti": self.jti
        }


@dataclass
class AuthUser:
    """Authenticated user."""
    
    user_id: UUID
    email: str
    role: UserRole
    permissions: list[Permission]
    
    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN


# ============================================================================
# Password Hashing
# ============================================================================

class PasswordHasher:
    """Secure password hashing using PBKDF2."""
    
    ITERATIONS = 100000
    SALT_LENGTH = 32
    HASH_LENGTH = 32
    
    @classmethod
    def hash(cls, password: str) -> str:
        """Hash a password with salt."""
        salt = secrets.token_bytes(cls.SALT_LENGTH)
        
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            cls.ITERATIONS,
            dklen=cls.HASH_LENGTH
        )
        
        # Combine salt and hash
        return f"{salt.hex()}${hash_bytes.hex()}"
    
    @classmethod
    def verify(cls, password: str, stored_hash: str) -> bool:
        """Verify a password against stored hash."""
        try:
            salt_hex, hash_hex = stored_hash.split('$')
            salt = bytes.fromhex(salt_hex)
            stored_bytes = bytes.fromhex(hash_hex)
            
            computed = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                cls.ITERATIONS,
                dklen=cls.HASH_LENGTH
            )
            
            return secrets.compare_digest(computed, stored_bytes)
        except Exception:
            return False


# ============================================================================
# JWT Service
# ============================================================================

class JWTService:
    """JWT token generation and validation."""
    
    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str | None = None,
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        configured_secret = settings.security.secret_key
        if hasattr(configured_secret, "get_secret_value"):
            configured_secret = configured_secret.get_secret_value()

        self.secret_key = secret_key or str(configured_secret)
        self.algorithm = algorithm or getattr(settings.security, "algorithm", "HS256")
        self.access_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_expire = timedelta(days=refresh_token_expire_days)
    
    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: UserRole
    ) -> str:
        """Create JWT access token."""
        permissions = ROLE_PERMISSIONS.get(role, [])
        
        payload = TokenPayload(
            sub=user_id,
            email=email,
            role=role,
            permissions=permissions,
            exp=datetime.utcnow() + self.access_expire
        )

        payload_dict = payload.to_dict()
        payload_dict["type"] = "access"
        return jwt.encode(payload_dict, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token."""
        payload = {
            "sub": user_id,
            "type": "refresh",
            "iat": int(datetime.utcnow().timestamp()),
            "exp": int((datetime.utcnow() + self.refresh_expire).timestamp()),
            "jti": str(uuid4()),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> TokenPayload:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            token_type = payload.get("type", "access")
            if token_type != "access":
                raise AuthenticationException("Invalid token type")

            return TokenPayload(
                sub=payload["sub"],
                email=payload["email"],
                role=UserRole(payload["role"]),
                permissions=[Permission(p) for p in payload.get("permissions", [])],
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"]),
                jti=payload["jti"],
            )
        except ExpiredSignatureError:
            raise AuthenticationException("Token expired")
        except JWTError as e:
            raise AuthenticationException(f"Invalid token: {e}")
    
    def decode_refresh_token(self, token: str) -> dict[str, Any]:
        """Decode and validate refresh token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "refresh":
                raise AuthenticationException("Invalid token type")
            return payload
        except ExpiredSignatureError:
            raise AuthenticationException("Refresh token expired")
        except JWTError as e:
            raise AuthenticationException(f"Invalid refresh token: {e}")


# ============================================================================
# API Key Service
# ============================================================================

class APIKeyService:
    """API key management for programmatic access."""
    
    PREFIX = "aida_"
    KEY_LENGTH = 32
    
    def __init__(self):
        self._keys: dict[str, dict] = {}  # In production, use database
    
    def generate_key(
        self,
        user_id: str,
        name: str,
        permissions: list[Permission] = None
    ) -> tuple[str, str]:
        """Generate new API key. Returns (key_id, secret)."""
        key_id = f"{self.PREFIX}{secrets.token_hex(8)}"
        secret = secrets.token_urlsafe(self.KEY_LENGTH)
        
        # Store hashed secret
        self._keys[key_id] = {
            "user_id": user_id,
            "name": name,
            "secret_hash": PasswordHasher.hash(secret),
            "permissions": permissions or [Permission.READ_DATA],
            "created_at": datetime.utcnow(),
            "last_used": None
        }
        
        return key_id, secret
    
    def validate_key(self, key_id: str, secret: str) -> Optional[dict]:
        """Validate API key."""
        if key_id not in self._keys:
            return None
        
        key_data = self._keys[key_id]
        
        if not PasswordHasher.verify(secret, key_data["secret_hash"]):
            return None
        
        # Update last used
        key_data["last_used"] = datetime.utcnow()
        
        return key_data
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False


# ============================================================================
# Authentication Service
# ============================================================================

class AuthenticationService:
    """
    Complete authentication service.
    
    Features:
    - JWT token authentication
    - Password hashing
    - Role-based access control
    - API key authentication
    - Session management
    """
    
    def __init__(self):
        self.jwt = JWTService()
        self.api_keys = APIKeyService()
        self.hasher = PasswordHasher()
        
        # In-memory user store (use database in production)
        self._users: dict[str, dict] = {}
        self._sessions: dict[str, dict] = {}
    
    def register_user(
        self,
        email: str,
        password: str,
        role: UserRole = UserRole.ANALYST
    ) -> dict[str, Any]:
        """Register a new user."""
        if email in self._users:
            raise AuthenticationException("User already exists")
        
        user_id = str(uuid4())
        
        self._users[email] = {
            "user_id": user_id,
            "email": email,
            "password_hash": self.hasher.hash(password),
            "role": role,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        return {
            "user_id": user_id,
            "email": email,
            "role": role.value
        }
    
    def authenticate(
        self,
        email: str,
        password: str
    ) -> dict[str, str]:
        """Authenticate user and return tokens."""
        if email not in self._users:
            raise AuthenticationException("Invalid credentials")
        
        user = self._users[email]
        
        if not self.hasher.verify(password, user["password_hash"]):
            raise AuthenticationException("Invalid credentials")
        
        if not user["is_active"]:
            raise AuthenticationException("Account is disabled")
        
        # Generate tokens
        access_token = self.jwt.create_access_token(
            user["user_id"],
            email,
            user["role"]
        )
        
        refresh_token = self.jwt.create_refresh_token(user["user_id"])
        
        # Store session
        self._sessions[user["user_id"]] = {
            "refresh_token": refresh_token,
            "created_at": datetime.utcnow()
        }
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    def validate_token(self, token: str) -> AuthUser:
        """Validate access token and return user."""
        payload = self.jwt.decode_token(token)
        
        return AuthUser(
            user_id=UUID(payload.sub),
            email=payload.email,
            role=payload.role,
            permissions=payload.permissions
        )
    
    def refresh_tokens(self, refresh_token: str) -> dict[str, str]:
        """Refresh access token."""
        payload = self.jwt.decode_refresh_token(refresh_token)
        user_id = payload["sub"]

        session = self._sessions.get(user_id)
        if not session or session.get("refresh_token") != refresh_token:
            raise AuthenticationException("Refresh token is not valid for this session")

        # Find user by id
        user: Optional[dict[str, Any]] = None
        for u in self._users.values():
            if u["user_id"] == user_id:
                user = u
                break

        if not user:
            raise AuthenticationException("User not found")

        access_token = self.jwt.create_access_token(
            user_id=user["user_id"],
            email=user["email"],
            role=user["role"],
        )
        new_refresh_token = self.jwt.create_refresh_token(user["user_id"])

        self._sessions[user["user_id"]] = {
            "refresh_token": new_refresh_token,
            "created_at": datetime.utcnow(),
        }

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
        }
    
    def logout(self, user_id: str) -> bool:
        """Logout user (invalidate session)."""
        if user_id in self._sessions:
            del self._sessions[user_id]
            return True
        return False
    
    def check_permission(
        self,
        user: AuthUser,
        permission: Permission
    ) -> bool:
        """Check if user has permission."""
        if not user.has_permission(permission):
            raise AuthorizationException(
                f"Permission denied: {permission.value}"
            )
        return True


# Factory function
def get_auth_service() -> AuthenticationService:
    """Get authentication service instance."""
    return _AUTH_SERVICE


_AUTH_SERVICE = AuthenticationService()
