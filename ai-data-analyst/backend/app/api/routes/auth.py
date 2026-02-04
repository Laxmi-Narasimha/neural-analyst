# AI Enterprise Data Analyst - API Auth Routes
# Authentication endpoints with JWT

from __future__ import annotations

from typing import Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Header
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.auth_service import (
    AuthenticationService,
    get_auth_service,
    UserRole,
    Permission,
    AuthUser
)
from app.core.exceptions import AuthenticationException, AuthorizationException
from app.core.logging import get_logger
from app.services.database import get_db_session

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    role: Optional[str] = "analyst"


class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class UserResponse(BaseModel):
    """User info response."""
    user_id: str
    email: str
    role: str
    permissions: list[str]


class APIKeyRequest(BaseModel):
    """API key creation request."""
    name: str
    permissions: Optional[list[str]] = None


class APIKeyResponse(BaseModel):
    """API key response."""
    key_id: str
    secret: str
    name: str
    created_at: datetime


# ============================================================================
# Dependencies
# ============================================================================

async def get_current_user(
    authorization: str = Header(None),
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> AuthUser:
    """Extract and validate user from token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Extract token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    token = parts[1]
    
    try:
        user = auth_service.validate_token(token)

        # Enforce that the user exists/is active in the DB (FKs rely on this).
        from app.models import User

        result = await db.execute(
            select(User).where(
                User.id == user.user_id,
                User.is_deleted == False,  # noqa: E712
            )
        )
        db_user = result.scalars().first()
        if not db_user or not db_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User is not active",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


def require_permission(permission: Permission):
    """Dependency to require specific permission."""
    async def check(
        user: AuthUser = Depends(get_current_user),
        auth_service: AuthenticationService = Depends(get_auth_service)
    ):
        try:
            auth_service.check_permission(user, permission)
            return user
        except AuthorizationException as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )
    return check


def require_admin():
    """Dependency to require admin role."""
    async def check(user: AuthUser = Depends(get_current_user)):
        if not user.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return user
    return check


# ============================================================================
# Routes
# ============================================================================

@router.post("/register", response_model=dict)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Register a new user."""
    try:
        role = UserRole(request.role) if request.role else UserRole.ANALYST
        
        result = await auth_service.register_user(
            db=db,
            email=request.email,
            password=request.password,
            role=role,
            full_name=request.full_name,
        )
        
        return {
            "status": "success",
            "message": "User registered successfully",
            "user": result
        }
        
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Authenticate user and return tokens."""
    try:
        tokens = await auth_service.authenticate(
            db=db,
            email=request.email,
            password=request.password
        )
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"]
        )
        
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Refresh access token."""
    try:
        tokens = await auth_service.refresh_tokens(db=db, refresh_token=request.refresh_token)
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"]
        )
        
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/logout")
async def logout(
    user: AuthUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Logout user (invalidate session)."""
    auth_service.logout(str(user.user_id))
    return {"status": "success", "message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: AuthUser = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        user_id=str(user.user_id),
        email=user.email,
        role=user.role.value,
        permissions=[p.value for p in user.permissions]
    )


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    user: AuthUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Create a new API key."""
    permissions = None
    if request.permissions:
        permissions = [Permission(p) for p in request.permissions]
    
    key_id, secret = auth_service.api_keys.generate_key(
        user_id=str(user.user_id),
        name=request.name,
        permissions=permissions
    )
    
    return APIKeyResponse(
        key_id=key_id,
        secret=secret,
        name=request.name,
        created_at=datetime.utcnow()
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: AuthUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Revoke an API key."""
    if auth_service.api_keys.revoke_key(key_id):
        return {"status": "success", "message": "API key revoked"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="API key not found"
    )


# Admin routes
@router.get("/admin/users", dependencies=[Depends(require_admin())])
async def list_users(
    db: AsyncSession = Depends(get_db_session),
):
    """List all users (admin only)."""
    from app.models import User

    result = await db.execute(select(User).where(User.is_deleted == False))  # noqa: E712
    users = []
    for u in result.scalars().all():
        users.append(
            {
                "user_id": str(u.id),
                "email": u.email,
                "full_name": u.full_name,
                "is_active": u.is_active,
                "is_superuser": u.is_superuser,
                "is_verified": u.is_verified,
                "created_at": u.created_at.isoformat(),
            }
        )
    return {"users": users, "total": len(users)}
