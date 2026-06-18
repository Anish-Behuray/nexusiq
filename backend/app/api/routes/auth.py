"""
backend/app/api/routes/auth.py
================================
Authentication endpoints: register, login, me.

CONCEPT: FastAPI APIRouter
----------------------------
FastAPI uses Routers to group related endpoints.
Each router handles one "domain" (auth, documents, queries).
The main app.py then mounts all routers with prefixes.

This keeps files small and focused — a key principle in
production codebases.

ENDPOINTS:
  POST /auth/register  — create a new user
  POST /auth/login     — get a JWT token
  GET  /auth/me        — get current user info
  POST /auth/refresh   — get a new token (Phase 2)
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from loguru import logger

from backend.app.db.database import get_db
from backend.app.models.models import User, UserRole
from backend.app.schemas.schemas import (
    UserCreate, UserResponse, LoginRequest, TokenResponse
)
from backend.app.core.auth import (
    hash_password, verify_password, create_access_token, get_current_user
)
from backend.app.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
def register(user_data: UserCreate, db: Session = Depends(get_db)) -> User:
    """
    Register a new user account.
    
    For Phase 1, registration is open (anyone can sign up).
    In Phase 2 (enterprise), registration will require an admin invite.
    
    WHAT HAPPENS:
    1. Check if email already exists (must be unique)
    2. Hash the password with bcrypt
    3. Save user to DB
    4. Return user info (WITHOUT password)
    """
    # Check for duplicate email
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{user_data.email}' is already registered",
        )
    
    # Create user — hash password BEFORE saving
    new_user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hash_password(user_data.password),  # Never store plain text
        role=user_data.role,
        tenant_id="default",  # Phase 2: will be org-specific
    )
    
    db.add(new_user)
    db.flush()  # Flush to get the generated ID without committing
    
    logger.info(f"New user registered: {new_user.email} [{new_user.role}]")
    return new_user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get JWT access token",
)
def login(credentials: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Authenticate a user and return a JWT token.
    
    SECURITY PRACTICE: We return the SAME error message for
    "user not found" and "wrong password". 
    
    WHY: If we returned different messages, an attacker could:
    1. Try emails until they find "wrong password" (vs "not found")
    2. Now they know which emails have accounts
    3. Then brute-force the password
    
    Generic "Invalid credentials" prevents email enumeration attacks.
    """
    # Look up user
    user = db.query(User).filter(
        User.email == credentials.email,
        User.is_active == True
    ).first()
    
    # Validate — same error message for not-found AND wrong-password
    if not user or not verify_password(credentials.password, user.hashed_password):
        logger.warning(f"Failed login attempt for email: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",  # Generic message — intentional
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create JWT token
    expire_delta = timedelta(minutes=settings.access_token_expire_minutes)
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role.value,
        tenant_id=user.tenant_id,
        expires_delta=expire_delta,
    )
    
    logger.info(f"User logged in: {user.email}")
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def get_me(current_user: User = Depends(get_current_user)) -> User:
    """
    Returns the currently authenticated user's profile.
    
    Requires: valid JWT token in Authorization header.
    This is a common pattern for "who am I?" checks in frontends.
    """
    return current_user
