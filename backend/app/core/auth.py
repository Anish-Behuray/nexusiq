"""
backend/app/core/auth.py
=========================
Authentication utilities: JWT tokens and password hashing.

CONCEPT: How JWT authentication works
---------------------------------------
JWT = JSON Web Token. It's the industry standard for stateless auth.

Flow:
1. User sends email + password to POST /auth/login
2. Server verifies password against bcrypt hash in DB
3. Server creates a JWT token containing: {user_id, role, tenant_id, expiry}
4. Server signs the token with SECRET_KEY (only the server knows this)
5. Client stores the token (browser localStorage or memory)
6. Client sends token in every request: "Authorization: Bearer <token>"
7. Server verifies the signature — if valid, extracts user info
8. No DB lookup needed on every request — the token IS the session

WHY STATELESS MATTERS:
- Chatbot: server stores session → doesn't scale (100K users = huge memory)
- JWT: server stores nothing → scales infinitely → can run on many servers

SECURITY NOTES:
- We sign tokens with SECRET_KEY. If someone steals the key, they can forge tokens.
  → Keep SECRET_KEY secret. Never commit it. Rotate periodically.
- Tokens expire (30 min by default). Short expiry = less damage if stolen.
- Passwords are hashed with bcrypt. bcrypt is slow by design (makes brute force hard).
- We use passlib to abstract hashing — changing algorithm only requires one line change.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from loguru import logger

from backend.app.core.config import get_settings
from backend.app.db.database import get_db
from backend.app.models.models import User, UserRole

settings = get_settings()

# ─── Password Hashing ───────────────────────────────────────
# CryptContext handles password hashing.
# bcrypt is the industry standard — deliberately slow to prevent brute force.
# "deprecated='auto'" means old hashes are automatically upgraded.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """
    Convert plain text password to bcrypt hash.
    
    Each hash is unique even for the same password (bcrypt adds a random salt).
    This means two users with the same password have different hashes.
    
    Example:
        hash_password("MyPassword1") 
        → "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    """
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check if plain password matches the stored hash.
    Returns True if match, False otherwise.
    
    Never compare passwords directly — always use this function.
    """
    return pwd_context.verify(plain_password, hashed_password)


# ─── JWT Token Operations ───────────────────────────────────
ALGORITHM = "HS256"  # HMAC with SHA-256 — standard symmetric JWT algorithm


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    tenant_id: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.
    
    The "payload" (claims) are:
    - sub: subject (user_id) — standard JWT claim
    - email, role, tenant_id: our custom claims
    - exp: expiry time — standard JWT claim (auto-validated by jose)
    - iat: issued at — useful for audit logs
    
    The token is SIGNED but not encrypted.
    Anyone can decode the payload with base64 decoding.
    The signature just proves it wasn't tampered with.
    → Never put sensitive data (passwords, secrets) in JWT payload.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    
    payload: Dict[str, Any] = {
        "sub": user_id,           # Standard: subject
        "email": email,
        "role": role,
        "tenant_id": tenant_id,
        "exp": expire,            # Standard: expiry (jose validates this)
        "iat": datetime.now(timezone.utc),  # Standard: issued at
        "type": "access",
    }
    
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    jose.jwt.decode() automatically:
    - Verifies the signature (was this token signed by us?)
    - Checks expiry (is exp in the future?)
    - Raises JWTError if either check fails
    
    SECURITY: We verify the algorithm explicitly to prevent "alg:none" attacks
    — a well-known vulnerability where attackers set algorithm to "none"
    to bypass signature verification.
    """
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[ALGORITHM]  # Explicit allowlist — never accept "none"
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── FastAPI Security Dependency ────────────────────────────
# HTTPBearer extracts the token from "Authorization: Bearer <token>" header
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency: extracts and validates the current user from JWT.
    
    Usage in any protected endpoint:
        @router.get("/protected")
        def protected_route(user: User = Depends(get_current_user)):
            return {"user": user.email}
    
    This runs BEFORE the endpoint function.
    If the token is invalid/missing, it raises 401 before your code runs.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = decode_token(credentials.credentials)
    user_id: str = payload.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    # Look up user in DB to ensure they still exist and are active
    # (Token could be valid but user might have been deactivated)
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )
    
    return user


# ─── Role-Based Access Control (RBAC) ───────────────────────
def require_role(*allowed_roles: UserRole):
    """
    Factory function that creates role-checking dependencies.
    
    Usage:
        @router.delete("/documents/{id}")
        def delete_doc(
            user: User = Depends(require_role(UserRole.ADMIN, UserRole.MANAGER))
        ):
            ...
    
    This cleanly enforces that only admins and managers can delete docs.
    Viewers and analysts get 403 Forbidden.
    
    INTERVIEW CONCEPT: This is "RBAC" — Role-Based Access Control.
    It's a dependency function that returns another dependency function.
    This pattern (higher-order functions for auth) is common in FastAPI.
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            logger.warning(
                f"Access denied: {current_user.email} ({current_user.role}) "
                f"tried to access resource requiring {allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not authorized for this action. "
                       f"Required: {[r.value for r in allowed_roles]}",
            )
        return current_user
    
    return role_checker


# ─── Convenience Aliases ────────────────────────────────────
# These make endpoint definitions clean and self-documenting
require_admin = require_role(UserRole.ADMIN)
require_manager_or_above = require_role(UserRole.ADMIN, UserRole.MANAGER)
require_analyst_or_above = require_role(UserRole.ADMIN, UserRole.MANAGER, UserRole.ANALYST)
