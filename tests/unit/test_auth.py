"""
tests/unit/test_auth.py
=========================
Unit tests for authentication utilities.

These test PURE FUNCTIONS — no database, no API, no external calls.
Fast, reliable, and runnable without any infrastructure.
"""

import pytest
import time
from datetime import timedelta
from jose import jwt

from backend.app.core.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
)
from backend.app.core.config import get_settings

settings = get_settings()


class TestPasswordHashing:
    """Test bcrypt password hashing."""
    
    def test_hash_is_not_plaintext(self):
        """Stored hash must never equal the plain password."""
        plain = "MyPassword1"
        hashed = hash_password(plain)
        assert hashed != plain
    
    def test_hash_starts_with_bcrypt_prefix(self):
        """Valid bcrypt hashes start with $2b$."""
        hashed = hash_password("TestPassword1")
        assert hashed.startswith("$2b$"), f"Expected bcrypt hash, got: {hashed[:10]}"
    
    def test_same_password_different_hashes(self):
        """
        Bcrypt uses random salts — same password = different hashes.
        This is a critical security property.
        If hashes were deterministic, attackers could precompute 
        all common password hashes (rainbow table attack).
        """
        password = "SamePassword1"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        assert hash1 != hash2, "Same password should produce different hashes (random salt)"
    
    def test_verify_correct_password(self):
        """Correct password should verify successfully."""
        plain = "CorrectPassword1"
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True
    
    def test_verify_wrong_password(self):
        """Wrong password should fail verification."""
        hashed = hash_password("RightPassword1")
        assert verify_password("WrongPassword1", hashed) is False
    
    def test_verify_empty_password(self):
        """Empty string should not verify against a real hash."""
        hashed = hash_password("RealPassword1")
        assert verify_password("", hashed) is False


class TestJWTTokens:
    """Test JWT token creation and decoding."""
    
    def test_token_is_string(self):
        """create_access_token should return a string."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            role="viewer",
            tenant_id="default",
        )
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_token_has_three_parts(self):
        """
        JWT tokens are three base64 strings separated by dots.
        Format: header.payload.signature
        """
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            role="viewer",
            tenant_id="default",
        )
        parts = token.split(".")
        assert len(parts) == 3, f"JWT should have 3 parts, got {len(parts)}"
    
    def test_decode_returns_correct_payload(self):
        """Decoded token should contain the data we put in."""
        user_id = "user-abc-123"
        email = "jane@company.com"
        role = "analyst"
        tenant = "acme-corp"
        
        token = create_access_token(
            user_id=user_id,
            email=email,
            role=role,
            tenant_id=tenant,
        )
        
        payload = decode_token(token)
        
        assert payload["sub"] == user_id
        assert payload["email"] == email
        assert payload["role"] == role
        assert payload["tenant_id"] == tenant
    
    def test_expired_token_raises_error(self):
        """Expired tokens should raise HTTPException."""
        from fastapi import HTTPException
        
        # Create token with -1 minute expiry (already expired)
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            role="viewer",
            tenant_id="default",
            expires_delta=timedelta(minutes=-1),
        )
        
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        
        assert exc_info.value.status_code == 401
    
    def test_tampered_token_raises_error(self):
        """
        A modified token should fail signature verification.
        This tests the security of JWT — tampering is detectable.
        """
        from fastapi import HTTPException
        
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            role="viewer",
            tenant_id="default",
        )
        
        # Tamper with the token (flip some characters in the payload)
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + "TAMPERED" + "." + parts[2]
        
        with pytest.raises(HTTPException):
            decode_token(tampered)
    
    def test_wrong_secret_raises_error(self):
        """Token signed with a different secret should be rejected."""
        from fastapi import HTTPException
        
        # Sign with a different secret
        wrong_secret_token = jwt.encode(
            {"sub": "user-123", "email": "test@example.com"},
            "WRONG_SECRET",
            algorithm="HS256"
        )
        
        with pytest.raises(HTTPException):
            decode_token(wrong_secret_token)
    
    def test_token_contains_type_claim(self):
        """Token should have 'type': 'access' claim."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            role="viewer",
            tenant_id="default",
        )
        payload = decode_token(token)
        assert payload.get("type") == "access"
