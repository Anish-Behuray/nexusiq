"""
tests/integration/test_auth_api.py
=====================================
Integration tests for the /auth/* endpoints.

DIFFERENCE FROM UNIT TESTS:
Unit tests: test one function in isolation (no DB, no HTTP)
Integration tests: test multiple components working together (HTTP → FastAPI → DB)

These tests call the actual FastAPI endpoints via TestClient
and verify the full request/response cycle.
"""

import pytest


class TestRegisterEndpoint:
    """Tests for POST /api/v1/auth/register."""
    
    def test_register_success(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "newuser@example.com",
            "full_name": "New User",
            "password": "ValidPassword1",
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert "id" in data
        # CRITICAL: password must never be returned
        assert "password" not in data
        assert "hashed_password" not in data
    
    def test_register_duplicate_email(self, client, test_user):
        """Registering with an existing email should return 409."""
        response = client.post("/api/v1/auth/register", json={
            "email": test_user.email,  # Already registered
            "full_name": "Duplicate User",
            "password": "ValidPassword1",
        })
        
        assert response.status_code == 409
        assert "already registered" in response.json()["detail"].lower()
    
    def test_register_weak_password(self, client):
        """Weak passwords should be rejected with 422."""
        response = client.post("/api/v1/auth/register", json={
            "email": "weak@example.com",
            "full_name": "Test User",
            "password": "weak",  # Too short, no uppercase, no number
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_register_invalid_email(self, client):
        """Invalid email format should return 422."""
        response = client.post("/api/v1/auth/register", json={
            "email": "not-an-email",
            "full_name": "Test User",
            "password": "ValidPassword1",
        })
        
        assert response.status_code == 422


class TestLoginEndpoint:
    """Tests for POST /api/v1/auth/login."""
    
    def test_login_success(self, client, test_user):
        response = client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "TestPassword1",
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["email"] == test_user.email
    
    def test_login_wrong_password(self, client, test_user):
        """Wrong password should return 401 with generic error."""
        response = client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "WrongPassword1",
        })
        
        assert response.status_code == 401
        # Error should be GENERIC — no info about whether email exists
        assert response.json()["detail"] == "Invalid credentials"
    
    def test_login_nonexistent_email(self, client):
        """Non-existent email should return the SAME 401 as wrong password."""
        response = client.post("/api/v1/auth/login", json={
            "email": "doesnotexist@example.com",
            "password": "SomePassword1",
        })
        
        assert response.status_code == 401
        # Same generic message — prevents email enumeration
        assert response.json()["detail"] == "Invalid credentials"
    
    def test_login_deactivated_user(self, client, db_session, test_user):
        """Deactivated users should not be able to login."""
        test_user.is_active = False
        db_session.flush()
        
        response = client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "TestPassword1",
        })
        
        assert response.status_code == 401


class TestMeEndpoint:
    """Tests for GET /api/v1/auth/me."""
    
    def test_me_with_valid_token(self, client, auth_headers, test_user):
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["id"] == test_user.id
    
    def test_me_without_token(self, client):
        """Unauthenticated request should return 401."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
    
    def test_me_with_invalid_token(self, client):
        """Invalid token should return 401."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer totally.invalid.token"}
        )
        assert response.status_code == 401
