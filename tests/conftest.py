"""
tests/conftest.py
==================
Shared test fixtures for all tests.

CONCEPT: pytest fixtures
--------------------------
Fixtures are reusable setup functions.
Instead of repeating test setup in every test function,
you define it once as a fixture and inject it where needed.

@pytest.fixture(scope="function"): runs fresh for each test
@pytest.fixture(scope="session"):  runs once for the entire test session

We use an in-memory SQLite DB for tests — 
no file created, no cleanup needed, blazing fast.
"""

import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment BEFORE importing application code
# This prevents the app from loading production credentials
os.environ["OPENAI_API_KEY"] = "sk-test-key-not-real"
os.environ["SECRET_KEY"] = "test-secret-key-at-least-32-chars-long"
os.environ["APP_ENV"] = "development"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from backend.app.main import app
from backend.app.db.database import Base, get_db
from backend.app.models.models import User, UserRole
from backend.app.core.auth import hash_password, create_access_token


# ─── Test Database ──────────────────────────────────────────
@pytest.fixture(scope="session")
def test_engine():
    """
    In-memory SQLite engine for testing.
    
    WHY IN-MEMORY:
    - No file to clean up
    - Instant (no disk I/O)
    - Isolated (each test session starts fresh)
    - No risk of corrupting real data
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(test_engine):
    """
    Provides a DB session that rolls back after each test.
    
    WHY ROLLBACK PATTERN:
    Instead of deleting test data after each test,
    we wrap the test in a transaction and roll it back.
    This is faster and guarantees isolation between tests.
    """
    TestingSession = sessionmaker(bind=test_engine)
    session = TestingSession()
    
    yield session
    
    # Rollback after each test — no cleanup needed
    session.rollback()
    session.close()


# ─── FastAPI Test Client ─────────────────────────────────────
@pytest.fixture(scope="function")
def client(db_session):
    """
    FastAPI TestClient with overridden database.
    
    Dependency overrides replace production dependencies
    with test versions — without modifying application code.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


# ─── User Fixtures ───────────────────────────────────────────
@pytest.fixture
def test_user(db_session) -> User:
    """A regular viewer user for testing."""
    user = User(
        email="testuser@example.com",
        full_name="Test User",
        hashed_password=hash_password("TestPassword1"),
        role=UserRole.ANALYST,
        tenant_id="test-tenant",
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture
def admin_user(db_session) -> User:
    """An admin user for testing privileged operations."""
    user = User(
        email="admin@example.com",
        full_name="Admin User",
        hashed_password=hash_password("AdminPassword1"),
        role=UserRole.ADMIN,
        tenant_id="test-tenant",
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture
def auth_headers(test_user) -> dict:
    """JWT auth headers for authenticated API calls."""
    token = create_access_token(
        user_id=test_user.id,
        email=test_user.email,
        role=test_user.role.value,
        tenant_id=test_user.tenant_id,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(admin_user) -> dict:
    """JWT auth headers for admin API calls."""
    token = create_access_token(
        user_id=admin_user.id,
        email=admin_user.email,
        role=admin_user.role.value,
        tenant_id=admin_user.tenant_id,
    )
    return {"Authorization": f"Bearer {token}"}
