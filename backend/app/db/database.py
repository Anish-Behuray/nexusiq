"""
backend/app/db/database.py
===========================
Database connection and session management.

CONCEPT: Why SQLAlchemy ORM?
------------------------------
SQLAlchemy lets you work with databases using Python classes instead
of raw SQL strings. Benefits:

1. Database-agnostic: switch from SQLite → PostgreSQL → MySQL
   by only changing DATABASE_URL. Zero code changes.

2. Type safety: Python dataclasses that map to DB tables

3. Migration support: Alembic tracks schema changes like git tracks code

CONCEPT: The Session pattern
------------------------------
A "session" is a database transaction context. Best practice:
- Open a session per request
- Do your work
- Commit (save) or rollback (undo) 
- Always close the session

FastAPI's dependency injection handles this automatically via
the get_db() generator below.

ARCHITECTURE NOTE:
Phase 1: SQLite (file-based, zero config, perfect for development)
Phase 2: PostgreSQL (production-grade, supports concurrent users,
         enables row-level security for multi-tenancy)

Switching is as simple as changing DATABASE_URL in .env.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from loguru import logger

from backend.app.core.config import get_settings

settings = get_settings()

# ─── Engine Configuration ───────────────────────────────────
# The engine is the low-level connection to the database.
# We configure it differently for SQLite vs PostgreSQL.

if "sqlite" in settings.database_url:
    # SQLite special config:
    # - check_same_thread=False: SQLite by default only allows
    #   one thread. FastAPI uses multiple threads, so we disable this.
    # - StaticPool: reuse same connection (needed for in-memory SQLite in tests)
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.debug,  # Log SQL queries in debug mode
    )
    
    # Enable WAL mode for SQLite — allows concurrent reads
    # (Important: without this, Streamlit + FastAPI both reading
    # the same SQLite file would cause locking errors)
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL — used in Phase 2+
    # pool_size: max persistent connections
    # max_overflow: extra connections when pool is full
    engine = create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Check connection is alive before using
        echo=settings.debug,
    )

# ─── Session Factory ────────────────────────────────────────
# SessionLocal is a class — calling it creates a new session object.
# autocommit=False: we control when to commit (safer)
# autoflush=False: we control when to sync to DB
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ─── Base Model ─────────────────────────────────────────────
# All our SQLAlchemy models inherit from this.
# It provides the metadata registry that Alembic reads for migrations.
Base = declarative_base()


# ─── Dependency: get_db ─────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.
    
    The 'yield' makes this a generator — FastAPI calls the code
    before yield to set up, then the code after yield for teardown.
    
    Usage in FastAPI endpoints:
        @router.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    INTERVIEW CONCEPT: "context manager as dependency"
    This pattern ensures sessions are ALWAYS closed, even if an
    exception occurs — preventing connection pool exhaustion.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database error, rolling back: {e}")
        db.rollback()
        raise
    finally:
        db.close()  # Always executes, even on exception


def create_tables() -> None:
    """
    Create all tables defined in SQLAlchemy models.
    Called once at application startup.
    
    In Phase 2, we'll use Alembic migrations instead —
    migrations track schema changes and allow rollbacks.
    """
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created")
