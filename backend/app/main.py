"""
backend/app/main.py  — NexusIQ FastAPI entry point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging
from backend.app.db.database import create_tables
from backend.app.api.routes import auth, documents, queries

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────
    setup_logging()
    logger.info(f"🚀 NexusIQ starting up | env={settings.app_env}")

    create_tables()
    logger.info("✅ Database ready")

    # Weaviate startup check.
    # BUG FIX: is_ready() and is_live() are both async in weaviate-client v4.
    # Calling them without await returns a coroutine (truthy) not a bool.
    # Use is_connected() which is synchronous, or just catch the connection error.
    # We don't fail startup if Weaviate is down — it's checked per-request.
    try:
        from ingestion.embedders.vector_store import get_weaviate_client
        client = get_weaviate_client()
        # is_connected() is the only synchronous health method in v4
        if client.is_connected():
            logger.info("✅ Weaviate connected")
        else:
            logger.warning("⚠️  Weaviate not connected — start it with: docker compose up weaviate -d")
    except Exception as e:
        logger.warning(f"⚠️  Weaviate unavailable at startup (will retry on first request): {e}")

    logger.info("✅ NexusIQ ready to serve requests")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("NexusIQ shutting down...")
    try:
        from ingestion.embedders.vector_store import _weaviate_client
        if _weaviate_client is not None:
            _weaviate_client.close()
    except Exception:
        pass


app = FastAPI(
    title="NexusIQ API",
    description="""
**NexusIQ** — Autonomous Enterprise Intelligence Platform

Ingests documents, answers questions with citations, tracks every query.

## Authentication
Use `/api/v1/auth/login` to get a Bearer token, then include it as:
`Authorization: Bearer <token>`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://your-production-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(auth.router,      prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(queries.router,   prefix=API_PREFIX)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check — used by load balancers and monitoring."""
    from backend.app.db.database import engine
    from sqlalchemy import text

    services = {}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        services["database"] = "ok"
    except Exception as e:
        services["database"] = f"error: {e}"

    # BUG FIX: is_ready() is async in weaviate-client v4 — use is_connected() instead.
    # Calling is_ready() without await returns a coroutine object (always truthy).
    try:
        from ingestion.embedders.vector_store import get_weaviate_client
        client = get_weaviate_client()
        services["weaviate"] = "ok" if client.is_connected() else "unavailable"
    except Exception:
        services["weaviate"] = "unavailable"

    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.app_env,
        "services": services,
    }


@app.get("/", tags=["System"])
async def root():
    return {"message": "NexusIQ API is running", "docs": "/docs", "health": "/health"}