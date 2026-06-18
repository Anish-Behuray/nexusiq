"""
backend/app/core/config.py
==========================
Central configuration module for NexusIQ.

CONCEPT: Why a dedicated config module?
----------------------------------------
In production systems, configuration is never hardcoded. Instead, you:
1. Define what settings exist and their types (this file, using Pydantic)
2. Store actual values in environment variables (.env file locally, 
   AWS Secrets Manager / GCP Secret Manager in production)
3. Access settings through a single object — never through os.environ directly

This pattern is called "12-factor app configuration" and is used by
every serious production system. It means:
- One place to see ALL settings
- Type safety (Pydantic validates types on startup)
- Automatic .env loading
- Easy to mock in tests

Pydantic BaseSettings automatically reads from:
  1. Environment variables (highest priority)
  2. .env file 
  3. Default values defined here (lowest priority)
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All NexusIQ configuration in one place.
    
    Pydantic validates types on startup — if OPENAI_API_KEY is missing,
    the app crashes immediately with a clear error rather than failing
    silently at the first API call.
    """
    
    # ─── Model Configuration ────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file in project root
        env_file_encoding="utf-8",
        case_sensitive=False,      # OPENAI_API_KEY == openai_api_key
        extra="ignore",            # Ignore unknown env vars (don't crash)
    )

    # ─── Application ────────────────────────────────────────
    app_name: str = "NexusIQ"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ─── API Server ─────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ─── LLM Providers ──────────────────────────────────────
    # Field(...) means "required — no default value"
    openai_api_key: str = Field(..., description="OpenAI API key — required")
    cohere_api_key: str = Field("", description="Cohere API key — for Phase 2 re-ranking")

    # ─── Database ───────────────────────────────────────────
    database_url: str = "sqlite:///./nexusiq.db"

    # ─── Security ───────────────────────────────────────────
    secret_key: str = Field(..., description="JWT signing secret — must be long random string")
    access_token_expire_minutes: int = 30

    # ─── Vector Database ────────────────────────────────────
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""  # Empty = no auth (local dev)

    # ─── LangSmith Observability ────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "nexusiq-dev"

    # ─── RAG Pipeline ───────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"   # Cheaper model for dev; switch to gpt-4o for prod
    retrieval_top_k: int = 10         # Fetch top-10 before re-ranking
    rerank_top_k: int = 5             # Keep top-5 after re-ranking

    # ─── Ingestion ──────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    docs_dir: str = "./data/raw"

    # ─── Computed Properties ────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def weaviate_collection_name(self) -> str:
        """
        Weaviate collection names are environment-scoped.
        This prevents dev data from polluting production indexes.
        """
        return f"NexusIQ_{self.app_env.capitalize()}"

    @field_validator("secret_key")
    @classmethod
    def secret_key_must_be_strong(cls, v: str) -> str:
        """
        INDUSTRY PRACTICE: Enforce minimum secret key length.
        Short keys are cryptographically weak.
        In production this should be 64+ chars.
        """
        if len(v) < 16 and v != "your-secret-key-here-change-in-production":
            raise ValueError("SECRET_KEY must be at least 16 characters")
        return v


# ─── Singleton Pattern ──────────────────────────────────────
# @lru_cache means this function is only called ONCE.
# Every time get_settings() is called, it returns the same object.
# This avoids re-reading the .env file on every request.
#
# INTERVIEW CONCEPT: This is the "singleton pattern" applied to configuration.
# FastAPI's dependency injection uses this extensively.

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    
    Usage:
        from backend.app.core.config import get_settings
        settings = get_settings()
        print(settings.llm_model)
    
    In FastAPI endpoints:
        @app.get("/")
        def root(settings: Settings = Depends(get_settings)):
            ...
    """
    return Settings()
