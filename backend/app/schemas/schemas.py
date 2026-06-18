"""
backend/app/schemas/schemas.py
================================
Pydantic v2 schemas for API request/response validation.

CONCEPT: Why separate schemas from ORM models?
-----------------------------------------------
SQLAlchemy models define the database structure.
Pydantic schemas define what the API accepts and returns.

They're different because:
1. The DB stores hashed_password — the API must NEVER return it
2. The API accepts plain_text password — the DB must NEVER store it
3. The API might have computed fields (e.g., doc_count) not in DB
4. You might want different response shapes for different endpoints

This separation enforces security and flexibility.

PATTERN: Request → Schema validates input → Model saves to DB
         DB row → Model → Schema serializes output → API response
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, field_validator
from backend.app.models.models import UserRole, DocumentStatus, QueryStatus


# ─── Base Schema Pattern ────────────────────────────────────
# We use a "Base + Create + Response" pattern per entity.
# Base: shared fields
# Create: fields needed when creating (includes things like password)
# Response: what the API returns (excludes sensitive fields)


# ══════════════════════════════════════════════════════════════
# USER SCHEMAS
# ══════════════════════════════════════════════════════════════

class UserBase(BaseModel):
    """Fields shared across all user-related schemas."""
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole = UserRole.VIEWER


class UserCreate(UserBase):
    """
    What the API receives when creating a new user.
    Includes plain text password — validated here, hashed before DB storage.
    """
    password: str = Field(..., min_length=8, description="Min 8 characters")
    
    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        """
        Basic password strength check.
        INDUSTRY PRACTICE: Validate at the API boundary, not in the DB.
        """
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v


class UserResponse(UserBase):
    """
    What the API returns when a user is requested.
    Note: NO password field. This is intentional and critical.
    """
    id: str
    tenant_id: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}  # Allows creating from SQLAlchemy model


class UserUpdate(BaseModel):
    """Fields that can be updated on a user (all optional)."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


# ══════════════════════════════════════════════════════════════
# AUTHENTICATION SCHEMAS
# ══════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    """What the client sends to log in."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """What the server returns after successful login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until token expires
    user: UserResponse


# ══════════════════════════════════════════════════════════════
# DOCUMENT SCHEMAS
# ══════════════════════════════════════════════════════════════

class DocumentResponse(BaseModel):
    """What the API returns for a document."""
    id: str
    filename: str
    file_type: str
    file_size_bytes: Optional[int]
    status: DocumentStatus
    chunk_count: int
    total_tokens: int
    description: Optional[str]
    source_url: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class IngestionStatusResponse(BaseModel):
    """Real-time status during document ingestion."""
    document_id: str
    filename: str
    status: DocumentStatus
    progress_pct: float = 0.0  # 0-100
    chunks_created: int = 0
    error_message: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# QUERY / CHAT SCHEMAS
# ══════════════════════════════════════════════════════════════

class Source(BaseModel):
    """A source document that contributed to an answer."""
    document_id: str
    document_name: str
    chunk_content: str   # The actual text that was used
    relevance_score: float
    page_number: Optional[int] = None
    section_header: Optional[str] = None


class QueryRequest(BaseModel):
    """What the client sends when asking a question."""
    question: str = Field(
        ..., 
        min_length=3, 
        max_length=2000,
        description="The question to answer from the knowledge base"
    )
    # In Phase 2: session_id for conversation memory
    session_id: Optional[str] = None
    
    # Optional filter: only search specific documents
    document_ids: Optional[List[str]] = None
    
    # Whether to include source citations in response
    include_sources: bool = True


class QueryResponse(BaseModel):
    """
    What the API returns after answering a query.
    
    DESIGN DECISION: Always include sources.
    This is what differentiates RAG from a plain chatbot —
    every answer is traceable to a specific document and chunk.
    Interviewers love asking: "How do you handle hallucinations?"
    Answer: "Every claim is grounded in retrieved context,
    and we show the source so users can verify."
    """
    query_id: str
    question: str
    answer: str
    sources: List[Source]          # Which chunks answered this
    confidence_score: float        # 0-1, derived from retrieval scores
    faithfulness_score: Optional[float] = None  # RAGAS (computed async)
    latency_ms: int
    tokens_used: int
    model_used: str


class QueryHistoryResponse(BaseModel):
    """Summary of a past query (for history/audit views)."""
    id: str
    question: str
    answer: Optional[str]
    status: QueryStatus
    faithfulness_score: Optional[float]
    latency_ms: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}


# ══════════════════════════════════════════════════════════════
# SYSTEM / HEALTH SCHEMAS
# ══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """
    Health check response — used by load balancers and monitoring.
    A 200 response means the service is healthy.
    """
    status: str = "healthy"
    version: str
    environment: str
    services: dict  # e.g., {"database": "ok", "weaviate": "ok"}


class StatsResponse(BaseModel):
    """Dashboard statistics."""
    total_documents: int
    indexed_documents: int
    total_queries: int
    avg_faithfulness_score: Optional[float]
    avg_latency_ms: Optional[float]
