"""
backend/app/models/models.py
=============================
SQLAlchemy ORM models — these define our database schema.

CONCEPT: ORM Models vs Pydantic Schemas
-----------------------------------------
You'll see two types of "models" in this project:

1. SQLAlchemy Models (this file):
   - Define database tables and columns
   - Used for DB read/write operations
   - Stored in backend/app/models/

2. Pydantic Schemas (backend/app/schemas/):
   - Define API request/response shapes
   - Used for data validation and serialization
   - What FastAPI sends to and receives from clients

The separation exists because what you store in the DB
and what you expose via API are often different
(e.g., you store password_hash but never return it in the API).

TABLES WE'RE CREATING:
  - users: who can access the system
  - documents: files ingested into the knowledge base
  - chunks: the split pieces of documents stored in vector DB
  - queries: audit log of every question asked
  - query_results: which chunks answered which query
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, 
    DateTime, Text, ForeignKey, JSON, Enum
)
from sqlalchemy.orm import relationship
import uuid
import enum

from backend.app.db.database import Base


# ─── Helper: UUID primary keys ──────────────────────────────
# We use UUIDs (not sequential integers) as primary keys.
# WHY: Sequential IDs leak information (user #3 vs user #1000000)
# and make horizontal scaling harder. UUIDs are random and globally unique.
def generate_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Always store timestamps in UTC. Never local time."""
    return datetime.now(timezone.utc)


# ─── Enums ──────────────────────────────────────────────────
class UserRole(str, enum.Enum):
    """
    Role-based access control roles.
    str + enum.Enum means roles are stored as strings in DB,
    which is human-readable in database queries.
    """
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"       # Just uploaded, not yet processed
    PROCESSING = "processing" # Being chunked and embedded
    INDEXED = "indexed"       # In vector DB, queryable
    FAILED = "failed"         # Processing failed


class QueryStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── User Model ─────────────────────────────────────────────
class User(Base):
    """
    Represents a system user.
    
    DESIGN DECISION: We store email (unique identifier), hashed password,
    role, and tenant_id. The tenant_id enables multi-tenancy in Phase 2 —
    users at Company A see only Company A's documents.
    
    SECURITY NOTE: We NEVER store plain text passwords.
    Only the bcrypt hash is stored. Even if the DB is compromised,
    attackers cannot recover passwords.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    tenant_id = Column(String, default="default", nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships — SQLAlchemy auto-joins these
    documents = relationship("Document", back_populates="uploaded_by_user")
    queries = relationship("Query", back_populates="user")

    def __repr__(self) -> str:
        return f"<User {self.email} [{self.role}]>"


# ─── Document Model ─────────────────────────────────────────
class Document(Base):
    """
    Represents an ingested document in the knowledge base.
    
    ARCHITECTURE NOTE: Documents live in TWO places:
    1. This table (metadata: filename, status, who uploaded, when)
    2. Weaviate (the actual content as vector embeddings)
    
    The doc_id links them — we store doc_id in both places.
    This is called "polyglot persistence" — using the right DB for each need:
    - PostgreSQL for structured metadata, search, audit
    - Weaviate for semantic similarity search
    """
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # "pdf", "md", "txt", "docx"
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    tenant_id = Column(String, default="default", nullable=False, index=True)
    
    # Who uploaded it
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=True)
    uploaded_by_user = relationship("User", back_populates="documents")
    
    # Processing metadata
    chunk_count = Column(Integer, default=0)  # How many chunks were created
    total_tokens = Column(Integer, default=0)  # Total tokens embedded
    embedding_model = Column(String, nullable=True)  # Which model was used
    
    # Error tracking
    error_message = Column(Text, nullable=True)  # If status=FAILED, why?
    
    # Source info (for citations)
    source_url = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    chunks = relationship("DocumentChunk", back_populates="document")

    def __repr__(self) -> str:
        return f"<Document {self.filename} [{self.status}]>"


# ─── Document Chunk Model ───────────────────────────────────
class DocumentChunk(Base):
    """
    Represents a single chunk of a document.
    
    CONCEPT: Why track chunks in the relational DB?
    
    Weaviate stores the chunk content + vector for semantic search.
    But PostgreSQL stores metadata about each chunk:
    - Which document it came from (for citations)
    - Which page/section (for UI display)
    - Token count (for context window management)
    
    When we retrieve chunks from Weaviate, we use chunk_id to look up
    this metadata in PostgreSQL for the full context.
    
    The weaviate_id links the PostgreSQL record to the Weaviate object.
    """
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    weaviate_id = Column(String, unique=True, nullable=True)  # ID in Weaviate
    
    chunk_index = Column(Integer, nullable=False)  # Position in document
    content = Column(Text, nullable=False)          # The actual text
    token_count = Column(Integer, nullable=False)
    
    # Location metadata (for citations in UI)
    page_number = Column(Integer, nullable=True)
    section_header = Column(String, nullable=True)  # "Chapter 3: Refund Policy"
    
    # Additional metadata stored as flexible JSON
    # e.g., {"heading_level": 2, "code_language": "python"}
    # Note: 'metadata' is reserved by SQLAlchemy's Declarative API, so we use 'extra_metadata'
    extra_metadata = Column(JSON, default={})
    
    created_at = Column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="chunks")


# ─── Query Model ────────────────────────────────────────────
class Query(Base):
    """
    Audit log of every query made to the system.
    
    ENTERPRISE CONCEPT: Audit logs are non-negotiable in enterprise AI.
    Every question, who asked it, what was returned, and how confident
    the system was — all stored for:
    - Debugging: "why did the system answer X incorrectly?"
    - Compliance: "what information did user Y access?"
    - Product analytics: "what are users asking most?"
    - Evaluation: ground truth for RAGAS scoring
    """
    __tablename__ = "queries"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    tenant_id = Column(String, default="default", nullable=False, index=True)
    
    # The question and answer
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    
    # Agent execution metadata
    status = Column(Enum(QueryStatus), default=QueryStatus.PENDING)
    agent_trace_id = Column(String, nullable=True)  # LangSmith trace ID
    
    # Quality metrics (filled in after RAGAS evaluation)
    faithfulness_score = Column(Float, nullable=True)
    relevancy_score = Column(Float, nullable=True)
    
    # Performance metrics
    latency_ms = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    estimated_cost_usd = Column(Float, nullable=True)
    
    # Error info
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="queries")
    retrieved_chunks = relationship("QueryChunkResult", back_populates="query")


# ─── Query-Chunk Result (Many-to-Many) ──────────────────────
class QueryChunkResult(Base):
    """
    Which chunks were retrieved and used to answer a specific query.
    
    This is a many-to-many join table: one query uses many chunks,
    one chunk can answer many queries.
    
    Storing retrieval scores lets us analyze our retrieval quality:
    "For question X, we retrieved chunk Y with score 0.92"
    — useful for improving the RAG pipeline over time.
    """
    __tablename__ = "query_chunk_results"

    id = Column(String, primary_key=True, default=generate_uuid)
    query_id = Column(String, ForeignKey("queries.id"), nullable=False, index=True)
    chunk_id = Column(String, ForeignKey("document_chunks.id"), nullable=True)
    
    retrieval_score = Column(Float, nullable=True)  # Cosine similarity score
    rerank_score = Column(Float, nullable=True)      # Score after re-ranking
    rank_position = Column(Integer, nullable=True)   # 1 = most relevant
    was_used = Column(Boolean, default=True)         # Was it in the final context?
    
    query = relationship("Query", back_populates="retrieved_chunks")
