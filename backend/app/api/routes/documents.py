"""
backend/app/api/routes/documents.py
Document management: upload, list, status, delete.
"""

import os
import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from loguru import logger

from backend.app.db.database import get_db
from backend.app.models.models import User, Document, DocumentStatus
from backend.app.schemas.schemas import (
    DocumentResponse, DocumentListResponse, IngestionStatusResponse
)
from backend.app.core.auth import get_current_user, require_manager_or_above
from backend.app.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_FILE_SIZE_MB = 50


@router.post(
    "/upload",
    response_model=IngestionStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion into the knowledge base",
)
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IngestionStatusResponse:

    # ── Validate file type ───────────────────────────────────
    filename = file.filename or "unnamed"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' not supported. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # ── Validate file size ───────────────────────────────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    # ── Save file to disk ────────────────────────────────────
    os.makedirs(settings.docs_dir, exist_ok=True)
    safe_filename = f"{int(time.time())}_{filename.replace(' ', '_')}"
    file_path = os.path.join(settings.docs_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"File saved: {file_path} ({size_mb:.2f}MB)")

    # ── Create DB record ─────────────────────────────────────
    document = Document(
        filename=filename,
        file_type=ext.lstrip("."),
        file_size_bytes=len(content),
        status=DocumentStatus.PENDING,
        description=description,
        source_url=source_url,
        uploaded_by=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    db.add(document)
    db.flush()
    logger.info(f"Document record created: {document.id} | '{filename}' by {current_user.email}")

    # ── Run ingestion pipeline ───────────────────────────────
    # BUG FIX: pass current_user.tenant_id so chunks are stored
    # under the same tenant namespace as the document record.
    # Previously tenant_id was never passed, defaulting to "default"
    # which caused a namespace mismatch between the document DB record
    # and the Weaviate chunks — making retrieval return nothing.
    try:
        await _run_ingestion_pipeline(
            document_id=document.id,
            file_path=file_path,
            tenant_id=current_user.tenant_id,
            db=db,
        )
    except Exception as e:
        logger.error(f"Ingestion failed for {document.id}: {e}")
        document.status = DocumentStatus.FAILED
        document.error_message = str(e)

    return IngestionStatusResponse(
        document_id=document.id,
        filename=filename,
        status=document.status,
        chunks_created=document.chunk_count,
    )


async def _run_ingestion_pipeline(
    document_id: str,
    file_path: str,
    tenant_id: str,
    db: Session,
) -> None:
    from ingestion.pipeline.ingest import run_ingestion

    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise ValueError(f"Document {document_id} not found")

    document.status = DocumentStatus.PROCESSING
    db.flush()

    # BUG FIX: tenant_id is now passed through correctly
    result = await run_ingestion(
        document_id=document_id,
        file_path=file_path,
        tenant_id=tenant_id,
    )

    document.chunk_count = result["chunks_created"]
    document.total_tokens = result["total_tokens"]
    document.embedding_model = result["embedding_model"]
    document.status = DocumentStatus.INDEXED

    logger.info(
        f"Ingestion complete: {document_id} | "
        f"{result['chunks_created']} chunks | "
        f"{result['total_tokens']} tokens"
    )


@router.get("/", response_model=DocumentListResponse)
def list_documents(
    page: int = 1,
    page_size: int = 20,
    status_filter: Optional[DocumentStatus] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    query = db.query(Document).filter(Document.tenant_id == current_user.tenant_id)
    if status_filter:
        query = query.filter(Document.status == status_filter)
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in documents],
        total=total, page=page, page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Document:
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.tenant_id == current_user.tenant_id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_manager_or_above)],
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.tenant_id == current_user.tenant_id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(document)
    logger.info(f"Document deleted: {document_id} by {current_user.email}")