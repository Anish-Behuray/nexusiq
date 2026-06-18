"""
backend/app/api/routes/queries.py
===================================
Query endpoints — the heart of the RAG system.

This is where user questions come in and answers go out.
It connects the FastAPI layer to the agent orchestration layer.

ENDPOINTS:
  POST /queries/ask          — ask a question, get a cited answer
  GET  /queries/history      — user's query history
  GET  /queries/{id}         — details of a specific query
  GET  /queries/{id}/feedback — submit feedback (thumbs up/down)
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from loguru import logger

from backend.app.db.database import get_db
from backend.app.models.models import User, Query, QueryStatus
from backend.app.schemas.schemas import (
    QueryRequest, QueryResponse, QueryHistoryResponse, Source
)
from backend.app.core.auth import get_current_user
from backend.app.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/queries", tags=["Queries"])


@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question to the knowledge base",
)
async def ask_question(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QueryResponse:
    """
    The main RAG query endpoint.
    
    FLOW:
    1. Validate and log the incoming question
    2. Create a Query record (audit trail — always first)
    3. Call the RAG agent pipeline
    4. Update Query record with answer and metrics
    5. Schedule async RAGAS evaluation (background task)
    6. Return cited answer to user
    
    DESIGN PATTERN: We create the DB audit record BEFORE calling the AI.
    This ensures that even if the AI call fails, we have a record
    of the attempt. Critical for debugging and compliance.
    
    BACKGROUND TASKS: FastAPI's BackgroundTasks runs after the response
    is sent. We use it for RAGAS evaluation — it's expensive and the
    user doesn't need to wait for it.
    """
    start_time = time.time()
    
    # ─── Create audit record ────────────────────────────────
    query_record = Query(
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
        question=request.question,
        status=QueryStatus.PROCESSING,
    )
    db.add(query_record)
    db.flush()  # Get query_record.id
    
    logger.info(
        f"Query received: {query_record.id} | "
        f"User: {current_user.email} | "
        f"Q: '{request.question[:80]}...'"
    )
    
    # ─── Run RAG pipeline ───────────────────────────────────
    try:
        from agents.graph.rag_agent import run_rag_query
        
        result = await run_rag_query(
            question=request.question,
            tenant_id=current_user.tenant_id,
            user_role=current_user.role.value,
        )
        
        # ─── Update audit record ────────────────────────────
        latency_ms = int((time.time() - start_time) * 1000)
        
        query_record.answer = result["answer"]
        query_record.status = QueryStatus.COMPLETED
        query_record.latency_ms = latency_ms
        query_record.tokens_used = result.get("tokens_used", 0)
        query_record.estimated_cost_usd = result.get("cost_usd", 0.0)
        query_record.agent_trace_id = result.get("trace_id")
        
        # ─── Schedule background evaluation ─────────────────
        # RAGAS evaluation runs AFTER we return the response
        # The user gets their answer fast; evaluation happens async
        background_tasks.add_task(
            _evaluate_query_async,
            query_id=query_record.id,
            question=request.question,
            answer=result["answer"],
            contexts=[s["content"] for s in result["sources"]],
        )
        
        logger.info(
            f"Query answered: {query_record.id} | "
            f"Latency: {latency_ms}ms | "
            f"Sources: {len(result['sources'])}"
        )
        
        return QueryResponse(
            query_id=query_record.id,
            question=request.question,
            answer=result["answer"],
            sources=[
                Source(
                    document_id=s["document_id"],
                    document_name=s["document_name"],
                    chunk_content=s["content"],
                    relevance_score=s["score"],
                    page_number=s.get("page_number"),
                    section_header=s.get("section_header"),
                )
                for s in result["sources"]
            ],
            confidence_score=result.get("confidence", 0.0),
            latency_ms=latency_ms,
            tokens_used=result.get("tokens_used", 0),
            model_used=settings.llm_model,
        )
        
    except Exception as e:
        # Update audit record with failure
        query_record.status = QueryStatus.FAILED
        query_record.error_message = str(e)
        logger.error(f"Query failed: {query_record.id} | Error: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


async def _evaluate_query_async(
    query_id: str,
    question: str,
    answer: str,
    contexts: list[str],
) -> None:
    """
    Background task: run RAGAS evaluation after response is sent.
    
    RAGAS measures:
    - Faithfulness: Is the answer supported by the retrieved context?
      (Detects hallucinations — the AI making things up)
    - Answer Relevancy: Does the answer actually address the question?
    
    Results are stored in the Query record for dashboard analytics.
    
    In Phase 1: We run RAGAS here but skip it if no RAGAS key available.
    In Phase 2: We'll run this in a Celery worker for better isolation.
    """
    try:
        # Import here to avoid circular imports and slow startup
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        
        eval_data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })
        
        scores = evaluate(eval_data, metrics=[faithfulness, answer_relevancy])
        
        # Update DB with scores
        from backend.app.db.database import SessionLocal
        with SessionLocal() as db:
            query = db.query(Query).filter(Query.id == query_id).first()
            if query:
                query.faithfulness_score = scores["faithfulness"]
                query.relevancy_score = scores["answer_relevancy"]
                db.commit()
                
        logger.info(
            f"RAGAS eval complete: {query_id} | "
            f"Faithfulness: {scores['faithfulness']:.3f} | "
            f"Relevancy: {scores['answer_relevancy']:.3f}"
        )
        
    except Exception as e:
        # RAGAS failures should never block the main flow
        logger.warning(f"RAGAS evaluation failed for {query_id}: {e}")


@router.get(
    "/history",
    response_model=list[QueryHistoryResponse],
    summary="Get query history for current user",
)
def get_query_history(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Query]:
    """
    Returns the user's recent query history.
    Useful for the chat UI to show previous conversations.
    """
    queries = db.query(Query).filter(
        Query.user_id == current_user.id
    ).order_by(Query.created_at.desc()).limit(limit).all()
    
    return queries


@router.get(
    "/{query_id}",
    response_model=QueryHistoryResponse,
    summary="Get details of a specific query",
)
def get_query(
    query_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Query:
    """
    Get a specific query's details including RAGAS scores.
    Used for debugging and quality monitoring.
    """
    query = db.query(Query).filter(
        Query.id == query_id,
        Query.user_id == current_user.id,  # Users can only see their own queries
    ).first()
    
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    return query
