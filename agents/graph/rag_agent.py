"""
agents/graph/rag_agent.py
==========================
LangGraph RAG agent — retrieve → generate → critique → END.

Embedding: sentence-transformers all-MiniLM-L6-v2 (local, free)
LLM:       GPT-4o-mini via OpenAI API (requires credits)
           OR any OpenAI-compatible local endpoint (e.g. Ollama)
"""

import os
import time
from typing import TypedDict, Optional
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from ingestion.embedders.vector_store import semantic_search
from agents.prompts.system_prompts import RAG_SYSTEM_PROMPT, QA_CRITIQUE_PROMPT
from backend.app.core.config import get_settings

settings = get_settings()


# ─── State ──────────────────────────────────────────────────

class RAGState(TypedDict):
    question:          str
    tenant_id:         str
    user_role:         str
    retrieved_chunks:  list
    context_text:      str
    draft_answer:      str
    final_answer:      str
    sources:           list
    faithfulness_score: float
    relevance_score:   float
    critique_notes:    list
    needs_refinement:  bool
    tokens_used:       int
    latency_ms:        int
    trace_id:          Optional[str]
    error:             Optional[str]
    refinement_attempts: int


# ─── LLM factory ────────────────────────────────────────────

def _get_llm():
    """
    Return a ChatOpenAI instance.
    temperature=0 for deterministic, factual RAG responses.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )


# ─── Nodes ──────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    """
    Search Weaviate using a locally-generated query embedding.
    Uses near_vector() internally — no OpenAI call at this stage.
    """
    logger.info(f"[retrieve] Searching: '{state['question'][:60]}'")

    chunks = semantic_search(
        query=state["question"],
        tenant_id=state["tenant_id"],
        top_k=settings.retrieval_top_k,
    )

    if not chunks:
        logger.warning("[retrieve] No chunks found — knowledge base empty or no matching content")
        return {
            "retrieved_chunks": [],
            "context_text": "No relevant documents found in the knowledge base.",
            "sources": [],
        }

    # Format numbered context block for the LLM prompt
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page = f", page {chunk['page_number']}" if chunk.get("page_number") else ""
        section = f", section '{chunk['section_header']}'" if chunk.get("section_header") else ""
        parts.append(
            f"[{i}] Source: {chunk['document_name']}{page}{section}\n"
            f"Relevance: {chunk['score']:.2f}\n"
            f"Content:\n{chunk['content']}"
        )

    context_text = "\n---\n".join(parts)
    logger.info(f"[retrieve] {len(chunks)} chunks | top score: {chunks[0]['score']:.3f}")

    return {
        "retrieved_chunks": chunks,
        "context_text": context_text,
        "sources": chunks,
    }


def generate_node(state: RAGState) -> dict:
    """
    Call the LLM with retrieved context to generate a cited answer.
    This is the only node that calls the OpenAI API.
    """
    logger.info("[generate] Generating answer...")

    if not state["retrieved_chunks"]:
        return {
            "draft_answer": (
                "I couldn't find any relevant information in the knowledge base "
                "to answer your question. Please upload relevant documents first."
            ),
            "tokens_used": 0,
            "refinement_attempts": state.get("refinement_attempts", 0),
        }

    llm = _get_llm()

    user_message = (
        f"Question: {state['question']}\n\n"
        f"Context Documents:\n{state['context_text']}\n\n"
        "Please answer the question based solely on the provided context. "
        "Include citations for every factual claim."
    )

    response = llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = response.usage_metadata.get("total_tokens", 0)

    logger.info(f"[generate] {len(response.content)} chars | {tokens_used} tokens")

    return {
        "draft_answer": response.content,
        "tokens_used": state.get("tokens_used", 0) + tokens_used,
        "refinement_attempts": state.get("refinement_attempts", 0),
    }


def critique_node(state: RAGState) -> dict:
    """
    Self-critique: check if the answer is grounded in the context.
    Skipped if no chunks were retrieved (nothing to verify against).
    """
    if not state["retrieved_chunks"]:
        return {
            "final_answer":      state["draft_answer"],
            "faithfulness_score": 0.0,
            "relevance_score":   0.0,
            "critique_notes":    [],
            "needs_refinement":  False,
        }

    logger.info("[critique] Evaluating answer quality...")

    llm = _get_llm()
    prompt = QA_CRITIQUE_PROMPT.format(
        context=state["context_text"],
        question=state["question"],
        answer=state["draft_answer"],
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = response.usage_metadata.get("total_tokens", 0)

    import json
    faithfulness, relevance, issues, needs_revision = 0.75, 0.75, [], False
    revised = None
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        critique = json.loads(content)
        faithfulness  = float(critique.get("faithfulness_score", 0.75))
        relevance     = float(critique.get("relevance_score", 0.75))
        issues        = critique.get("issues_found", [])
        needs_revision = critique.get("verdict") == "needs_revision"
        revised       = critique.get("revised_answer")
    except Exception as e:
        logger.warning(f"[critique] Could not parse JSON: {e}")

    if revised and needs_revision:
        return {
            "final_answer":      revised,
            "faithfulness_score": faithfulness,
            "relevance_score":   relevance,
            "critique_notes":    issues,
            "needs_refinement":  False,
            "tokens_used": state.get("tokens_used", 0) + tokens_used,
        }

    needs_refinement = needs_revision and state.get("refinement_attempts", 0) < 1
    logger.info(f"[critique] faithfulness={faithfulness:.2f} relevance={relevance:.2f}")

    return {
        "final_answer":      state["draft_answer"],
        "faithfulness_score": faithfulness,
        "relevance_score":   relevance,
        "critique_notes":    issues,
        "needs_refinement":  needs_refinement,
        "tokens_used": state.get("tokens_used", 0) + tokens_used,
    }


def refine_node(state: RAGState) -> dict:
    """Re-generate the answer addressing critique issues."""
    logger.info(f"[refine] Fixing issues: {state['critique_notes']}")

    llm = _get_llm()
    message = (
        f"Your previous answer had these issues:\n"
        + "\n".join(f"- {i}" for i in state["critique_notes"])
        + f"\n\nOriginal question: {state['question']}\n\n"
        f"Context:\n{state['context_text']}\n\n"
        f"Previous answer:\n{state['draft_answer']}\n\n"
        "Provide an improved answer fixing all issues. Only use the provided context."
    )

    response = llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])
    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = response.usage_metadata.get("total_tokens", 0)

    return {
        "final_answer":      response.content,
        "draft_answer":      response.content,
        "needs_refinement":  False,
        "refinement_attempts": state.get("refinement_attempts", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + tokens_used,
    }


# ─── Routing ────────────────────────────────────────────────

def should_refine(state: RAGState) -> str:
    return "refine" if state.get("needs_refinement") else END


# ─── Graph ──────────────────────────────────────────────────

_graph = None


def _build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("critique", critique_node)
    g.add_node("refine",   refine_node)
    g.add_edge(START,      "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "critique")
    g.add_edge("refine",   "critique")
    g.add_conditional_edges("critique", should_refine, {"refine": "refine", END: END})
    return g.compile()


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ─── Entry point ────────────────────────────────────────────

async def run_rag_query(
    question: str,
    tenant_id: str = "default",
    user_role: str = "viewer",
) -> dict:
    """Run the full RAG pipeline for one question."""
    start_time = time.time()

    # Disable LangSmith tracing if no valid key
    if not (settings.langchain_api_key or "").strip() or not settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    initial: RAGState = {
        "question":           question,
        "tenant_id":          tenant_id,
        "user_role":          user_role,
        "retrieved_chunks":   [],
        "context_text":       "",
        "draft_answer":       "",
        "final_answer":       "",
        "sources":            [],
        "faithfulness_score": 0.0,
        "relevance_score":    0.0,
        "critique_notes":     [],
        "needs_refinement":   False,
        "tokens_used":        0,
        "latency_ms":         0,
        "trace_id":           None,
        "error":              None,
        "refinement_attempts": 0,
    }

    final = await _get_graph().ainvoke(initial)

    latency_ms = int((time.time() - start_time) * 1000)
    sources = final.get("sources", [])
    scores = [s.get("score", 0) for s in sources]
    confidence = sum(scores) / len(scores) if scores else 0.0

    return {
        "answer":            final.get("final_answer", "Unable to generate an answer."),
        "sources":           sources,
        "confidence":        round(confidence, 3),
        "faithfulness_score": final.get("faithfulness_score"),
        "tokens_used":       final.get("tokens_used", 0),
        "latency_ms":        latency_ms,
        "trace_id":          None,
        "cost_usd":          round(final.get("tokens_used", 0) / 1_000_000 * 0.20, 6),
    }