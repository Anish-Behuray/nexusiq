"""
ingestion/chunkers/chunker.py
===============================
Chunking: splitting documents into smaller pieces for embedding.

CONCEPT: Why do we chunk documents?
--------------------------------------
LLMs have a context window limit (e.g., GPT-4o can handle ~128K tokens).
But more importantly, embedding models have limits too (typically 512-8192 tokens).

The deeper reason: PRECISION.

If you embed an entire 50-page PDF, the embedding is an average of the whole document.
When you search for "refund policy," a document about HR policies AND refund policies 
might score similarly — because the embedding blurs everything together.

When you chunk the document into 512-token pieces:
- Chunk A: "...employee vacation policies..." → far from query
- Chunk B: "...our refund policy is 30 days..." → very close to query ✓

Chunking → better precision in retrieval.

CHUNKING STRATEGIES (know these for interviews):

1. Fixed-size chunking:
   Split every N tokens, with M token overlap.
   Simple. Fast. But cuts sentences mid-way. ❌
   
2. Sentence-based chunking (what we use):
   Split on sentence boundaries. Never cuts a sentence.
   Groups sentences until hitting the token limit. ✓
   
3. Semantic chunking (Phase 2):
   Uses embedding similarity to find natural topic boundaries.
   Groups text by "ideas" not just size. Most accurate. Expensive. ✓✓
   
4. Hierarchical chunking (Phase 2):
   Creates BOTH large "parent" chunks and small "child" chunks.
   Retrieve small chunks (precision) but return large chunks (context).
   The best of both worlds. Used by LlamaIndex's ParentDocumentRetriever.

WHAT WE BUILD HERE: Sentence-based chunking with metadata preservation.
The metadata from the loader (page_number, section_header, document_id)
must survive chunking — every chunk needs to know where it came from.
"""

import tiktoken
from typing import Any
from loguru import logger
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


@dataclass
class Chunk:
    """
    A single chunk of text, ready for embedding.
    
    We use a dataclass instead of a dict because:
    - Type hints enforce structure
    - IDE autocomplete works
    - Easier to evolve without breaking callers
    
    CRITICAL FIELDS:
    - content: the text that gets embedded
    - metadata: preserved from the original document (for citations)
    - token_count: helps us track costs and context window usage
    """
    content: str
    metadata: dict[str, Any]
    token_count: int
    chunk_index: int   # Position within this document


# ─── Token Counter ──────────────────────────────────────────
# tiktoken is OpenAI's tokenizer — it counts tokens the SAME way
# the API counts them. This lets us:
# 1. Verify chunks fit in embedding model limits
# 2. Accurately estimate API costs before sending
# 3. Enforce context window budgets
# NOTE: Loaded lazily on first use to avoid startup network calls.

_tokenizer = None

def _get_tokenizer():
    """Returns the tiktoken tokenizer, loading it on first call."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback: approximate token count (1 token ≈ 4 chars)
            return None
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens using OpenAI's tokenizer, with character fallback."""
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        # Fallback approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text))


# ─── Main Chunker ───────────────────────────────────────────

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split a list of Documents into Chunks for embedding.
    
    Uses LlamaIndex's SentenceSplitter which:
    1. Splits on sentence boundaries (respects punctuation)
    2. Groups sentences together until chunk_size is reached
    3. Adds overlap between chunks (so no context is cut off)
    
    The overlap means adjacent chunks share some sentences.
    Example with chunk_size=4, overlap=1:
      Text: [S1, S2, S3, S4, S5, S6]
      Chunk 1: [S1, S2, S3, S4]
      Chunk 2: [S4, S5, S6]  ← S4 repeated for continuity
    
    Args:
        documents: List of Document objects from the loader
        chunk_size: Max tokens per chunk (default 512)
        chunk_overlap: Token overlap between adjacent chunks (default 50)
    
    Returns:
        List of Chunk objects with content + metadata
    """
    if not documents:
        return []
    
    # LlamaIndex SentenceSplitter — industry standard for sentence-aware chunking
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # paragraph_separator: split at double newlines first (natural boundaries)
        paragraph_separator="\n\n",
        # secondary_chunking_regex: fallback for very long sentences
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
    )
    
    all_chunks = []
    chunk_index = 0
    
    for doc in documents:
        if not doc.text or not doc.text.strip():
            logger.debug(f"Skipping empty document section")
            continue
        
        # SentenceSplitter returns LlamaIndex TextNode objects
        nodes: list[TextNode] = splitter.get_nodes_from_documents([doc])
        
        for node in nodes:
            content = node.get_content()
            
            if not content.strip():
                continue
            
            # Quality filter: skip chunks that are too short to be useful
            # (e.g., a page with only a page number "- 47 -")
            token_count = count_tokens(content)
            if token_count < 10:
                logger.debug(f"Skipping very short chunk ({token_count} tokens)")
                continue
            
            # Merge metadata: document-level + node-level
            # node.metadata has the original doc metadata (from loader)
            chunk_metadata = {
                **doc.metadata,     # document_id, filename, page_number, etc.
                **node.metadata,    # Any LlamaIndex-added metadata
                "chunk_size_tokens": token_count,
                # This chunk_index is global across the entire ingestion job
                "chunk_position": chunk_index,
            }
            
            chunk = Chunk(
                content=content,
                metadata=chunk_metadata,
                token_count=token_count,
                chunk_index=chunk_index,
            )
            all_chunks.append(chunk)
            chunk_index += 1
    
    # ─── Validation Report ──────────────────────────────────
    if all_chunks:
        token_counts = [c.token_count for c in all_chunks]
        avg_tokens = sum(token_counts) / len(token_counts)
        logger.info(
            f"Chunking complete: {len(all_chunks)} chunks | "
            f"avg {avg_tokens:.0f} tokens | "
            f"range: {min(token_counts)}-{max(token_counts)} tokens"
        )
        
        # Warn if any chunks are too large
        oversized = [c for c in all_chunks if c.token_count > chunk_size * 1.2]
        if oversized:
            logger.warning(
                f"{len(oversized)} chunks exceed chunk_size by >20%. "
                f"This may indicate very long sentences in the document."
            )
    else:
        logger.warning("No chunks produced! Check document content.")
    
    return all_chunks


def estimate_ingestion_cost(chunks: list[Chunk], model: str = "text-embedding-3-small") -> float:
    """
    Estimate the cost of embedding all chunks.
    
    WHY THIS MATTERS FOR INTERVIEWS:
    "I built cost awareness into the ingestion pipeline. Before
    sending 10,000 chunks to the API, the system estimates the cost
    so we can catch accidental expensive ingestions early."
    
    Pricing (as of 2024):
    - text-embedding-3-small: $0.02 / 1M tokens
    - text-embedding-3-large: $0.13 / 1M tokens
    """
    total_tokens = sum(c.token_count for c in chunks)
    
    price_per_million = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }
    
    price = price_per_million.get(model, 0.02)
    estimated_cost = (total_tokens / 1_000_000) * price
    
    logger.info(
        f"Ingestion cost estimate: {total_tokens:,} tokens × "
        f"${price}/M = ${estimated_cost:.4f}"
    )
    
    return estimated_cost
