"""
ingestion/embedders/vector_store.py
=====================================
Weaviate v4.7.1 — using local embeddings (no OpenAI vectorizer).

KEY ARCHITECTURE CHANGE vs previous version:
  Before: Weaviate called OpenAI to vectorize on insert and query
          → requires OPENAI_API_KEY with credits, fails with 429

  After:  We generate vectors locally using sentence-transformers,
          then pass them explicitly to Weaviate (none vectorizer).
          Weaviate stores and indexes our vectors without calling any API.
          On query, we embed the question locally and pass the vector directly.
          → zero cost, zero API keys, works fully offline

Weaviate collection uses:
  vectorizer_config = Configure.Vectorizer.none()   ← no external calls
  On insert: batch.add_object(properties=..., vector=our_vector)
  On query:  collection.query.near_vector(vector=our_query_vector)
"""

import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.query import MetadataQuery
from typing import Optional, List
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.embedders.embedder import embed_texts, embed_query
from backend.app.core.config import get_settings

settings = get_settings()

_weaviate_client: Optional[weaviate.WeaviateClient] = None


def _parse_weaviate_url(url: str):
    """Parse WEAVIATE_URL → (host, port). Handles http://localhost:8080 etc."""
    url = url.strip().rstrip("/")
    url = url.replace("https://", "").replace("http://", "")
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        try:
            return host.strip(), int(port_str)
        except ValueError:
            return url.strip(), 8080
    return url.strip(), 8080


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Return a singleton Weaviate v4 client.

    Fix preserved: strip() the api_key before truthy check.
    Any whitespace-only value must NOT trigger connect_to_weaviate_cloud()
    because that function hardcodes port=443 and constructs 'localhost:8080:443'.
    """
    global _weaviate_client

    if _weaviate_client is not None and _weaviate_client.is_connected():
        return _weaviate_client

    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
        except Exception:
            pass
        _weaviate_client = None

    logger.info(f"Connecting to Weaviate at {settings.weaviate_url}")

    api_key = (settings.weaviate_api_key or "").strip()

    if api_key:
        logger.info("Using Weaviate Cloud connection")
        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
        )
    else:
        host, port = _parse_weaviate_url(settings.weaviate_url)
        logger.info(f"Using local Weaviate: host={host}, http={port}, grpc=50051")
        _weaviate_client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051,
        )

    logger.info("✅ Weaviate client connected")
    return _weaviate_client


# ─── Schema ─────────────────────────────────────────────────

def ensure_collection_exists(collection_name: str) -> None:
    """
    Create the Weaviate collection if it does not exist.

    CRITICAL CHANGE: vectorizer_config=Configure.Vectorizer.none()
    This tells Weaviate NOT to call any external API for vectorization.
    We supply our own vectors on every insert and query.
    """
    client = get_weaviate_client()

    if client.collections.exists(collection_name):
        logger.debug(f"Collection '{collection_name}' already exists")
        return

    logger.info(f"Creating Weaviate collection '{collection_name}' (local vectors, no API)")

    client.collections.create(
        name=collection_name,
        properties=[
            Property(name="content",        data_type=DataType.TEXT, index_filterable=False),
            Property(name="document_id",    data_type=DataType.TEXT, index_filterable=True),
            Property(name="filename",       data_type=DataType.TEXT, index_filterable=True),
            Property(name="chunk_index",    data_type=DataType.INT,  index_filterable=False),
            Property(name="page_number",    data_type=DataType.INT,  index_filterable=True),
            Property(name="section_header", data_type=DataType.TEXT, index_filterable=False),
            Property(name="token_count",    data_type=DataType.INT,  index_filterable=False),
            Property(name="tenant_id",      data_type=DataType.TEXT, index_filterable=True),
        ],
        # No external vectorizer — we pass vectors explicitly
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
    )
    logger.info(f"✅ Collection '{collection_name}' created with local vectorizer")


# ─── Store ──────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def store_chunks(
    chunks: list,
    document_id: str,
    tenant_id: str,
    collection_name: Optional[str] = None,
) -> List[str]:
    """
    Generate local embeddings for all chunks and store in Weaviate.

    Change from previous version: we call embed_texts() to get vectors,
    then pass vector=... to batch.add_object() instead of relying on
    Weaviate's built-in text2vec-openai module.
    """
    if not chunks:
        return []

    col_name = collection_name or settings.weaviate_collection_name
    ensure_collection_exists(col_name)

    # Generate all embeddings in one batch (efficient)
    texts = [chunk.content for chunk in chunks]
    vectors = embed_texts(texts)

    client = get_weaviate_client()
    collection = client.collections.get(col_name)
    weaviate_ids = []

    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(chunks, vectors):
            obj = {
                "content":     chunk.content,
                "document_id": document_id,
                "filename":    chunk.metadata.get("filename", ""),
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "tenant_id":   tenant_id,
            }
            if chunk.metadata.get("page_number") is not None:
                obj["page_number"] = chunk.metadata["page_number"]
            if chunk.metadata.get("section_header"):
                obj["section_header"] = chunk.metadata["section_header"]

            # Pass our locally-generated vector explicitly
            result = batch.add_object(properties=obj, vector=vector)
            weaviate_ids.append(str(result))

    logger.info(f"Stored {len(weaviate_ids)} chunks with local embeddings in '{col_name}'")
    return weaviate_ids


# ─── Search ─────────────────────────────────────────────────

def semantic_search(
    query: str,
    tenant_id: str,
    top_k: int = 10,
    document_ids: Optional[List[str]] = None,
    collection_name: Optional[str] = None,
) -> List[dict]:
    """
    Embed the query locally, then search Weaviate with near_vector().

    Change: uses near_vector(vector=...) instead of near_text(query=...)
    because near_text() triggers Weaviate's built-in vectorizer (OpenAI).
    near_vector() uses our pre-computed local embedding directly.
    """
    col_name = collection_name or settings.weaviate_collection_name
    client = get_weaviate_client()

    if not client.collections.exists(col_name):
        logger.warning(f"Collection '{col_name}' does not exist — no documents ingested yet.")
        return []

    # Embed the query with the same model used during ingestion
    query_vector = embed_query(query)
    if not query_vector:
        logger.error("Failed to embed query")
        return []

    collection = client.collections.get(col_name)

    from weaviate.classes.query import Filter
    filters = Filter.by_property("tenant_id").equal(tenant_id)
    if document_ids:
        filters = filters & Filter.by_property("document_id").contains_any(document_ids)

    # near_vector: search by pre-computed vector — no OpenAI call
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        filters=filters,
        return_metadata=MetadataQuery(distance=True),
        return_properties=[
            "content", "document_id", "filename",
            "chunk_index", "page_number", "section_header", "token_count",
        ],
    )

    results = []
    for obj in response.objects:
        distance = obj.metadata.distance if obj.metadata else 1.0
        score = max(0.0, 1.0 - (distance or 1.0))
        results.append({
            "weaviate_id":    str(obj.uuid),
            "content":        obj.properties.get("content", ""),
            "document_id":    obj.properties.get("document_id", ""),
            "document_name":  obj.properties.get("filename", ""),
            "chunk_index":    obj.properties.get("chunk_index", 0),
            "page_number":    obj.properties.get("page_number"),
            "section_header": obj.properties.get("section_header"),
            "score":          round(score, 4),
        })

    logger.info(f"Semantic search: '{query[:50]}' → {len(results)} results")
    return results


def delete_document_chunks(
    document_id: str,
    tenant_id: str,
    collection_name: Optional[str] = None,
) -> int:
    """Delete all Weaviate chunks for a given document."""
    col_name = collection_name or settings.weaviate_collection_name
    client = get_weaviate_client()

    if not client.collections.exists(col_name):
        return 0

    from weaviate.classes.query import Filter
    collection = client.collections.get(col_name)
    result = collection.data.delete_many(
        where=(
            Filter.by_property("document_id").equal(document_id)
            & Filter.by_property("tenant_id").equal(tenant_id)
        )
    )
    count = result.successful if result else 0
    logger.info(f"Deleted {count} chunks for document {document_id}")
    return count