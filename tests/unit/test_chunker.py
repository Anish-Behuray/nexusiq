"""
tests/unit/test_chunker.py
============================
Unit tests for the chunking module.

CONCEPT: Why write tests for AI systems?
-----------------------------------------
AI systems have non-deterministic components (LLM outputs vary).
But infrastructure components — chunkers, loaders, validators —
are deterministic and must be tested like any software.

WHAT WE TEST:
1. Chunker produces the right number of chunks
2. Chunk metadata is preserved from original documents
3. Token counts are accurate
4. Edge cases: empty docs, very short docs, very long docs

TESTING PATTERN: AAA (Arrange, Act, Assert)
- Arrange: set up test data
- Act: call the function being tested
- Assert: verify the output

RUN TESTS:
  pytest tests/ -v
  pytest tests/unit/ -v --cov=ingestion --cov-report=term-missing
"""

import pytest
from llama_index.core import Document
from ingestion.chunkers.chunker import chunk_documents, count_tokens, Chunk


class TestCountTokens:
    """Test the token counting utility."""
    
    def test_empty_string(self):
        # With our character-based fallback, empty string returns max(1, 0//4) = 1.
        # With real tiktoken it returns 0. Either is acceptable.
        result = count_tokens("")
        assert result in (0, 1), f"Expected 0 or 1 for empty string, got {result}"
    
    def test_single_word(self):
        # "Hello" = 1 token in cl100k_base
        assert count_tokens("Hello") == 1
    
    def test_known_sentence(self):
        # Predictable sentence for regression testing
        text = "The quick brown fox jumps over the lazy dog."
        count = count_tokens(text)
        # Should be around 10 tokens for this sentence
        assert 8 <= count <= 12, f"Expected 8-12 tokens, got {count}"
    
    def test_long_text(self):
        # 500 words should produce roughly 700 tokens (avg ~1.4 tokens/word)
        words = ["word"] * 500
        text = " ".join(words)
        count = count_tokens(text)
        assert 400 <= count <= 800, f"Expected 400-800 tokens, got {count}"


class TestChunkDocuments:
    """Test the document chunking pipeline."""
    
    def _make_document(self, text: str, filename: str = "test.pdf", 
                       page: int = 1) -> Document:
        """Helper to create a Document for testing."""
        return Document(
            text=text,
            metadata={
                "document_id": "test-doc-123",
                "filename": filename,
                "file_type": "pdf",
                "page_number": page,
                "section_header": "Test Section",
            }
        )
    
    def test_basic_chunking_produces_chunks(self):
        """A document with enough text should produce at least one chunk."""
        doc = self._make_document("This is test content. " * 50)
        chunks = chunk_documents([doc])
        
        assert len(chunks) > 0, "Should produce at least one chunk"
    
    def test_chunks_are_chunk_instances(self):
        """All chunks should be Chunk dataclass instances."""
        doc = self._make_document("Sample text. " * 30)
        chunks = chunk_documents([doc])
        
        for chunk in chunks:
            assert isinstance(chunk, Chunk), f"Expected Chunk, got {type(chunk)}"
    
    def test_metadata_preserved_in_chunks(self):
        """Critical: chunk metadata must include document_id for citations."""
        doc = self._make_document("Content for metadata test. " * 20)
        chunks = chunk_documents([doc])
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "document_id" in chunk.metadata, "document_id must be in chunk metadata"
            assert chunk.metadata["document_id"] == "test-doc-123"
            assert "filename" in chunk.metadata
    
    def test_chunk_size_respected(self):
        """Chunks should not significantly exceed the configured chunk_size."""
        long_text = "This is a sentence with several words. " * 200  # ~800 tokens
        doc = self._make_document(long_text)
        
        chunk_size = 256
        chunks = chunk_documents([doc], chunk_size=chunk_size)
        
        # Allow 20% tolerance (sentence splitter can't always split exactly)
        max_allowed = chunk_size * 1.2
        for chunk in chunks:
            assert chunk.token_count <= max_allowed, (
                f"Chunk has {chunk.token_count} tokens, "
                f"expected max {max_allowed}"
            )
    
    def test_multiple_documents_chunked(self):
        """Multiple input documents should all be chunked."""
        docs = [
            self._make_document(f"Content for document {i}. " * 20, f"doc{i}.pdf")
            for i in range(3)
        ]
        
        chunks = chunk_documents(docs)
        
        # Should have chunks from all documents
        doc_filenames = {c.metadata.get("filename") for c in chunks}
        assert len(doc_filenames) == 3, f"Expected 3 source files, got {doc_filenames}"
    
    def test_empty_document_list(self):
        """Empty input should return empty output, not crash."""
        chunks = chunk_documents([])
        assert chunks == [], f"Expected empty list, got {chunks}"
    
    def test_empty_document_text_skipped(self):
        """Documents with empty text should not produce chunks."""
        empty_doc = Document(text="", metadata={"document_id": "empty", "filename": "empty.txt"})
        chunks = chunk_documents([empty_doc])
        # Empty doc should produce no chunks
        assert len(chunks) == 0
    
    def test_chunk_indices_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        doc = self._make_document("Sequential content. " * 100)
        chunks = chunk_documents([doc])
        
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i, (
                    f"Expected index {i}, got {chunk.chunk_index}"
                )
    
    def test_token_count_accuracy(self):
        """Token counts on chunks should be accurate."""
        doc = self._make_document("Checking tokens. " * 50)
        chunks = chunk_documents([doc])
        
        for chunk in chunks:
            actual_count = count_tokens(chunk.content)
            # Allow 5% tolerance (rounding in splitter)
            assert abs(chunk.token_count - actual_count) <= max(5, actual_count * 0.05), (
                f"Token count mismatch: stored {chunk.token_count}, "
                f"actual {actual_count}"
            )
    
    def test_very_short_text(self):
        """Very short text (less than min_chunk_tokens) may produce zero chunks."""
        short_doc = self._make_document("Hi.")
        # Should not crash on very short text
        chunks = chunk_documents([short_doc])
        # Either 0 or 1 chunks — both acceptable
        assert len(chunks) in [0, 1]
    
    def test_chunk_content_not_empty(self):
        """No chunk should have empty content."""
        doc = self._make_document("Non-empty content here. " * 30)
        chunks = chunk_documents([doc])
        
        for chunk in chunks:
            assert chunk.content.strip(), "Chunk content should not be empty"


class TestChunkOverlap:
    """Test that chunk overlap is working correctly."""
    
    def test_overlap_causes_shared_content(self):
        """
        With overlap > 0, adjacent chunks should share some text.
        This ensures no context is cut off at chunk boundaries.
        """
        # Create a predictable document with numbered sentences
        sentences = [f"This is sentence number {i}. " for i in range(50)]
        text = " ".join(sentences)
        
        doc = Document(
            text=text,
            metadata={"document_id": "overlap-test", "filename": "test.txt"}
        )
        
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
        
        if len(chunks) >= 2:
            # Check that consecutive chunks share some words
            chunk1_words = set(chunks[0].content.lower().split())
            chunk2_words = set(chunks[1].content.lower().split())
            shared_words = chunk1_words & chunk2_words
            
            # They should share at least some words (from the overlap)
            assert len(shared_words) > 0, (
                "Adjacent chunks with overlap > 0 should share some content"
            )
