from pathlib import Path
import numpy as np
import pytest

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import DocumentChunk
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.vector_store import FAISSVectorStore


def test_embedding_service_cache_and_dimension():
    embedder = EmbeddingService()
    dim = embedder.get_embedding_dimension()
    assert dim == 768

    text = "Artificial intelligence workflow testing passage."
    vec1 = embedder.embed_query(text)
    assert len(vec1) == 768
    # Test normalization (L2 norm should equal ~1.0)
    norm = np.linalg.norm(vec1)
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # Ensure cache hit
    initial_cache_size = len(embedder._cache)
    vec2 = embedder.embed_query(text)
    assert vec1 == vec2
    assert len(embedder._cache) == initial_cache_size


def test_faiss_vector_store_lifecycle(tmp_path: Path):
    store_dir = tmp_path / "test_vectorstore"
    embedder = EmbeddingService()
    store = FAISSVectorStore(storage_dir=store_dir, embedder=embedder)

    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_alpha",
            filename="alpha.pdf",
            page_number=1,
            chunk_index=0,
            text="Acme Corp announced record revenue of fifty million dollars.",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="doc_alpha",
            filename="alpha.pdf",
            page_number=2,
            chunk_index=1,
            text="Employee count grew to four hundred and fifty people in 2024.",
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            document_id="doc_beta",
            filename="beta.pdf",
            page_number=1,
            chunk_index=0,
            text="Beta Corp focuses primarily on solar panel hardware manufacturing.",
        ),
    ]

    # Add chunks
    store.add_chunks(chunks)
    assert store.count() == 3

    # Test similarity search
    query_vec = embedder.embed_query("What was the company's revenue?")
    matches = store.similarity_search(query_vec, top_k=2)
    assert len(matches) == 2
    top_meta, top_score = matches[0]
    assert top_meta["document_id"] == "doc_alpha"
    assert "revenue" in top_meta["text"].lower()

    # Test persistence and reloading from disk
    reloaded_store = FAISSVectorStore(storage_dir=store_dir, embedder=embedder)
    assert reloaded_store.count() == 3

    # Test deletion of specific document
    deleted_count = reloaded_store.delete_document("doc_alpha")
    assert deleted_count == 2
    assert reloaded_store.count() == 1

    # Ensure only beta remains
    remaining_matches = reloaded_store.similarity_search(query_vec, top_k=5)
    assert len(remaining_matches) == 1
    assert remaining_matches[0][0]["document_id"] == "doc_beta"


def test_semantic_retriever_threshold_and_filters(tmp_path: Path):
    store_dir = tmp_path / "retriever_test"
    embedder = EmbeddingService()
    store = FAISSVectorStore(storage_dir=store_dir, embedder=embedder)

    chunks = [
        DocumentChunk(
            chunk_id="chk_rev",
            document_id="doc_1",
            filename="report.pdf",
            page_number=5,
            chunk_index=0,
            text="Fiscal year 2024 net income reached twelve million dollars.",
        ),
        DocumentChunk(
            chunk_id="chk_cust",
            document_id="doc_1",
            filename="report.pdf",
            page_number=8,
            chunk_index=1,
            text="Customer satisfaction score was ninety-four percent.",
        ),
    ]
    store.add_chunks(chunks)

    retriever = SemanticRetriever(vector_store=store, embedder=embedder)

    # Test retrieval
    results = retriever.retrieve(query="What was the net income?", top_k=1)
    assert len(results) == 1
    assert results[0].page_number == 5
    assert results[0].filename == "report.pdf"
    assert "twelve million" in results[0].text

    # Test strict threshold rejection on completely irrelevant query
    unrelated_results = retriever.retrieve(query="quantum teleportation in deep space", threshold=0.7)
    assert len(unrelated_results) == 0
