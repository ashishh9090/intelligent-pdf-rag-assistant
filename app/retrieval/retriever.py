from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService, embedding_service
from app.retrieval.vector_store import BaseVectorStore, get_vector_store


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_citation(self, citation_num: int) -> str:
        """Standardized source citation string."""
        return f"[Source {citation_num}: {self.filename}, Page {self.page_number}]"


class SemanticRetriever:
    """Retrieves top-k relevant document chunks using dense semantic vector search."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedder: Optional[EmbeddingService] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or embedding_service

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes semantic search for a user query.
        Applies configurable top_k, similarity threshold, and optional document filters.
        """
        k = top_k if top_k is not None else settings.TOP_K
        thresh = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

        logger.info(f"Query received for retrieval: '{query[:100]}...' | top_k={k}, threshold={thresh}")

        # Compute normalized query vector
        query_vector = self.embedder.embed_query(query)

        # Retrieve candidate matches from vector store
        raw_matches = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=k * 2 if filter_document_ids else k,
            threshold=thresh,
        )

        results: List[RetrievedChunk] = []
        for meta, score in raw_matches:
            # Apply document filter if user filtered to specific documents
            doc_id = meta.get("document_id", "")
            if filter_document_ids and doc_id not in filter_document_ids:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    document_id=doc_id,
                    filename=meta.get("filename", "unknown.pdf"),
                    page_number=meta.get("page_number", 1),
                    text=meta.get("text", ""),
                    similarity_score=round(score, 4),
                    metadata=meta.get("metadata", {}),
                )
            )
            if len(results) >= k:
                break

        # Log retrieval results
        scores_str = ", ".join([f"{r.similarity_score:.3f}" for r in results])
        logger.info(
            f"Retrieved {len(results)} chunks (Scores: [{scores_str}]) "
            f"for query: '{query[:60]}...'"
        )

        return results


# Default retriever singleton
retriever = SemanticRetriever()
