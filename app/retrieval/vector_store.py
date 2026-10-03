import json
import os
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService, embedding_service
from app.ingestion.chunker import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract interface to decouple vector store implementation (FAISS, Chroma, Qdrant)."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks and their embeddings into the store."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count of deleted chunks."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search top_k most similar chunks. Returns (metadata, score)."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist index and metadata to disk."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Load index and metadata from disk. Returns True if loaded, False if not found."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns total number of chunks indexed."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store implementation with persistent metadata and cosine similarity search.
    Supports local persistence, dynamic insertion, document-level deletion, and metadata retrieval.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        embedder: Optional[EmbeddingService] = None,
    ):
        self.storage_dir = Path(storage_dir or settings.VECTORSTORE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "faiss_index.bin"
        self.metadata_file = self.storage_dir / "chunks_metadata.json"

        self.embedder = embedder or embedding_service
        self.index: Optional[faiss.IndexIDMap2] = None
        # metadata store: id_int (str key in json) -> chunk metadata dict
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.next_id: int = 1
        self._lock = threading.Lock()

        # Attempt to load existing index on initialization
        self.load()

    def _init_new_index(self, dimension: int) -> None:
        """Initializes a new FAISS IndexIDMap2 wrapping IndexFlatIP for cosine similarity."""
        base_index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap2(base_index)
        self.metadata_store = {}
        self.next_id = 1
        logger.info(f"Initialized new FAISS IndexIDMap2 (dimension={dimension})")

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embeds and indexes document chunks."""
        if not chunks:
            return

        with self._lock:
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_documents(texts)
            np_vectors = np.array(vectors, dtype=np.float32)

            dim = np_vectors.shape[1]
            if self.index is None:
                self._init_new_index(dim)

            num_new = len(chunks)
            ids = np.arange(self.next_id, self.next_id + num_new, dtype=np.int64)

            # Add to FAISS index with explicit IDs
            self.index.add_with_ids(np_vectors, ids)

            # Store metadata
            for i, chunk in enumerate(chunks):
                assigned_id = int(ids[i])
                meta = chunk.to_dict()
                meta["id"] = assigned_id
                self.metadata_store[str(assigned_id)] = meta

            self.next_id += num_new
            self.save()
            logger.info(f"Added {num_new} chunks to FAISS index. Total indexed: {self.count()}")

    def delete_document(self, document_id: str) -> int:
        """Removes all chunks associated with a specific document_id."""
        with self._lock:
            if self.index is None or self.count() == 0:
                return 0

            # Find all IDs matching document_id
            ids_to_remove = [
                int(id_str)
                for id_str, meta in self.metadata_store.items()
                if meta.get("document_id") == document_id
            ]

            if not ids_to_remove:
                logger.info(f"No chunks found for document_id '{document_id}' to delete.")
                return 0

            # Remove from FAISS index
            id_array = np.array(ids_to_remove, dtype=np.int64)
            self.index.remove_ids(id_array)

            # Remove from metadata store
            for id_int in ids_to_remove:
                self.metadata_store.pop(str(id_int), None)

            self.save()
            logger.info(f"Deleted {len(ids_to_remove)} chunks for document_id '{document_id}'.")
            return len(ids_to_remove)

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes cosine similarity search against the index.
        Returns list of (chunk_metadata, similarity_score).
        """
        with self._lock:
            if self.index is None or self.count() == 0:
                logger.warning("Vector store is empty. Returning 0 results.")
                return []

            k = min(top_k, self.count())
            q_vec = np.array([query_vector], dtype=np.float32)

            distances, indices = self.index.search(q_vec, k)
            results: List[Tuple[Dict[str, Any], float]] = []

            for score, id_val in zip(distances[0], indices[0]):
                if id_val == -1:
                    continue

                sim_score = float(score)
                # Filter by similarity threshold if specified
                if threshold is not None and sim_score < threshold:
                    logger.debug(f"Chunk id {id_val} score {sim_score:.4f} below threshold {threshold}")
                    continue

                meta = self.metadata_store.get(str(id_val))
                if meta:
                    results.append((meta, sim_score))

            return results

    def save(self) -> None:
        """Persists the FAISS binary index and chunks metadata JSON file atomically."""
        if self.index is None:
            return

        temp_index = self.storage_dir / "faiss_index.bin.tmp"
        temp_meta = self.storage_dir / "chunks_metadata.json.tmp"

        faiss.write_index(self.index, str(temp_index))
        state = {
            "next_id": self.next_id,
            "chunks": self.metadata_store,
        }
        with open(temp_meta, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        temp_index.replace(self.index_file)
        temp_meta.replace(self.metadata_file)
        logger.debug(f"Persisted vector store index ({self.count()} chunks) to {self.storage_dir}")

    def load(self) -> bool:
        """Loads FAISS index and metadata from disk if available."""
        with self._lock:
            if not self.index_file.exists() or not self.metadata_file.exists():
                logger.info("No existing vector store index found on disk. Ready for new documents.")
                return False

            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.next_id = state.get("next_id", 1)
                    self.metadata_store = state.get("chunks", {})

                logger.info(
                    f"Successfully loaded vector store index with {self.count()} chunks "
                    f"from {self.storage_dir}"
                )
                return True
            except Exception as e:
                logger.error(f"Error loading vector store from {self.storage_dir}: {e}", exc_info=True)
                return False

    def count(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal


# Factory and lifecycle functions specified in Project Requirement 5
_default_vector_store: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    """Returns the application vector store singleton."""
    global _default_vector_store
    if _default_vector_store is None:
        _default_vector_store = FAISSVectorStore()
    return _default_vector_store


def create_vector_store(chunks: Optional[List[DocumentChunk]] = None) -> FAISSVectorStore:
    """Creates/initializes vector store, optionally adding initial chunks."""
    store = get_vector_store()
    if chunks:
        store.add_chunks(chunks)
    return store


def load_vector_store() -> FAISSVectorStore:
    """Explicitly loads or re-loads the persisted vector store."""
    store = get_vector_store()
    store.load()
    return store


def update_vector_store(chunks: List[DocumentChunk]) -> FAISSVectorStore:
    """Appends new chunks to the vector store."""
    store = get_vector_store()
    store.add_chunks(chunks)
    return store


def delete_document_from_vector_store(document_id: str) -> int:
    """Deletes all chunks for a document ID from the vector store."""
    store = get_vector_store()
    return store.delete_document(document_id)
