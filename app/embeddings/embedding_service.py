import hashlib
import threading
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.logging import logger


class EmbeddingService(Embeddings):
    """
    Singleton service for generating normalized dense embeddings using BAAI/bge-base-en-v1.5.
    Implements LangChain's Embeddings interface and provides in-memory vector caching.
    """

    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
    ):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.normalize = normalize if normalize is not None else settings.EMBEDDING_NORMALIZE
        self._model = None
        self._cache: Dict[str, List[float]] = {}
        self._initialized = True
        logger.info(f"Initialized EmbeddingService with model: {self.model_name}, device: {self.device}")

    @property
    def model(self):
        """Lazy loader for the SentenceTransformer model to avoid blocking on module import."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer

                    torch.set_num_threads(1)
                    logger.info(f"Loading SentenceTransformer embedding model '{self.model_name}'...")
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
        return self._model

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a batch of texts, utilizing an internal vector cache to avoid redundant computation.
        """
        if not texts:
            return []

        embeddings_result: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_compute: List[str] = []
        indices_to_compute: List[int] = []

        # Check cache
        for idx, text in enumerate(texts):
            h = self._hash_text(text)
            if h in self._cache:
                embeddings_result[idx] = self._cache[h]
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(idx)

        # Compute missing embeddings in batches
        if texts_to_compute:
            logger.info(
                f"Generating embeddings for {len(texts_to_compute)} new chunks "
                f"({len(texts) - len(texts_to_compute)} retrieved from cache)..."
            )
            raw_vectors = self.model.encode(
                texts_to_compute,
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )

            # Store in cache and result list
            for sub_idx, original_idx in enumerate(indices_to_compute):
                vec = raw_vectors[sub_idx].tolist()
                h = self._hash_text(texts[original_idx])
                self._cache[h] = vec
                embeddings_result[original_idx] = vec

        return [v for v in embeddings_result if v is not None]

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        For BGE models, query embedding uses the same normalization.
        """
        h = self._hash_text(text)
        if h in self._cache:
            return self._cache[h]

        vector = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).tolist()
        self._cache[h] = vector
        return vector

    def get_embedding_dimension(self) -> int:
        """Returns the embedding vector dimensionality (e.g. 768 for bge-base)."""
        sample_vec = self.embed_query("dimension probe")
        return len(sample_vec)


# Shared default singleton instance
embedding_service = EmbeddingService()
