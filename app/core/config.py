import os
from pathlib import Path
from typing import List, Optional

# Prevent OpenMP and PyTorch multi-threading conflicts on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory: root of the project (rag-document-chatbot)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Project
    PROJECT_NAME: str = "Intelligent PDF Knowledge Assistant – RAG"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Storage Paths
    DATA_DIR: Path = Field(default=BASE_DIR / "data")
    DOCUMENTS_DIR: Path = Field(default=BASE_DIR / "data" / "documents")
    VECTORSTORE_DIR: Path = Field(default=BASE_DIR / "data" / "vectorstore")
    DATABASE_URL: str = Field(default=f"sqlite:///{BASE_DIR}/data/chatbot.db")

    # Ingestion & Chunking
    CHUNK_SIZE: int = Field(default=800, ge=100, le=4000)
    CHUNK_OVERLAP: int = Field(default=150, ge=0, le=1000)

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_DEVICE: str = "cpu"

    # Retrieval
    TOP_K: int = Field(default=5, ge=1, le=50)
    SIMILARITY_THRESHOLD: float = Field(default=0.35, ge=0.0, le=1.0)

    # LLM Settings
    LLM_PROVIDER: str = "hf_local"  # "hf_local", "hf_api", "mock"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
    HUGGINGFACE_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_NEW_TOKENS: int = 512

    # Memory & Conversation
    MAX_HISTORY_TURNS: int = 10

    # Security & Upload Validation
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: List[str] = ["pdf"]

    # Server Network
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_URL: str = "http://localhost:8000"

    def ensure_directories(self) -> None:
        """Ensure all required runtime data directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
