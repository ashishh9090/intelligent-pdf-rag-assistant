import time
from typing import Any, Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.database import get_db
from app.database.models import DocumentModel
from app.retrieval.vector_store import get_vector_store

router = APIRouter(tags=["Health"])

START_TIME = time.time()


@router.get("/health", summary="System Health & Readiness Check")
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Returns application health status, indexed counts, and configuration summary."""
    try:
        doc_count = db.query(DocumentModel).count()
        vector_count = get_vector_store().count()
        uptime_seconds = round(time.time() - START_TIME, 2)

        return {
            "status": "healthy",
            "uptime_seconds": uptime_seconds,
            "project_name": settings.PROJECT_NAME,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL_NAME,
            "indexed_documents_count": doc_count,
            "indexed_chunks_count": vector_count,
            "vector_store_type": "FAISS",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
