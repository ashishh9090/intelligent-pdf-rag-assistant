from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.core.config import settings
from app.core.logging import logger
from app.database.database import init_db
from app.retrieval.vector_store import load_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info(f"Starting {settings.PROJECT_NAME} in '{settings.ENVIRONMENT}' mode...")
    settings.ensure_directories()
    init_db()
    load_vector_store()
    logger.info("Application initialized successfully and ready to serve requests.")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Production-style Retrieval-Augmented Generation (RAG) backend for querying uploaded "
        "PDF documents using dense embeddings (BAAI/bge-base-en-v1.5), FAISS vector store, "
        "and strictly grounded citation-backed LLM generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for Streamlit frontend and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred. Please consult the server logs."},
    )


# Register API routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )
