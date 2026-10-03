import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.core.security import compute_file_hash, sanitize_filename, validate_pdf_content
from app.database.database import get_db
from app.database.models import DocumentModel
from app.ingestion.chunker import Chunker
from app.ingestion.pdf_loader import PDFLoader
from app.retrieval.vector_store import delete_document_from_vector_store, update_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    upload_time: str
    page_count: int
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)


class UploadSummaryResponse(BaseModel):
    total_uploaded: int
    indexed: List[DocumentResponse]
    skipped_duplicates: List[str]
    failed: List[Dict[str, str]]


@router.post(
    "/upload",
    response_model=UploadSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index one or multiple PDF documents",
)
async def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> UploadSummaryResponse:
    """
    Accepts one or multiple PDF files, validates them, extracts page-by-page text,
    generates chunks with rich metadata, computes embeddings, and indexes in FAISS.
    Prevents duplicate indexing using SHA-256 file hashes.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided in the upload request.",
        )

    indexed_docs: List[DocumentResponse] = []
    skipped_duplicates: List[str] = []
    failed_docs: List[Dict[str, str]] = []

    chunker = Chunker()

    for file in files:
        raw_filename = file.filename or "uploaded.pdf"
        safe_name = sanitize_filename(raw_filename)

        try:
            content = await file.read()

            # 1. Security & Format Validation
            is_valid, reason = validate_pdf_content(content, safe_name)
            if not is_valid:
                logger.warning(f"File validation rejected '{raw_filename}': {reason}")
                failed_docs.append({"filename": raw_filename, "error": reason})
                continue

            # 2. Duplicate Detection via SHA-256
            file_hash = compute_file_hash(content)
            existing_doc = db.query(DocumentModel).filter(DocumentModel.file_hash == file_hash).first()
            if existing_doc:
                logger.info(
                    f"Skipping duplicate upload for '{raw_filename}' "
                    f"(matches existing document '{existing_doc.filename}' ID: {existing_doc.document_id})"
                )
                skipped_duplicates.append(f"{raw_filename} (already indexed as {existing_doc.filename})")
                continue

            # 3. Document ID and Storage
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            saved_filename = f"{doc_id}_{safe_name}"
            target_path = settings.DOCUMENTS_DIR / saved_filename

            with open(target_path, "wb") as f:
                f.write(content)

            # 4. Page-by-page PDF extraction
            pages = PDFLoader.load_pdf(
                file_path_or_bytes=content,
                filename=safe_name,
                document_id=doc_id,
            )

            # 5. Chunking with metadata retention
            chunks = chunker.chunk_pages(pages)
            if not chunks:
                logger.warning(f"No extractable text chunks produced for '{safe_name}' (scanned or blank).")

            # 6. Indexing in Vector Store
            if chunks:
                update_vector_store(chunks)

            # 7. Persist document metadata in SQLite
            doc_record = DocumentModel(
                document_id=doc_id,
                filename=safe_name,
                file_hash=file_hash,
                upload_time=datetime.now(timezone.utc),
                page_count=len(pages),
                chunk_count=len(chunks),
                file_path=str(target_path),
            )
            db.add(doc_record)
            db.commit()
            db.refresh(doc_record)

            logger.info(
                f"Successfully indexed document: '{safe_name}' (ID: {doc_id}) "
                f"with {len(pages)} pages and {len(chunks)} chunks."
            )

            indexed_docs.append(
                DocumentResponse(
                    document_id=doc_record.document_id,
                    filename=doc_record.filename,
                    file_hash=doc_record.file_hash,
                    upload_time=doc_record.upload_time.isoformat(),
                    page_count=doc_record.page_count,
                    chunk_count=doc_record.chunk_count,
                )
            )

        except Exception as e:
            logger.error(f"Error processing file '{raw_filename}': {e}", exc_info=True)
            failed_docs.append({"filename": raw_filename, "error": str(e)})

    return UploadSummaryResponse(
        total_uploaded=len(files),
        indexed=indexed_docs,
        skipped_duplicates=skipped_duplicates,
        failed=failed_docs,
    )


@router.get(
    "",
    response_model=List[DocumentResponse],
    summary="List all indexed documents",
)
def list_documents(db: Session = Depends(get_db)) -> List[DocumentResponse]:
    """Retrieves all indexed PDF documents stored in the database."""
    docs = db.query(DocumentModel).order_by(DocumentModel.upload_time.desc()).all()
    return [
        DocumentResponse(
            document_id=d.document_id,
            filename=d.filename,
            file_hash=d.file_hash,
            upload_time=d.upload_time.isoformat(),
            page_count=d.page_count,
            chunk_count=d.chunk_count,
        )
        for d in docs
    ]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an indexed document",
)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Deletes document chunks from the vector store, removes physical PDF from disk,
    and deletes the record from the database.
    """
    doc = db.query(DocumentModel).filter(DocumentModel.document_id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    # 1. Remove chunks from Vector Store
    deleted_chunks = delete_document_from_vector_store(document_id)

    # 2. Remove physical file from disk
    file_path = Path(doc.file_path)
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"Removed document file from disk: {file_path}")
        except Exception as file_err:
            logger.warning(f"Could not remove file {file_path} from disk: {file_err}")

    # 3. Remove record from Database
    db.delete(doc)
    db.commit()

    logger.info(f"Document '{doc.filename}' (ID: {document_id}) deleted successfully ({deleted_chunks} chunks purged).")

    return {
        "status": "success",
        "message": f"Document '{doc.filename}' deleted successfully.",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
    }
