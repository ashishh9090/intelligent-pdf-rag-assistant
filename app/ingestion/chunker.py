import uuid
from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.pdf_loader import PDFPageData


class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        filename: str,
        page_number: int,
        chunk_index: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.filename = filename
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.text = text
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
        }


class Chunker:
    """Chunks page text into semantically cohesive passages while preserving rich metadata."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_pages(self, pages: List[PDFPageData]) -> List[DocumentChunk]:
        """
        Chunks pages page-by-page to guarantee page number attribution.
        Each chunk receives a unique chunk_id and full metadata.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            page_splits = self.splitter.split_text(page.text)
            for split_idx, split_text in enumerate(page_splits):
                cleaned_split = split_text.strip()
                if not cleaned_split:
                    continue

                chunk_id = f"{page.document_id}_p{page.page_number}_c{split_idx}_{uuid.uuid4().hex[:6]}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    text=cleaned_split,
                    metadata={
                        "char_count": len(cleaned_split),
                        "page_chunk_index": split_idx,
                    },
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        logger.info(
            f"Chunked document (ID: {pages[0].document_id if pages else 'N/A'}) into "
            f"{len(all_chunks)} chunks using chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )
        return all_chunks
