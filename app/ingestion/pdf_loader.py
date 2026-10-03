from pathlib import Path
from typing import Any, Dict, List, Union
import pymupdf

from app.core.logging import logger
from app.ingestion.text_cleaner import clean_text


class PDFPageData:
    def __init__(
        self,
        document_id: str,
        filename: str,
        page_number: int,
        text: str,
        char_count: int,
    ):
        self.document_id = document_id
        self.filename = filename
        self.page_number = page_number
        self.text = text
        self.char_count = char_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
        }


class PDFLoader:
    """Extracts text page-by-page from PDF documents safely."""

    @staticmethod
    def load_pdf(
        file_path_or_bytes: Union[str, Path, bytes],
        filename: str,
        document_id: str,
    ) -> List[PDFPageData]:
        """
        Loads PDF and extracts text per page with robust error handling.
        Gracefully handles empty/scanned pages without crashing.
        """
        pages: List[PDFPageData] = []
        doc = None
        try:
            if isinstance(file_path_or_bytes, bytes):
                doc = pymupdf.open(stream=file_path_or_bytes, filetype="pdf")
            else:
                doc = pymupdf.open(str(file_path_or_bytes))

            total_pages = len(doc)
            logger.info(f"Extracting text from '{filename}' (ID: {document_id}), total pages: {total_pages}")

            for page_idx in range(total_pages):
                page_number = page_idx + 1
                try:
                    page = doc.load_page(page_idx)
                    raw_text = page.get_text("text")
                    cleaned = clean_text(raw_text)

                    if not cleaned:
                        logger.warning(
                            f"Page {page_number}/{total_pages} of '{filename}' has no extractable text "
                            f"(might be an image/scanned page). Skipping page gracefully."
                        )

                    pages.append(
                        PDFPageData(
                            document_id=document_id,
                            filename=filename,
                            page_number=page_number,
                            text=cleaned,
                            char_count=len(cleaned),
                        )
                    )
                except Exception as page_err:
                    logger.error(
                        f"Error extracting text from page {page_number} of '{filename}': {page_err}. "
                        f"Skipping page to prevent crash."
                    )
                    pages.append(
                        PDFPageData(
                            document_id=document_id,
                            filename=filename,
                            page_number=page_number,
                            text="",
                            char_count=0,
                        )
                    )

            return pages

        except Exception as e:
            logger.error(f"Failed to load or parse PDF '{filename}': {e}", exc_info=True)
            raise ValueError(f"Could not process PDF file '{filename}': {str(e)}")
        finally:
            if doc is not None:
                doc.close()
