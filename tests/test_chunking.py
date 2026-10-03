from pathlib import Path
import pytest

from app.ingestion.chunker import Chunker
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_cleaner import clean_text


def test_clean_text_normalizations():
    # Test hyphenated words broken across newline
    raw = "The organi-\nzation reported great prog-\nress."
    cleaned = clean_text(raw)
    assert "organization" in cleaned
    assert "progress" in cleaned

    # Test whitespace and control characters
    raw_spaces = "Hello \t  world!\x00\x05\n\n\n\nParagraph 2."
    cleaned_spaces = clean_text(raw_spaces)
    assert "Hello world!" in cleaned_spaces
    assert "\n\nParagraph 2." in cleaned_spaces
    assert "\x00" not in cleaned_spaces


def test_pdf_loader_preserves_metadata(sample_pdf_2024: Path):
    doc_id = "doc_test_123"
    filename = "annual_report_2024.pdf"

    pages = PDFLoader.load_pdf(sample_pdf_2024, filename=filename, document_id=doc_id)
    assert len(pages) == 3

    # Check page 1 metadata
    assert pages[0].document_id == doc_id
    assert pages[0].filename == filename
    assert pages[0].page_number == 1
    assert "Acme Corp" in pages[0].text

    # Check page 2 metadata
    assert pages[1].page_number == 2
    assert "revenue of $50 million" in pages[1].text

    # Check page 3 metadata
    assert pages[2].page_number == 3
    assert "450 full-time" in pages[2].text


def test_pdf_loader_graceful_blank_page(sample_pdf_with_blank_page: Path):
    doc_id = "doc_blank_test"
    filename = "blank_page_doc.pdf"

    pages = PDFLoader.load_pdf(sample_pdf_with_blank_page, filename=filename, document_id=doc_id)
    assert len(pages) == 3

    # Page 1 has text
    assert pages[0].text == "Text on page 1."
    # Page 2 is empty, did not crash
    assert pages[1].text == ""
    assert pages[1].char_count == 0
    # Page 3 has text
    assert pages[2].text == "Text on page 3."


def test_chunker_metadata_and_custom_sizes(sample_pdf_2024: Path):
    pages = PDFLoader.load_pdf(sample_pdf_2024, filename="annual_report_2024.pdf", document_id="doc_chk_99")

    # Test with custom small chunk size
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 0
    for chunk in chunks:
        # Every chunk must maintain metadata as per requirement 2 & 3
        assert chunk.document_id == "doc_chk_99"
        assert chunk.filename == "annual_report_2024.pdf"
        assert chunk.page_number in [1, 2, 3]
        assert chunk.chunk_id != ""
        assert len(chunk.text) > 0

        as_dict = chunk.to_dict()
        assert "document_id" in as_dict
        assert "filename" in as_dict
        assert "page_number" in as_dict
        assert "chunk_id" in as_dict
        assert "text" in as_dict
