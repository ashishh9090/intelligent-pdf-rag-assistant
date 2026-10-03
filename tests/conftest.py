import os
import shutil
import tempfile
from pathlib import Path
import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.database.database import get_db
from app.database.models import Base
from app.main import app
from app.retrieval.vector_store import FAISSVectorStore, get_vector_store


@pytest.fixture(scope="session")
def sample_pdf_2024(tmp_path_factory) -> Path:
    """Creates a 3-page synthetic PDF for testing RAG extraction, chunking, and citations."""
    temp_dir = tmp_path_factory.mktemp("sample_pdfs")
    pdf_path = temp_dir / "annual_report_2024.pdf"

    doc = pymupdf.open()
    # Page 1
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "Acme Corp Annual Report 2024.\n"
        "Executive Summary:\n"
        "Acme Corp is a leading provider of enterprise AI workflow orchestration and cloud platforms.\n"
        "The company was founded in 2018 and has expanded across North America and Europe.",
    )
    # Page 2
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "Financial Performance 2024:\n"
        "In fiscal year 2024, Acme Corp reported total revenue of $50 million, representing a 15% increase.\n"
        "Gross profit reached $30 million with a 60% gross margin.\n"
        "Research and Development spending was $10 million.",
    )
    # Page 3
    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "Workforce & Operations:\n"
        "Total employee headcount reached 450 full-time team members by the end of 2024.\n"
        "Customer satisfaction score (CSAT) averaged 94% across all service tiers.",
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture(scope="session")
def sample_pdf_2025(tmp_path_factory) -> Path:
    """Creates a second synthetic PDF for multi-document comparisons."""
    temp_dir = tmp_path_factory.mktemp("sample_pdfs")
    pdf_path = temp_dir / "annual_report_2025.pdf"

    doc = pymupdf.open()
    # Page 1
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "Acme Corp Annual Report 2025.\n"
        "Strategic Growth:\n"
        "Accelerating enterprise adoption of generative knowledge systems.",
    )
    # Page 2
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "Financial Highlights 2025:\n"
        "In fiscal year 2025, Acme Corp reported total revenue of $65 million, marking a 30% increase.\n"
        "Operating income rose to $18 million.",
    )
    # Page 3
    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "Workforce 2025:\n"
        "Total global headcount expanded to 600 employees with a new European hub.",
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture(scope="session")
def sample_pdf_with_blank_page(tmp_path_factory) -> Path:
    """Creates a PDF with a blank page to test graceful empty page handling."""
    temp_dir = tmp_path_factory.mktemp("sample_pdfs")
    pdf_path = temp_dir / "blank_page_doc.pdf"

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Text on page 1.")
    p2 = doc.new_page()  # Page 2 is completely empty
    p3 = doc.new_page()
    p3.insert_text((50, 72), "Text on page 3.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


from sqlalchemy.pool import StaticPool


@pytest.fixture
def test_db_session():
    """In-memory SQLite database session with StaticPool for unit tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(test_db_session):
    """FastAPI test client with database override."""
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
