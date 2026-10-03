from pathlib import Path
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_chunks_count" in data
    assert data["vector_store_type"] == "FAISS"


def test_upload_single_pdf(client: TestClient, sample_pdf_2024: Path):
    with open(sample_pdf_2024, "rb") as f:
        files = [("files", ("annual_report_2024.pdf", f.read(), "application/pdf"))]

    response = client.post("/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 1
    assert len(data["indexed"]) == 1
    doc = data["indexed"][0]
    assert doc["filename"] == "annual_report_2024.pdf"
    assert doc["page_count"] == 3
    assert doc["chunk_count"] > 0
    assert len(doc["file_hash"]) == 64


def test_duplicate_upload_rejection(client: TestClient, sample_pdf_2024: Path):
    with open(sample_pdf_2024, "rb") as f:
        content = f.read()

    files = [("files", ("annual_report_2024.pdf", content, "application/pdf"))]
    # First upload
    res1 = client.post("/documents/upload", files=files)
    assert res1.status_code == 201

    # Second upload with identical file content
    res2 = client.post("/documents/upload", files=files)
    assert res2.status_code == 201
    data2 = res2.json()
    assert len(data2["indexed"]) == 0
    assert len(data2["skipped_duplicates"]) == 1
    assert "already indexed" in data2["skipped_duplicates"][0]


def test_upload_multiple_pdfs(client: TestClient, sample_pdf_2024: Path, sample_pdf_2025: Path):
    with open(sample_pdf_2024, "rb") as f1, open(sample_pdf_2025, "rb") as f2:
        files = [
            ("files", ("doc_2024.pdf", f1.read(), "application/pdf")),
            ("files", ("doc_2025.pdf", f2.read(), "application/pdf")),
        ]

    response = client.post("/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 2
    assert len(data["indexed"]) == 2


def test_invalid_file_upload(client: TestClient):
    files = [("files", ("malicious.exe", b"NOT A PDF CONTENT", "application/octet-stream"))]
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert len(data["failed"]) == 1
    assert "Only PDF documents are allowed" in data["failed"][0]["error"]


def test_list_and_delete_document(client: TestClient, sample_pdf_2024: Path):
    with open(sample_pdf_2024, "rb") as f:
        files = [("files", ("to_delete.pdf", f.read(), "application/pdf"))]

    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["indexed"][0]["document_id"]

    # List documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["document_id"] == doc_id for d in docs)

    # Delete document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Verify not in list
    list_after = client.get("/documents").json()
    assert not any(d["document_id"] == doc_id for d in list_after)
