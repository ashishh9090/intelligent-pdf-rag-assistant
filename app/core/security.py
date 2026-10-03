import hashlib
import re
from pathlib import Path
from typing import Tuple
from fastapi import HTTPException, status
from app.core.config import settings

# PDF header signature magic bytes
PDF_MAGIC_BYTES = b"%PDF-"


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes filename by stripping directory traversal sequences,
    non-whitelisted characters, and ensuring a safe base name.
    """
    # Extract only the base name (prevents ../ or /etc/passwd attacks)
    base_name = Path(filename).name
    # Remove any dangerous characters except alphanumeric, dashes, underscores, and dots
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)
    if not clean_name or clean_name == ".pdf":
        clean_name = f"document_{hashlib.md5(filename.encode()).hexdigest()[:8]}.pdf"
    return clean_name


def validate_pdf_content(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validates that the file:
    1. Has a valid .pdf extension.
    2. Does not exceed maximum allowed size.
    3. Starts with the valid PDF magic header (%PDF-).
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False, f"Invalid file extension '.{ext}'. Only PDF documents are allowed."

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, f"File size exceeds limit of {settings.MAX_UPLOAD_SIZE_MB}MB."

    if len(file_bytes) < len(PDF_MAGIC_BYTES) or not file_bytes.startswith(PDF_MAGIC_BYTES):
        return False, "File does not appear to be a valid PDF document (missing %PDF- header)."

    return True, "Valid PDF"


def compute_file_hash(content: bytes) -> str:
    """Computes SHA-256 hash of byte content for deduplication."""
    return hashlib.sha256(content).hexdigest()
