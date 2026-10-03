import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw text extracted from PDF:
    - Replaces non-breaking spaces and excessive whitespace.
    - Joins words broken across line breaks with hyphens (e.g. 'infor-\nmation' -> 'information').
    - Removes non-printable/control characters while preserving paragraph structure.
    - Trims leading and trailing spaces.
    """
    if not text:
        return ""

    # Replace non-breaking spaces and control characters (except newline and tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    cleaned = cleaned.replace("\u00a0", " ")

    # Rejoin hyphenated words broken across newlines: word-\nword -> wordword
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

    # Replace multiple consecutive spaces/tabs within a line with a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Normalize excessive newlines (more than 2 consecutive newlines -> 2)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
