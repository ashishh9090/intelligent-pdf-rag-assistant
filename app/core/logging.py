import logging
import sys
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "rag_assistant") -> logging.Logger:
    """Configures and returns a structured application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger("rag_assistant")


def sanitize_for_log(data: Any) -> Any:
    """Mask potential API keys or sensitive authorization tokens."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if any(secret_term in k.lower() for secret_term in ["token", "key", "secret", "password", "auth"]):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = sanitize_for_log(v)
        return sanitized
    return data
