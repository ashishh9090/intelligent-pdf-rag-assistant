from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.core.logging import logger
from app.database.models import Base

# Ensure parent directory for database exists
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite:///"):
    sqlite_path = Path(db_url.replace("sqlite:///", ""))
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
