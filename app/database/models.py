import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    username = Column(String(128), unique=True, nullable=False, default="guest")
    created_at = Column(DateTime, default=utc_now, nullable=False)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=True)
    title = Column(String(256), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.timestamp")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, index=True)
    conversation_id = Column(String(64), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # JSON serialized source citations
    timestamp = Column(DateTime, default=utc_now, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    @property
    def sources(self) -> List[Dict[str, Any]]:
        if not self.sources_json:
            return []
        try:
            return json.loads(self.sources_json)
        except Exception:
            return []

    @sources.setter
    def sources(self, value: List[Dict[str, Any]]) -> None:
        self.sources_json = json.dumps(value, ensure_ascii=False)


class DocumentModel(Base):
    __tablename__ = "documents"

    document_id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(256), nullable=False)
    file_hash = Column(String(64), unique=True, index=True, nullable=False)
    upload_time = Column(DateTime, default=utc_now, nullable=False)
    page_count = Column(Integer, default=0, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    file_path = Column(String(512), nullable=False)
