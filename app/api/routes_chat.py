from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.database.database import get_db
from app.database.models import Conversation, Message
from app.generation.rag_chain import Citation, rag_chain

router = APIRouter(tags=["Chat & Conversations"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum cosine similarity score"
    )
    filter_document_ids: Optional[List[str]] = Field(
        default=None, description="Optional restriction to specific document IDs"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: List[Citation] = Field(default_factory=list)
    query: str
    rewritten_query: str
    has_answer: bool
    conversation_id: str


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageItem]


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about uploaded PDF documents",
)
def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """
    Submits a user question to the RAG pipeline.
    Retrieves grounded document passages, contextualizes follow-ups, queries the LLM,
    and returns factual answer with structured citations.
    """
    try:
        response = rag_chain.run(
            query=request.query,
            db=db,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filter_document_ids=request.filter_document_ids,
        )
        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            query=response.query,
            rewritten_query=response.rewritten_query,
            has_answer=response.has_answer,
            conversation_id=response.conversation_id,
        )
    except Exception as e:
        logger.error(f"Error during RAG chat execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering your question: {str(e)}",
        )


@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="List all conversations",
)
def list_conversations(db: Session = Depends(get_db)) -> List[ConversationSummary]:
    """Lists all user conversations with summary metadata."""
    conversations = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    summaries = []
    for conv in conversations:
        summaries.append(
            ConversationSummary(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at.isoformat(),
                updated_at=conv.updated_at.isoformat(),
                message_count=len(conv.messages),
            )
        )
    return summaries


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation history by ID",
)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ConversationDetail:
    """Retrieves full conversation turns and citations for a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    msg_items = [
        MessageItem(
            id=m.id,
            role=m.role,
            content=m.content,
            timestamp=m.timestamp.isoformat(),
            sources=m.sources,
        )
        for m in conv.messages
    ]

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        messages=msg_items,
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete conversation session",
)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Deletes conversation and all its messages."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    db.delete(conv)
    db.commit()
    logger.info(f"Deleted conversation session: {conversation_id}")
    return {"status": "success", "message": f"Conversation '{conversation_id}' deleted."}
