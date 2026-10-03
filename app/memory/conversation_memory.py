import re
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.database.models import Conversation, Message


class ConversationMemory:
    """Manages multi-turn conversation history and context assembly from SQLite."""

    def __init__(self, max_turns: Optional[int] = None):
        self.max_turns = max_turns or settings.MAX_HISTORY_TURNS

    def get_or_create_conversation(self, db: Session, conversation_id: Optional[str] = None) -> Conversation:
        """Retrieves existing conversation by ID or creates a new conversation."""
        if conversation_id:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conv:
                return conv

        new_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        conv = Conversation(id=new_id, title="New Conversation")
        db.add(conv)
        db.commit()
        db.refresh(conv)
        logger.info(f"Created new conversation session: {new_id}")
        return conv

    def add_message(
        self,
        db: Session,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Appends a new message to the conversation."""
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        msg = Message(
            id=msg_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        if sources:
            msg.sources = sources

        db.add(msg)
        # Update conversation title from first user message if it's still default
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv and conv.title == "New Conversation" and role == "user":
            conv.title = content[:60].strip()

        db.commit()
        db.refresh(msg)
        return msg

    def get_recent_history(
        self,
        db: Session,
        conversation_id: str,
        limit_turns: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieves recent turns as [{"role": "user"|"assistant", "content": "..."}].
        Respects max_history_turns to prevent prompt saturation.
        """
        limit = limit_turns or self.max_turns
        # Each turn consists of user + assistant message (2 messages)
        max_messages = limit * 2

        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.desc())
            .limit(max_messages)
            .all()
        )
        # Return in chronological order
        ordered = list(reversed(messages))
        return [{"role": m.role, "content": m.content} for m in ordered]

    def format_history_for_prompt(self, history: List[Dict[str, str]]) -> str:
        """Formats conversation history into a clean dialog transcript."""
        if not history:
            return ""
        formatted = []
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg['content']}")
        return "\n".join(formatted)


class QueryRewriter:
    """
    Intelligently rewrites conversational follow-up questions into standalone search queries.
    Prevents conversation history from overriding document retrieval while maintaining context.
    """

    PRONOUN_PATTERN = re.compile(
        r"\b(it|its|they|them|their|this|that|these|those|he|she|him|her|the same|also|previous)\b",
        re.IGNORECASE,
    )

    @classmethod
    def is_likely_followup(cls, query: str, history: List[Dict[str, str]]) -> bool:
        """Determines if the current query references earlier conversation entities."""
        if not history:
            return False
        # If query has pronoun or is very short (under 5 words), likely a follow up
        if cls.PRONOUN_PATTERN.search(query):
            return True
        if len(query.strip().split()) <= 4 and not query.lower().startswith(("what", "who", "where", "when")):
            return True
        return False

    @classmethod
    def rewrite_query(
        cls,
        current_query: str,
        history: List[Dict[str, str]],
        llm: Any = None,
    ) -> str:
        """
        Rewrites a contextual follow-up question into an independent semantic search query.
        Never exposes internal chain-of-thought.
        """
        if not history:
            return current_query.strip()

        # If not a follow-up, return directly to avoid unnecessary LLM latency
        if not cls.is_likely_followup(current_query, history):
            return current_query.strip()

        # If an LLM is available, use it for contextual rewriting
        if llm is not None:
            prompt = (
                "Given the following conversation history and a follow-up question, "
                "rephrase the follow-up question to be a standalone, self-contained search query. "
                "Do NOT answer the question. Only output the rewritten search query with no quotes or preamble.\n\n"
                f"Conversation History:\n"
            )
            # Take last 2-4 messages for history context
            for msg in history[-4:]:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            prompt += f"Follow-up Question: {current_query}\nStandalone Search Query:"

            try:
                rewritten = llm.generate(prompt=prompt, max_new_tokens=64)
                rewritten = rewritten.strip().strip('"').strip("'").split("\n")[0]
                if rewritten and len(rewritten) > 3:
                    logger.info(f"Rewrote query '{current_query}' -> '{rewritten}'")
                    return rewritten
            except Exception as e:
                logger.warning(f"LLM query rewriting encountered error: {e}. Falling back to heuristic.")

        # Heuristic fallback if LLM is unavailable: synthesize with the latest user topic
        last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
        if last_user_msgs:
            last_query = last_user_msgs[-1]
            # Replace pronouns with key topic if found
            rewritten_heuristic = f"{current_query} (context: {last_query})"
            logger.info(f"Heuristic query rewrite: '{current_query}' -> '{rewritten_heuristic}'")
            return rewritten_heuristic

        return current_query.strip()


conversation_memory = ConversationMemory()
