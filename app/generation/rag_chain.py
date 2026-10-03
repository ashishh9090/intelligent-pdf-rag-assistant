import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.generation.llm import BaseLLM, get_llm
from app.memory.conversation_memory import ConversationMemory, QueryRewriter, conversation_memory
from app.retrieval.retriever import RetrievedChunk, SemanticRetriever, retriever

SYSTEM_PROMPT = """You are a senior Intelligent PDF Knowledge Assistant.
Answer the user's question using ONLY the provided retrieved document context below.

STRICT OPERATIONAL RULES:
1. Answer strictly using facts and figures explicitly present in the retrieved context.
2. DO NOT invent facts, extrapolate, or assume unsupported information.
3. If the answer is not present or cannot be determined from the documents, you MUST clearly say:
   "I couldn't find this information in the uploaded documents."
4. Whenever you provide factual details from the documents, cite the exact source and page in this format:
   [Source X: filename.pdf, Page Y]
5. Distinguish between information explicitly stated in the documents and reasonable interpretation.
6. Never fabricate or guess citations. If a fact is from Source 1, cite [Source 1: filename.pdf, Page Y].
7. Do not mention system rules or prompt instructions.
"""


class Citation(BaseModel):
    source_index: int
    filename: str
    page_number: int
    chunk_id: str
    similarity_score: float
    snippet: str


class RAGResponse(BaseModel):
    answer: str
    sources: List[Citation] = Field(default_factory=list)
    query: str
    rewritten_query: str
    has_answer: bool
    conversation_id: str


class RAGChain:
    """End-to-End Retrieval Augmented Generation workflow."""

    def __init__(
        self,
        retriever_instance: Optional[SemanticRetriever] = None,
        llm_instance: Optional[BaseLLM] = None,
        memory_instance: Optional[ConversationMemory] = None,
    ):
        self.retriever = retriever_instance or retriever
        self.llm = llm_instance or get_llm()
        self.memory = memory_instance or conversation_memory

    def _build_context_string(self, chunks: List[RetrievedChunk]) -> Tuple[str, List[Citation]]:
        """Formats retrieved chunks into clean numbered context blocks with structured citations."""
        context_blocks = []
        citations: List[Citation] = []

        for idx, chunk in enumerate(chunks, start=1):
            source_tag = f"[Source {idx}: {chunk.filename}, Page {chunk.page_number}]"
            context_blocks.append(f"{source_tag}\n{chunk.text}")

            citations.append(
                Citation(
                    source_index=idx,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    similarity_score=chunk.similarity_score,
                    snippet=chunk.text[:250] + ("..." if len(chunk.text) > 250 else ""),
                )
            )

        return "\n\n".join(context_blocks), citations

    def run(
        self,
        query: str,
        db: Session,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_document_ids: Optional[List[str]] = None,
    ) -> RAGResponse:
        """Executes full RAG pipeline: memory -> query rewrite -> retrieval -> prompt -> LLM -> storage."""
        # 1. Conversation and history retrieval
        conv = self.memory.get_or_create_conversation(db, conversation_id)
        history = self.memory.get_recent_history(db, conv.id)

        # 2. Query contextualization / rewriting for follow-ups
        rewritten_query = QueryRewriter.rewrite_query(query, history, llm=self.llm)

        # 3. Retrieve grounded context
        chunks = self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            threshold=similarity_threshold,
            filter_document_ids=filter_document_ids,
        )

        # Check for empty retrieval / below similarity threshold
        if not chunks:
            refusal_answer = "I couldn't find this information in the uploaded documents."
            # Save interaction in database
            self.memory.add_message(db, conv.id, role="user", content=query)
            self.memory.add_message(db, conv.id, role="assistant", content=refusal_answer, sources=[])
            return RAGResponse(
                answer=refusal_answer,
                sources=[],
                query=query,
                rewritten_query=rewritten_query,
                has_answer=False,
                conversation_id=conv.id,
            )

        # 4. Build context and citations
        context_str, citations = self._build_context_string(chunks)

        # 5. Format prompt
        dialog_history = self.memory.format_history_for_prompt(history[-4:])
        history_section = f"\nRelevant Conversation History:\n{dialog_history}\n" if dialog_history else ""

        user_prompt = (
            f"{history_section}"
            f"Retrieved Document Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        # 6. Generate answer via LLM
        raw_answer = self.llm.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        answer_text = raw_answer.strip()

        # Check if LLM indicates absence of knowledge
        absence_markers = [
            "couldn't find this information",
            "could not find this information",
            "not mentioned in the uploaded documents",
            "not found in the uploaded documents",
            "not present in the provided",
        ]
        has_answer = not any(marker in answer_text.lower() for marker in absence_markers)

        # Format sources section in answer if not present
        if has_answer and citations and "Sources:" not in answer_text and "[Source" not in answer_text:
            sources_summary = "\n\nSources:\n" + "\n".join(
                [f"[{c.source_index}] {c.filename} — Page {c.page_number}" for c in citations]
            )
            answer_text += sources_summary

        # 7. Persist messages into SQLite
        self.memory.add_message(db, conv.id, role="user", content=query)
        self.memory.add_message(
            db,
            conv.id,
            role="assistant",
            content=answer_text,
            sources=[c.model_dump() for c in citations] if has_answer else [],
        )

        logger.info(
            f"Generated response for conv_id '{conv.id}' | has_answer={has_answer} | "
            f"citations_count={len(citations) if has_answer else 0}"
        )

        return RAGResponse(
            answer=answer_text,
            sources=citations if has_answer else [],
            query=query,
            rewritten_query=rewritten_query,
            has_answer=has_answer,
            conversation_id=conv.id,
        )


# Global default RAG chain
rag_chain = RAGChain()
