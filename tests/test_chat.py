from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.generation.llm import MockLLM
from app.generation.rag_chain import rag_chain
from app.memory.conversation_memory import QueryRewriter


def test_query_rewriter():
    history = [
        {"role": "user", "content": "What was the company revenue in 2024?"},
        {"role": "assistant", "content": "The revenue was $50 million."},
    ]
    follow_up = "How much did it increase?"

    # Without LLM (heuristic mode)
    rewritten = QueryRewriter.rewrite_query(follow_up, history, llm=None)
    assert "increase" in rewritten.lower()
    assert "revenue" in rewritten.lower()

    # Query with no pronouns and clear question
    non_follow_up = "Where is the research headquarters located?"
    assert QueryRewriter.rewrite_query(non_follow_up, history, llm=None) == non_follow_up


def test_chat_factual_question_and_citations(client: TestClient, sample_pdf_2024: Path):
    # Upload document
    with open(sample_pdf_2024, "rb") as f:
        client.post(
            "/documents/upload",
            files=[("files", ("annual_report_2024.pdf", f.read(), "application/pdf"))],
        )

    # Chat query
    payload = {
        "query": "What was Acme Corp's revenue in fiscal year 2024?",
        "top_k": 3,
        "similarity_threshold": 0.2,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_answer"] is True
    assert len(data["sources"]) > 0

    # Verify citation attributes
    top_source = data["sources"][0]
    assert "2024.pdf" in top_source["filename"]
    assert top_source["page_number"] == 2
    assert "50 million" in top_source["snippet"]


def test_chat_hallucination_refusal(client: TestClient):
    """
    Requirement 17: Hallucination Test.
    System must refuse to answer when information is absent from documents.
    """
    payload = {
        "query": "What is the secret recipe for Martian rocket fuel invented in 1840?",
        "top_k": 3,
        "similarity_threshold": 0.45,  # High threshold for completely absent topic
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_answer"] is False
    assert len(data["sources"]) == 0
    assert "I couldn't find this information in the uploaded documents." in data["answer"]


def test_multi_turn_conversation_and_api(client: TestClient, sample_pdf_2024: Path):
    # Upload document
    with open(sample_pdf_2024, "rb") as f:
        client.post(
            "/documents/upload",
            files=[("files", ("annual_report_2024.pdf", f.read(), "application/pdf"))],
        )

    # Turn 1
    res1 = client.post(
        "/chat",
        json={"query": "What was the employee headcount at Acme Corp?"},
    )
    assert res1.status_code == 200
    conv_id = res1.json()["conversation_id"]
    assert conv_id != ""

    # Turn 2 (Follow-up using the same conversation_id)
    res2 = client.post(
        "/chat",
        json={
            "query": "What was the customer satisfaction score?",
            "conversation_id": conv_id,
        },
    )
    assert res2.status_code == 200
    assert res2.json()["conversation_id"] == conv_id

    # Retrieve conversation history
    conv_res = client.get(f"/conversations/{conv_id}")
    assert conv_res.status_code == 200
    conv_data = conv_res.json()
    assert conv_data["id"] == conv_id
    assert len(conv_data["messages"]) == 4  # 2 user + 2 assistant messages

    # Delete conversation
    del_res = client.delete(f"/conversations/{conv_id}")
    assert del_res.status_code == 200

    # Ensure deleted
    assert client.get(f"/conversations/{conv_id}").status_code == 404
