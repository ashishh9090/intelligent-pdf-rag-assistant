import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import pymupdf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup project path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.database.models import Base, DocumentModel
from app.generation.rag_chain import RAGChain
from app.generation.llm import get_llm
from app.ingestion.chunker import Chunker
from app.ingestion.pdf_loader import PDFLoader
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.vector_store import FAISSVectorStore


def create_eval_documents(doc_dir: Path) -> Dict[str, Path]:
    """Generates evaluation benchmark PDF documents if they do not exist."""
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = {}

    p2024 = doc_dir / "annual_report_2024.pdf"
    if not p2024.exists():
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text(
            (50, 72),
            "Acme Corp Annual Report 2024.\n"
            "Executive Summary:\n"
            "Acme Corp is a leading provider of enterprise AI workflow orchestration and cloud platforms.\n"
            "The company was founded in 2018 and has expanded across North America and Europe.",
        )
        p2 = doc.new_page()
        p2.insert_text(
            (50, 72),
            "Financial Performance 2024:\n"
            "In fiscal year 2024, Acme Corp reported total revenue of $50 million, representing a 15% increase.\n"
            "Gross profit reached $30 million with a 60% gross margin.\n"
            "Research and Development spending was $10 million.",
        )
        p3 = doc.new_page()
        p3.insert_text(
            (50, 72),
            "Workforce & Operations:\n"
            "Total employee headcount reached 450 full-time team members by the end of 2024.\n"
            "Customer satisfaction score (CSAT) averaged 94% across all service tiers.",
        )
        doc.save(str(p2024))
        doc.close()
    pdf_paths["annual_report_2024.pdf"] = p2024

    p2025 = doc_dir / "annual_report_2025.pdf"
    if not p2025.exists():
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text(
            (50, 72),
            "Acme Corp Annual Report 2025.\n"
            "Strategic Growth:\n"
            "Accelerating enterprise adoption of generative knowledge systems.",
        )
        p2 = doc.new_page()
        p2.insert_text(
            (50, 72),
            "Financial Highlights 2025:\n"
            "In fiscal year 2025, Acme Corp reported total revenue of $65 million, marking a 30% increase.\n"
            "Operating income rose to $18 million.",
        )
        p3 = doc.new_page()
        p3.insert_text(
            (50, 72),
            "Workforce 2025:\n"
            "Total global headcount expanded to 600 employees with a new European hub.",
        )
        doc.save(str(p2025))
        doc.close()
    pdf_paths["annual_report_2025.pdf"] = p2025

    return pdf_paths


def run_evaluation(provider: str = "mock"):
    print("=" * 80)
    print(f"STARTING RAG BENCHMARK EVALUATION (20 QUESTIONS) [Provider: {provider}]")
    print("=" * 80)

    # 1. Setup isolated database and vector store for evaluation
    eval_dir = BASE_DIR / "data" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_db_path = eval_dir / "eval_chatbot.db"
    if eval_db_path.exists():
        eval_db_path.unlink()

    engine = create_engine(f"sqlite:///{eval_db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    eval_vectorstore_dir = eval_dir / "vectorstore"
    if eval_vectorstore_dir.exists():
        import shutil
        shutil.rmtree(eval_vectorstore_dir)
    eval_vectorstore_dir.mkdir(parents=True, exist_ok=True)

    vector_store = FAISSVectorStore(storage_dir=eval_vectorstore_dir)
    retriever_inst = SemanticRetriever(vector_store=vector_store)
    llm_inst = get_llm(provider)
    rag = RAGChain(retriever_instance=retriever_inst, llm_instance=llm_inst)

    # 2. Ingest benchmark documents
    doc_paths = create_eval_documents(eval_dir / "docs")
    chunker = Chunker(chunk_size=600, chunk_overlap=100)

    for fname, path in doc_paths.items():
        doc_id = f"eval_{Path(fname).stem}"
        pages = PDFLoader.load_pdf(path, filename=fname, document_id=doc_id)
        chunks = chunker.chunk_pages(pages)
        vector_store.add_chunks(chunks)

        record = DocumentModel(
            document_id=doc_id,
            filename=fname,
            file_hash=f"hash_{fname}",
            page_count=len(pages),
            chunk_count=len(chunks),
            file_path=str(path),
        )
        db.add(record)
    db.commit()

    print(f"Indexed {len(doc_paths)} evaluation documents ({vector_store.count()} chunks).\n")

    # 3. Load Questions
    questions_file = BASE_DIR / "evaluation" / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Evaluation counters
    total_q = len(questions)
    retrieval_hits = 0
    total_expected_retrieval_count = 0
    total_retrieved_relevant_count = 0
    total_retrieved_count = 0
    correct_refusals = 0
    expected_refusals = 0
    faithful_grounded_count = 0
    correct_citations_count = 0
    eval_records = []

    for item in questions:
        q_id = item["id"]
        cat = item["category"]
        question = item["question"]
        expected_docs = set(item.get("expected_documents", []))
        expected_pages = set(item.get("expected_pages", []))
        requires_refusal = item.get("requires_refusal", False)
        conv_context = item.get("conversation_context", [])

        # Create fresh conversation session
        conv = rag.memory.get_or_create_conversation(db)
        if conv_context:
            for c_msg in conv_context:
                rag.memory.add_message(db, conv.id, role=c_msg["role"], content=c_msg["content"])

        # Execute RAG Chain
        resp = rag.run(
            query=question,
            db=db,
            conversation_id=conv.id,
            top_k=3,
            similarity_threshold=0.35,
        )

        retrieved_docs = set(c.filename for c in resp.sources)
        retrieved_pages = set(c.page_number for c in resp.sources)

        # 1. Retrieval Metrics
        if requires_refusal:
            expected_refusals += 1
            # For unanswerable questions, hit rate is satisfied if 0 irrelevant chunks or refused
            hit = 1 if not resp.has_answer else 0
            precision = 1.0 if len(resp.sources) == 0 else 0.0
            recall = 1.0 if len(resp.sources) == 0 else 0.0
            if not resp.has_answer and "couldn't find this information" in resp.answer.lower():
                correct_refusals += 1
                grounded = True
            else:
                grounded = False
            citation_correct = 1.0 if len(resp.sources) == 0 else 0.0
        else:
            # Check overlap with expected docs
            doc_overlap = expected_docs.intersection(retrieved_docs)
            page_overlap = expected_pages.intersection(retrieved_pages)
            hit = 1 if len(doc_overlap) > 0 else 0

            precision = len(doc_overlap) / len(retrieved_docs) if retrieved_docs else 0.0
            recall = len(doc_overlap) / len(expected_docs) if expected_docs else 1.0

            total_retrieved_count += len(retrieved_docs)
            total_retrieved_relevant_count += len(doc_overlap)
            total_expected_retrieval_count += len(expected_docs)

            # Groundedness & Relevance
            grounded = resp.has_answer and ("[" in resp.answer or len(resp.sources) > 0)
            # Check citation correctness: citation must reference expected docs and pages
            citation_correct = 1.0 if (doc_overlap and (not expected_pages or page_overlap)) else 0.0

        if hit:
            retrieval_hits += 1
        if grounded:
            faithful_grounded_count += 1
        if citation_correct:
            correct_citations_count += 1

        eval_records.append({
            "id": q_id,
            "category": cat,
            "question": question,
            "rewritten_query": resp.rewritten_query,
            "answer": resp.answer[:120] + "..." if len(resp.answer) > 120 else resp.answer,
            "hit": bool(hit),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "grounded": grounded,
            "citation_correct": bool(citation_correct),
            "sources": [f"{s.filename}:P{s.page_number}" for s in resp.sources],
        })

        status_sym = "✅" if hit and grounded else "⚠️"
        print(f"[{q_id:02d}/20] {status_sym} [{cat}] {question[:55]}...")

    # Aggregate Metrics Calculation
    hit_rate = round(retrieval_hits / total_q, 4)
    avg_precision = round(
        total_retrieved_relevant_count / total_retrieved_count if total_retrieved_count > 0 else 1.0, 4
    )
    avg_recall = round(
        total_retrieved_relevant_count / total_expected_retrieval_count if total_expected_retrieval_count > 0 else 1.0, 4
    )
    groundedness_rate = round(faithful_grounded_count / total_q, 4)
    citation_accuracy = round(correct_citations_count / total_q, 4)
    refusal_accuracy = round(correct_refusals / expected_refusals if expected_refusals > 0 else 1.0, 4)

    metrics = {
        "total_questions_evaluated": total_q,
        "hit_rate": hit_rate,
        "retrieval_precision": avg_precision,
        "retrieval_recall": avg_recall,
        "faithfulness_groundedness": groundedness_rate,
        "citation_correctness": citation_accuracy,
        "negative_refusal_accuracy": refusal_accuracy,
        "breakdown": eval_records,
    }

    # Save to results.json
    results_path = BASE_DIR / "evaluation" / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY RESULTS:")
    print("=" * 80)
    print(f"Total Questions Evaluated:  {total_q}")
    print(f"Retrieval Hit Rate:         {hit_rate * 100:.1f}%")
    print(f"Retrieval Precision:        {avg_precision * 100:.1f}%")
    print(f"Retrieval Recall:           {avg_recall * 100:.1f}%")
    print(f"Faithfulness / Grounded:    {groundedness_rate * 100:.1f}%")
    print(f"Citation Correctness:       {citation_accuracy * 100:.1f}%")
    print(f"Hallucination Refusal Acc:  {refusal_accuracy * 100:.1f}%")
    print("=" * 80)
    print(f"Detailed evaluation saved to: {results_path}\n")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline accuracy on benchmark dataset.")
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "hf_local", "hf_api"],
        help="LLM provider for evaluation (default: 'mock' for high-speed benchmark)",
    )
    args = parser.parse_args()
    run_evaluation(provider=args.provider)
