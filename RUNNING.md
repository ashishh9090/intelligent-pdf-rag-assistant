# Quick Running & Verification Guide

This document provides exact, battle-tested terminal commands to run and verify the **Intelligent PDF Knowledge Assistant – RAG** locally and in Docker.

---

## 1. Local Environment Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended: `uv` or `python3 -m venv`)

### Create & Activate Virtual Environment
```bash
cd rag-document-chatbot
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 2. Running Backend (FastAPI)

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Backend API URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Alternative ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `curl -s http://localhost:8000/health | jq`

---

## 3. Running Frontend (Streamlit)

In a separate terminal:
```bash
cd rag-document-chatbot
source .venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```
- **Streamlit Web UI**: `http://localhost:8501`

---

## 4. Running Automated Tests

Run the full pytest suite with verbose output:
```bash
source .venv/bin/activate
pytest -v
```

Run specific test modules:
```bash
# Test Ingestion & Page Extraction
pytest tests/test_chunking.py -v

# Test FAISS Vector Store Lifecycle & Search
pytest tests/test_retrieval.py -v

# Test PDF Upload, Hash Deduplication & Delete API
pytest tests/test_documents.py -v

# Test Multi-Turn Chat, Citations & Hallucination Refusal
pytest tests/test_chat.py -v
```

---

## 5. Running RAG Benchmark Evaluation

Execute the 20-question benchmark evaluation suite:
```bash
source .venv/bin/activate
python evaluation/evaluate.py
```
This tests:
- Retrieval Precision & Recall
- Hit Rate
- Faithfulness / Groundedness
- Citation Correctness
- Negative Hallucination Refusal Accuracy

Results are output to terminal and stored in `evaluation/results.json`.

---

## 6. Docker Deployment

### Build and Launch Everything via Docker Compose
```bash
docker compose up --build
```

### Stop Containers
```bash
docker compose down
```
