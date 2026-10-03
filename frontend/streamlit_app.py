import os
import time
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# ==============================================================================
# Configuration & Backend API Client
# ==============================================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Intelligent PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich styling and glassmorphism badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 999px;
        background-color: #E3F2FD;
        color: #1565C0;
        margin-right: 0.5rem;
    }
    .source-box {
        border-left: 3px solid #1E88E5;
        padding-left: 0.8rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: rgba(30, 136, 229, 0.05);
        border-radius: 0 4px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_check_health() -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def api_list_documents() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.sidebar.error(f"Error fetching documents: {e}")
    return []


def api_upload_files(files) -> Dict[str, Any]:
    upload_list = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    res = requests.post(f"{BACKEND_URL}/documents/upload", files=upload_list, timeout=120)
    return res.json() if res.status_code in [200, 201] else {"error": res.text}


def api_delete_document(doc_id: str) -> bool:
    try:
        res = requests.delete(f"{BACKEND_URL}/documents/{doc_id}", timeout=10)
        return res.status_code == 200
    except Exception:
        return False


def api_send_chat(
    query: str,
    conv_id: Optional[str],
    top_k: int,
    similarity_threshold: float,
    filter_docs: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    payload = {
        "query": query,
        "conversation_id": conv_id,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "filter_document_ids": filter_docs if filter_docs else None,
    }
    try:
        res = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=90)
        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"Chat error ({res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Network error communicating with backend: {e}")
    return None


def api_get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{BACKEND_URL}/conversations/{conv_id}", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


# ==============================================================================
# State Initialization
# ==============================================================================
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# ==============================================================================
# Sidebar: Control Panel & Document Management
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")

    # Backend Connection Status
    health = api_check_health()
    if health and health.get("status") == "healthy":
        st.success(f"🟢 Backend Online | Chunks: {health.get('indexed_chunks_count', 0)}")
    else:
        st.error("🔴 Backend Offline (Ensure FastAPI is running)")

    st.markdown("---")
    st.markdown("### 📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload one or multiple PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index PDFs", use_container_width=True, type="primary"):
            with st.spinner("Extracting pages, chunking, and computing embeddings..."):
                resp = api_upload_files(uploaded_files)
                if "error" in resp:
                    st.error(f"Upload failed: {resp['error']}")
                else:
                    indexed_count = len(resp.get("indexed", []))
                    st.success(f"✅ Successfully indexed {indexed_count} document(s)!")
                    if resp.get("skipped_duplicates"):
                        st.info(f"Duplicate files skipped: {', '.join(resp['skipped_duplicates'])}")
                    st.session_state["uploader_key"] += 1
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Indexed Documents")
    documents = api_list_documents()

    if not documents:
        st.caption("No documents indexed yet. Upload a PDF above to begin.")
    else:
        for doc in documents:
            col_text, col_del = st.columns([4, 1])
            with col_text:
                st.markdown(
                    f"**📄 {doc['filename']}**  \n"
                    f"<small>Pages: {doc['page_count']} | Chunks: {doc['chunk_count']}</small>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{doc['document_id']}", help=f"Delete {doc['filename']}"):
                    if api_delete_document(doc["document_id"]):
                        st.toast(f"Deleted {doc['filename']}")
                        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5, step=1)
    similarity_thresh = st.slider(
        "Similarity Threshold", min_value=0.0, max_value=0.8, value=0.35, step=0.05
    )

    # Multi-document selection filter
    selected_doc_filter = None
    if documents:
        doc_options = {d["filename"]: d["document_id"] for d in documents}
        chosen_filter_names = st.multiselect(
            "Filter by Specific Documents",
            options=list(doc_options.keys()),
            help="Leave empty to search across all uploaded documents.",
        )
        if chosen_filter_names:
            selected_doc_filter = [doc_options[name] for name in chosen_filter_names]

    st.markdown("---")
    if st.button("🔄 New Chat Session", use_container_width=True):
        st.session_state["conversation_id"] = None
        st.session_state["messages"] = []
        st.rerun()

# ==============================================================================
# Main View: Header & Chat Interface
# ==============================================================================
st.markdown('<div class="main-title">Intelligent PDF Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production RAG System with Semantic Retrieval, Strict Document Grounding, & Verifiable Citations</div>',
    unsafe_allow_html=True,
)

# Architectural Badges
st.markdown(
    """
    <span class="badge">Dense Embedding: BAAI/bge-base-en-v1.5</span>
    <span class="badge">Vector Database: FAISS</span>
    <span class="badge">Chunker: RecursiveCharacterTextSplitter</span>
    <span class="badge">Memory: Multi-Turn SQLite</span>
    """,
    unsafe_allow_html=True,
)
st.write("")

# Render conversation messages
for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and sources:
            st.markdown("##### 📌 Verified Citations:")
            for s in sources:
                st.markdown(
                    f"📄 **{s.get('filename')}** — Page {s.get('page_number')} "
                    f"*(Similarity: {s.get('similarity_score', 0):.2f})*"
                )

            with st.expander("🔍 Inspect Retrieved Context Snippets", expanded=False):
                for s in sources:
                    st.markdown(
                        f"**[Source {s.get('source_index', '')}: {s.get('filename')}, Page {s.get('page_number')}]**"
                    )
                    st.markdown(f'<div class="source-box">{s.get("snippet", "")}</div>', unsafe_allow_html=True)

# User Chat Input
user_input = st.chat_input("Ask a question about your uploaded documents...")

if user_input:
    # Append user question to UI
    st.session_state["messages"].append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Query Backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and formulating answer..."):
            resp = api_send_chat(
                query=user_input,
                conv_id=st.session_state["conversation_id"],
                top_k=top_k,
                similarity_threshold=similarity_thresh,
                filter_docs=selected_doc_filter,
            )

        if resp:
            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
            st.session_state["conversation_id"] = resp.get("conversation_id")

            st.markdown(answer)

            if sources:
                st.markdown("##### 📌 Verified Citations:")
                for s in sources:
                    st.markdown(
                        f"📄 **{s.get('filename')}** — Page {s.get('page_number')} "
                        f"*(Similarity: {s.get('similarity_score', 0):.2f})*"
                    )

                with st.expander("🔍 Inspect Retrieved Context Snippets", expanded=False):
                    for s in sources:
                        st.markdown(
                            f"**[Source {s.get('source_index', '')}: {s.get('filename')}, Page {s.get('page_number')}]**"
                        )
                        st.markdown(f'<div class="source-box">{s.get("snippet", "")}</div>', unsafe_allow_html=True)

            # Store in session state
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
