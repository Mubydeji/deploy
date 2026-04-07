import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="NFTI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_DATA_PATHS = [
    "nfti_pages.jsonl",
    "data/nfti_pages.jsonl",
    "./nfti_pages.jsonl",
]

SAMPLE_QUESTIONS = [
    "What is Click-On Kaduna?",
    "What does DSFP stand for?",
    "Summarize NFTI in one paragraph.",
    "Who is this program for?",
    "What problem is the initiative trying to solve?",
    "List the most important takeaways about Click-On Kaduna.",
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"], [data-testid="collapsedControl"] {
                display: none !important;
            }
            #MainMenu, footer, header {
                visibility: hidden;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1d4ed8 100%);
                color: white;
                padding: 2rem;
                border-radius: 24px;
                margin-bottom: 1.25rem;
                box-shadow: 0 20px 60px rgba(15, 23, 42, 0.22);
            }
            .hero h1 {
                margin: 0 0 0.5rem 0;
                font-size: 2.2rem;
                line-height: 1.1;
            }
            .hero p {
                margin: 0;
                color: rgba(255,255,255,0.88);
                font-size: 1rem;
            }
            .stat-card {
                background: #ffffff;
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }
            .stat-label {
                font-size: 0.82rem;
                color: #64748b;
                margin-bottom: 0.35rem;
            }
            .stat-value {
                font-size: 1.45rem;
                font-weight: 700;
                color: #0f172a;
            }
            .section-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: #0f172a;
                margin: 0.25rem 0 0.75rem 0;
            }
            .answer-card {
                background: #ffffff;
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 22px;
                padding: 1.25rem;
                box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
            }
            .source-card {
                background: #ffffff;
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 1rem;
                margin-bottom: 0.9rem;
            }
            .source-title {
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.35rem;
            }
            .muted {
                color: #64748b;
                font-size: 0.92rem;
            }
            .pill {
                display: inline-block;
                padding: 0.35rem 0.6rem;
                border-radius: 999px;
                background: #eff6ff;
                color: #1d4ed8;
                font-size: 0.82rem;
                font-weight: 600;
                margin-right: 0.4rem;
                margin-bottom: 0.4rem;
            }
            .small-gap {
                margin-top: 0.35rem;
            }
            .stTextInput input {
                border-radius: 16px !important;
            }
            .stButton > button {
                border-radius: 14px !important;
                border: 1px solid rgba(15, 23, 42, 0.08) !important;
                padding: 0.55rem 0.95rem !important;
                font-weight: 600 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


def read_pages_from_jsonl_bytes(file_bytes: bytes) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for raw_line in file_bytes.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pages.append(json.loads(line))
    return pages


@st.cache_resource(show_spinner=True)
def prepare_knowledge_base(file_bytes: bytes):
    pages = read_pages_from_jsonl_bytes(file_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents: list[dict[str, Any]] = []

    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"{page.get('url', 'unknown')}#chunk-{i}",
                    "text": chunk,
                    "metadata": {
                        "title": page.get("title", ""),
                        "url": page.get("url", ""),
                        "h1": page.get("h1", ""),
                        "published_date": page.get("published_date", ""),
                        "chunk_index": i,
                    },
                }
            )

    embedding_model = load_embedding_model()
    temp_dir = tempfile.mkdtemp(prefix="chroma_streamlit_")
    client_db = chromadb.PersistentClient(path=temp_dir)
    content_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    collection_name = f"nfti_kb_{content_hash}"
    collection = client_db.get_or_create_collection(name=collection_name)

    batch_size = 64
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        ids = [doc["id"] for doc in batch]
        texts = [doc["text"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return pages, documents, collection


def search_knowledge_base(collection, query: str, n_results: int = 4) -> dict[str, Any]:
    embedding_model = load_embedding_model()
    query_embedding = embedding_model.encode([query], show_progress_bar=False)[0].tolist()
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)


def answer_question(client, collection, query: str, n_results: int = 4, min_passage_length: int = 120):
    results = search_knowledge_base(collection, query=query, n_results=n_results)
    retrieved_chunks = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]

    usable_chunks: list[str] = []
    usable_metadata: list[dict[str, Any]] = []
    for chunk, meta in zip(retrieved_chunks, retrieved_metadata):
        if chunk and len(chunk.strip()) >= min_passage_length:
            usable_chunks.append(chunk)
            usable_metadata.append(meta)

    if not usable_chunks:
        return {
            "answer": "I don't know based on the provided NFTI materials.",
            "sources": [],
            "link": None,
            "warning": None,
        }

    context = "\n\n---\n\n".join(
        [
            f"Title: {meta.get('title', '')}\n"
            f"URL: {meta.get('url', '')}\n"
            f"Passage: {chunk}"
            for chunk, meta in zip(usable_chunks, usable_metadata)
        ]
    )

    prompt = f"""
You are a concise question-answering assistant for Click-On Kaduna, DSFP, NFTI, and related materials.

Answer ONLY from the retrieved context.
Be brief, direct, and factual.
Do not invent facts.
If the answer is not supported by the context, say exactly:
I don't know based on the provided NFTI materials.

Question:
{query}

Context:
{context}
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer_text = (response.text or "").strip()
        warning = None
    except genai_errors.ClientError as e:
        message = str(e)
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            answer_text = "Gemini quota is temporarily exhausted for this API key. Please wait a moment and try again, or switch to a billed key."
            warning = "quota"
        else:
            raise

    sources = []
    for meta, chunk in zip(usable_metadata, usable_chunks):
        sources.append(
            {
                "title": meta.get("title", "Untitled") or "Untitled",
                "url": meta.get("url", ""),
                "passage": chunk,
                "published_date": meta.get("published_date", ""),
                "section": meta.get("h1", ""),
            }
        )

    return {
        "answer": answer_text,
        "sources": sources,
        "link": sources[0]["url"] if sources else None,
        "warning": warning,
    }


def load_default_file_bytes() -> tuple[bytes | None, str | None]:
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read(), os.path.basename(path)
    return None, None


def render_hero(file_name: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">Grounded answers</div>
            <div class="pill">Private knowledge base</div>
            <div class="pill">No sidebar setup</div>
            <h1>NFTI Knowledge Assistant</h1>
            <p>Ask focused questions about Click-On Kaduna, DSFP, NFTI, and related source material. Answers are generated from your backend JSONL knowledge base and Gemini secret.</p>
            <div class="small-gap muted">Current data file: <strong>{file_name}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats(pages: list[dict[str, Any]], documents: list[dict[str, Any]], file_name: str) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Pages loaded</div><div class="stat-value">{len(pages):,}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Search chunks</div><div class="stat-value">{len(documents):,}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Backend file</div><div class="stat-value" style="font-size:1rem">{file_name}</div></div>',
            unsafe_allow_html=True,
        )


def render_source_card(idx: int, source: dict[str, Any]) -> None:
    title = source.get("title", "Untitled")
    url = source.get("url", "")
    section = source.get("section", "")
    published_date = source.get("published_date", "")
    passage = source.get("passage", "")

    meta_bits = []
    if section:
        meta_bits.append(f"Section: {section}")
    if published_date:
        meta_bits.append(f"Published: {published_date}")
    meta_line = " • ".join(meta_bits)

    st.markdown('<div class="source-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='source-title'>{idx}. {title}</div>", unsafe_allow_html=True)
    if meta_line:
        st.markdown(f"<div class='muted'>{meta_line}</div>", unsafe_allow_html=True)
    if url:
        st.markdown(f"[Open source ↗]({url})")
    st.write(passage)
    st.markdown("</div>", unsafe_allow_html=True)


def set_query(text: str) -> None:
    st.session_state["query_input"] = text


def main() -> None:
    inject_css()

    secret_api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not secret_api_key:
        st.error("GEMINI_API_KEY is missing from Streamlit Secrets.")
        st.stop()

    file_bytes, file_name = load_default_file_bytes()
    if not file_bytes or not file_name:
        st.error("Backend data file not found. Add nfti_pages.jsonl to the repo root or data/nfti_pages.jsonl.")
        st.stop()

    client = get_gemini_client(secret_api_key)

    render_hero(file_name)

    with st.spinner("Preparing knowledge base..."):
        pages, documents, collection = prepare_knowledge_base(file_bytes)

    render_stats(pages, documents, file_name)

    st.markdown("<div class='section-title'>Ask better questions</div>", unsafe_allow_html=True)
    chip_cols = st.columns(3)
    for i, prompt in enumerate(SAMPLE_QUESTIONS):
        with chip_cols[i % 3]:
            st.button(prompt, use_container_width=True, on_click=set_query, args=(prompt,))

    st.markdown("<div class='section-title'>Search the knowledge base</div>", unsafe_allow_html=True)
    with st.form("ask_form", clear_on_submit=False):
        left, right = st.columns([5, 1.2])
        with left:
            query = st.text_input(
                "Question",
                key="query_input",
                placeholder="Ask something specific about Click-On Kaduna, DSFP, or NFTI...",
                label_visibility="collapsed",
            )
        with right:
            top_k = st.selectbox("Sources", [3, 4, 5, 6], index=1)
        submitted = st.form_submit_button("Ask now", use_container_width=True)

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if submitted and query.strip():
        with st.spinner("Searching and generating answer..."):
            result = answer_question(client, collection, query=query.strip(), n_results=int(top_k))
        st.session_state["latest_result"] = result
        st.session_state["latest_query"] = query.strip()
        st.session_state["history"] = [
            {"query": query.strip(), "answer": result["answer"]},
            *st.session_state["history"],
        ][:6]

    latest_result = st.session_state.get("latest_result")
    latest_query = st.session_state.get("latest_query")

    content_col, side_col = st.columns([1.55, 0.95])

    with content_col:
        if latest_result and latest_query:
            st.markdown("<div class='section-title'>Answer</div>", unsafe_allow_html=True)
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.caption(f"Question: {latest_query}")
            if latest_result.get("warning") == "quota":
                st.warning(latest_result["answer"])
            else:
                st.write(latest_result["answer"])
            if latest_result.get("link"):
                st.markdown(f"**Best source:** [Open primary source ↗]({latest_result['link']})")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Evidence</div>", unsafe_allow_html=True)
            for idx, source in enumerate(latest_result.get("sources", [])[: int(top_k)], start=1):
                render_source_card(idx, source)
        else:
            st.info("Choose a sample question or type your own to get started.")

    with side_col:
        st.markdown("<div class='section-title'>Recent questions</div>", unsafe_allow_html=True)
        history = st.session_state.get("history", [])
        if history:
            for item in history:
                with st.container(border=True):
                    st.caption(item["query"])
                    st.write(item["answer"][:180] + ("..." if len(item["answer"]) > 180 else ""))
        else:
            st.markdown("<div class='muted'>Your recent questions will show up here.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>How this works</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="source-card">
                <div class="muted">1. The app loads your backend JSONL knowledge base.</div>
                <div class="muted">2. It retrieves the most relevant passages with embeddings.</div>
                <div class="muted">3. Gemini answers only from those retrieved passages.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
