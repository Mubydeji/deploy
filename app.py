
import os
import json
import hashlib
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional imports for retrieval
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page config + style
# -----------------------------
st.set_page_config(
    page_title="Nftify",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        .app-shell {
            background: linear-gradient(180deg, #0b1020 0%, #0e1326 100%);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 8px;
        }

        .brand-badge {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            background: linear-gradient(135deg, #7c3aed, #2563eb);
            box-shadow: 0 10px 30px rgba(37,99,235,0.35);
        }

        .brand-text h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
            color: white;
            letter-spacing: -0.02em;
        }

        .brand-text p {
            margin: 4px 0 0 0;
            color: #b8c0e0;
            font-size: 0.98rem;
        }

        .panel {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px;
            height: 100%;
        }

        .section-label {
            color: #a5b4fc;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        .answer-card {
            background: linear-gradient(180deg, rgba(37,99,235,0.10), rgba(255,255,255,0.02));
            border: 1px solid rgba(96,165,250,0.22);
            border-radius: 20px;
            padding: 20px;
        }

        .source-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 12px;
        }

        .source-title {
            font-weight: 700;
            color: white;
            margin-bottom: 6px;
        }

        .source-meta {
            color: #aab4d6;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }

        .empty-state {
            text-align: center;
            padding: 50px 20px;
            color: #b8c0e0;
        }

        .footer-note {
            color: #9aa5ce;
            font-size: 0.9rem;
            margin-top: 10px;
        }

        .stTextArea textarea {
            border-radius: 16px !important;
            min-height: 120px !important;
        }

        div[data-testid="stButton"] > button {
            border-radius: 14px !important;
            font-weight: 700 !important;
            padding: 0.65rem 1rem !important;
            margin-top: 1.7rem !important;
        }

        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(99,102,241,0.12);
            border: 1px solid rgba(129,140,248,0.18);
            color: #c7d2fe;
            font-size: 0.85rem;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        a {
            color: #93c5fd !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utilities
# -----------------------------
DATA_CANDIDATES = [
    Path("nfti_pages.jsonl"),
    Path("data/nfti_pages.jsonl"),
]

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_api_key() -> str | None:
    return st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))


def find_data_file() -> Path | None:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def row_to_text(row: dict) -> tuple[str, dict]:
    title = row.get("title") or row.get("name") or row.get("heading") or "Untitled"
    url = row.get("url") or row.get("source") or ""
    content = (
        row.get("content")
        or row.get("text")
        or row.get("body")
        or row.get("description")
        or ""
    )
    metadata = {
        "title": title,
        "url": url,
        "source": url or title,
    }
    text = f"Title: {title}\nURL: {url}\n\n{content}".strip()
    return text, metadata


@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=True)
def build_collection(data_file: str):
    path = Path(data_file)
    rows = load_jsonl(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    embedder = get_embedder()

    client = chromadb.Client()
    collection_name = "nftify_" + hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]
    collection = client.create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    idx = 0
    for row in rows:
        text, metadata = row_to_text(row)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"doc_{idx}")
            idx += 1

    embeddings = embedder.encode(documents, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection, len(rows), len(documents)


def retrieve(collection, query: str, n_results: int = 5):
    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results


def build_prompt(query: str, results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    context_blocks = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        title = meta.get("title", f"Source {i+1}")
        url = meta.get("url", "")
        context_blocks.append(
            f"[Source {i+1}]\nTitle: {title}\nURL: {url}\nContent:\n{doc}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are Nftify, a precise assistant for answering questions from the provided knowledge base.

Rules:
- Answer using the retrieved context.
- Be clear, direct, and helpful.
- If the context is not enough, say that clearly.
- Do not invent facts.
- When useful, summarize the relevant sources naturally.

Knowledge base context:
{context}

User question:
{query}
""".strip()


def generate_answer(client, query: str, results: dict) -> str:
    prompt = build_prompt(query, results)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text if getattr(response, "text", None) else "No answer returned."


def render_sources(results: dict):
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        st.info("No matching sources found.")
        return

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        title = meta.get("title", f"Source {i+1}")
        url = meta.get("url", "")
        snippet = doc[:350].strip()

        st.markdown('<div class="source-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="source-title">{i+1}. {title}</div>', unsafe_allow_html=True)
        if url:
            st.markdown(
                f'<div class="source-meta"><a href="{url}" target="_blank">{url}</a></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="source-meta">Internal knowledge snippet</div>', unsafe_allow_html=True)
        st.write(snippet + ("..." if len(doc) > 350 else ""))
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Main app
# -----------------------------
def main():
    api_key = get_api_key()
    data_file = find_data_file()

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="brand">
            <div class="brand-badge">🖼️</div>
            <div class="brand-text">
                <h1>Nftify</h1>
                <p>Ask about your NFT knowledge base and get grounded answers with source-backed context.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1], gap="large")

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Suggested questions</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div>
                <span class="pill">What is ClickOn?</span>
                <span class="pill">Summarize this collection</span>
                <span class="pill">What are the key features?</span>
                <span class="pill">How does this project work?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="footer-note">Nftify answers from the indexed knowledge base and shows supporting sources below.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Ask Nftify</div>', unsafe_allow_html=True)
        query = st.text_area(
            "Question",
            label_visibility="collapsed",
            placeholder="Ask a question about the NFT knowledge base...",
        )
        controls_left, controls_right = st.columns([1, 1], gap="small")
        with controls_left:
            top_k = st.slider("Sources to retrieve", min_value=3, max_value=8, value=5)
        with controls_right:
            st.write("")
            st.write("")
            ask_clicked = st.button("Ask Nftify", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not api_key:
        st.warning("Add GEMINI_API_KEY to Streamlit Secrets to use Nftify.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    if not data_file:
        st.warning("Add nfti_pages.jsonl to the repo root or data/ folder.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    client = genai.Client(api_key=api_key)

    try:
        collection, row_count, chunk_count = build_collection(str(data_file))
    except Exception as e:
        st.error("Failed to build the search index.")
        st.exception(e)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    if ask_clicked and query.strip():
        with st.spinner("Searching and composing answer..."):
            try:
                results = retrieve(collection, query=query.strip(), n_results=top_k)
                answer = generate_answer(client, query=query.strip(), results=results)

                st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("Sources", expanded=True):
                    render_sources(results)

            except genai_errors.ClientError as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    st.warning(
                        "Gemini quota is currently exhausted for this API key. Please try again later or switch to a billed key/project."
                    )
                else:
                    st.error("Gemini request failed.")
                    st.exception(e)
            except Exception as e:
                st.error("Something went wrong while answering your question.")
                st.exception(e)

    elif not query.strip():
        st.markdown(
            """
            <div class="empty-state">
                <h3 style="margin-bottom:8px; color:white;">Start with a question</h3>
                <div>Nftify will search your indexed NFT dataset, answer clearly, and show the supporting sources.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
