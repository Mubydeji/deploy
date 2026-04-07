import os
import json
from pathlib import Path

import numpy as np
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Nftify",
    page_icon="🤖",
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
            max-width: 1100px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 1.2rem;
        }

        .brand-logo {
            width: 52px;
            height: 52px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            background: linear-gradient(135deg, #7c3aed, #2563eb);
            box-shadow: 0 10px 28px rgba(37, 99, 235, 0.28);
            flex-shrink: 0;
        }

        .brand-title {
            color: white;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin: 0;
        }

        .brand-subtitle {
            color: #c4cae0;
            font-size: 1rem;
            margin-top: 0.35rem;
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
            margin-bottom: 10px;
        }

        .chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .chip {
            display: inline-block;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.10);
            border: 1px solid rgba(129, 140, 248, 0.18);
            color: #dce2ff;
            font-size: 0.95rem;
        }

        .helper {
            color: #9aa5ce;
            font-size: 0.95rem;
            margin-top: 1rem;
            line-height: 1.6;
        }

        .source-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }

        .source-title {
            font-weight: 700;
            margin-bottom: 6px;
            color: white;
        }

        .source-url {
            font-size: 0.88rem;
            color: #a8b2d4;
            margin-bottom: 8px;
            word-break: break-all;
        }

        .empty-wrap {
            border: 1px dashed rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 2.2rem 1.25rem;
            text-align: center;
            color: #b8c0e0;
            margin-top: 1rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 14px !important;
            font-weight: 700 !important;
            min-height: 48px !important;
        }

        .stTextArea textarea {
            border-radius: 16px !important;
            min-height: 150px !important;
        }

        .stChatMessage {
            border-radius: 18px;
        }

        a {
            color: #93c5fd !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
            if line:
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
    metadata = {"title": title, "url": url, "source": url or title}
    text = f"Title: {title}\nURL: {url}\n\n{content}".strip()
    return text, metadata


@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=True)
def build_index(data_file: str):
    path = Path(data_file)
    rows = load_jsonl(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    metadatas = []

    for row in rows:
        text, metadata = row_to_text(row)
        for chunk in splitter.split_text(text):
            documents.append(chunk)
            metadatas.append(metadata)

    if not documents:
        raise ValueError("No documents found in nfti_pages.jsonl")

    embedder = get_embedder()
    embeddings = np.array(embedder.encode(documents, show_progress_bar=False), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return {"documents": documents, "metadatas": metadatas, "embeddings": embeddings}


def retrieve(index: dict, query: str, n_results: int = 5) -> dict:
    embedder = get_embedder()
    query_vec = np.array(embedder.encode([query], show_progress_bar=False)[0], dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        norm = 1.0
    query_vec = query_vec / norm
    scores = index["embeddings"] @ query_vec
    top_idx = np.argsort(scores)[::-1][:n_results]
    return {
        "documents": [[index["documents"][i] for i in top_idx]],
        "metadatas": [[index["metadatas"][i] for i in top_idx]],
        "distances": [[float(scores[i]) for i in top_idx]],
    }


def build_history_text(messages: list[dict], limit: int = 6) -> str:
    lines = []
    for msg in messages[-limit:]:
        content = msg.get("content", "").strip()
        if content:
            lines.append(f'{msg.get("role", "user").title()}: {content}')
    return "\n".join(lines)


def build_prompt(query: str, results: dict, messages: list[dict]) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    context_blocks = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        title = meta.get("title", f"Source {i+1}")
        url = meta.get("url", "")
        context_blocks.append(f"[Source {i+1}]\nTitle: {title}\nURL: {url}\nContent:\n{doc}")

    return f"""
You are Nftify, a grounded assistant for an NFT knowledge base.

Instructions:
- Answer using the retrieved context first.
- Use the recent conversation only to maintain continuity.
- Be concise, clear, and direct.
- If the context is insufficient, say so clearly.
- Do not invent facts.

Recent conversation:
{build_history_text(messages)}

Knowledge base context:
{chr(10).join(context_blocks)}

User question:
{query}
""".strip()


def generate_answer(client, query: str, results: dict, messages: list[dict]) -> str:
    prompt = build_prompt(query, results, messages)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text if getattr(response, "text", None) else "No answer returned."


def render_sources(results: dict):
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        st.info("No sources found.")
        return

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        title = meta.get("title", f"Source {i+1}")
        url = meta.get("url", "")
        snippet = doc[:320].strip()

        st.markdown('<div class="source-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="source-title">{i+1}. {title}</div>', unsafe_allow_html=True)
        if url:
            st.markdown(
                f'<div class="source-url"><a href="{url}" target="_blank">{url}</a></div>',
                unsafe_allow_html=True,
            )
        st.write(snippet + ("..." if len(doc) > 320 else ""))
        st.markdown("</div>", unsafe_allow_html=True)


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "source_map" not in st.session_state:
        st.session_state.source_map = {}


def main():
    init_state()

    api_key = get_api_key()
    data_file = find_data_file()

    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-logo">🤖</div>
            <div>
                <div class="brand-title">Nftify</div>
                <div class="brand-subtitle">Ask about your NFT knowledge base and get grounded answers with source-backed context.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.7, 1], gap="large")

    with top_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Ask Nftify</div>', unsafe_allow_html=True)
        query = st.text_area(
            "Question",
            label_visibility="collapsed",
            placeholder="Ask a question about the NFT knowledge base...",
        )
        action_col, slider_col = st.columns([1.15, 1], gap="medium")
        with action_col:
            ask_clicked = st.button("Ask Nftify", type="primary", use_container_width=True)
        with slider_col:
            top_k = st.slider("Sources to retrieve", min_value=3, max_value=8, value=5)
        st.markdown('</div>', unsafe_allow_html=True)

    with top_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Suggested questions</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="chip-grid">
                <span class="chip">What is ClickOn?</span>
                <span class="chip">Summarize this collection</span>
                <span class="chip">What are the key features?</span>
                <span class="chip">How does this project work?</span>
            </div>
            <div class="helper">Nftify answers from the indexed knowledge base and shows supporting sources below.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if not api_key:
        st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
        st.stop()

    if not data_file:
        st.error("Missing nfti_pages.jsonl in repo root or data folder.")
        st.stop()

    index = build_index(str(data_file))
    client = genai.Client(api_key=api_key)

    if ask_clicked and query.strip():
        st.session_state.messages.append({"role": "user", "content": query})

        try:
            with st.spinner("Thinking..."):
                results = retrieve(index, query=query, n_results=top_k)
                answer = generate_answer(client, query, results, st.session_state.messages[:-1])

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.source_map[len(st.session_state.messages) - 1] = results

        except genai_errors.ClientError as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                answer = "Gemini quota is exhausted for this API key. Try again later or switch to a billed key."
            else:
                answer = f"Gemini request failed: {e}"
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            answer = f"Something went wrong: {e}"
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        st.markdown("")
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and idx in st.session_state.source_map:
                    with st.expander("Sources"):
                        render_sources(st.session_state.source_map[idx])
    else:
        st.markdown(
            """
            <div class="empty-wrap">
                <h3 style="margin-bottom:8px; color:white;">Start the conversation</h3>
                <div>Ask Nftify a question to search the knowledge base and continue with follow-ups.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
