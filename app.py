
import os
import json
import hashlib
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
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
            max-width: 980px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .topbar {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 1rem;
            padding: 0.25rem 0 0.75rem 0;
        }

        .logo {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
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
            color: #b5bfd9;
            font-size: 0.98rem;
            margin-top: 0.35rem;
        }

        .helper {
            color: #9aa5ce;
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 0.75rem 0 1.25rem 0;
        }

        .chip {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.10);
            border: 1px solid rgba(129, 140, 248, 0.18);
            color: #d6dcff;
            font-size: 0.9rem;
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
        for chunk in splitter.split_text(text):
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

    return collection


def retrieve(collection, query: str, n_results: int = 5) -> dict:
    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )


def build_history_text(messages: list[dict], limit: int = 6) -> str:
    recent = messages[-limit:]
    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(lines)


def build_prompt(query: str, results: dict, messages: list[dict]) -> str:
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

    history_text = build_history_text(messages)
    context = "\n\n".join(context_blocks)

    return f"""
You are Nftify, a grounded assistant for an NFT knowledge base.

Instructions:
- Answer using the retrieved context first.
- Use the recent conversation only to maintain continuity.
- Be concise, clear, and direct.
- If the context is insufficient, say so clearly.
- Do not invent facts.

Recent conversation:
{history_text}

Knowledge base context:
{context}

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
        st.markdown('</div>', unsafe_allow_html=True)


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
        <div class="topbar">
            <div class="logo">🖼️</div>
            <div>
                <div class="brand-title">Nftify</div>
                <div class="brand-subtitle">Chat with your knowledge base.</div>
                <div class="helper">Answers are grounded in your indexed content and can continue across turns.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="chips">
            <span class="chip">What is ClickOn?</span>
            <span class="chip">Summarize this collection</span>
            <span class="chip">What are the key features?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
        st.stop()

    if not data_file:
        st.error("Missing nfti_pages.jsonl in repo root or data folder.")
        st.stop()

    client = genai.Client(api_key=api_key)
    collection = build_collection(str(data_file))

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-wrap">
                <h3 style="margin-bottom:8px; color:white;">Start the conversation</h3>
                <div>Ask Nftify a question and continue naturally with follow-ups.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and idx in st.session_state.source_map:
                with st.expander("Sources"):
                    render_sources(st.session_state.source_map[idx])

    query = st.chat_input("Ask Nftify anything about your knowledge base")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    results = retrieve(collection, query=query, n_results=5)
                    answer = generate_answer(
                        client=client,
                        query=query,
                        results=results,
                        messages=st.session_state.messages[:-1],
                    )

                st.markdown(answer)
                assistant_index = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.source_map[assistant_index] = results

                with st.expander("Sources"):
                    render_sources(results)

            except genai_errors.ClientError as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    error_text = "Gemini quota is exhausted for this API key. Try again later or switch to a billed key."
                else:
                    error_text = f"Gemini request failed: {e}"

                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})

            except Exception as e:
                error_text = f"Something went wrong: {e}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})


if __name__ == "__main__":
    main()
