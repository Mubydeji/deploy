import os
import json
from pathlib import Path

import numpy as np
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Nftify",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- STYLE ----------------
st.markdown(
    """
    <style>
        /* Remove Streamlit UI junk */
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarCollapsedControl"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"] {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            height: 0rem;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 0.5rem !important;
            padding-bottom: 1.5rem;
        }

        /* Header */
        .brand-row {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 0.6rem;
        }

        .brand-logo {
            width: 50px;
            height: 50px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            background: linear-gradient(135deg, #7c3aed, #2563eb);
        }

        .brand-title {
            color: white;
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
        }

        .brand-subtitle {
            color: #c4cae0;
            font-size: 0.95rem;
        }

        /* Panels */
        .panel {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px;
        }

        .section-label {
            color: #a5b4fc;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        /* Chips */
        .chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .chip {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(129, 140, 248, 0.18);
            color: #dce2ff;
            font-size: 0.9rem;
        }

        /* Inputs */
        .stTextArea textarea {
            border-radius: 14px !important;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
        }

        /* Sources */
        .source-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 12px;
            margin-bottom: 8px;
        }

        .source-title {
            font-weight: 600;
            color: white;
        }

        .source-url {
            font-size: 0.85rem;
            color: #9aa5ce;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- DATA ----------------
DATA_CANDIDATES = [
    Path("nfti_pages.jsonl"),
    Path("data/nfti_pages.jsonl"),
]

def get_api_key():
    return st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

def find_data_file():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    return None

def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def row_to_text(row):
    return (
        f"{row.get('title','')}\n{row.get('content','')}",
        {"title": row.get("title",""), "url": row.get("url","")}
    )

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_index(file):
    rows = load_jsonl(Path(file))
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    docs, metas = [], []

    for r in rows:
        text, meta = row_to_text(r)
        for c in splitter.split_text(text):
            docs.append(c)
            metas.append(meta)

    emb = np.array(get_embedder().encode(docs), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    return {"docs": docs, "metas": metas, "emb": emb}

def retrieve(index, query, k=5):
    q = get_embedder().encode([query])[0]
    q /= np.linalg.norm(q)
    scores = index["emb"] @ q
    idx = np.argsort(scores)[::-1][:k]

    return {
        "documents": [[index["docs"][i] for i in idx]],
        "metadatas": [[index["metas"][i] for i in idx]],
    }

def render_sources(res):
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        st.markdown(f"""
        <div class="source-card">
            <div class="source-title">{i+1}. {meta.get("title","")}</div>
            <div class="source-url">{meta.get("url","")}</div>
            {doc[:200]}...
        </div>
        """, unsafe_allow_html=True)

# ---------------- APP ----------------
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.sources = {}

    api = get_api_key()
    data = find_data_file()

    # Header
    st.markdown("""
    <div class="brand-row">
        <div class="brand-logo">🤖</div>
        <div>
            <div class="brand-title">Nftify</div>
            <div class="brand-subtitle">Ask your NFT knowledge base</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not api or not data:
        st.error("Missing API key or dataset")
        return

    index = build_index(str(data))
    client = genai.Client(api_key=api)

    # Input
    col1, col2 = st.columns([2,1])

    with col1:
        query = st.text_area("", placeholder="Ask something...")
        ask = st.button("Ask Nftify", use_container_width=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Examples</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="chip-grid">
            <span class="chip">What is ClickOn?</span>
            <span class="chip">Summarize this</span>
            <span class="chip">Key features?</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Run
    if ask and query:
        st.session_state.messages.append(("user", query))

        res = retrieve(index, query)
        prompt = query + "\n\n" + "\n".join(res["documents"][0])

        try:
            r = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            ans = r.text
        except Exception:
            ans = "Error or quota exceeded"

        st.session_state.messages.append(("bot", ans))
        st.session_state.sources[len(st.session_state.messages)] = res

    # Chat display
    for i, (role, msg) in enumerate(st.session_state.messages):
        with st.chat_message("user" if role=="user" else "assistant"):
            st.write(msg)
            if role == "bot" and i in st.session_state.sources:
                with st.expander("Sources"):
                    render_sources(st.session_state.sources[i])


if __name__ == "__main__":
    main()
