import hashlib
import json
import os
import tempfile
from typing import Any

import chromadb
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="NFTI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DEFAULT_DATA_PATHS = ["nfti_pages.jsonl", "data/nfti_pages.jsonl"]


st.markdown(
    """
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}
        #MainMenu, header, footer {visibility:hidden;}
        .block-container {max-width: 760px; padding-top: 2rem; padding-bottom: 2rem;}
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


def load_default_file_bytes() -> bytes | None:
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    return None


def read_pages(file_bytes: bytes) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for raw_line in file_bytes.decode("utf-8").splitlines():
        line = raw_line.strip()
        if line:
            pages.append(json.loads(line))
    return pages


@st.cache_resource(show_spinner=True)
def prepare_collection(file_bytes: bytes):
    pages = read_pages(file_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            documents.append(chunk)
            metadatas.append(
                {
                    "title": page.get("title", "") or "Untitled",
                    "url": page.get("url", ""),
                }
            )
            ids.append(f"{page.get('url', 'page')}#chunk-{i}")

    model = load_embedding_model()
    temp_dir = tempfile.mkdtemp(prefix="chroma_streamlit_")
    db = chromadb.PersistentClient(path=temp_dir)
    collection = db.get_or_create_collection(name=f"nfti_{hashlib.md5(file_bytes).hexdigest()[:12]}")

    batch_size = 64
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=embeddings)

    return collection


def answer_question(client, collection, query: str, n_results: int = 4) -> dict[str, Any]:
    model = load_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    sources = []
    context_parts = []

    for doc, meta in zip(docs, metas):
        if not doc.strip():
            continue
        sources.append({
            "title": meta.get("title", "Untitled"),
            "url": meta.get("url", ""),
            "text": doc,
        })
        context_parts.append(f"Title: {meta.get('title', '')}\nURL: {meta.get('url', '')}\nText: {doc}")

    if not context_parts:
        return {"answer": "I don't know based on the provided NFTI materials.", "sources": []}

    prompt = f"""
Answer only from the context below.
Be concise and direct.
If the answer is not supported by the context, say: I don't know based on the provided NFTI materials.

Question: {query}

Context:
{chr(10).join(context_parts)}
""".strip()

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = (response.text or "").strip()
    except genai_errors.ClientError as e:
        message = str(e)
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            answer = "Gemini quota is temporarily exhausted. Please try again shortly."
        else:
            raise

    return {"answer": answer, "sources": sources}


def main() -> None:
    api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        st.error("GEMINI_API_KEY is missing from Streamlit Secrets.")
        st.stop()

    file_bytes = load_default_file_bytes()
    if not file_bytes:
        st.error("nfti_pages.jsonl was not found in the repo root or data folder.")
        st.stop()

    client = get_gemini_client(api_key)
    collection = prepare_collection(file_bytes)

    st.title("NFTI Assistant")
    query = st.text_input("Ask a question", placeholder="Ask about Click-On Kaduna, DSFP, or NFTI")

    if st.button("Ask", use_container_width=True) and query.strip():
        with st.spinner("Thinking..."):
            result = answer_question(client, collection, query.strip())
        st.write(result["answer"])

        if result["sources"]:
            with st.expander("Sources"):
                for source in result["sources"][:3]:
                    st.markdown(f"**{source['title']}**")
                    if source["url"]:
                        st.markdown(f"[{source['url']}]({source['url']})")
                    st.caption(source["text"][:300] + ("..." if len(source["text"]) > 300 else ""))


if __name__ == "__main__":
    main()
