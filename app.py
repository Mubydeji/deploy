import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from google import genai
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


def chunk_documents(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return documents



def build_vector_store(documents: list[dict[str, Any]], collection_name: str):
    embedding_model = load_embedding_model()
    temp_dir = tempfile.mkdtemp(prefix="chroma_streamlit_")
    client_db = chromadb.PersistentClient(path=temp_dir)
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

    return collection


@st.cache_resource(show_spinner=True)
def prepare_knowledge_base(file_bytes: bytes):
    pages = read_pages_from_jsonl_bytes(file_bytes)
    documents = chunk_documents(pages)

    content_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    collection_name = f"nfti_kb_{content_hash}"
    collection = build_vector_store(documents, collection_name)
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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    sources = []
    for meta, chunk in zip(usable_metadata, usable_chunks):
        sources.append(
            {
                "title": meta.get("title", "Untitled"),
                "url": meta.get("url", ""),
                "passage": chunk,
            }
        )

    return {
        "answer": response.text.strip(),
        "sources": sources,
        "link": sources[0]["url"] if sources else None,
    }



def load_default_file_bytes() -> tuple[bytes | None, str | None]:
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read(), os.path.basename(path)
    return None, None



def load_secret_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""



def main():
    st.title("📚 NFTI Knowledge Assistant")
    st.caption("Ask questions grounded in your NFTI scraped materials.")

    api_key = load_secret_api_key()
    file_bytes, file_name = load_default_file_bytes()
    top_k = 4

    if not api_key:
        st.error(
            "Missing GEMINI_API_KEY. Add it to Streamlit Secrets before running this app."
        )
        st.code('GEMINI_API_KEY = "your_api_key_here"')
        st.stop()

    if not file_bytes or not file_name:
        st.error(
            "Missing nfti_pages.jsonl. Put the file in the repo root or in data/nfti_pages.jsonl before deployment."
        )
        st.code("deploy/\n├── app.py\n├── nfti_pages.jsonl\n└── requirements.txt")
        st.stop()

    client = get_gemini_client(api_key)

    with st.spinner("Preparing knowledge base..."):
        pages, documents, collection = prepare_knowledge_base(file_bytes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pages loaded", len(pages))
    col2.metric("Chunks created", len(documents))
    col3.metric("Data file", file_name)

    sample_prompts = [
        "What is Click-On Kaduna?",
        "What does DSFP stand for?",
        "Summarize NFTI in one paragraph.",
    ]

    st.markdown("### Ask a question")
    selected_prompt = st.selectbox("Try a sample prompt", [""] + sample_prompts)
    query = st.text_input(
        "Question",
        value=selected_prompt,
        placeholder="Ask something about the NFTI materials...",
    )

    if query:
        with st.spinner("Searching and generating answer..."):
            result = answer_question(client, collection, query=query, n_results=top_k)

        st.markdown("### Answer")
        st.write(result["answer"])

        if result.get("link"):
            st.markdown(f"**Best source:** {result['link']}")

        st.markdown("### Top sources")
        for idx, source in enumerate(result.get("sources", [])[:top_k], start=1):
            with st.expander(f"Source {idx}: {source['title'] or 'Untitled'}"):
                if source.get("url"):
                    st.markdown(f"**URL:** {source['url']}")
                st.write(source.get("passage", ""))


if __name__ == "__main__":
    main()
