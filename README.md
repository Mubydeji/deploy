# Nftify — Domain-Specific RAG Question-Answering System
### Grounded AI answers from a curated NFTI knowledge base

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

---

## Overview

Nftify is a domain-specific AI assistant built on a full Retrieval-Augmented Generation pipeline. It answers questions about Click-On Kaduna, the DSFP programme, and NFTI using a curated knowledge base — eliminating the hallucination problem that makes generic LLMs unreliable for specialised queries.

Instead of asking a general-purpose LLM to recall facts it may have never seen, Nftify retrieves the most relevant content from a vector-indexed knowledge base first, then generates a grounded, source-backed answer.

---

## Live Demo

🔗 [nftifybot.streamlit.app](https://nftifybot.streamlit.app)

---

## The Problem

Generic LLMs have no reliable knowledge of niche programmes like NFTI or Click-On Kaduna. When asked, they either hallucinate plausible-sounding but wrong answers, or admit they do not know. Neither is useful.

Nftify fixes this by grounding every answer in a verified, domain-specific knowledge base. If the answer is not in the knowledge base, Nftify says so — it does not invent one.

---

## Architecture

    User Query
        ↓
    Sentence Transformer (all-MiniLM-L6-v2)
        ↓ Query embedding
    ChromaDB Vector Index
        ↓ Top-K semantic retrieval
    Context Assembly
        ↓ Retrieved chunks + metadata
    Gemini 2.5 Flash
        ↓ Grounded generation
    Answer + Sources

---

## Pipeline Details

**1. Data Ingestion**
Web scraping with Trafilatura to extract clean text from NFTI-related pages. Raw content stored as structured JSONL with title, URL, and body fields.

**2. Text Chunking**
RecursiveCharacterTextSplitter with 900-character chunks and 150-character overlap. Overlap prevents context loss at chunk boundaries.

**3. Embedding and Indexing**
Sentence Transformers (all-MiniLM-L6-v2) for dense vector embeddings. ChromaDB for in-memory vector storage and similarity search. Embeddings cached on first load — no recomputation on repeat queries.

**4. Retrieval**
Semantic similarity search returns top-K most relevant chunks. Adjustable retrieval size (3–8 sources) via UI slider.

**5. Generation**
Retrieved chunks assembled into a structured prompt. Gemini 2.5 Flash generates a grounded, source-aware answer. Model instructed to acknowledge gaps rather than hallucinate.

---

## Project Structure

    nftify-rag/
    ├── app.py                  # Full Streamlit application
    ├── nfti_pages.jsonl        # Curated NFTI knowledge base
    ├── requirements.txt        # Dependencies
    └── README.md

---

## Key Design Decisions

**Why RAG over fine-tuning?**
Fine-tuning is expensive, slow to update, and still hallucinates. RAG is modular — update the knowledge base without retraining anything.

**Why all-MiniLM-L6-v2?**
Fast, lightweight, and strong on semantic similarity for short to medium text chunks. Runs without a GPU.

**Why ChromaDB?**
Simple, in-memory, no infrastructure required. Right tool for a focused single-domain knowledge base at this scale.

**Why chunk overlap?**
Answers often span chunk boundaries. 150-character overlap ensures no key sentence is split across two non-adjacent chunks.

---

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | UI and deployment |
| Gemini 2.5 Flash | Answer generation |
| Sentence Transformers | Query and document embeddings |
| ChromaDB | Vector storage and retrieval |
| LangChain Text Splitters | Recursive chunking |
| Trafilatura | Web scraping and content extraction |

---

## Author

**Mubarak Adesola Adedeji**
Data Analyst · AI Developer | Python · SQL · R · Power BI
[LinkedIn](https://linkedin.com/in/mubarak-adedeji-776804273) · [GitHub](https://github.com/Mubydeji)

---

## License

MIT License — free to use, adapt, and build on with attribution.
